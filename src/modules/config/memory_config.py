#!/usr/bin/env python3
"""
Memory Optimization Configuration for BLIP3-o Multi-GPU Training
Provides adaptive memory settings based on available hardware and dataset size

Key features:
1. Automatic memory limit detection
2. Adaptive batch size and cache settings
3. GPU memory optimization
4. Shard management strategies
"""

import torch
import psutil
import os
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class MemoryConfig:
    """Memory optimization configuration"""
    # Batch size settings
    batch_size_per_gpu: int
    gradient_accumulation_steps: int
    max_shard_cache: int
    samples_per_shard_load: int
    
    # Memory limits
    gpu_memory_limit_gb: float
    cpu_memory_limit_gb: float
    max_workers: int
    
    # Optimization flags
    use_fp16: bool
    use_gradient_checkpointing: bool
    pin_memory: bool
    
    # Safety margins
    memory_safety_margin: float = 0.85
    gpu_memory_safety_margin: float = 0.90


class MemoryOptimizer:
    """Automatic memory optimization for BLIP3-o training"""
    
    def __init__(self, world_size: int = 1, task_mode: str = "clip_denoising"):
        self.world_size = world_size
        self.task_mode = task_mode
        
        # Get system specs
        self.gpu_count = torch.cuda.device_count()
        self.gpu_memory_gb = self._get_gpu_memory_gb()
        self.cpu_memory_gb = self._get_cpu_memory_gb()
        self.cpu_count = os.cpu_count()
        
        logger.info(f"Memory Optimizer initialized:")
        logger.info(f"  GPUs: {self.gpu_count} x {self.gpu_memory_gb:.1f}GB")
        logger.info(f"  CPU RAM: {self.cpu_memory_gb:.1f}GB")
        logger.info(f"  CPU cores: {self.cpu_count}")
        logger.info(f"  World size: {world_size}")
        logger.info(f"  Task: {task_mode}")

    def _get_gpu_memory_gb(self) -> float:
        """Get GPU memory in GB"""
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            # Get memory of first GPU (assume all GPUs are the same)
            props = torch.cuda.get_device_properties(0)
            return props.total_memory / (1024**3)
        return 0.0

    def _get_cpu_memory_gb(self) -> float:
        """Get CPU memory in GB"""
        return psutil.virtual_memory().total / (1024**3)

    def _estimate_model_memory_gb(self, model_size: str) -> float:
        """Estimate model memory usage in GB"""
        # Rough estimates for different model sizes
        model_memory_estimates = {
            "tiny": 0.5,
            "small": 1.0,
            "base": 2.0,
            "large": 4.0,
        }
        
        base_memory = model_memory_estimates.get(model_size, 2.0)
        
        # Add memory for optimizer states (AdamW: ~3x model parameters)
        optimizer_memory = base_memory * 3
        
        # Add memory for gradients
        gradient_memory = base_memory
        
        # Total model memory
        total_memory = base_memory + optimizer_memory + gradient_memory
        
        return total_memory

    def _estimate_batch_memory_gb(self, batch_size: int) -> float:
        """Estimate memory usage for a batch"""
        if self.task_mode == "eva_denoising":
            # EVA: [B, 256, 4096] input + conditioning + target
            tokens = 256
            dim = 4096
            tensors_per_sample = 3  # input, conditioning, target
        elif self.task_mode == "clip_denoising":
            # CLIP: [B, 256, 1024] input + target, [B, 256, 4096] conditioning
            tokens = 256
            clip_dim = 1024
            eva_dim = 4096
            # Estimate as weighted average
            dim = (clip_dim * 2 + eva_dim) / 3  # 2 CLIP tensors, 1 EVA tensor
            tensors_per_sample = 3
        else:
            # Conservative estimate
            tokens = 256
            dim = 2048
            tensors_per_sample = 3
        
        # Calculate memory: batch_size * tokens * dim * tensors * 4 bytes (fp32)
        memory_bytes = batch_size * tokens * dim * tensors_per_sample * 4
        
        # Add overhead for intermediate computations (2x)
        memory_bytes *= 2
        
        return memory_bytes / (1024**3)

    def get_optimal_config(
        self,
        max_shards: int,
        model_size: str = "base",
        target_effective_batch_size: int = 32,
        prefer_large_batches: bool = True
    ) -> MemoryConfig:
        """Get optimal memory configuration for the given constraints"""
        
        # Estimate model memory requirements
        model_memory_gb = self._estimate_model_memory_gb(model_size)
        
        # Calculate available GPU memory per device
        available_gpu_memory = self.gpu_memory_gb * 0.90  # 90% safety margin
        memory_for_batches = available_gpu_memory - model_memory_gb
        
        # Calculate optimal batch size per GPU
        if prefer_large_batches:
            # Try larger batch sizes first
            batch_sizes_to_try = [8, 6, 4, 3, 2, 1]
        else:
            # Try smaller batch sizes first (more memory efficient)
            batch_sizes_to_try = [2, 3, 4, 6, 8, 1]
        
        optimal_batch_size = 1
        optimal_grad_accum = target_effective_batch_size // self.world_size
        
        for batch_size in batch_sizes_to_try:
            batch_memory = self._estimate_batch_memory_gb(batch_size)
            
            if batch_memory <= memory_for_batches:
                optimal_batch_size = batch_size
                # Calculate gradient accumulation to reach target effective batch size
                effective_batch_per_gpu = target_effective_batch_size // self.world_size
                optimal_grad_accum = max(1, effective_batch_per_gpu // batch_size)
                break
        
        # Calculate shard cache settings based on available CPU memory
        available_cpu_memory = self.cpu_memory_gb * 0.80  # 80% safety margin
        
        # Estimate memory per cached shard
        if self.task_mode == "eva_denoising":
            shard_memory_estimate = 2.0  # GB per shard (EVA embeddings)
        elif self.task_mode == "clip_denoising":
            shard_memory_estimate = 3.0  # GB per shard (CLIP + EVA embeddings)
        else:
            shard_memory_estimate = 2.5  # Conservative estimate
        
        max_shard_cache = max(1, int(available_cpu_memory // shard_memory_estimate))
        max_shard_cache = min(max_shard_cache, 5)  # Cap at 5 for efficiency
        
        # Samples per shard load (smaller for limited memory)
        if available_cpu_memory > 64:
            samples_per_shard_load = 1000
        elif available_cpu_memory > 32:
            samples_per_shard_load = 750
        elif available_cpu_memory > 16:
            samples_per_shard_load = 500
        else:
            samples_per_shard_load = 250
        
        # Worker count optimization
        max_workers = min(4, self.cpu_count // 4, max_shard_cache)
        
        # Use FP16 for large models or limited memory
        use_fp16 = (model_size in ["large"] or self.gpu_memory_gb < 24)
        
        # Use gradient checkpointing for memory savings
        use_gradient_checkpointing = (model_size in ["large"] or self.gpu_memory_gb < 32)
        
        config = MemoryConfig(
            batch_size_per_gpu=optimal_batch_size,
            gradient_accumulation_steps=optimal_grad_accum,
            max_shard_cache=max_shard_cache,
            samples_per_shard_load=samples_per_shard_load,
            gpu_memory_limit_gb=available_gpu_memory,
            cpu_memory_limit_gb=available_cpu_memory,
            max_workers=max_workers,
            use_fp16=use_fp16,
            use_gradient_checkpointing=use_gradient_checkpointing,
            pin_memory=torch.cuda.is_available(),
        )
        
        # Log the configuration
        self._log_config(config, model_size, max_shards, target_effective_batch_size)
        
        return config

    def _log_config(
        self,
        config: MemoryConfig,
        model_size: str,
        max_shards: int,
        target_effective_batch_size: int
    ):
        """Log the optimized configuration"""
        effective_batch_size = (
            config.batch_size_per_gpu * 
            self.world_size * 
            config.gradient_accumulation_steps
        )
        
        logger.info(f"🔧 Optimized Memory Configuration:")
        logger.info(f"  Model size: {model_size}")
        logger.info(f"  Max shards: {max_shards}")
        logger.info(f"  Target effective batch size: {target_effective_batch_size}")
        logger.info(f"  Actual effective batch size: {effective_batch_size}")
        logger.info(f"")
        logger.info(f"📦 Batch Configuration:")
        logger.info(f"  Batch size per GPU: {config.batch_size_per_gpu}")
        logger.info(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
        logger.info(f"  Total GPUs: {self.world_size}")
        logger.info(f"  Effective batch size: {effective_batch_size}")
        logger.info(f"")
        logger.info(f"💾 Memory Configuration:")
        logger.info(f"  Shard cache: {config.max_shard_cache} shards")
        logger.info(f"  Samples per load: {config.samples_per_shard_load}")
        logger.info(f"  Workers: {config.max_workers}")
        logger.info(f"  GPU memory limit: {config.gpu_memory_limit_gb:.1f}GB")
        logger.info(f"  CPU memory limit: {config.cpu_memory_limit_gb:.1f}GB")
        logger.info(f"")
        logger.info(f"⚡ Optimizations:")
        logger.info(f"  FP16: {config.use_fp16}")
        logger.info(f"  Gradient checkpointing: {config.use_gradient_checkpointing}")
        logger.info(f"  Pin memory: {config.pin_memory}")

    def get_recommended_job_resources(
        self,
        max_shards: int,
        model_size: str = "base"
    ) -> Dict[str, Any]:
        """Get recommended SLURM job resources"""
        
        # GPU requirements
        if max_shards <= 10:
            recommended_gpus = 2
        elif max_shards <= 25:
            recommended_gpus = 4
        else:
            recommended_gpus = 4  # Cap at 4 for single node
        
        # Memory requirements
        model_memory_gb = self._estimate_model_memory_gb(model_size)
        
        # CPU memory: base requirement + shard storage + safety margin
        base_memory_gb = 16  # Base system requirements
        shard_memory_gb = max_shards * 0.5  # Rough estimate
        safety_margin = 1.5
        
        total_cpu_memory_gb = (base_memory_gb + shard_memory_gb) * safety_margin
        total_cpu_memory_gb = min(total_cpu_memory_gb, 160)  # Cap at typical node limit
        
        # CPU requirements
        cpus_per_task = 8  # Good balance for I/O and compute
        
        # Time requirements (rough estimates)
        base_time_hours = 2
        shard_factor = max_shards / 10
        model_factor = {"tiny": 0.5, "small": 0.75, "base": 1.0, "large": 1.5}.get(model_size, 1.0)
        
        estimated_time_hours = base_time_hours * shard_factor * model_factor
        estimated_time_hours = min(estimated_time_hours, 12)  # Cap at 12 hours
        
        return {
            "nodes": 1,
            "gpus": recommended_gpus,
            "ntasks": recommended_gpus,
            "cpus_per_task": cpus_per_task,
            "memory_gb": int(total_cpu_memory_gb),
            "time_hours": int(estimated_time_hours),
            "partition": "gpu_h100" if recommended_gpus >= 4 else "gpu",
            "estimated_params": {
                "model_memory_gb": model_memory_gb,
                "shard_memory_gb": shard_memory_gb,
                "safety_factor": safety_margin,
            }
        }

    def diagnose_oom_issue(self, error_log: str = None) -> Dict[str, Any]:
        """Diagnose OOM issues and provide recommendations"""
        
        # Current memory usage
        current_memory = psutil.virtual_memory()
        gpu_memory_info = []
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_memory_info.append({
                    "gpu": i,
                    "allocated_gb": torch.cuda.memory_allocated(i) / (1024**3),
                    "reserved_gb": torch.cuda.memory_reserved(i) / (1024**3),
                    "total_gb": torch.cuda.get_device_properties(i).total_memory / (1024**3),
                })
        
        diagnosis = {
            "cpu_memory": {
                "total_gb": current_memory.total / (1024**3),
                "available_gb": current_memory.available / (1024**3),
                "used_percent": current_memory.percent,
                "status": "critical" if current_memory.percent > 90 else "warning" if current_memory.percent > 80 else "ok"
            },
            "gpu_memory": gpu_memory_info,
            "recommendations": []
        }
        
        # Generate recommendations
        if current_memory.percent > 90:
            diagnosis["recommendations"].append("Reduce max_shard_cache (try 1-2)")
            diagnosis["recommendations"].append("Reduce samples_per_shard_load (try 250-500)")
            diagnosis["recommendations"].append("Reduce batch_size_per_gpu (try 1-2)")
        
        for gpu_info in gpu_memory_info:
            gpu_usage_percent = (gpu_info["allocated_gb"] / gpu_info["total_gb"]) * 100
            if gpu_usage_percent > 90:
                diagnosis["recommendations"].append(f"GPU {gpu_info['gpu']}: Enable FP16 training")
                diagnosis["recommendations"].append(f"GPU {gpu_info['gpu']}: Enable gradient checkpointing")
                diagnosis["recommendations"].append(f"GPU {gpu_info['gpu']}: Reduce batch size")
        
        if error_log:
            if "CUDA out of memory" in error_log:
                diagnosis["recommendations"].append("CUDA OOM detected: Reduce batch size or enable FP16")
            if "Cannot allocate memory" in error_log:
                diagnosis["recommendations"].append("CPU OOM detected: Reduce shard cache or samples per load")
        
        return diagnosis


def get_optimal_memory_config(
    world_size: int,
    max_shards: int,
    task_mode: str = "clip_denoising",
    model_size: str = "base",
    target_effective_batch_size: int = 32
) -> MemoryConfig:
    """Convenience function to get optimal memory configuration"""
    
    optimizer = MemoryOptimizer(world_size=world_size, task_mode=task_mode)
    return optimizer.get_optimal_config(
        max_shards=max_shards,
        model_size=model_size,
        target_effective_batch_size=target_effective_batch_size
    )


def get_slurm_resources(
    max_shards: int,
    model_size: str = "base",
    task_mode: str = "clip_denoising"
) -> Dict[str, Any]:
    """Get recommended SLURM job resources"""
    
    optimizer = MemoryOptimizer(world_size=1, task_mode=task_mode)  # World size doesn't matter for resource calculation
    return optimizer.get_recommended_job_resources(max_shards=max_shards, model_size=model_size)


def print_memory_recommendations(
    max_shards: int,
    model_size: str = "base",
    task_mode: str = "clip_denoising"
):
    """Print memory optimization recommendations"""
    
    print(f"🧠 Memory Optimization Recommendations")
    print("=" * 60)
    print(f"Task: {task_mode}")
    print(f"Model size: {model_size}")
    print(f"Max shards: {max_shards}")
    print()
    
    # Get SLURM resources
    resources = get_slurm_resources(max_shards, model_size, task_mode)
    
    print(f"📊 Recommended SLURM Resources:")
    print(f"#SBATCH --nodes={resources['nodes']}")
    print(f"#SBATCH --gpus={resources['gpus']}")
    print(f"#SBATCH --ntasks={resources['ntasks']}")
    print(f"#SBATCH --cpus-per-task={resources['cpus_per_task']}")
    print(f"#SBATCH --mem={resources['memory_gb']}G")
    print(f"#SBATCH --time={resources['time_hours']}:00:00")
    print(f"#SBATCH --partition={resources['partition']}")
    print()
    
    # Get memory config for different world sizes
    for world_size in [2, 4]:
        if world_size <= resources['gpus']:
            print(f"🔧 Optimal Config for {world_size} GPUs:")
            config = get_optimal_memory_config(
                world_size=world_size,
                max_shards=max_shards,
                task_mode=task_mode,
                model_size=model_size
            )
            print(f"  Batch size per GPU: {config.batch_size_per_gpu}")
            print(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
            print(f"  Effective batch size: {config.batch_size_per_gpu * world_size * config.gradient_accumulation_steps}")
            print(f"  Max shard cache: {config.max_shard_cache}")
            print(f"  Samples per load: {config.samples_per_shard_load}")
            print(f"  FP16: {config.use_fp16}")
            print()


if __name__ == "__main__":
    # Example usage
    print_memory_recommendations(max_shards=35, model_size="base", task_mode="clip_denoising")