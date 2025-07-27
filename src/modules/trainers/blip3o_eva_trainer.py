#!/usr/bin/env python3
"""
DDP-Aware BLIP3-o Trainer with Memory Optimization and Gradient Accumulation
Fixes OOM issues by implementing proper distributed training and memory management

Key features:
1. DistributedDataParallel training support
2. Gradient accumulation for large effective batch sizes
3. Memory monitoring and optimization
4. Rank-aware logging and evaluation
5. Proper synchronization across ranks
"""

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from typing import Dict, Any, Optional, List, Tuple
import logging
import time
import numpy as np
from pathlib import Path
import json
import gc
from collections import deque
import math
import psutil

logger = logging.getLogger(__name__)


class DDPDenoisingTrainer:
    """
    DDP-aware trainer for both EVA and CLIP denoising with memory optimization
    """
    
    def __init__(
        self,
        model,
        loss_fn,
        train_dataloader,
        eval_dataloader=None,
        # Training configuration
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        num_epochs: int = 10,
        warmup_steps: int = 100,
        max_grad_norm: float = 1.0,
        gradient_accumulation_steps: int = 4,
        fp16: bool = False,
        # Evaluation
        eval_every_n_steps: int = 200,
        eval_num_samples: int = 500,
        eval_inference_steps: int = 50,
        # Debugging
        debug_mode: bool = False,
        overfit_test_size: Optional[int] = None,
        log_every_n_steps: int = 10,
        save_every_n_steps: int = 500,
        # Task configuration
        task_mode: Optional[str] = None,
        # Output
        output_dir: str = "./checkpoints",
        # DDP
        device: Optional[torch.device] = None,
        rank: int = 0,
        world_size: int = 1,
        # WandB integration
        wandb_instance=None,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        
        # Training config
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.fp16 = fp16
        
        # Evaluation config
        self.eval_every_n_steps = eval_every_n_steps
        self.eval_num_samples = eval_num_samples
        self.eval_inference_steps = eval_inference_steps
        
        # Debugging config
        self.debug_mode = debug_mode
        self.overfit_test_size = overfit_test_size
        self.log_every_n_steps = log_every_n_steps
        self.save_every_n_steps = save_every_n_steps
        
        # Task configuration
        self.task_mode = task_mode
        
        # DDP configuration
        self.device = device or torch.device("cpu")
        self.rank = rank
        self.world_size = world_size
        self.is_main_process = (rank == 0)
        
        # Output
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # WandB integration (only on rank 0)
        self.wandb = wandb_instance if self.is_main_process else None
        self.use_wandb = (wandb_instance is not None and self.is_main_process)
        
        # Setup optimizer and scheduler
        self._setup_optimizer_and_scheduler()
        
        # Setup mixed precision
        if self.fp16:
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None
        
        # Tracking variables
        self.global_step = 0
        self.current_epoch = 0
        self.best_eval_similarity = 0.0
        self.best_loss = float('inf')
        
        # Metrics tracking (all ranks track for debugging, but only rank 0 logs)
        self.loss_history = deque(maxlen=1000)
        self.similarity_history = deque(maxlen=1000)
        self.eval_similarity_history = deque(maxlen=1000)
        self.lr_history = deque(maxlen=1000)
        self.grad_norm_history = deque(maxlen=1000)
        self.sphere_violation_history = deque(maxlen=1000)
        
        # Task-specific tracking
        self.eva_similarity_history = deque(maxlen=1000)
        self.clip_similarity_history = deque(maxlen=1000)
        
        # Memory monitoring
        self.memory_history = deque(maxlen=100)
        
        # Overfitting test data (only on rank 0)
        self.overfit_batch = None
        if self.overfit_test_size and self.is_main_process:
            self._prepare_overfit_test()
        
        # Auto-detect task mode if not provided
        if self.task_mode is None:
            self.task_mode = self._detect_task_mode()
        
        # Log initialization (only on rank 0)
        if self.is_main_process:
            self._log_initialization()

    def _log_initialization(self):
        """Log initialization info (rank 0 only)"""
        task_info = self._get_task_info()
        logger.info("DDP Universal Denoising Trainer initialized")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Rank: {self.rank}/{self.world_size}")
        logger.info(f"  Task: {task_info['task']}")
        logger.info(f"  Input: {task_info['input']}")
        logger.info(f"  Conditioning: {task_info['conditioning']}")
        logger.info(f"  Target: {task_info['target']}")
        logger.info(f"  Learning rate: {self.learning_rate}")
        logger.info(f"  Epochs: {self.num_epochs}")
        logger.info(f"  Gradient accumulation: {self.gradient_accumulation_steps}")
        logger.info(f"  Prediction type: {getattr(self.model.module if hasattr(self.model, 'module') else self.model, 'config', {}).get('prediction_type', 'velocity')}")
        logger.info(f"  Overfit test: {self.overfit_test_size if self.overfit_test_size else 'Disabled'}")
        logger.info(f"  Mixed precision: {self.fp16}")
        logger.info(f"  WandB logging: {self.use_wandb}")

    def get_memory_usage_gb(self) -> float:
        """Get current memory usage in GB"""
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            return memory_mb / 1024
        except:
            return 0.0

    def _cleanup_memory(self):
        """Aggressive memory cleanup"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def _detect_task_mode(self) -> str:
        """Auto-detect task mode from model configuration"""
        model_ref = self.model.module if hasattr(self.model, 'module') else self.model
        
        if hasattr(model_ref, 'config') and hasattr(model_ref.config, 'task_mode'):
            return model_ref.config.task_mode
        
        # Try to detect from first batch (only on rank 0)
        if self.is_main_process:
            try:
                first_batch = next(iter(self.train_dataloader))
                if 'task_mode' in first_batch:
                    detected_mode = first_batch['task_mode']
                    logger.info(f"Auto-detected task mode: {detected_mode}")
                    return detected_mode
            except Exception as e:
                logger.warning(f"Could not auto-detect task mode: {e}")
        
        # Default fallback
        return "clip_denoising"

    def _get_task_info(self) -> Dict[str, str]:
        """Get task-specific information"""
        if self.task_mode == "eva_denoising":
            return {
                "task": "EVA-CLIP Denoising",
                "input": "Noisy EVA embeddings [B, N, 4096]",
                "conditioning": "Clean EVA embeddings [B, N, 4096]",
                "target": "Clean EVA embeddings [B, N, 4096]",
            }
        elif self.task_mode == "clip_denoising":
            return {
                "task": "CLIP-ViT Denoising with EVA Conditioning",
                "input": "Noisy CLIP embeddings [B, N, 1024]",
                "conditioning": "Clean EVA embeddings [B, N, 4096]",
                "target": "Clean CLIP embeddings [B, N, 1024]",
            }
        else:
            return {
                "task": "Unknown Task",
                "input": "Unknown",
                "conditioning": "Unknown", 
                "target": "Unknown",
            }

    def _setup_optimizer_and_scheduler(self):
        """Setup optimizer and learning rate scheduler"""
        # Use AdamW with weight decay
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
            eps=1e-8
        )
        
        # Estimate total steps for scheduler
        try:
            dataloader_length = len(self.train_dataloader)
            total_steps = (dataloader_length * self.num_epochs) // self.gradient_accumulation_steps
        except (TypeError, AttributeError):
            if self.is_main_process:
                logger.warning("Cannot determine exact dataloader length, using estimate")
            estimated_samples_per_epoch = 10000 // self.world_size  # Divide by world size for DDP
            batch_size = getattr(self.train_dataloader, 'batch_size', 4)
            estimated_batches_per_epoch = estimated_samples_per_epoch // batch_size
            total_steps = (estimated_batches_per_epoch * self.num_epochs) // self.gradient_accumulation_steps
        
        if self.warmup_steps > 0:
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=self.warmup_steps
            )
            cosine_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps - self.warmup_steps,
                eta_min=self.learning_rate * 0.01
            )
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[self.warmup_steps]
            )
        else:
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,
                eta_min=self.learning_rate * 0.01
            )
        
        if self.is_main_process:
            logger.info(f"Optimizer and scheduler setup complete")
            logger.info(f"  Estimated total steps: {total_steps}")
            logger.info(f"  Warmup steps: {self.warmup_steps}")
            logger.info(f"  Gradient accumulation: {self.gradient_accumulation_steps}")

    def _prepare_overfit_test(self):
        """Prepare overfitting test batch (rank 0 only)"""
        if not self.is_main_process:
            return
            
        logger.info(f"Preparing overfitting test with {self.overfit_test_size} samples...")
        
        try:
            first_batch = next(iter(self.train_dataloader))
            actual_size = min(self.overfit_test_size, first_batch['batch_size'])
            
            self.overfit_batch = {}
            for key, value in first_batch.items():
                if torch.is_tensor(value) and value.dim() > 0:
                    self.overfit_batch[key] = value[:actual_size].clone().detach()
                elif isinstance(value, list):
                    self.overfit_batch[key] = value[:actual_size]
                else:
                    self.overfit_batch[key] = value
            
            self.overfit_batch['batch_size'] = actual_size
            logger.info(f"Overfitting test prepared with {actual_size} samples")
            
        except Exception as e:
            logger.error(f"Failed to prepare overfitting test: {e}")
            self.overfit_batch = None

    def _compute_loss(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute loss for a batch with DDP support"""
        # Move batch to device
        for key, value in batch.items():
            if torch.is_tensor(value):
                batch[key] = value.to(self.device)
        
        # Use overfit batch if specified (only on rank 0)
        if self.overfit_batch is not None and self.is_main_process:
            for key, value in self.overfit_batch.items():
                if torch.is_tensor(value):
                    batch[key] = value.to(self.device)
                else:
                    batch[key] = value
        
        # Extract inputs (universal interface)
        x_t = batch['hidden_states']
        timestep = batch['timestep']
        conditioning = batch['encoder_hidden_states']
        target = batch['target_embeddings']
        velocity_target = batch.get('velocity_target')
        noise = batch.get('noise')
        task_mode = batch.get('task_mode', self.task_mode)
        
        # Forward pass with mixed precision
        if self.fp16:
            with torch.amp.autocast('cuda'):
                model_output = self.model(
                    hidden_states=x_t,
                    timestep=timestep,
                    encoder_hidden_states=conditioning,
                    return_dict=False
                )
                
                loss, metrics = self.loss_fn(
                    model_output=model_output,
                    target_samples=target,
                    timesteps=timestep,
                    conditioning=conditioning,
                    noise=noise,
                    x_t=x_t,
                    task_mode=task_mode,
                    return_metrics=True
                )
        else:
            model_output = self.model(
                hidden_states=x_t,
                timestep=timestep,
                encoder_hidden_states=conditioning,
                return_dict=False
            )
            
            loss, metrics = self.loss_fn(
                model_output=model_output,
                target_samples=target,
                timesteps=timestep,
                conditioning=conditioning,
                noise=noise,
                x_t=x_t,
                task_mode=task_mode,
                return_metrics=True
            )
        
        # Scale loss by gradient accumulation steps
        loss = loss / self.gradient_accumulation_steps
        
        return loss, metrics

    def _backward_and_step(self, loss: torch.Tensor, step_in_accumulation: int) -> float:
        """Backward pass with gradient accumulation"""
        # Backward pass
        if self.fp16:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Only step optimizer after accumulation is complete
        if (step_in_accumulation + 1) % self.gradient_accumulation_steps == 0:
            # Compute gradient norm before clipping
            grad_norm = 0.0
            for param in self.model.parameters():
                if param.grad is not None:
                    grad_norm += param.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5
            
            # Gradient clipping
            if self.max_grad_norm > 0:
                if self.fp16:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            # Optimizer step
            if self.fp16:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            self.scheduler.step()
            
            return grad_norm
        else:
            return 0.0  # No step taken

    def _evaluate(self, num_samples: Optional[int] = None) -> Dict[str, float]:
        """Run evaluation (only on rank 0)"""
        if not self.is_main_process or self.eval_dataloader is None:
            return {}
        
        if num_samples is None:
            num_samples = self.eval_num_samples
        
        model_ref = self.model.module if hasattr(self.model, 'module') else self.model
        model_ref.eval()
        
        all_similarities = []
        all_angular_distances = []
        all_sphere_violations = []
        samples_processed = 0
        task_mode_detected = None
        
        with torch.no_grad():
            for batch in self.eval_dataloader:
                if samples_processed >= num_samples:
                    break
                
                try:
                    input_embeddings = batch['input_embeddings'].to(self.device)
                    conditioning = batch['encoder_hidden_states'].to(self.device)
                    target = batch['target_embeddings'].to(self.device)
                    task_mode_detected = batch.get('task_mode', self.task_mode)
                except KeyError as e:
                    logger.warning(f"Missing key in evaluation batch: {e}")
                    continue
                
                # Denoise using the model
                denoised = model_ref.denoise(
                    noisy_embeddings=input_embeddings,
                    conditioning=conditioning,
                    num_inference_steps=self.eval_inference_steps,
                )
                
                # Compute similarity
                target_norm = F.normalize(target, p=2, dim=-1)
                denoised_norm = F.normalize(denoised, p=2, dim=-1)
                
                similarity = F.cosine_similarity(denoised_norm, target_norm, dim=-1)
                per_image_similarity = similarity.mean(dim=1)
                
                angular_distance = torch.acos(torch.clamp(similarity, -1 + 1e-7, 1 - 1e-7)).mean(dim=1)
                
                denoised_norms = torch.norm(denoised, dim=-1)
                sphere_violation = torch.abs(denoised_norms - 1.0).mean(dim=1)
                
                all_similarities.append(per_image_similarity.cpu())
                all_angular_distances.append(angular_distance.cpu())
                all_sphere_violations.append(sphere_violation.cpu())
                samples_processed += input_embeddings.shape[0]
        
        model_ref.train()
        
        if not all_similarities:
            return {}
        
        all_sims = torch.cat(all_similarities)
        all_angular = torch.cat(all_angular_distances)
        all_violations = torch.cat(all_sphere_violations)
        
        # Task-specific metrics
        if task_mode_detected == "eva_denoising":
            metric_prefix = "eval_eva"
            high_thresh, very_high_thresh, excellent_thresh = 0.7, 0.8, 0.9
        elif task_mode_detected == "clip_denoising":
            metric_prefix = "eval_clip"
            high_thresh, very_high_thresh, excellent_thresh = 0.6, 0.7, 0.8
        else:
            metric_prefix = "eval_generic"
            high_thresh, very_high_thresh, excellent_thresh = 0.6, 0.7, 0.8
        
        return {
            f'{metric_prefix}_similarity': all_sims.mean().item(),
            f'{metric_prefix}_similarity_std': all_sims.std().item(),
            f'{metric_prefix}_angular_distance': all_angular.mean().item(),
            f'{metric_prefix}_sphere_violation': all_violations.mean().item(),
            f'{metric_prefix}_high_quality': (all_sims > high_thresh).float().mean().item(),
            f'{metric_prefix}_very_high_quality': (all_sims > very_high_thresh).float().mean().item(),
            f'{metric_prefix}_excellent_quality': (all_sims > excellent_thresh).float().mean().item(),
            f'{metric_prefix}_samples': samples_processed,
            f'{metric_prefix}_similarity_min': all_sims.min().item(),
            f'{metric_prefix}_similarity_max': all_sims.max().item(),
            'eval_task_mode': task_mode_detected or self.task_mode,
        }

    def _safe_deque_slice(self, deque_obj: deque, slice_size: int = 10) -> List[float]:
        """Safely get last N elements from deque"""
        if len(deque_obj) == 0:
            return []
        deque_list = list(deque_obj)
        return deque_list[-slice_size:] if len(deque_list) >= slice_size else deque_list

    def _safe_format_value(self, key: str, value: Any, decimal_places: int = 4) -> str:
        """Safely format a value for logging"""
        try:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if decimal_places == 6:
                    return f"{key}: {value:.6f}"
                else:
                    return f"{key}: {value:.4f}"
            elif isinstance(value, str):
                return f"{key}: {value}"
            elif isinstance(value, bool):
                return f"{key}: {value}"
            else:
                return f"{key}: {str(value)}"
        except (ValueError, TypeError):
            return f"{key}: {str(value)}"

    def _log_metrics(self, loss: float, metrics: Dict[str, float], grad_norm: float):
        """Log training metrics (only on rank 0)"""
        if not self.is_main_process:
            return
        
        # Store metrics
        self.loss_history.append(loss * self.gradient_accumulation_steps)  # Unscaled loss
        if 'prediction_similarity' in metrics:
            self.similarity_history.append(metrics['prediction_similarity'])
        if 'eval_similarity' in metrics:
            self.eval_similarity_history.append(metrics['eval_similarity'])
        if 'sphere_violation' in metrics:
            self.sphere_violation_history.append(metrics['sphere_violation'])
        
        # Memory tracking
        current_memory = self.get_memory_usage_gb()
        self.memory_history.append(current_memory)
        
        # Task-specific similarity tracking
        task_mode = metrics.get('task_mode', self.task_mode)
        eval_sim = metrics.get('eval_similarity', metrics.get('prediction_similarity', 0))
        
        if task_mode == "eva_denoising":
            self.eva_similarity_history.append(eval_sim)
        elif task_mode == "clip_denoising":
            self.clip_similarity_history.append(eval_sim)
        
        self.lr_history.append(self.optimizer.param_groups[0]['lr'])
        if grad_norm > 0:  # Only when step was taken
            self.grad_norm_history.append(grad_norm)
        
        # Update best metrics
        unscaled_loss = loss * self.gradient_accumulation_steps
        if eval_sim > self.best_eval_similarity:
            self.best_eval_similarity = eval_sim
        if unscaled_loss < self.best_loss:
            self.best_loss = unscaled_loss
        
        # WandB logging
        if self.use_wandb and self.wandb:
            try:
                wandb_metrics = {
                    'train/loss': unscaled_loss,
                    'train/scaled_loss': loss,
                    'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                    'train/step': self.global_step,
                    'train/epoch': self.current_epoch,
                    'train/memory_gb': current_memory,
                    'train/world_size': self.world_size,
                }
                
                if grad_norm > 0:
                    wandb_metrics['train/grad_norm'] = grad_norm
                
                # Add loss function metrics
                for key, value in metrics.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        if key in ['prediction_similarity', 'eval_similarity']:
                            wandb_metrics[f'train/{key}'] = value
                        elif key == 'sphere_violation':
                            wandb_metrics[f'train/{key}'] = value
                        elif key in ['loss', 'total_loss']:
                            wandb_metrics[f'train/{key}'] = value
                        elif not key.startswith('task_'):
                            wandb_metrics[f'train/metrics/{key}'] = value
                
                # Add best metrics
                wandb_metrics['train/best_loss'] = self.best_loss
                wandb_metrics['train/best_eval_similarity'] = self.best_eval_similarity
                
                self.wandb.log(wandb_metrics, step=self.global_step)
                
            except Exception as e:
                logger.warning(f"Failed to log to WandB: {e}")
        
        # Console logging
        if self.global_step % self.log_every_n_steps == 0:
            task_name = "EVA" if task_mode == "eva_denoising" else "CLIP" if task_mode == "clip_denoising" else "UNK"
            log_msg = f"Step {self.global_step} [{task_name}]: Loss={unscaled_loss:.6f}"
            
            if 'prediction_similarity' in metrics:
                pred_sim = metrics['prediction_similarity']
                eval_sim = metrics.get('eval_similarity', pred_sim)
                quality = metrics.get('quality_assessment', 'unknown')
                log_msg += f", PredSim={pred_sim:.4f}, EvalSim={eval_sim:.4f} ({quality})"
            
            if 'sphere_violation' in metrics:
                sphere_viol = metrics['sphere_violation']
                log_msg += f", SphereViol={sphere_viol:.6f}"
            
            if 'output_dim' in metrics:
                out_dim = metrics['output_dim']
                cond_dim = metrics.get('conditioning_dim', 'unknown')
                log_msg += f", Dims={out_dim}|{cond_dim}"
            
            if grad_norm > 0:
                log_msg += f", GradNorm={grad_norm:.3f}"
            
            log_msg += f", LR={self.optimizer.param_groups[0]['lr']:.2e}"
            log_msg += f", Mem={current_memory:.1f}GB"
            log_msg += f", GPU{self.rank}/{self.world_size}"
            
            if self.overfit_batch is not None:
                log_msg += " [OVERFIT TEST]"
            
            logger.info(log_msg)

    def _log_evaluation_metrics(self, eval_metrics: Dict[str, float]):
        """Log evaluation metrics (only on rank 0)"""
        if not self.is_main_process or not self.use_wandb or not self.wandb or not eval_metrics:
            return
        
        try:
            wandb_eval_metrics = {}
            for key, value in eval_metrics.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    wandb_eval_metrics[f'eval/{key}'] = value
                elif isinstance(value, str) and key == 'eval_task_mode':
                    wandb_eval_metrics['eval/task_mode'] = value
            
            wandb_eval_metrics['eval/step'] = self.global_step
            wandb_eval_metrics['eval/epoch'] = self.current_epoch
            
            self.wandb.log(wandb_eval_metrics, step=self.global_step)
            
        except Exception as e:
            logger.warning(f"Failed to log evaluation metrics to WandB: {e}")

    def _save_checkpoint(self):
        """Save model checkpoint (only on rank 0)"""
        if not self.is_main_process:
            return
        
        checkpoint_path = self.output_dir / f"checkpoint_step_{self.global_step}.pt"
        
        # Get model state dict (handle DDP)
        model_state_dict = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
        
        checkpoint = {
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'current_epoch': self.current_epoch,
            'best_eval_similarity': self.best_eval_similarity,
            'best_loss': self.best_loss,
            'task_mode': self.task_mode,
            'ddp_info': {
                'rank': self.rank,
                'world_size': self.world_size,
                'gradient_accumulation_steps': self.gradient_accumulation_steps,
            },
            'loss_history': list(self.loss_history),
            'similarity_history': list(self.similarity_history),
            'eval_similarity_history': list(self.eval_similarity_history),
            'memory_history': list(self.memory_history),
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def _sync_metrics_across_ranks(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Synchronize metrics across all ranks for consistent logging"""
        if self.world_size <= 1:
            return metrics
        
        synced_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                tensor_value = torch.tensor(value, device=self.device)
                dist.all_reduce(tensor_value, op=dist.ReduceOp.AVG)
                synced_metrics[key] = tensor_value.item()
            else:
                synced_metrics[key] = value
        
        return synced_metrics

    def train(self) -> Dict[str, Any]:
        """Main DDP training loop"""
        if self.is_main_process:
            task_info = self._get_task_info()
            logger.info(f"Starting DDP {task_info['task']} training...")
            
            model_ref = self.model.module if hasattr(self.model, 'module') else self.model
            param_count = sum(p.numel() for p in model_ref.parameters() if p.requires_grad)
            logger.info(f"  Model parameters: {param_count:,}")
            logger.info(f"  World size: {self.world_size}")
            logger.info(f"  Gradient accumulation: {self.gradient_accumulation_steps}")
            
            effective_batch_size = (
                getattr(self.train_dataloader, 'batch_size', 4) * 
                self.world_size * 
                self.gradient_accumulation_steps
            )
            logger.info(f"  Effective batch size: {effective_batch_size}")
        
        # Synchronize all ranks before starting
        if self.world_size > 1:
            dist.barrier()
        
        self.model.train()
        start_time = time.time()
        
        try:
            for epoch in range(self.num_epochs):
                self.current_epoch = epoch
                
                if self.is_main_process:
                    logger.info(f"Starting epoch {epoch + 1}/{self.num_epochs}")
                
                epoch_loss = 0.0
                epoch_steps = 0
                accumulation_step = 0
                
                for batch_idx, batch in enumerate(self.train_dataloader):
                    step_start_time = time.time()
                    
                    # Compute loss
                    try:
                        loss, metrics = self._compute_loss(batch)
                    except Exception as e:
                        if self.is_main_process:
                            logger.error(f"Error computing loss at step {self.global_step}: {e}")
                        continue
                    
                    # Backward pass
                    try:
                        grad_norm = self._backward_and_step(loss, accumulation_step)
                    except Exception as e:
                        if self.is_main_process:
                            logger.error(f"Error in backward pass at step {self.global_step}: {e}")
                        continue
                    
                    # Update metrics
                    epoch_loss += loss.item()
                    accumulation_step += 1
                    
                    # Only increment global step after full accumulation
                    if accumulation_step % self.gradient_accumulation_steps == 0:
                        self.global_step += 1
                        epoch_steps += 1
                        
                        # Synchronize metrics across ranks for consistent logging
                        if metrics:
                            metrics = self._sync_metrics_across_ranks(metrics)
                        
                        # Log metrics (only on rank 0)
                        self._log_metrics(loss.item(), metrics or {}, grad_norm)
                        
                        # Run evaluation (only on rank 0)
                        if (self.global_step % self.eval_every_n_steps == 0 and 
                            self.is_main_process):
                            logger.info(f"Running evaluation at step {self.global_step}...")
                            eval_metrics = self._evaluate()
                            
                            if eval_metrics:
                                task_mode = eval_metrics.get('eval_task_mode', self.task_mode)
                                metric_prefix = "eval_eva" if task_mode == "eva_denoising" else "eval_clip"
                                
                                logger.info(f"Evaluation results for {task_mode}:")
                                for key, value in eval_metrics.items():
                                    formatted_value = self._safe_format_value(key, value, decimal_places=4)
                                    logger.info(f"  {formatted_value}")
                                
                                self._log_evaluation_metrics(eval_metrics)
                                
                                # Update best eval similarity
                                main_sim_key = f'{metric_prefix}_similarity'
                                if main_sim_key in eval_metrics and eval_metrics[main_sim_key] > self.best_eval_similarity:
                                    self.best_eval_similarity = eval_metrics[main_sim_key]
                                    logger.info(f"New best {task_mode} similarity: {self.best_eval_similarity:.4f}")
                        
                        # Save checkpoint (only on rank 0)
                        if (self.global_step % self.save_every_n_steps == 0 and 
                            self.is_main_process):
                            self._save_checkpoint()
                        
                        # Memory cleanup
                        if self.global_step % 100 == 0:
                            self._cleanup_memory()
                    
                    # Synchronize periodically
                    if self.world_size > 1 and batch_idx % 50 == 0:
                        dist.barrier()
                
                # End of epoch summary (only on rank 0)
                if self.is_main_process:
                    avg_epoch_loss = epoch_loss / max(accumulation_step, 1) * self.gradient_accumulation_steps
                    logger.info(f"Epoch {epoch + 1} completed:")
                    logger.info(f"  Average loss: {avg_epoch_loss:.6f}")
                    logger.info(f"  Steps: {epoch_steps}")
                    logger.info(f"  Memory: {self.get_memory_usage_gb():.1f}GB")
                
                # Synchronize at end of epoch
                if self.world_size > 1:
                    dist.barrier()
        
        except KeyboardInterrupt:
            if self.is_main_process:
                logger.info("Training interrupted by user")
        except Exception as e:
            if self.is_main_process:
                logger.error(f"Training failed with error: {e}")
            raise
        
        finally:
            # Final synchronization
            if self.world_size > 1:
                dist.barrier()
            
            # Final checkpoint (only on rank 0)
            if self.is_main_process:
                self._save_checkpoint()
                
                # Final evaluation
                logger.info("Running final evaluation...")
                final_eval = self._evaluate(num_samples=self.eval_num_samples * 2)
                
                if final_eval:
                    self._log_evaluation_metrics(final_eval)
                
                total_time = time.time() - start_time
                
                # Training summary
                summary = {
                    'training_completed': True,
                    'task_mode': self.task_mode,
                    'total_time_seconds': total_time,
                    'total_steps': self.global_step,
                    'final_epoch': self.current_epoch,
                    'best_loss': self.best_loss,
                    'best_eval_similarity': self.best_eval_similarity,
                    'final_eval': final_eval,
                    'ddp_info': {
                        'world_size': self.world_size,
                        'gradient_accumulation_steps': self.gradient_accumulation_steps,
                        'effective_batch_size': (
                            getattr(self.train_dataloader, 'batch_size', 4) * 
                            self.world_size * 
                            self.gradient_accumulation_steps
                        ),
                    },
                    'memory_stats': {
                        'final_memory_gb': self.get_memory_usage_gb(),
                        'peak_memory_gb': max(self.memory_history) if self.memory_history else 0,
                        'avg_memory_gb': sum(self.memory_history) / len(self.memory_history) if self.memory_history else 0,
                    },
                    'overfit_test': self.overfit_batch is not None,
                    'loss_history': list(self.loss_history),
                    'similarity_history': list(self.similarity_history),
                    'memory_history': list(self.memory_history),
                    'wandb_url': self.wandb.url if self.use_wandb and self.wandb else None,
                }
                
                # Save training summary
                summary_path = self.output_dir / "ddp_training_summary.json"
                with open(summary_path, 'w') as f:
                    json.dump(summary, f, indent=2)
                
                logger.info(f"DDP {task_info['task']} training completed!")
                logger.info(f"  Total time: {total_time:.1f} seconds")
                logger.info(f"  Total steps: {self.global_step}")
                logger.info(f"  Best loss: {self.best_loss:.6f}")
                logger.info(f"  Best similarity: {self.best_eval_similarity:.4f}")
                logger.info(f"  World size: {self.world_size}")
                logger.info(f"  Peak memory: {max(self.memory_history) if self.memory_history else 0:.1f}GB")
                
                return summary
            else:
                # Non-main processes return minimal summary
                return {'training_completed': True, 'rank': self.rank}


def create_ddp_trainer(
    model,
    loss_fn,
    train_dataloader,
    eval_dataloader=None,
    learning_rate: float = 1e-4,
    num_epochs: int = 10,
    gradient_accumulation_steps: int = 4,
    output_dir: str = "./checkpoints",
    task_mode: Optional[str] = None,
    overfit_test_size: Optional[int] = None,
    debug_mode: bool = False,
    device: Optional[torch.device] = None,
    rank: int = 0,
    world_size: int = 1,
    wandb_instance=None,
    **kwargs
) -> DDPDenoisingTrainer:
    """Factory function to create DDP trainer"""
    
    return DDPDenoisingTrainer(
        model=model,
        loss_fn=loss_fn,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        gradient_accumulation_steps=gradient_accumulation_steps,
        output_dir=output_dir,
        task_mode=task_mode,
        overfit_test_size=overfit_test_size,
        debug_mode=debug_mode,
        device=device,
        rank=rank,
        world_size=world_size,
        wandb_instance=wandb_instance,
        **kwargs
    )