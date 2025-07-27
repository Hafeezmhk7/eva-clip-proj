#!/usr/bin/env python3
"""
FIXED DDP-Aware BLIP3-o Trainer with Robust Memory Management
Addresses OOM issues and DDP synchronization problems
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
import os

logger = logging.getLogger(__name__)


class FixedDDPDenoisingTrainer:
    """
    FIXED DDP-aware trainer with robust memory management and error handling
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
        num_epochs: int = 5,
        warmup_steps: int = 100,
        max_grad_norm: float = 1.0,
        gradient_accumulation_steps: int = 8,
        fp16: bool = True,
        # Evaluation
        eval_every_n_steps: int = 250,
        eval_num_samples: int = 300,
        eval_inference_steps: int = 25,
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
        # Memory optimization
        memory_cleanup_steps: int = 50,
        max_memory_gb: float = 32.0,
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
        self.task_mode = task_mode or self._detect_task_mode()
        
        # DDP configuration
        self.device = device or torch.device("cpu")
        self.rank = rank
        self.world_size = world_size
        self.is_main_process = (rank == 0)
        self.is_distributed = (world_size > 1)
        
        # Output
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Memory optimization
        self.memory_cleanup_steps = memory_cleanup_steps
        self.max_memory_gb = max_memory_gb
        
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
        
        # Metrics tracking
        self.loss_history = deque(maxlen=1000)
        self.similarity_history = deque(maxlen=1000)
        self.memory_history = deque(maxlen=100)
        self.grad_norm_history = deque(maxlen=1000)
        
        # Error tracking
        self.error_count = 0
        self.max_errors = 10
        
        # Memory monitoring
        self.peak_memory_gb = 0.0
        self.memory_warnings = 0
        
        # Overfitting test data (only on rank 0)
        self.overfit_batch = None
        if self.overfit_test_size and self.is_main_process:
            self._prepare_overfit_test()
        
        # Log initialization
        if self.is_main_process:
            self._log_initialization()

    def _log_initialization(self):
        """Log initialization info (rank 0 only)"""
        task_info = self._get_task_info()
        logger.info("FIXED DDP Universal Denoising Trainer initialized")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Rank: {self.rank}/{self.world_size}")
        logger.info(f"  Distributed: {self.is_distributed}")
        logger.info(f"  Task: {task_info['task']}")
        logger.info(f"  Learning rate: {self.learning_rate}")
        logger.info(f"  Epochs: {self.num_epochs}")
        logger.info(f"  Gradient accumulation: {self.gradient_accumulation_steps}")
        logger.info(f"  Mixed precision: {self.fp16}")
        logger.info(f"  Memory limit: {self.max_memory_gb}GB")
        logger.info(f"  Overfit test: {self.overfit_test_size if self.overfit_test_size else 'Disabled'}")

    def get_memory_usage_gb(self) -> float:
        """Get current memory usage in GB"""
        try:
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated(self.device) / 1024**3
            else:
                process = psutil.Process()
                return process.memory_info().rss / 1024**3
        except:
            return 0.0

    def _aggressive_memory_cleanup(self):
        """Aggressive memory cleanup"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # Additional cleanup for DDP
            if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                torch.cuda.reset_peak_memory_stats()

    def _check_memory_usage(self):
        """Check and manage memory usage"""
        current_memory = self.get_memory_usage_gb()
        self.memory_history.append(current_memory)
        
        if current_memory > self.peak_memory_gb:
            self.peak_memory_gb = current_memory
        
        # Memory warning threshold
        if current_memory > self.max_memory_gb * 0.8:
            self.memory_warnings += 1
            if self.is_main_process and self.memory_warnings % 10 == 1:
                logger.warning(f"High memory usage: {current_memory:.1f}GB (limit: {self.max_memory_gb}GB)")
        
        # Emergency cleanup
        if current_memory > self.max_memory_gb * 0.95:
            if self.is_main_process:
                logger.warning(f"Emergency memory cleanup triggered at {current_memory:.1f}GB")
            self._aggressive_memory_cleanup()

    def _detect_task_mode(self) -> str:
        """Auto-detect task mode from model configuration"""
        model_ref = self.model.module if hasattr(self.model, 'module') else self.model
        
        if hasattr(model_ref, 'config') and hasattr(model_ref.config, 'task_mode'):
            return model_ref.config.task_mode
        
        return "clip_denoising"  # Default

    def _get_task_info(self) -> Dict[str, str]:
        """Get task-specific information"""
        if self.task_mode == "eva_denoising":
            return {
                "task": "EVA-CLIP Denoising",
                "input": "Noisy EVA [B, N, 4096]",
                "target": "Clean EVA [B, N, 4096]",
            }
        elif self.task_mode == "clip_denoising":
            return {
                "task": "CLIP-ViT Denoising with EVA Conditioning",
                "input": "Noisy CLIP [B, N, 1024]",
                "target": "Clean CLIP [B, N, 1024]",
            }
        else:
            return {"task": "Unknown", "input": "Unknown", "target": "Unknown"}

    def _setup_optimizer_and_scheduler(self):
        """Setup optimizer and learning rate scheduler"""
        # Get parameters to optimize
        if hasattr(self.model, 'module'):
            params = self.model.module.parameters()
        else:
            params = self.model.parameters()
        
        # Use AdamW with conservative settings
        self.optimizer = AdamW(
            params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
            eps=1e-8
        )
        
        # Estimate total steps
        try:
            if hasattr(self.train_dataloader, '__len__'):
                dataloader_length = len(self.train_dataloader)
            else:
                # Conservative estimate for iterable datasets
                estimated_samples = 10000 // max(self.world_size, 1)
                batch_size = getattr(self.train_dataloader, 'batch_size', 2)
                dataloader_length = estimated_samples // batch_size
            
            total_steps = (dataloader_length * self.num_epochs) // self.gradient_accumulation_steps
        except:
            total_steps = 1000  # Fallback
        
        # Setup scheduler with warmup
        if self.warmup_steps > 0:
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=self.warmup_steps
            )
            cosine_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=max(total_steps - self.warmup_steps, 1),
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
                T_max=max(total_steps, 1),
                eta_min=self.learning_rate * 0.01
            )
        
        if self.is_main_process:
            logger.info(f"Optimizer setup: {total_steps} estimated steps, warmup: {self.warmup_steps}")

    def _prepare_overfit_test(self):
        """Prepare overfitting test batch (rank 0 only)"""
        if not self.is_main_process:
            return
            
        try:
            logger.info(f"Preparing overfitting test with {self.overfit_test_size} samples...")
            
            first_batch = next(iter(self.train_dataloader))
            actual_size = min(self.overfit_test_size, first_batch.get('batch_size', len(first_batch.get('keys', []))))
            
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

    def _safe_move_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Safely move batch to device with error handling"""
        try:
            for key, value in batch.items():
                if torch.is_tensor(value):
                    batch[key] = value.to(self.device, non_blocking=True)
            return batch
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(f"OOM error moving batch to device: {e}")
                self._aggressive_memory_cleanup()
                # Try again with smaller batch or fallback
                raise
            else:
                raise

    def _compute_loss_robust(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Robust loss computation with error handling"""
        
        # Use overfit batch if specified (only on rank 0)
        if self.overfit_batch is not None and self.is_main_process:
            batch = self.overfit_batch.copy()
        
        # Move batch to device
        batch = self._safe_move_to_device(batch)
        
        # Extract inputs
        try:
            x_t = batch['hidden_states']
            timestep = batch['timestep']
            conditioning = batch['encoder_hidden_states']
            target = batch['target_embeddings']
            velocity_target = batch.get('velocity_target')
            noise = batch.get('noise')
            task_mode = batch.get('task_mode', self.task_mode)
        except KeyError as e:
            raise ValueError(f"Missing required key in batch: {e}")
        
        # Forward pass with mixed precision and error handling
        try:
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
        
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(f"OOM during forward pass: {e}")
                self._aggressive_memory_cleanup()
                raise
            else:
                raise
        
        # Scale loss for gradient accumulation
        loss = loss / self.gradient_accumulation_steps
        
        # Check for numerical issues
        if torch.isnan(loss) or torch.isinf(loss):
            logger.error(f"Numerical issue in loss: {loss.item()}")
            raise ValueError("NaN or Inf in loss")
        
        return loss, metrics or {}

    def _backward_and_step_robust(self, loss: torch.Tensor, step_in_accumulation: int) -> float:
        """Robust backward pass with error handling"""
        
        try:
            # Backward pass
            if self.fp16:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Only step optimizer after accumulation is complete
            if (step_in_accumulation + 1) % self.gradient_accumulation_steps == 0:
                
                # Compute gradient norm before clipping
                grad_norm = 0.0
                param_count = 0
                
                if hasattr(self.model, 'module'):
                    parameters = self.model.module.parameters()
                else:
                    parameters = self.model.parameters()
                
                for param in parameters:
                    if param.grad is not None:
                        grad_norm += param.grad.data.norm(2).item() ** 2
                        param_count += 1
                
                grad_norm = grad_norm ** 0.5
                
                # Check for gradient explosion
                if grad_norm > 100:
                    logger.warning(f"Large gradient norm detected: {grad_norm:.2f}")
                
                # Gradient clipping
                if self.max_grad_norm > 0:
                    if self.fp16:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters() if not hasattr(self.model, 'module') else self.model.module.parameters(),
                            self.max_grad_norm
                        )
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters() if not hasattr(self.model, 'module') else self.model.module.parameters(),
                            self.max_grad_norm
                        )
                
                # Optimizer step
                if self.fp16:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad(set_to_none=True)  # More memory efficient
                self.scheduler.step()
                
                return grad_norm
            else:
                return 0.0  # No step taken
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(f"OOM during backward pass: {e}")
                self._aggressive_memory_cleanup()
                raise
            else:
                raise

    def _sync_metrics_across_ranks(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Synchronize metrics across all ranks"""
        if not self.is_distributed:
            return metrics
        
        synced_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                try:
                    tensor_value = torch.tensor(value, device=self.device, dtype=torch.float32)
                    dist.all_reduce(tensor_value, op=dist.ReduceOp.AVG)
                    synced_metrics[key] = tensor_value.item()
                except Exception as e:
                    logger.warning(f"Failed to sync metric {key}: {e}")
                    synced_metrics[key] = value
            else:
                synced_metrics[key] = value
        
        return synced_metrics

    def _log_metrics(self, loss: float, metrics: Dict[str, float], grad_norm: float):
        """Log training metrics (only on rank 0)"""
        if not self.is_main_process:
            return
        
        # Store metrics
        unscaled_loss = loss * self.gradient_accumulation_steps
        self.loss_history.append(unscaled_loss)
        
        if 'prediction_similarity' in metrics:
            self.similarity_history.append(metrics['prediction_similarity'])
        
        current_memory = self.get_memory_usage_gb()
        self.memory_history.append(current_memory)
        
        if grad_norm > 0:
            self.grad_norm_history.append(grad_norm)
        
        # Update best metrics
        eval_sim = metrics.get('eval_similarity', metrics.get('prediction_similarity', 0))
        if eval_sim > self.best_eval_similarity:
            self.best_eval_similarity = eval_sim
        if unscaled_loss < self.best_loss:
            self.best_loss = unscaled_loss
        
        # Console logging
        if self.global_step % self.log_every_n_steps == 0:
            task_name = "EVA" if self.task_mode == "eva_denoising" else "CLIP"
            log_msg = f"Step {self.global_step} [{task_name}]: Loss={unscaled_loss:.6f}"
            
            if 'prediction_similarity' in metrics:
                sim = metrics['prediction_similarity']
                quality = metrics.get('quality_assessment', 'unknown')
                log_msg += f", Sim={sim:.4f} ({quality})"
            
            if grad_norm > 0:
                log_msg += f", GradNorm={grad_norm:.3f}"
            
            log_msg += f", LR={self.optimizer.param_groups[0]['lr']:.2e}"
            log_msg += f", Mem={current_memory:.1f}GB"
            
            if self.overfit_batch is not None:
                log_msg += " [OVERFIT]"
            
            logger.info(log_msg)

    def _evaluate(self, num_samples: Optional[int] = None) -> Dict[str, float]:
        """Run evaluation (only on rank 0)"""
        if not self.is_main_process or self.eval_dataloader is None:
            return {}
        
        if num_samples is None:
            num_samples = self.eval_num_samples
        
        model_ref = self.model.module if hasattr(self.model, 'module') else self.model
        model_ref.eval()
        
        all_similarities = []
        samples_processed = 0
        
        with torch.no_grad():
            for batch in self.eval_dataloader:
                if samples_processed >= num_samples:
                    break
                
                try:
                    batch = self._safe_move_to_device(batch)
                    
                    input_embeddings = batch['input_embeddings']
                    conditioning = batch['encoder_hidden_states']
                    target = batch['target_embeddings']
                    
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
                    
                    all_similarities.append(per_image_similarity.cpu())
                    samples_processed += input_embeddings.shape[0]
                    
                except Exception as e:
                    logger.warning(f"Error in evaluation batch: {e}")
                    continue
        
        model_ref.train()
        
        if not all_similarities:
            return {}
        
        all_sims = torch.cat(all_similarities)
        
        # Task-specific metrics
        if self.task_mode == "eva_denoising":
            metric_prefix = "eval_eva"
            high_thresh, excellent_thresh = 0.7, 0.9
        else:
            metric_prefix = "eval_clip"
            high_thresh, excellent_thresh = 0.6, 0.8
        
        return {
            f'{metric_prefix}_similarity': all_sims.mean().item(),
            f'{metric_prefix}_similarity_std': all_sims.std().item(),
            f'{metric_prefix}_high_quality': (all_sims > high_thresh).float().mean().item(),
            f'{metric_prefix}_excellent_quality': (all_sims > excellent_thresh).float().mean().item(),
            f'{metric_prefix}_samples': samples_processed,
        }

    def _save_checkpoint(self):
        """Save checkpoint (only on rank 0)"""
        if not self.is_main_process:
            return
        
        checkpoint_path = self.output_dir / f"checkpoint_step_{self.global_step}.pt"
        
        try:
            # Get model state dict
            if hasattr(self.model, 'module'):
                model_state_dict = self.model.module.state_dict()
            else:
                model_state_dict = self.model.state_dict()
            
            checkpoint = {
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'global_step': self.global_step,
                'current_epoch': self.current_epoch,
                'best_eval_similarity': self.best_eval_similarity,
                'best_loss': self.best_loss,
                'task_mode': self.task_mode,
                'loss_history': list(self.loss_history),
                'similarity_history': list(self.similarity_history),
                'memory_history': list(self.memory_history),
                'peak_memory_gb': self.peak_memory_gb,
            }
            
            if self.scaler is not None:
                checkpoint['scaler_state_dict'] = self.scaler.state_dict()
            
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def train(self) -> Dict[str, Any]:
        """FIXED: Main training loop with robust error handling"""
        
        if self.is_main_process:
            task_info = self._get_task_info()
            logger.info(f"Starting FIXED DDP {task_info['task']} training...")
            logger.info(f"  World size: {self.world_size}")
            logger.info(f"  Gradient accumulation: {self.gradient_accumulation_steps}")
            
            if hasattr(self.model, 'module'):
                param_count = sum(p.numel() for p in self.model.module.parameters() if p.requires_grad)
            else:
                param_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            logger.info(f"  Model parameters: {param_count:,}")
        
        # Synchronize all ranks before starting
        if self.is_distributed:
            dist.barrier()
        
        self.model.train()
        start_time = time.time()
        
        try:
            for epoch in range(self.num_epochs):
                self.current_epoch = epoch
                
                if self.is_main_process:
                    logger.info(f"Starting epoch {epoch + 1}/{self.num_epochs}")
                
                epoch_loss = 0.0
                accumulation_step = 0
                successful_batches = 0
                
                for batch_idx, batch in enumerate(self.train_dataloader):
                    try:
                        # Memory check
                        self._check_memory_usage()
                        
                        # Compute loss
                        loss, metrics = self._compute_loss_robust(batch)
                        
                        # Backward pass
                        grad_norm = self._backward_and_step_robust(loss, accumulation_step)
                        
                        # Update metrics
                        epoch_loss += loss.item()
                        accumulation_step += 1
                        
                        # Only increment global step after full accumulation
                        if accumulation_step % self.gradient_accumulation_steps == 0:
                            self.global_step += 1
                            successful_batches += 1
                            
                            # Sync metrics across ranks
                            if metrics and self.is_distributed:
                                metrics = self._sync_metrics_across_ranks(metrics)
                            
                            # Log metrics
                            self._log_metrics(loss.item(), metrics, grad_norm)
                            
                            # Run evaluation
                            if (self.global_step % self.eval_every_n_steps == 0 and 
                                self.is_main_process):
                                eval_metrics = self._evaluate()
                                if eval_metrics:
                                    logger.info("Evaluation results:")
                                    for key, value in eval_metrics.items():
                                        logger.info(f"  {key}: {value:.4f}")
                            
                            # Save checkpoint
                            if (self.global_step % self.save_every_n_steps == 0 and 
                                self.is_main_process):
                                self._save_checkpoint()
                        
                        # Periodic memory cleanup
                        if batch_idx % self.memory_cleanup_steps == 0:
                            self._aggressive_memory_cleanup()
                        
                        # Sync periodically for distributed training
                        if self.is_distributed and batch_idx % 100 == 0:
                            dist.barrier()
                    
                    except Exception as e:
                        self.error_count += 1
                        
                        if self.is_main_process:
                            logger.error(f"Error in batch {batch_idx}: {e}")
                        
                        # Emergency cleanup
                        self._aggressive_memory_cleanup()
                        
                        # Skip batch and continue
                        if self.error_count < self.max_errors:
                            continue
                        else:
                            logger.error(f"Too many errors ({self.error_count}), stopping training")
                            break
                
                # End of epoch summary
                if self.is_main_process and successful_batches > 0:
                    avg_loss = epoch_loss / max(accumulation_step, 1) * self.gradient_accumulation_steps
                    logger.info(f"Epoch {epoch + 1} completed:")
                    logger.info(f"  Average loss: {avg_loss:.6f}")
                    logger.info(f"  Successful batches: {successful_batches}")
                    logger.info(f"  Memory: {self.get_memory_usage_gb():.1f}GB")
                    logger.info(f"  Errors: {self.error_count}")
                
                # Epoch sync
                if self.is_distributed:
                    dist.barrier()
        
        except KeyboardInterrupt:
            if self.is_main_process:
                logger.info("Training interrupted by user")
        except Exception as e:
            if self.is_main_process:
                logger.error(f"Training failed: {e}")
            raise
        
        finally:
            # Final sync and cleanup
            if self.is_distributed:
                dist.barrier()
            
            if self.is_main_process:
                self._save_checkpoint()
                
                # Final evaluation
                final_eval = self._evaluate()
                
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
                    'error_count': self.error_count,
                    'peak_memory_gb': self.peak_memory_gb,
                    'memory_warnings': self.memory_warnings,
                    'world_size': self.world_size,
                    'gradient_accumulation_steps': self.gradient_accumulation_steps,
                }
                
                logger.info(f"FIXED DDP training completed!")
                logger.info(f"  Total time: {total_time:.1f}s")
                logger.info(f"  Total steps: {self.global_step}")
                logger.info(f"  Best loss: {self.best_loss:.6f}")
                logger.info(f"  Best similarity: {self.best_eval_similarity:.4f}")
                logger.info(f"  Peak memory: {self.peak_memory_gb:.1f}GB")
                logger.info(f"  Errors: {self.error_count}")
                
                return summary
            else:
                return {'training_completed': True, 'rank': self.rank}


def create_fixed_ddp_trainer(
    model,
    loss_fn,
    train_dataloader,
    eval_dataloader=None,
    learning_rate: float = 1e-4,
    num_epochs: int = 5,
    gradient_accumulation_steps: int = 8,
    max_memory_gb: float = 32.0,
    output_dir: str = "./checkpoints",
    task_mode: Optional[str] = None,
    device: Optional[torch.device] = None,
    rank: int = 0,
    world_size: int = 1,
    **kwargs
) -> FixedDDPDenoisingTrainer:
    """Factory function to create fixed DDP trainer"""
    
    return FixedDDPDenoisingTrainer(
        model=model,
        loss_fn=loss_fn,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_memory_gb=max_memory_gb,
        output_dir=output_dir,
        task_mode=task_mode,
        device=device,
        rank=rank,
        world_size=world_size,
        **kwargs
    )