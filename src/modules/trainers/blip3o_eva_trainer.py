#!/usr/bin/env python3
"""
FIXED Enhanced BLIP3-o Trainer - Support for Both EVA and CLIP Denoising with WandB Integration
Key features:
1. Universal training loop for both EVA and CLIP denoising
2. Task-specific evaluation metrics
3. Automatic task detection and validation
4. Comprehensive monitoring and debugging
5. WandB logging integration for metrics visualization

FIXES:
- Fixed deque slice indexing issue
- Fixed string formatting for non-numeric values
- Better type checking for metrics logging
- Safe access to deque elements
"""

import torch
import torch.nn.functional as F
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

logger = logging.getLogger(__name__)


class UniversalDenoisingTrainer:
    """
    Universal trainer for both EVA and CLIP denoising with comprehensive monitoring and WandB integration
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
        fp16: bool = False,
        # Evaluation
        eval_every_n_steps: int = 100,
        eval_num_samples: int = 500,
        eval_inference_steps: int = 50,
        # Debugging
        debug_mode: bool = False,
        overfit_test_size: Optional[int] = None,
        log_every_n_steps: int = 10,
        save_every_n_steps: int = 500,
        # Task configuration
        task_mode: Optional[str] = None,  # NEW: "eva_denoising" or "clip_denoising"
        # Output
        output_dir: str = "./checkpoints",
        # Device
        device: Optional[torch.device] = None,
        # WandB integration - NEW!
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
        self.task_mode = task_mode  # Will be auto-detected if None
        
        # Output
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Device
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.model = self.model.to(self.device)
        
        # WandB integration - NEW!
        self.wandb = wandb_instance
        self.use_wandb = wandb_instance is not None
        
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
        
        # Metrics tracking (universal for both tasks)
        self.loss_history = deque(maxlen=1000)
        self.similarity_history = deque(maxlen=1000)
        self.eval_similarity_history = deque(maxlen=1000)
        self.lr_history = deque(maxlen=1000)
        self.grad_norm_history = deque(maxlen=1000)
        self.sphere_violation_history = deque(maxlen=1000)
        
        # Task-specific tracking
        self.eva_similarity_history = deque(maxlen=1000)
        self.clip_similarity_history = deque(maxlen=1000)
        
        # Overfitting test data
        self.overfit_batch = None
        if self.overfit_test_size:
            self._prepare_overfit_test()
        
        # Auto-detect task mode if not provided
        if self.task_mode is None:
            self.task_mode = self._detect_task_mode()
        
        # Log initialization
        task_info = self._get_task_info()
        logger.info("Universal Denoising Trainer initialized")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Task: {task_info['task']}")
        logger.info(f"  Input: {task_info['input']}")
        logger.info(f"  Conditioning: {task_info['conditioning']}")
        logger.info(f"  Target: {task_info['target']}")
        logger.info(f"  Learning rate: {self.learning_rate}")
        logger.info(f"  Epochs: {self.num_epochs}")
        logger.info(f"  Prediction type: {getattr(self.model.config, 'prediction_type', 'velocity')}")
        logger.info(f"  Overfit test: {self.overfit_test_size if self.overfit_test_size else 'Disabled'}")
        logger.info(f"  Mixed precision: {self.fp16}")
        logger.info(f"  WandB logging: {self.use_wandb}")
        if self.use_wandb and self.wandb:
            logger.info(f"  WandB URL: {self.wandb.url}")

    def _detect_task_mode(self) -> str:
        """Auto-detect task mode from model configuration"""
        if hasattr(self.model, 'config') and hasattr(self.model.config, 'task_mode'):
            return self.model.config.task_mode
        
        # Try to detect from first batch
        try:
            first_batch = next(iter(self.train_dataloader))
            if 'task_mode' in first_batch:
                detected_mode = first_batch['task_mode']
                logger.info(f"Auto-detected task mode: {detected_mode}")
                return detected_mode
        except Exception as e:
            logger.warning(f"Could not auto-detect task mode: {e}")
        
        # Default fallback
        logger.warning("Could not detect task mode, defaulting to eva_denoising")
        return "eva_denoising"

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
        
        # Setup learning rate scheduler with warmup
        try:
            dataloader_length = len(self.train_dataloader)
            total_steps = dataloader_length * self.num_epochs
        except (TypeError, AttributeError):
            logger.warning("Cannot determine exact dataloader length, using estimate")
            estimated_samples_per_epoch = 10000
            batch_size = getattr(self.train_dataloader, 'batch_size', 16)
            estimated_batches_per_epoch = estimated_samples_per_epoch // batch_size
            total_steps = estimated_batches_per_epoch * self.num_epochs
        
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
        
        logger.info(f"Optimizer and scheduler setup complete")
        logger.info(f"  Estimated total steps: {total_steps}")
        logger.info(f"  Warmup steps: {self.warmup_steps}")

    def _prepare_overfit_test(self):
        """Prepare overfitting test batch"""
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
        """Compute loss for a batch (universal for both tasks)"""
        # Move batch to device
        for key, value in batch.items():
            if torch.is_tensor(value):
                batch[key] = value.to(self.device)
        
        # Use overfit batch if specified
        if self.overfit_batch is not None:
            for key, value in self.overfit_batch.items():
                if torch.is_tensor(value):
                    batch[key] = value.to(self.device)
                else:
                    batch[key] = value
        
        # Extract inputs (universal interface)
        x_t = batch['hidden_states']                    # [B, N, input_dim] - Current flow state
        timestep = batch['timestep']                    # [B] - Timesteps
        conditioning = batch['encoder_hidden_states']   # [B, N, conditioning_dim] - Conditioning
        target = batch['target_embeddings']             # [B, N, output_dim] - Target
        velocity_target = batch.get('velocity_target')  # [B, N, input_dim] - Velocity target
        noise = batch.get('noise')                      # [B, N, input_dim] - Noise
        task_mode = batch.get('task_mode', self.task_mode)  # Task mode
        
        # Forward pass
        if self.fp16:
            with torch.amp.autocast('cuda'):
                model_output = self.model(
                    hidden_states=x_t,
                    timestep=timestep,
                    encoder_hidden_states=conditioning,
                    return_dict=False
                )
                
                # Compute universal spherical flow matching loss
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
        
        return loss, metrics

    def _backward_and_step(self, loss: torch.Tensor) -> float:
        """Backward pass and optimizer step"""
        # Backward pass
        if self.fp16:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
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

    def _evaluate(self, num_samples: Optional[int] = None) -> Dict[str, float]:
        """Run universal denoising evaluation"""
        if self.eval_dataloader is None:
            return {}
        
        if num_samples is None:
            num_samples = self.eval_num_samples
        
        self.model.eval()
        
        all_similarities = []
        all_angular_distances = []
        all_sphere_violations = []
        samples_processed = 0
        task_mode_detected = None
        
        with torch.no_grad():
            for batch in self.eval_dataloader:
                if samples_processed >= num_samples:
                    break
                
                # Move to device - use the correct keys from universal collate function
                try:
                    # Use the actual keys from the universal collate function
                    input_embeddings = batch['input_embeddings'].to(self.device)
                    conditioning = batch['encoder_hidden_states'].to(self.device)  # This is the correct key!
                    target = batch['target_embeddings'].to(self.device)
                    task_mode_detected = batch.get('task_mode', self.task_mode)
                except KeyError as e:
                    logger.warning(f"Missing key in evaluation batch: {e}")
                    logger.warning(f"Available keys: {list(batch.keys())}")
                    # Skip this batch and continue
                    continue
                
                # Denoise using the model
                denoised = self.model.denoise(
                    noisy_embeddings=input_embeddings,
                    conditioning=conditioning,
                    num_inference_steps=self.eval_inference_steps,
                )
                
                # Compute similarity
                target_norm = F.normalize(target, p=2, dim=-1)
                denoised_norm = F.normalize(denoised, p=2, dim=-1)
                
                # Cosine similarity (main metric)
                similarity = F.cosine_similarity(denoised_norm, target_norm, dim=-1)
                per_image_similarity = similarity.mean(dim=1)
                
                # Angular distance
                cos_sim_clamped = torch.clamp(similarity, -1 + 1e-7, 1 - 1e-7)
                angular_distance = torch.acos(cos_sim_clamped).mean(dim=1)
                
                # Sphere constraint violation
                denoised_norms = torch.norm(denoised, dim=-1)
                sphere_violation = torch.abs(denoised_norms - 1.0).mean(dim=1)
                
                all_similarities.append(per_image_similarity.cpu())
                all_angular_distances.append(angular_distance.cpu())
                all_sphere_violations.append(sphere_violation.cpu())
                samples_processed += input_embeddings.shape[0]
        
        self.model.train()
        
        if not all_similarities:
            return {}
        
        all_sims = torch.cat(all_similarities)
        all_angular = torch.cat(all_angular_distances)
        all_violations = torch.cat(all_sphere_violations)
        
        # Task-specific metric names and thresholds
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
        
        # Convert to list and slice
        deque_list = list(deque_obj)
        return deque_list[-slice_size:] if len(deque_list) >= slice_size else deque_list

    def _safe_format_value(self, key: str, value: Any, decimal_places: int = 4) -> str:
        """Safely format a value for logging, handling different types"""
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
        """Log training metrics (task-aware) with WandB integration"""
        # Store metrics
        self.loss_history.append(loss)
        if 'prediction_similarity' in metrics:
            self.similarity_history.append(metrics['prediction_similarity'])
        if 'eval_similarity' in metrics:
            self.eval_similarity_history.append(metrics['eval_similarity'])
        if 'sphere_violation' in metrics:
            self.sphere_violation_history.append(metrics['sphere_violation'])
        
        # Task-specific similarity tracking
        task_mode = metrics.get('task_mode', self.task_mode)
        eval_sim = metrics.get('eval_similarity', metrics.get('prediction_similarity', 0))
        
        if task_mode == "eva_denoising":
            self.eva_similarity_history.append(eval_sim)
        elif task_mode == "clip_denoising":
            self.clip_similarity_history.append(eval_sim)
        
        self.lr_history.append(self.optimizer.param_groups[0]['lr'])
        self.grad_norm_history.append(grad_norm)
        
        # Update best metrics
        if eval_sim > self.best_eval_similarity:
            self.best_eval_similarity = eval_sim
        
        if loss < self.best_loss:
            self.best_loss = loss
        
        # WandB logging - NEW!
        if self.use_wandb and self.wandb:
            try:
                wandb_metrics = {
                    'train/loss': loss,
                    'train/grad_norm': grad_norm,
                    'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                    'train/step': self.global_step,
                    'train/epoch': self.current_epoch,
                }
                
                # Add loss function metrics
                for key, value in metrics.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        if key in ['prediction_similarity', 'eval_similarity']:
                            wandb_metrics[f'train/{key}'] = value
                        elif key == 'sphere_violation':
                            wandb_metrics[f'train/{key}'] = value
                        elif key == 'quality_assessment':
                            continue  # Skip string values
                        elif key in ['loss', 'total_loss']:
                            wandb_metrics[f'train/{key}'] = value
                        elif key.startswith('task_'):
                            continue  # Skip task metadata
                        else:
                            wandb_metrics[f'train/metrics/{key}'] = value
                
                # Add task-specific metrics
                if task_mode:
                    wandb_metrics['train/task_mode'] = task_mode
                
                # Add best metrics
                wandb_metrics['train/best_loss'] = self.best_loss
                wandb_metrics['train/best_eval_similarity'] = self.best_eval_similarity
                
                # Log to WandB
                self.wandb.log(wandb_metrics, step=self.global_step)
                
            except Exception as e:
                logger.warning(f"Failed to log to WandB: {e}")
        
        # Log to console
        if self.global_step % self.log_every_n_steps == 0:
            task_name = "EVA" if task_mode == "eva_denoising" else "CLIP" if task_mode == "clip_denoising" else "UNK"
            log_msg = f"Step {self.global_step} [{task_name}]: Loss={loss:.6f}"
            
            if 'prediction_similarity' in metrics:
                pred_sim = metrics['prediction_similarity']
                eval_sim = metrics.get('eval_similarity', pred_sim)
                quality = metrics.get('quality_assessment', 'unknown')
                log_msg += f", PredSim={pred_sim:.4f}, EvalSim={eval_sim:.4f} ({quality})"
            
            if 'sphere_violation' in metrics:
                sphere_viol = metrics['sphere_violation']
                log_msg += f", SphereViol={sphere_viol:.6f}"
            
            # Add dimensional info
            if 'output_dim' in metrics:
                out_dim = metrics['output_dim']
                cond_dim = metrics.get('conditioning_dim', 'unknown')
                log_msg += f", Dims={out_dim}|{cond_dim}"
            
            log_msg += f", GradNorm={grad_norm:.3f}"
            log_msg += f", LR={self.optimizer.param_groups[0]['lr']:.2e}"
            
            if self.overfit_batch is not None:
                log_msg += " [OVERFIT TEST]"
            
            logger.info(log_msg)
            
            # Detailed logging in debug mode
            if self.debug_mode:
                logger.info(f"  Detailed metrics for {task_mode}:")
                for key, value in metrics.items():
                    formatted_value = self._safe_format_value(key, value, decimal_places=6)
                    logger.info(f"    {formatted_value}")

    def _log_evaluation_metrics(self, eval_metrics: Dict[str, float]):
        """Log evaluation metrics with WandB integration"""
        if self.use_wandb and self.wandb and eval_metrics:
            try:
                wandb_eval_metrics = {}
                for key, value in eval_metrics.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        wandb_eval_metrics[f'eval/{key}'] = value
                    elif isinstance(value, str) and key == 'eval_task_mode':
                        wandb_eval_metrics['eval/task_mode'] = value
                
                # Add step information
                wandb_eval_metrics['eval/step'] = self.global_step
                wandb_eval_metrics['eval/epoch'] = self.current_epoch
                
                self.wandb.log(wandb_eval_metrics, step=self.global_step)
                
            except Exception as e:
                logger.warning(f"Failed to log evaluation metrics to WandB: {e}")

    def _save_checkpoint(self):
        """Save model checkpoint"""
        checkpoint_path = self.output_dir / f"checkpoint_step_{self.global_step}.pt"
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'current_epoch': self.current_epoch,
            'best_eval_similarity': self.best_eval_similarity,
            'best_loss': self.best_loss,
            'task_mode': self.task_mode,
            'loss_history': list(self.loss_history),
            'similarity_history': list(self.similarity_history),
            'eval_similarity_history': list(self.eval_similarity_history),
            'eva_similarity_history': list(self.eva_similarity_history),
            'clip_similarity_history': list(self.clip_similarity_history),
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Log checkpoint to WandB
        if self.use_wandb and self.wandb:
            try:
                self.wandb.log({
                    'checkpoint/step': self.global_step,
                    'checkpoint/best_loss': self.best_loss,
                    'checkpoint/best_similarity': self.best_eval_similarity,
                }, step=self.global_step)
            except Exception as e:
                logger.warning(f"Failed to log checkpoint info to WandB: {e}")

    def train(self) -> Dict[str, Any]:
        """Main training loop (universal for both tasks) with WandB integration"""
        task_info = self._get_task_info()
        
        logger.info(f"Starting {task_info['task']} training...")
        logger.info(f"  Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Handle dataloader length safely
        try:
            steps_per_epoch = len(self.train_dataloader)
            total_training_steps = steps_per_epoch * self.num_epochs
            logger.info(f"  Training steps per epoch: {steps_per_epoch}")
            logger.info(f"  Total training steps: {total_training_steps}")
        except (TypeError, AttributeError):
            logger.info(f"  Training steps per epoch: Unknown (IterableDataset)")
            logger.info(f"  Total training steps: Estimated")
        
        if self.overfit_batch is not None:
            logger.info(f"  OVERFITTING TEST MODE: Using {self.overfit_batch['batch_size']} samples")
        
        if self.use_wandb and self.wandb:
            logger.info(f"  WandB logging enabled: {self.wandb.url}")
        
        self.model.train()
        start_time = time.time()
        
        try:
            for epoch in range(self.num_epochs):
                self.current_epoch = epoch
                logger.info(f"Starting epoch {epoch + 1}/{self.num_epochs}")
                
                epoch_loss = 0.0
                epoch_steps = 0
                
                for batch_idx, batch in enumerate(self.train_dataloader):
                    step_start_time = time.time()
                    
                    # Compute loss
                    try:
                        loss, metrics = self._compute_loss(batch)
                    except Exception as e:
                        logger.error(f"Error computing loss at step {self.global_step}: {e}")
                        continue
                    
                    # Backward pass
                    try:
                        grad_norm = self._backward_and_step(loss)
                    except Exception as e:
                        logger.error(f"Error in backward pass at step {self.global_step}: {e}")
                        continue
                    
                    # Update metrics
                    epoch_loss += loss.item()
                    epoch_steps += 1
                    self.global_step += 1
                    
                    # Log metrics (includes WandB logging)
                    self._log_metrics(loss.item(), metrics or {}, grad_norm)
                    
                    # Run evaluation
                    if self.global_step % self.eval_every_n_steps == 0:
                        logger.info(f"Running evaluation at step {self.global_step}...")
                        eval_metrics = self._evaluate()
                        
                        if eval_metrics:
                            task_mode = eval_metrics.get('eval_task_mode', self.task_mode)
                            metric_prefix = "eval_eva" if task_mode == "eva_denoising" else "eval_clip"
                            
                            logger.info(f"Evaluation results for {task_mode}:")
                            for key, value in eval_metrics.items():
                                formatted_value = self._safe_format_value(key, value, decimal_places=4)
                                logger.info(f"  {formatted_value}")
                            
                            # Log evaluation metrics to WandB
                            self._log_evaluation_metrics(eval_metrics)
                            
                            # Update best eval similarity
                            main_sim_key = f'{metric_prefix}_similarity'
                            if main_sim_key in eval_metrics and eval_metrics[main_sim_key] > self.best_eval_similarity:
                                self.best_eval_similarity = eval_metrics[main_sim_key]
                                logger.info(f"New best {task_mode} similarity: {self.best_eval_similarity:.4f}")
                                
                                # Log new best to WandB
                                if self.use_wandb and self.wandb:
                                    try:
                                        self.wandb.log({
                                            'best/similarity': self.best_eval_similarity,
                                            'best/similarity_step': self.global_step,
                                        }, step=self.global_step)
                                    except Exception as e:
                                        logger.warning(f"Failed to log best metrics to WandB: {e}")
                    
                    # Save checkpoint
                    if self.global_step % self.save_every_n_steps == 0:
                        self._save_checkpoint()
                    
                    # Check for early success in overfitting test
                    if (self.overfit_batch is not None and 
                        metrics and 
                        metrics.get('eval_similarity', 0) > 0.9):
                        logger.info("🎉 OVERFITTING TEST PASSED! Model can learn effectively.")
                        break
                
                # End of epoch logging
                avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
                logger.info(f"Epoch {epoch + 1} completed:")
                logger.info(f"  Average loss: {avg_epoch_loss:.6f}")
                logger.info(f"  Best loss: {self.best_loss:.6f}")
                logger.info(f"  Best similarity: {self.best_eval_similarity:.4f}")
                
                # Log epoch summary to WandB
                if self.use_wandb and self.wandb:
                    try:
                        self.wandb.log({
                            'epoch/avg_loss': avg_epoch_loss,
                            'epoch/best_loss': self.best_loss,
                            'epoch/best_similarity': self.best_eval_similarity,
                            'epoch/number': epoch + 1,
                        }, step=self.global_step)
                    except Exception as e:
                        logger.warning(f"Failed to log epoch summary to WandB: {e}")
                
                # Early stopping for overfitting test - FIXED slice access
                if (self.overfit_batch is not None and 
                    len(self.eval_similarity_history) > 0):
                    # Safe access to last N elements
                    recent_similarities = self._safe_deque_slice(self.eval_similarity_history, 10)
                    if recent_similarities and max(recent_similarities) > 0.9:
                        logger.info("Overfitting test completed successfully!")
                        break
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            raise
        
        finally:
            # Final checkpoint
            self._save_checkpoint()
            
            # Final evaluation
            logger.info("Running final evaluation...")
            final_eval = self._evaluate(num_samples=self.eval_num_samples * 2)
            
            # Log final evaluation to WandB
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
                'overfit_test': self.overfit_batch is not None,
                'overfit_success': (self.overfit_batch is not None and 
                                  len(self.eval_similarity_history) > 0 and 
                                  max(self._safe_deque_slice(self.eval_similarity_history, 50)) > 0.8),
                'loss_history': list(self.loss_history),
                'similarity_history': list(self.similarity_history),
                'eval_similarity_history': list(self.eval_similarity_history),
                'eva_similarity_history': list(self.eva_similarity_history),
                'clip_similarity_history': list(self.clip_similarity_history),
                'lr_history': list(self.lr_history),
                'grad_norm_history': list(self.grad_norm_history),
                'sphere_violation_history': list(self.sphere_violation_history),
                'wandb_url': self.wandb.url if self.use_wandb and self.wandb else None,
            }
            
            # Save training summary
            summary_path = self.output_dir / "training_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"{task_info['task']} training completed!")
            logger.info(f"  Total time: {total_time:.1f} seconds")
            logger.info(f"  Total steps: {self.global_step}")
            logger.info(f"  Best loss: {self.best_loss:.6f}")
            logger.info(f"  Best similarity: {self.best_eval_similarity:.4f}")
            
            if final_eval:
                task_mode = final_eval.get('eval_task_mode', self.task_mode)
                metric_prefix = "eval_eva" if task_mode == "eva_denoising" else "eval_clip"
                main_sim_key = f'{metric_prefix}_similarity'
                
                if main_sim_key in final_eval:
                    final_sim = final_eval[main_sim_key]
                    logger.info(f"  Final {task_mode} evaluation:")
                    for key, value in final_eval.items():
                        formatted_value = self._safe_format_value(key, value, decimal_places=4)
                        logger.info(f"    {formatted_value}")
                    
                    # Success assessment
                    if task_mode == "eva_denoising":
                        if final_sim > 0.8:
                            logger.info("🎉 OUTSTANDING SUCCESS! EVA similarity > 0.8")
                        elif final_sim > 0.7:
                            logger.info("🎊 EXCELLENT SUCCESS! EVA similarity > 0.7")
                        elif final_sim > 0.5:
                            logger.info("✅ GOOD SUCCESS! EVA similarity > 0.5")
                        else:
                            logger.info(f"📈 Progress made: EVA similarity = {final_sim:.4f}")
                    
                    elif task_mode == "clip_denoising":
                        if final_sim > 0.7:
                            logger.info("🎉 OUTSTANDING SUCCESS! CLIP similarity > 0.7")
                        elif final_sim > 0.6:
                            logger.info("🎊 EXCELLENT SUCCESS! CLIP similarity > 0.6")
                        elif final_sim > 0.4:
                            logger.info("✅ GOOD SUCCESS! CLIP similarity > 0.4")
                        else:
                            logger.info(f"📈 Progress made: CLIP similarity = {final_sim:.4f}")
            
            # Overfitting test results
            if self.overfit_batch is not None:
                overfit_success = summary['overfit_success']
                logger.info(f"🧪 OVERFITTING TEST: {'✅ PASSED' if overfit_success else '❌ FAILED'}")
                if overfit_success:
                    logger.info("   ✅ Model can learn and memorize - architecture is working perfectly!")
                else:
                    logger.info("   ⚠️ Model struggles to overfit - may need hyperparameter tuning")
            
            if self.use_wandb and self.wandb:
                logger.info(f"📊 WandB training logs: {self.wandb.url}")
            
            return summary


def create_universal_trainer(
    model,
    loss_fn,
    train_dataloader,
    eval_dataloader=None,
    learning_rate: float = 1e-4,
    num_epochs: int = 10,
    output_dir: str = "./checkpoints",
    task_mode: Optional[str] = None,  # NEW: Auto-detect if None
    overfit_test_size: Optional[int] = None,
    debug_mode: bool = False,
    wandb_instance=None,  # NEW: WandB instance
    **kwargs
) -> UniversalDenoisingTrainer:
    """Factory function to create universal trainer for both EVA and CLIP denoising"""
    
    return UniversalDenoisingTrainer(
        model=model,
        loss_fn=loss_fn,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        output_dir=output_dir,
        task_mode=task_mode,
        overfit_test_size=overfit_test_size,
        debug_mode=debug_mode,
        wandb_instance=wandb_instance,  # NEW: Pass WandB instance
        **kwargs
    )


# Backward compatibility aliases
def create_spherical_eva_trainer(*args, **kwargs):
    """Backward compatibility: create EVA denoising trainer"""
    kwargs['task_mode'] = 'eva_denoising'
    return create_universal_trainer(*args, **kwargs)

def create_clip_denoising_trainer(*args, **kwargs):
    """NEW: Create CLIP denoising trainer"""
    kwargs['task_mode'] = 'clip_denoising'
    return create_universal_trainer(*args, **kwargs)

# Legacy alias
SphericalEVATrainer = UniversalDenoisingTrainer