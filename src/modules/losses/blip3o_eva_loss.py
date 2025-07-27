#!/usr/bin/env python3
"""
Enhanced Spherical Flow Matching Loss - Support for Both EVA and CLIP Denoising
Key features:
1. Universal spherical flow matching for any embedding dimension
2. EVA denoising: 4096-dim spherical flow matching
3. CLIP denoising: 1024-dim spherical flow matching with EVA conditioning
4. Automatic dimension detection and validation
5. Task-specific evaluation metrics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
import math
import logging

logger = logging.getLogger(__name__)


class UniversalSphericalFlowMatchingLoss(nn.Module):
    """
    Universal Spherical Flow Matching Loss for both EVA and CLIP denoising
    
    Supports:
    - EVA denoising: Input/Output 4096-dim, Conditioning 4096-dim
    - CLIP denoising: Input/Output 1024-dim, Conditioning 4096-dim
    - Automatic dimension detection and validation
    - Task-specific metrics and evaluation
    """
    
    def __init__(
        self,
        prediction_type: str = "velocity",
        loss_weight: float = 1.0,
        eps: float = 1e-8,
        # Boundary handling
        min_timestep: float = 1e-3,
        max_timestep: float = 1.0 - 1e-3,
        # Spherical regularization
        sphere_constraint_weight: float = 0.1,
        velocity_smoothness_weight: float = 0.0,
        angle_preservation_weight: float = 0.0,
        # Numerical stability
        min_angle_threshold: float = 1e-6,
        max_angle_threshold: float = math.pi - 1e-6,
        debug_mode: bool = False,
    ):
        super().__init__()
        
        self.prediction_type = prediction_type
        self.loss_weight = loss_weight
        self.eps = eps
        self.min_timestep = min_timestep
        self.max_timestep = max_timestep
        
        # Spherical regularization
        self.sphere_constraint_weight = sphere_constraint_weight
        self.velocity_smoothness_weight = velocity_smoothness_weight
        self.angle_preservation_weight = angle_preservation_weight
        
        # Numerical stability
        self.min_angle_threshold = min_angle_threshold
        self.max_angle_threshold = max_angle_threshold
        
        self.debug_mode = debug_mode
        
        # Validate inputs
        assert prediction_type in ["velocity", "target", "noise"]
        
        # Running statistics (universal for both tasks)
        self.register_buffer('step_count', torch.tensor(0))
        self.register_buffer('loss_ema', torch.tensor(0.0))
        self.register_buffer('similarity_ema', torch.tensor(0.0))
        self.register_buffer('sphere_violation_ema', torch.tensor(0.0))
        
        # Task-specific statistics
        self.register_buffer('eva_similarity_ema', torch.tensor(0.0))
        self.register_buffer('clip_similarity_ema', torch.tensor(0.0))
        
        logger.info(f"Universal Spherical Flow Matching Loss initialized:")
        logger.info(f"  Prediction type: {prediction_type}")
        logger.info(f"  Loss weight: {loss_weight}")
        logger.info(f"  Sphere constraint weight: {sphere_constraint_weight}")
        logger.info(f"  Supports: EVA denoising (4096D) & CLIP denoising (1024D)")
        logger.info(f"  Debug mode: {debug_mode}")

    def _clamp_timesteps(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Clamp timesteps to avoid numerical issues"""
        return torch.clamp(timesteps, min=self.min_timestep, max=self.max_timestep)

    def _update_ema(self, tensor_buffer: torch.Tensor, new_value: float, alpha: float = 0.01):
        """Update exponential moving average"""
        if tensor_buffer.item() == 0.0:
            tensor_buffer.data = torch.tensor(new_value)
        else:
            tensor_buffer.data = alpha * new_value + (1 - alpha) * tensor_buffer.data

    def _slerp(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Spherical linear interpolation between two unit vectors (universal for any dimension)
        
        Args:
            x0: Source vectors [B, N, D] on unit sphere
            x1: Target vectors [B, N, D] on unit sphere  
            t: Interpolation parameter [B, 1, 1] in [0, 1]
            
        Returns:
            Interpolated vectors [B, N, D] on unit sphere
        """
        # Ensure inputs are normalized
        x0 = F.normalize(x0, p=2, dim=-1)
        x1 = F.normalize(x1, p=2, dim=-1)
        
        # Compute angle between vectors
        cos_angle = torch.sum(x0 * x1, dim=-1, keepdim=True)
        cos_angle = torch.clamp(cos_angle, -1 + self.eps, 1 - self.eps)
        angle = torch.acos(cos_angle)
        
        # Handle small angles (linear interpolation for numerical stability)
        small_angle_mask = angle < self.min_angle_threshold
        large_angle_mask = angle > self.max_angle_threshold
        
        # For small angles, use linear interpolation + normalization
        linear_interp = (1 - t) * x0 + t * x1
        linear_interp = F.normalize(linear_interp, p=2, dim=-1)
        
        # For large angles (nearly antipodal), use a robust slerp
        sin_angle = torch.sin(angle)
        sin_angle = torch.clamp(sin_angle, min=self.eps)
        
        # Standard slerp formula
        w0 = torch.sin((1 - t) * angle) / sin_angle
        w1 = torch.sin(t * angle) / sin_angle
        
        slerp_result = w0 * x0 + w1 * x1
        slerp_result = F.normalize(slerp_result, p=2, dim=-1)
        
        # Choose interpolation method based on angle
        result = torch.where(small_angle_mask, linear_interp, slerp_result)
        result = torch.where(large_angle_mask, linear_interp, result)
        
        return result

    def _spherical_velocity(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute velocity field for spherical flow matching (universal for any dimension)
        """
        # Ensure inputs are normalized
        x0 = F.normalize(x0, p=2, dim=-1)
        x1 = F.normalize(x1, p=2, dim=-1)
        
        # Compute angle and interpolation weights
        cos_angle = torch.sum(x0 * x1, dim=-1, keepdim=True)
        cos_angle = torch.clamp(cos_angle, -1 + self.eps, 1 - self.eps)
        angle = torch.acos(cos_angle)
        
        # Handle edge cases
        small_angle_mask = angle < self.min_angle_threshold
        
        # For small angles, velocity is approximately x1 - x0 projected to tangent space
        simple_velocity = x1 - x0
        
        # For normal case, compute proper spherical velocity
        sin_angle = torch.sin(angle)
        sin_angle = torch.clamp(sin_angle, min=self.eps)
        
        # Velocity = angle * (x1 - cos(angle) * x0) / sin(angle)
        velocity = angle * (x1 - cos_angle * x0) / sin_angle
        
        # For small angles, use simple velocity
        velocity = torch.where(small_angle_mask, simple_velocity, velocity)
        
        return velocity

    def _detect_task_mode(self, model_output: torch.Tensor, target_samples: torch.Tensor) -> str:
        """Detect task mode based on tensor dimensions"""
        output_dim = model_output.shape[-1]
        target_dim = target_samples.shape[-1]
        
        if output_dim == 4096 and target_dim == 4096:
            return "eva_denoising"
        elif output_dim == 1024 and target_dim == 1024:
            return "clip_denoising"
        else:
            logger.warning(f"Unknown dimensions: output={output_dim}, target={target_dim}")
            return "unknown"

    def forward(
        self,
        model_output: torch.Tensor,       # [B, N, output_dim] - Model's prediction
        target_samples: torch.Tensor,    # [B, N, output_dim] - Clean target embeddings
        timesteps: torch.Tensor,         # [B] - Flow matching timesteps
        conditioning: torch.Tensor,      # [B, N, conditioning_dim] - Conditioning (EVA)
        noise: Optional[torch.Tensor] = None,
        x_t: Optional[torch.Tensor] = None,
        return_metrics: bool = True,
        task_mode: Optional[str] = None,  # NEW: Optional task mode hint
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[Dict[str, float]]]:
        """
        Compute universal spherical flow matching loss
        """
        
        batch_size, num_tokens, output_dim = model_output.shape
        device = model_output.device
        dtype = model_output.dtype
        
        # Detect task mode if not provided
        if task_mode is None:
            task_mode = self._detect_task_mode(model_output, target_samples)
        
        # Update step count
        self.step_count += 1
        
        # Clamp timesteps
        timesteps = self._clamp_timesteps(timesteps)
        
        # Ensure all inputs are normalized (critical for spherical flow)
        target_normalized = F.normalize(target_samples.detach(), p=2, dim=-1)
        conditioning_normalized = F.normalize(conditioning.detach(), p=2, dim=-1)
        
        # Create noise if not provided (same dimension as target)
        if noise is None:
            noise = torch.randn_like(target_normalized, device=device, dtype=dtype)
            noise = F.normalize(noise, p=2, dim=-1)
        
        # Expand timesteps for broadcasting [B, 1, 1]
        t = timesteps.view(batch_size, 1, 1).to(dtype)
        
        # SPHERICAL FLOW COMPUTATION (universal for any dimension)
        if self.prediction_type == "velocity":
            # Direct velocity prediction
            true_velocity = self._spherical_velocity(noise, target_normalized, t)
            target_for_loss = true_velocity
            
        elif self.prediction_type == "target":
            # Direct target prediction
            target_for_loss = target_normalized
            
        elif self.prediction_type == "noise":
            # Noise prediction (epsilon parameterization)
            target_for_loss = noise
        
        else:
            raise NotImplementedError(f"Prediction type {self.prediction_type} not implemented")
        
        # MAIN LOSS COMPUTATION
        if self.prediction_type == "velocity":
            # MSE loss on velocity field
            main_loss = F.mse_loss(model_output, target_for_loss, reduction='none')
        else:
            # For target/noise prediction, ensure outputs are normalized
            model_output_norm = F.normalize(model_output, p=2, dim=-1)
            main_loss = F.mse_loss(model_output_norm, target_for_loss, reduction='none')
        
        # Reduce loss: mean over tokens and embedding dimensions, then over batch
        main_loss = main_loss.mean(dim=(1, 2))  # [B]
        
        # SPHERICAL REGULARIZATION TERMS
        reg_loss = 0.0
        
        # Sphere constraint: ensure model outputs respect unit sphere
        if self.sphere_constraint_weight > 0:
            if self.prediction_type == "velocity":
                # For velocity, no direct sphere constraint
                sphere_violation = 0.0
            else:
                # For target/noise prediction, enforce unit norm
                output_norms = torch.norm(model_output, dim=-1)  # [B, N]
                sphere_violation = F.mse_loss(output_norms, torch.ones_like(output_norms))
                reg_loss += self.sphere_constraint_weight * sphere_violation
        
        # Velocity smoothness (optional)
        if self.velocity_smoothness_weight > 0 and self.prediction_type == "velocity":
            velocity_diff = model_output[:, 1:, :] - model_output[:, :-1, :]  # [B, N-1, D]
            smoothness_loss = torch.norm(velocity_diff, dim=-1).mean()
            reg_loss += self.velocity_smoothness_weight * smoothness_loss
        
        # Total loss
        total_loss = main_loss.mean() + reg_loss
        total_loss = total_loss * self.loss_weight
        
        # METRICS COMPUTATION
        metrics = {}
        if return_metrics:
            with torch.no_grad():
                # Normalize predictions for similarity computation
                if self.prediction_type == "velocity":
                    pred_normalized = F.normalize(model_output + self.eps, p=2, dim=-1)
                    target_norm = F.normalize(target_for_loss + self.eps, p=2, dim=-1)
                else:
                    pred_normalized = F.normalize(model_output + self.eps, p=2, dim=-1)
                    target_norm = target_for_loss
                
                # Cosine similarity
                cosine_sim = F.cosine_similarity(pred_normalized, target_norm, dim=-1)
                per_image_sim = cosine_sim.mean(dim=1)  # [B]
                mean_similarity = per_image_sim.mean().item()
                
                # Task-specific evaluation
                if x_t is not None and self.prediction_type == "velocity":
                    # Integrate velocity to get clean prediction
                    dt = 0.1
                    predicted_clean = x_t + dt * model_output
                    predicted_clean = F.normalize(predicted_clean, p=2, dim=-1)
                    
                    # Compare with actual clean
                    clean_similarity = F.cosine_similarity(predicted_clean, target_normalized, dim=-1)
                    clean_per_image_sim = clean_similarity.mean(dim=1)
                    eval_similarity = clean_per_image_sim.mean().item()
                else:
                    eval_similarity = mean_similarity
                
                # Update EMAs
                self._update_ema(self.loss_ema, main_loss.mean().item())
                self._update_ema(self.similarity_ema, mean_similarity)
                
                # Task-specific EMA updates
                if task_mode == "eva_denoising":
                    self._update_ema(self.eva_similarity_ema, eval_similarity)
                elif task_mode == "clip_denoising":
                    self._update_ema(self.clip_similarity_ema, eval_similarity)
                
                # Sphere constraint monitoring
                if self.prediction_type != "velocity":
                    output_norms = torch.norm(model_output, dim=-1).mean().item()
                    sphere_violation_val = abs(output_norms - 1.0)
                    self._update_ema(self.sphere_violation_ema, sphere_violation_val)
                else:
                    sphere_violation_val = 0.0
                
                # Compute norms for monitoring
                pred_norm = torch.norm(model_output, dim=-1).mean().item()
                target_norm_val = torch.norm(target_for_loss, dim=-1).mean().item()
                clean_norm = torch.norm(target_normalized, dim=-1).mean().item()
                noise_norm = torch.norm(noise, dim=-1).mean().item()
                conditioning_norm = torch.norm(conditioning_normalized, dim=-1).mean().item()
                
                # Error analysis
                error = model_output - target_for_loss
                error_norm = torch.norm(error, dim=-1).mean().item()
                relative_error = error_norm / (target_norm_val + self.eps)
                
                # Quality assessment (task-specific thresholds)
                if task_mode == "eva_denoising":
                    # EVA denoising thresholds
                    if eval_similarity > 0.8:
                        quality = "excellent"
                    elif eval_similarity > 0.5:
                        quality = "good"
                    elif eval_similarity > 0.2:
                        quality = "fair"
                    else:
                        quality = "poor"
                elif task_mode == "clip_denoising":
                    # CLIP denoising thresholds (may be different)
                    if eval_similarity > 0.7:
                        quality = "excellent"
                    elif eval_similarity > 0.4:
                        quality = "good"
                    elif eval_similarity > 0.15:
                        quality = "fair"
                    else:
                        quality = "poor"
                else:
                    quality = "unknown"
                
                metrics = {
                    # Core metrics
                    'loss': main_loss.mean().item(),
                    'total_loss': total_loss.item(),
                    'reg_loss': reg_loss if isinstance(reg_loss, float) else reg_loss.item(),
                    'prediction_similarity': mean_similarity,
                    'eval_similarity': eval_similarity,
                    'similarity_std': per_image_sim.std().item(),
                    
                    # Task-specific metrics
                    'task_mode': task_mode,
                    'output_dim': output_dim,
                    'target_dim': target_samples.shape[-1],
                    'conditioning_dim': conditioning.shape[-1],
                    
                    # Sphere constraint
                    'sphere_violation': sphere_violation_val,
                    'sphere_constraint_loss': sphere_violation if self.sphere_constraint_weight > 0 else 0.0,
                    
                    # Norm tracking
                    'pred_norm': pred_norm,
                    'target_norm': target_norm_val,
                    'clean_norm': clean_norm,
                    'noise_norm': noise_norm,
                    'conditioning_norm': conditioning_norm,
                    
                    # Error analysis
                    'error_norm': error_norm,
                    'relative_error': relative_error,
                    
                    # Training progress
                    'step_count': self.step_count.item(),
                    'loss_ema': self.loss_ema.item(),
                    'similarity_ema': self.similarity_ema.item(),
                    'sphere_violation_ema': self.sphere_violation_ema.item(),
                    'eva_similarity_ema': self.eva_similarity_ema.item(),
                    'clip_similarity_ema': self.clip_similarity_ema.item(),
                    'quality_assessment': quality,
                    
                    # Flow matching specific
                    'timestep_mean': timesteps.mean().item(),
                    'timestep_std': timesteps.std().item(),
                    'interpolation_weight': t.mean().item(),
                    
                    # Debugging info
                    'pred_min': model_output.min().item(),
                    'pred_max': model_output.max().item(),
                    'target_min': target_for_loss.min().item(),
                    'target_max': target_for_loss.max().item(),
                }
                
                # Check for numerical issues
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    metrics['numerical_issue'] = True
                    logger.error(f"[Step {self.step_count}] Numerical issue detected in {task_mode}!")
                    logger.error(f"  Loss: {total_loss.item()}")
                    logger.error(f"  Pred norm: {pred_norm}")
                    logger.error(f"  Target norm: {target_norm_val}")
                
                # Debug logging
                if self.debug_mode and self.step_count % 50 == 0:
                    logger.info(f"[Step {self.step_count}] {task_mode.upper()} Debug:")
                    logger.info(f"  Loss: {main_loss.mean().item():.6f} (quality: {quality})")
                    logger.info(f"  Prediction Sim: {mean_similarity:.4f}")
                    logger.info(f"  Eval Sim: {eval_similarity:.4f}")
                    logger.info(f"  Sphere Violation: {sphere_violation_val:.6f}")
                    logger.info(f"  Dims: out={output_dim}, target={target_samples.shape[-1]}, cond={conditioning.shape[-1]}")
                    logger.info(f"  Norms - Pred: {pred_norm:.3f}, Target: {target_norm_val:.3f}, Cond: {conditioning_norm:.3f}")
        
        return total_loss, metrics

    def compute_eval_loss(
        self,
        generated: torch.Tensor,  # Generated clean embeddings
        target: torch.Tensor,     # Target clean embeddings
        task_mode: Optional[str] = None,
    ) -> Dict[str, float]:
        """Compute universal evaluation metrics"""
        with torch.no_grad():
            # Detect task mode if not provided
            if task_mode is None:
                task_mode = self._detect_task_mode(generated, target)
            
            # Normalize both
            generated_norm = F.normalize(generated, p=2, dim=-1)
            target_norm = F.normalize(target, p=2, dim=-1)
            
            # Cosine similarity (main metric for both tasks)
            cosine_sim = F.cosine_similarity(generated_norm, target_norm, dim=-1)
            per_image_sim = cosine_sim.mean(dim=1)
            
            # MSE in normalized space
            mse_loss = F.mse_loss(generated_norm, target_norm)
            
            # Angular distance
            cos_sim_clamped = torch.clamp(cosine_sim, -1 + 1e-7, 1 - 1e-7)
            angular_distance = torch.acos(cos_sim_clamped).mean()
            
            # Task-specific quality metrics
            if task_mode == "eva_denoising":
                # EVA denoising quality thresholds
                high_quality = (per_image_sim > 0.7).float().mean().item()
                very_high_quality = (per_image_sim > 0.8).float().mean().item()
                excellent_quality = (per_image_sim > 0.9).float().mean().item()
                metric_prefix = "eval_eva"
            elif task_mode == "clip_denoising":
                # CLIP denoising quality thresholds
                high_quality = (per_image_sim > 0.6).float().mean().item()
                very_high_quality = (per_image_sim > 0.7).float().mean().item()
                excellent_quality = (per_image_sim > 0.8).float().mean().item()
                metric_prefix = "eval_clip"
            else:
                # Unknown task - use generic thresholds
                high_quality = (per_image_sim > 0.6).float().mean().item()
                very_high_quality = (per_image_sim > 0.7).float().mean().item()
                excellent_quality = (per_image_sim > 0.8).float().mean().item()
                metric_prefix = "eval_generic"
            
            # Sphere constraint violation
            generated_norms = torch.norm(generated, dim=-1)
            sphere_violation = F.mse_loss(generated_norms, torch.ones_like(generated_norms)).item()
            
            return {
                f'{metric_prefix}_similarity': per_image_sim.mean().item(),
                f'{metric_prefix}_mse_loss': mse_loss.item(),
                f'{metric_prefix}_angular_distance': angular_distance.item(),
                f'{metric_prefix}_high_quality_ratio': high_quality,
                f'{metric_prefix}_very_high_quality_ratio': very_high_quality,
                f'{metric_prefix}_excellent_quality_ratio': excellent_quality,
                f'{metric_prefix}_similarity_std': per_image_sim.std().item(),
                f'{metric_prefix}_sphere_violation': sphere_violation,
                f'{metric_prefix}_generated_norm_mean': generated_norms.mean().item(),
                f'{metric_prefix}_generated_norm_std': generated_norms.std().item(),
                'eval_task_mode': task_mode,
                'eval_output_dim': generated.shape[-1],
                'eval_target_dim': target.shape[-1],
            }


def create_universal_flow_loss(
    prediction_type: str = "velocity",
    loss_weight: float = 1.0,
    sphere_constraint_weight: float = 0.1,
    debug_mode: bool = False,
    **kwargs
) -> UniversalSphericalFlowMatchingLoss:
    """Factory function for universal spherical flow matching loss"""
    return UniversalSphericalFlowMatchingLoss(
        prediction_type=prediction_type,
        loss_weight=loss_weight,
        sphere_constraint_weight=sphere_constraint_weight,
        debug_mode=debug_mode,
        **kwargs
    )


# Backward compatibility aliases
def create_spherical_flow_loss(*args, **kwargs):
    """Backward compatibility: create EVA denoising loss"""
    return create_universal_flow_loss(*args, **kwargs)

def create_clip_flow_loss(*args, **kwargs):
    """NEW: Create CLIP denoising loss (same as universal)"""
    return create_universal_flow_loss(*args, **kwargs)

# Legacy alias
SphericalFlowMatchingLoss = UniversalSphericalFlowMatchingLoss