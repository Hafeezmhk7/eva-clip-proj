"""
Enhanced BLIP3-o Configuration - Support for CLIP Denoising with EVA Conditioning
src/modules/config/blip3o_config.py

NEW ADDITION:
- CLIP denoising mode: Input/Output CLIP [B, N, 1024], Conditioning EVA [B, N, 4096]
- EVA denoising mode: Input/Output EVA [B, N, 4096], Conditioning EVA [B, N, 4096]
"""

from transformers import PretrainedConfig
from typing import Dict, Any, Optional
from dataclasses import dataclass
import math


class BLIP3oDiTConfig(PretrainedConfig):
    """
    Enhanced configuration class for BLIP3-o DiT model supporting both EVA and CLIP denoising.
    
    Supported modes:
    1. EVA Denoising: Input/Output EVA [4096], Conditioning EVA [4096]
    2. CLIP Denoising: Input/Output CLIP [1024], Conditioning EVA [4096]
    """
    
    model_type = "blip3o_dit"
    
    def __init__(
        self,
        # Model architecture
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        num_key_value_heads: int = 4,
        intermediate_size: int = 3072,
        
        # Task configuration - NEW: Support for CLIP and EVA modes
        task_mode: str = "eva_denoising",  # "eva_denoising" or "clip_denoising"
        eva_embedding_size: int = 4096,   # EVA-CLIP dimension (conditioning)
        clip_embedding_size: int = 1024,  # CLIP-ViT dimension (input/output for CLIP mode)
        num_tokens: int = 256,
        
        # Training configuration
        max_position_embeddings: int = 256,
        dropout_prob: float = 0.0,
        
        # Normalization
        rms_norm_eps: float = 1e-6,
        use_rms_norm: bool = True,
        
        # Attention configuration
        attention_dropout: float = 0.0,
        use_3d_rope: bool = True,
        rope_theta: float = 10000.0,
        rope_scaling: Optional[Dict] = None,
        
        # Flow matching parameters
        prediction_type: str = "velocity",
        
        # Training optimizations
        use_gradient_checkpointing: bool = False,
        training_mode: str = "patch_only",
        zero_init_output: bool = True,
        
        # Architecture features
        use_sandwich_norm: bool = True,
        use_grouped_query_attention: bool = True,
        
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        # Core architecture
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        
        # Task configuration
        self.task_mode = task_mode
        self.eva_embedding_size = eva_embedding_size
        self.clip_embedding_size = clip_embedding_size
        self.num_tokens = num_tokens
        
        # Derived dimensions based on task mode
        if task_mode == "eva_denoising":
            self.input_embedding_size = eva_embedding_size   # 4096
            self.output_embedding_size = eva_embedding_size  # 4096
            self.conditioning_embedding_size = eva_embedding_size  # 4096
        elif task_mode == "clip_denoising":
            self.input_embedding_size = clip_embedding_size   # 1024
            self.output_embedding_size = clip_embedding_size  # 1024
            self.conditioning_embedding_size = eva_embedding_size  # 4096
        else:
            raise ValueError(f"Unknown task_mode: {task_mode}")
        
        # Training configuration
        self.max_position_embeddings = max_position_embeddings
        self.dropout_prob = dropout_prob
        
        # Normalization
        self.rms_norm_eps = rms_norm_eps
        self.use_rms_norm = use_rms_norm
        
        # Attention
        self.attention_dropout = attention_dropout
        self.use_3d_rope = use_3d_rope
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        
        # Flow matching
        self.prediction_type = prediction_type
        
        # Training optimizations
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.training_mode = training_mode
        self.zero_init_output = zero_init_output
        
        # Architecture features
        self.use_sandwich_norm = use_sandwich_norm
        self.use_grouped_query_attention = use_grouped_query_attention
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        # Check head dimension compatibility
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )
        
        # Check grouped-query attention compatibility
        if self.use_grouped_query_attention:
            if self.num_attention_heads % self.num_key_value_heads != 0:
                raise ValueError(
                    f"num_attention_heads ({self.num_attention_heads}) must be divisible by "
                    f"num_key_value_heads ({self.num_key_value_heads}) for grouped-query attention"
                )
        
        # Check task mode
        if self.task_mode not in ["eva_denoising", "clip_denoising"]:
            raise ValueError(f"task_mode must be 'eva_denoising' or 'clip_denoising', got {self.task_mode}")
        
        # Check number of tokens
        if self.num_tokens not in [256, 257]:
            raise ValueError(f"num_tokens must be 256 or 257, got {self.num_tokens}")
        
        # Check prediction type
        if self.prediction_type not in ["velocity", "epsilon", "target"]:
            raise ValueError(f"prediction_type must be 'velocity', 'epsilon' or 'target', got {self.prediction_type}")
        
        # Validate embedding dimensions
        if self.eva_embedding_size <= 0:
            raise ValueError(f"eva_embedding_size must be positive, got {self.eva_embedding_size}")
        if self.clip_embedding_size <= 0:
            raise ValueError(f"clip_embedding_size must be positive, got {self.clip_embedding_size}")
    
    def get_task_info(self) -> Dict[str, Any]:
        """Get task-specific information"""
        if self.task_mode == "eva_denoising":
            return {
                "task": "EVA-CLIP Denoising",
                "input": f"Noisy EVA embeddings [B, N, {self.eva_embedding_size}]",
                "conditioning": f"Clean EVA embeddings [B, N, {self.eva_embedding_size}]",
                "output": f"Clean EVA embeddings [B, N, {self.eva_embedding_size}]",
                "input_dim": self.eva_embedding_size,
                "output_dim": self.eva_embedding_size,
                "conditioning_dim": self.eva_embedding_size,
            }
        elif self.task_mode == "clip_denoising":
            return {
                "task": "CLIP-ViT Denoising with EVA Conditioning",
                "input": f"Noisy CLIP embeddings [B, N, {self.clip_embedding_size}]",
                "conditioning": f"Clean EVA embeddings [B, N, {self.eva_embedding_size}]",
                "output": f"Clean CLIP embeddings [B, N, {self.clip_embedding_size}]",
                "input_dim": self.clip_embedding_size,
                "output_dim": self.clip_embedding_size,
                "conditioning_dim": self.eva_embedding_size,
            }

    def get_parameter_count_estimate(self):
        """Estimate total parameter count"""
        # Input/output projections (task-specific)
        input_params = self.input_embedding_size * self.hidden_size
        output_params = self.hidden_size * self.output_embedding_size
        
        # EVA conditioning projection
        eva_conditioning_params = self.conditioning_embedding_size * self.hidden_size
        
        # Embeddings
        pos_embed_params = self.max_position_embeddings * self.hidden_size
        timestep_embed_params = 256 * self.hidden_size + self.hidden_size * self.hidden_size
        
        # Transformer layers
        layer_params = self.num_hidden_layers * (
            # Self-attention
            self.hidden_size * self.hidden_size * 3 +  # Q, K, V
            self.hidden_size * self.hidden_size +       # Output projection
            # Cross-attention with EVA conditioning
            self.hidden_size * self.hidden_size +       # Q projection
            self.hidden_size * self.hidden_size * 2 +   # K, V projections for conditioning
            self.hidden_size * self.hidden_size +       # Output projection
            # FFN
            self.hidden_size * self.intermediate_size +   # Up projection
            self.intermediate_size * self.hidden_size +   # Down projection
            # Norms
            self.hidden_size * 6                          # Multiple norms per layer
        )
        
        total_params = (
            input_params + output_params + eva_conditioning_params + 
            pos_embed_params + timestep_embed_params + layer_params
        )
        
        return total_params


def get_blip3o_config(
    model_size: str = "base",
    training_mode: str = "patch_only",
    task_mode: str = "eva_denoising",  # NEW: Support for both EVA and CLIP denoising
    **kwargs
) -> BLIP3oDiTConfig:
    """
    Get predefined BLIP3-o configuration with support for EVA and CLIP denoising tasks.
    
    Args:
        model_size: Model size - "tiny", "small", "base", "large"
        training_mode: "patch_only" (256 tokens) or "cls_patch" (257 tokens)
        task_mode: "eva_denoising" or "clip_denoising" - NEW!
        **kwargs: Additional configuration overrides
        
    Returns:
        BLIP3oDiTConfig instance
    """
    # Predefined configurations optimized for both tasks
    configs = {
        "tiny": {
            "hidden_size": 384,
            "num_hidden_layers": 6,
            "num_attention_heads": 6,
            "num_key_value_heads": 2,
            "intermediate_size": 1536,
        },
        "small": {
            "hidden_size": 512,
            "num_hidden_layers": 8,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "intermediate_size": 2048,
        },
        "base": {
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "num_key_value_heads": 4,
            "intermediate_size": 3072,
        },
        "large": {
            "hidden_size": 1024,
            "num_hidden_layers": 16,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "intermediate_size": 4096,
        },
    }
    
    if model_size not in configs:
        raise ValueError(f"Unknown model size: {model_size}. Available: {list(configs.keys())}")
    
    # Get base config
    config_dict = configs[model_size].copy()
    
    # Set token count based on training mode
    config_dict["num_tokens"] = 257 if training_mode == "cls_patch" else 256
    config_dict["max_position_embeddings"] = max(config_dict["num_tokens"], 257)
    config_dict["training_mode"] = training_mode
    config_dict["task_mode"] = task_mode  # NEW: Task mode
    
    # Default architecture features
    config_dict.update({
        "use_3d_rope": True,
        "use_sandwich_norm": True,
        "use_grouped_query_attention": True,
        "use_rms_norm": True,
        "zero_init_output": True,
        "dropout_prob": 0.0,
        "attention_dropout": 0.0,
    })
    
    # Apply overrides
    config_dict.update(kwargs)
    
    return BLIP3oDiTConfig(**config_dict)


@dataclass
class FlowMatchingConfig:
    """Configuration for flow matching training - works for both EVA and CLIP"""
    prediction_type: str = "velocity"
    normalize_targets: bool = True
    flow_type: str = "rectified"
    loss_scale: float = 1.0
    
    # Stability parameters
    min_timestep: float = 1e-3
    max_timestep: float = 1.0 - 1e-3
    clip_norm_max: float = 1.0
    
    # Boundary condition handling
    handle_boundaries: bool = True
    boundary_loss_weight: float = 0.1


@dataclass  
class TrainingConfig:
    """Configuration for training parameters - universal for both tasks"""
    num_epochs: int = 20
    batch_size: int = 16
    eval_batch_size: int = 16
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    lr_scheduler_type: str = "cosine"
    gradient_accumulation_steps: int = 2
    fp16: bool = True
    dataloader_num_workers: int = 0
    
    # Evaluation parameters
    eval_every_n_steps: int = 50
    eval_num_samples: int = 100
    eval_inference_steps: int = 50
    
    # Debugging
    debug_mode: bool = False
    track_gradients: bool = True
    overfit_test_size: Optional[int] = None
    
    # Robustness
    skip_corrupted_samples: bool = True
    validate_tensor_shapes: bool = True
    max_grad_norm: float = 1.0


def create_config_from_args(args) -> tuple:
    """Create configurations from command line arguments with task mode support"""
    model_config = get_blip3o_config(
        model_size=getattr(args, 'model_size', 'base'),
        training_mode=getattr(args, 'training_mode', 'patch_only'),
        task_mode=getattr(args, 'task_mode', 'eva_denoising'),  # NEW
        use_gradient_checkpointing=getattr(args, 'gradient_checkpointing', False),
    )
    
    flow_config = FlowMatchingConfig(
        prediction_type="velocity",
        normalize_targets=True,
        flow_type="rectified",
        loss_scale=1.0,
    )
    
    training_config = TrainingConfig(
        num_epochs=getattr(args, 'num_epochs', 20),
        batch_size=getattr(args, 'batch_size', 16),
        learning_rate=getattr(args, 'learning_rate', 5e-4),
        gradient_accumulation_steps=getattr(args, 'gradient_accumulation_steps', 2),
        fp16=getattr(args, 'fp16', True),
        debug_mode=getattr(args, 'debug_mode', False),
        overfit_test_size=getattr(args, 'overfit_test_size', None),
    )
    
    return model_config, flow_config, training_config


def print_task_info(config: BLIP3oDiTConfig):
    """Print task-specific information"""
    task_info = config.get_task_info()
    
    print(f"📋 {task_info['task']} Configuration:")
    print(f"   📥 Input: {task_info['input']}")
    print(f"   🎮 Conditioning: {task_info['conditioning']}")
    print(f"   📤 Output: {task_info['output']}")
    print(f"   🔧 Dimensions: {task_info['input_dim']} → {task_info['output_dim']} (cond: {task_info['conditioning_dim']})")
    print(f"   🎯 Task mode: {config.task_mode}")


# Export commonly used configurations for both tasks
DEFAULT_EVA_CONFIG = get_blip3o_config("base", "patch_only", "eva_denoising")
DEFAULT_CLIP_CONFIG = get_blip3o_config("base", "patch_only", "clip_denoising")  # NEW

# Task-specific configurations
EVA_DENOISING_CONFIGS = {
    size: get_blip3o_config(size, "patch_only", "eva_denoising") 
    for size in ["tiny", "small", "base", "large"]
}

CLIP_DENOISING_CONFIGS = {  # NEW
    size: get_blip3o_config(size, "patch_only", "clip_denoising") 
    for size in ["tiny", "small", "base", "large"]
}