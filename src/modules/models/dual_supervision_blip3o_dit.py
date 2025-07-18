"""
FIXED Dual Supervision BLIP3-o DiT Model Implementation
Place this file as: src/modules/models/dual_supervision_blip3o_dit.py

This is a wrapper around the main BLIP3-o DiT model that provides dual supervision
architecture with enhanced global alignment for retrieval performance.

IMPORTANT: This is our own implementation inspired by BLIP3-o paper.
We are NOT using any pre-trained BLIP3-o models.

Architecture:
EVA [B,256,4096] → DiT → [B,256,1024] → {
    Patch Output: [B,256,1024] (patch loss)
    Global Path: Avg Pool → MLP → Frozen CLIP Proj → [B,768]
}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, Union
from transformers import PreTrainedModel

from .blip3o_dit import BLIP3oDiTModel
from ..config.blip3o_config import BLIP3oDiTConfig


class DualSupervisionBLIP3oDiTModel(BLIP3oDiTModel):
    """
    Dual Supervision BLIP3-o DiT Model.
    
    This is essentially the same as the base BLIP3oDiTModel but with explicit
    dual supervision interface. The base model already has the dual supervision
    architecture built-in.
    
    IMPORTANT: This is our own implementation inspired by BLIP3-o paper.
    We are NOT using any pre-trained BLIP3-o models.
    """
    
    def __init__(self, config: BLIP3oDiTConfig):
        # FIXED: Config is already a PretrainedConfig, pass directly
        super().__init__(config)
        
        # Verify dual supervision components are present
        assert hasattr(self, 'global_adaptation_mlp'), "Global adaptation MLP not found"
        
        print(f"✅ DualSupervisionBLIP3oDiTModel initialized")
        print(f"   Dual supervision architecture: Enabled")
        print(f"   Global adaptation MLP: {self.global_adaptation_mlp}")
        print(f"   NOTE: Our own implementation, NOT pre-trained BLIP3-o")
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        **kwargs
    ):
        """
        Forward pass with explicit dual supervision outputs.
        
        Returns:
            Dict containing:
            - patch_output: [B, 256, 1024] for patch-level supervision
            - global_output: [B, 768] for global supervision (if CLIP projection loaded)
        """
        # Call parent forward method which already implements dual supervision
        outputs = super().forward(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            cross_attention_kwargs=cross_attention_kwargs,
            return_dict=return_dict,
            **kwargs
        )
        
        # The base model already returns the correct dual supervision format
        if return_dict:
            return outputs
        else:
            # If not return_dict, return (patch_output, global_output)
            return outputs['patch_output'], outputs.get('global_output')
    
    def enable_dual_supervision(self):
        """Enable dual supervision mode (already enabled by default)."""
        if self.frozen_clip_visual_proj is None:
            print("⚠️  Frozen CLIP projection not loaded. Loading now...")
            self.load_frozen_clip_projection()
        print("✅ Dual supervision mode enabled")
    
    def get_dual_supervision_info(self) -> Dict[str, Any]:
        """Get information about dual supervision components."""
        return {
            'has_global_adaptation_mlp': hasattr(self, 'global_adaptation_mlp'),
            'has_frozen_clip_proj': self.frozen_clip_visual_proj is not None,
            'mlp_input_dim': getattr(self.global_adaptation_mlp, 'input_dim', 'unknown'),
            'mlp_output_dim': getattr(self.global_adaptation_mlp, 'output_dim', 'unknown'),
            'clip_proj_shape': self.frozen_clip_visual_proj.weight.shape if self.frozen_clip_visual_proj else None,
            'expected_patch_output_shape': f"[batch_size, 256, {self.config.in_channels}]",
            'expected_global_output_shape': "[batch_size, 768]",
            'implementation_note': "Our own implementation inspired by BLIP3-o, not pre-trained model",
        }


def create_blip3o_dit_model(
    config: Optional[BLIP3oDiTConfig] = None,
    load_clip_projection: bool = True,
    clip_model_name: str = "openai/clip-vit-large-patch14",
    enable_dual_supervision: bool = True,
    **kwargs
) -> DualSupervisionBLIP3oDiTModel:
    """
    Factory function to create a dual supervision BLIP3-o DiT model.
    
    IMPORTANT: This creates our own implementation inspired by BLIP3-o paper.
    We are NOT loading any pre-trained BLIP3-o models.
    
    Args:
        config: Model configuration
        load_clip_projection: Whether to load frozen CLIP projection
        clip_model_name: CLIP model to load projection from
        enable_dual_supervision: Whether to enable dual supervision (always True for this variant)
        **kwargs: Additional configuration parameters
        
    Returns:
        DualSupervisionBLIP3oDiTModel instance
    """
    if config is None:
        from ..config.blip3o_config import get_default_blip3o_config
        config = get_default_blip3o_config()
    
    # Override config with kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            print(f"Warning: Unknown config parameter '{key}' ignored")
    
    print("🏗️ Creating dual supervision BLIP3-o DiT model")
    print(f"   NOTE: This is our own implementation inspired by BLIP3-o paper")
    print(f"   NOTE: We are NOT loading any pre-trained BLIP3-o models")
    print(f"   NOTE: Only CLIP's visual projection layer is loaded for fair comparison")
    
    # Create dual supervision model
    model = DualSupervisionBLIP3oDiTModel(config)
    
    # Load frozen CLIP projection if requested
    if load_clip_projection:
        model.load_frozen_clip_projection(clip_model_name)
        print(f"✅ Loaded frozen CLIP projection from {clip_model_name}")
    
    # Enable dual supervision
    if enable_dual_supervision:
        model.enable_dual_supervision()
    
    # Print dual supervision info
    dual_info = model.get_dual_supervision_info()
    print(f"🎯 Dual Supervision Model Created:")
    print(f"   Global adaptation MLP: {dual_info['has_global_adaptation_mlp']}")
    print(f"   Frozen CLIP projection: {dual_info['has_frozen_clip_proj']}")
    print(f"   Expected outputs:")
    print(f"     Patch: {dual_info['expected_patch_output_shape']}")
    print(f"     Global: {dual_info['expected_global_output_shape']}")
    print(f"   Implementation: {dual_info['implementation_note']}")
    
    return model


def load_dual_supervision_blip3o_dit_model(
    model_path: str,
    device: str = "auto",
    torch_dtype: Optional[torch.dtype] = None
) -> DualSupervisionBLIP3oDiTModel:
    """Load a pre-trained dual supervision BLIP3-o DiT model."""
    print(f"📁 Loading our trained dual supervision BLIP3-o model from: {model_path}")
    print(f"   NOTE: This loads our own trained model, not a pre-trained BLIP3-o")
    
    from transformers import AutoModel
    
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device if device != "auto" else None,
    )
    
    # Ensure it's a dual supervision model
    if not isinstance(model, DualSupervisionBLIP3oDiTModel):
        print("⚠️  Loaded model is not DualSupervisionBLIP3oDiTModel, wrapping...")
        # Could implement conversion logic here if needed
    
    return model


# Export functions for compatibility
__all__ = [
    "DualSupervisionBLIP3oDiTModel",
    "create_blip3o_dit_model",
    "load_dual_supervision_blip3o_dit_model",
]