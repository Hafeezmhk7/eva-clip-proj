#!/usr/bin/env python3
"""
Enhanced BLIP3-o Dataset - Support for Both EVA and CLIP Denoising
Key features:
1. EVA Denoising: Input/Target EVA [4096], Conditioning EVA [4096]
2. CLIP Denoising: Input/Target CLIP [1024], Conditioning EVA [4096]
3. Flexible spherical data handling for both modalities
4. Proper flow matching setup for different embedding types
"""

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import pickle
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union, Iterator
from pathlib import Path
import logging
import json
import random
import time
import gc
import torch.nn.functional as F
import math

logger = logging.getLogger(__name__)


class UniversalDenoisingDataset(IterableDataset):
    """
    Universal dataset for both EVA and CLIP denoising with proper spherical data handling
    
    Modes:
    1. EVA Denoising: Takes clean EVA as TARGET and CONDITIONING, creates noisy EVA for INPUT
    2. CLIP Denoising: Takes clean CLIP as TARGET and INPUT, uses clean EVA as CONDITIONING
    """
    
    def __init__(
        self,
        chunked_embeddings_dir: Union[str, Path],
        task_mode: str = "eva_denoising",  # "eva_denoising" or "clip_denoising"
        split: str = "train",
        training_mode: str = "patch_only",
        max_shards: Optional[int] = None,
        shuffle_shards: bool = True,
        shuffle_within_shard: bool = True,
        expected_tokens: Optional[int] = None,
        # Spherical noise parameters
        noise_schedule: str = "uniform",
        max_noise_level: float = 0.9,
        min_noise_level: float = 0.1,
        # Error handling
        skip_corrupted: bool = True,
        validate_shapes: bool = True,
        max_retries: int = 3,
    ):
        super().__init__()
        
        self.chunked_embeddings_dir = Path(chunked_embeddings_dir)
        self.task_mode = task_mode
        self.split = split
        self.training_mode = training_mode
        self.max_shards = max_shards
        self.shuffle_shards = shuffle_shards
        self.shuffle_within_shard = shuffle_within_shard
        self.skip_corrupted = skip_corrupted
        self.validate_shapes = validate_shapes
        self.max_retries = max_retries
        
        # Spherical noise parameters
        self.noise_schedule = noise_schedule
        self.max_noise_level = max_noise_level
        self.min_noise_level = min_noise_level
        
        # Determine expected tokens
        if expected_tokens is None:
            self.expected_tokens = 257 if training_mode == "cls_patch" else 256
        else:
            self.expected_tokens = expected_tokens
        
        # Setup random state
        self.rng = random.Random(42)
        
        # Validate task mode
        if task_mode not in ["eva_denoising", "clip_denoising"]:
            raise ValueError(f"task_mode must be 'eva_denoising' or 'clip_denoising', got {task_mode}")
        
        # Load manifest and prepare shards
        self._load_manifest()
        self._prepare_shard_list()
        
        # Current state
        self.current_shard_idx = 0
        self.current_shard_data = None
        self.current_sample_idx = 0
        self.total_samples_processed = 0
        
        # Log task configuration
        if task_mode == "eva_denoising":
            logger.info(f"EVA Denoising Dataset initialized:")
            logger.info(f"  INPUT: Noisy EVA embeddings [B, N, 4096]")
            logger.info(f"  CONDITIONING: Clean EVA embeddings [B, N, 4096]")
            logger.info(f"  TARGET: Clean EVA embeddings [B, N, 4096]")
        elif task_mode == "clip_denoising":
            logger.info(f"CLIP Denoising Dataset initialized:")
            logger.info(f"  INPUT: Noisy CLIP embeddings [B, N, 1024]")
            logger.info(f"  CONDITIONING: Clean EVA embeddings [B, N, 4096]")
            logger.info(f"  TARGET: Clean CLIP embeddings [B, N, 1024]")
        
        logger.info(f"  Directory: {self.chunked_embeddings_dir}")
        logger.info(f"  Mode: {self.training_mode} ({self.expected_tokens} tokens)")
        logger.info(f"  Noise schedule: {self.noise_schedule}")
        logger.info(f"  Noise range: [{self.min_noise_level}, {self.max_noise_level}]")
        logger.info(f"  Shards: {len(self.shard_files) if hasattr(self, 'shard_files') else 'Loading...'}")

    def _load_manifest(self):
        """Load embeddings manifest"""
        manifest_path = self.chunked_embeddings_dir / "embeddings_manifest.json"
        
        try:
            if manifest_path.exists():
                with open(manifest_path, 'r') as f:
                    self.manifest = json.load(f)
                logger.info(f"Loaded manifest: {self.manifest.get('total_shards', 0)} shards, {self.manifest.get('total_samples', 0):,} samples")
            else:
                self.manifest = {"total_shards": 0, "total_samples": 0}
                logger.warning(f"No manifest found at {manifest_path}")
        except Exception as e:
            logger.warning(f"Failed to load manifest: {e}")
            self.manifest = {"total_shards": 0, "total_samples": 0}

    def _prepare_shard_list(self):
        """Prepare list of shard files"""
        # Look for shard files with different patterns
        mode_suffix = "cls_patch" if self.training_mode == "cls_patch" else "patch_only"
        patterns = [
            f"embeddings_shard_*_{mode_suffix}.pkl",
            f"*_{mode_suffix}.pkl",
            "embeddings_shard_*.pkl",
            "*.pkl"
        ]
        
        shard_files = []
        for pattern in patterns:
            shard_files = list(self.chunked_embeddings_dir.glob(pattern))
            if shard_files:
                logger.info(f"Found {len(shard_files)} files with pattern: {pattern}")
                break
        
        if not shard_files:
            raise FileNotFoundError(f"No shard files found in {self.chunked_embeddings_dir}")
        
        # Sort files
        shard_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x.stem))) if any(c.isdigit() for c in x.stem) else 0)
        
        # Apply max shards limit
        if self.max_shards is not None:
            shard_files = shard_files[:self.max_shards]
        
        # Filter existing files
        self.shard_files = [f for f in shard_files if f.exists()]
        
        if self.shuffle_shards:
            self.rng.shuffle(self.shard_files)
        
        logger.info(f"Prepared {len(self.shard_files)} shard files")

    def _load_shard(self, shard_path: Path) -> Optional[Dict[str, Any]]:
        """Load a single shard with error handling"""
        for attempt in range(self.max_retries):
            try:
                with open(shard_path, 'rb') as f:
                    shard_data = pickle.load(f)
                
                # Validate and process shard
                self._validate_and_process_shard(shard_data, shard_path)
                
                return shard_data
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{self.max_retries} failed for {shard_path}: {e}")
                if attempt == self.max_retries - 1:
                    if self.skip_corrupted:
                        logger.warning(f"Skipping corrupted shard: {shard_path}")
                        return None
                    else:
                        raise
                time.sleep(0.1)

    def _validate_and_process_shard(self, shard_data: Dict[str, Any], shard_path: Path):
        """Validate and process shard data for both EVA and CLIP modes"""
        # Check required keys based on task mode
        if self.task_mode == "eva_denoising":
            required_keys = ['eva_blip3o_embeddings', 'captions']
            if 'eva_blip3o_embeddings' not in shard_data:
                raise ValueError(f"Missing EVA embeddings in shard {shard_path}")
        elif self.task_mode == "clip_denoising":
            required_keys = ['clip_blip3o_embeddings', 'eva_blip3o_embeddings', 'captions']
            if 'clip_blip3o_embeddings' not in shard_data:
                raise ValueError(f"Missing CLIP embeddings in shard {shard_path}")
            if 'eva_blip3o_embeddings' not in shard_data:
                raise ValueError(f"Missing EVA embeddings in shard {shard_path}")
        
        for key in required_keys:
            if key not in shard_data:
                raise ValueError(f"Missing key '{key}' in shard {shard_path}")
        
        # Get and validate embeddings based on task mode
        if self.task_mode == "eva_denoising":
            eva_emb = shard_data['eva_blip3o_embeddings']
            
            # Convert to tensors if needed
            if not torch.is_tensor(eva_emb):
                eva_emb = torch.tensor(eva_emb, dtype=torch.float32)
                shard_data['eva_blip3o_embeddings'] = eva_emb
            
            # Validate shapes
            if self.validate_shapes:
                if eva_emb.dim() != 3:
                    raise ValueError(f"Expected 3D tensor for EVA, got: {eva_emb.shape}")
                if eva_emb.shape[2] != 4096:
                    raise ValueError(f"Expected EVA dim 4096, got: {eva_emb.shape[2]}")
        
        elif self.task_mode == "clip_denoising":
            clip_emb = shard_data['clip_blip3o_embeddings']
            eva_emb = shard_data['eva_blip3o_embeddings']
            
            # Convert to tensors if needed
            if not torch.is_tensor(clip_emb):
                clip_emb = torch.tensor(clip_emb, dtype=torch.float32)
                shard_data['clip_blip3o_embeddings'] = clip_emb
            if not torch.is_tensor(eva_emb):
                eva_emb = torch.tensor(eva_emb, dtype=torch.float32)
                shard_data['eva_blip3o_embeddings'] = eva_emb
            
            # Validate shapes
            if self.validate_shapes:
                if clip_emb.dim() != 3:
                    raise ValueError(f"Expected 3D tensor for CLIP, got: {clip_emb.shape}")
                if eva_emb.dim() != 3:
                    raise ValueError(f"Expected 3D tensor for EVA, got: {eva_emb.shape}")
                if clip_emb.shape[2] != 1024:
                    raise ValueError(f"Expected CLIP dim 1024, got: {clip_emb.shape[2]}")
                if eva_emb.shape[2] != 4096:
                    raise ValueError(f"Expected EVA dim 4096, got: {eva_emb.shape[2]}")
                if clip_emb.shape[0] != eva_emb.shape[0]:
                    raise ValueError(f"Batch size mismatch: CLIP {clip_emb.shape[0]} vs EVA {eva_emb.shape[0]}")
                if clip_emb.shape[1] != eva_emb.shape[1]:
                    raise ValueError(f"Token count mismatch: CLIP {clip_emb.shape[1]} vs EVA {eva_emb.shape[1]}")
        
        # Handle token count adaptation for both embeddings
        if self.task_mode == "eva_denoising":
            eva_emb = shard_data['eva_blip3o_embeddings']
            current_tokens = eva_emb.shape[1]
            
            if current_tokens != self.expected_tokens:
                logger.debug(f"Adapting EVA from {current_tokens} to {self.expected_tokens} tokens")
                shard_data['eva_blip3o_embeddings'] = self._adapt_token_count(eva_emb, current_tokens)
        
        elif self.task_mode == "clip_denoising":
            clip_emb = shard_data['clip_blip3o_embeddings']
            eva_emb = shard_data['eva_blip3o_embeddings']
            current_tokens = clip_emb.shape[1]
            
            if current_tokens != self.expected_tokens:
                logger.debug(f"Adapting both embeddings from {current_tokens} to {self.expected_tokens} tokens")
                shard_data['clip_blip3o_embeddings'] = self._adapt_token_count(clip_emb, current_tokens)
                shard_data['eva_blip3o_embeddings'] = self._adapt_token_count(eva_emb, current_tokens)
        
        # Apply normalization
        shard_data = self._normalize_embeddings(shard_data)

    def _adapt_token_count(self, embeddings: torch.Tensor, current_tokens: int) -> torch.Tensor:
        """Adapt token count for embeddings"""
        if current_tokens == 256 and self.expected_tokens == 257:
            # Add CLS token (average of patches)
            cls_token = embeddings.mean(dim=1, keepdim=True)
            return torch.cat([cls_token, embeddings], dim=1)
        elif current_tokens == 257 and self.expected_tokens == 256:
            # Remove CLS token
            return embeddings[:, 1:, :]
        else:
            raise ValueError(f"Cannot adapt from {current_tokens} to {self.expected_tokens} tokens")

    def _normalize_embeddings(self, shard_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply L2 normalization to embeddings"""
        eps = 1e-8
        
        if self.task_mode == "eva_denoising":
            eva_emb = shard_data['eva_blip3o_embeddings']
            
            # Check for NaN/Inf
            if torch.isnan(eva_emb).any() or torch.isinf(eva_emb).any():
                logger.warning("Found NaN/Inf in EVA embeddings")
                eva_emb = torch.nan_to_num(eva_emb, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Apply L2 normalization
            eva_normalized = F.normalize(eva_emb + eps, p=2, dim=-1)
            eva_norm = torch.norm(eva_normalized, dim=-1).mean().item()
            
            if abs(eva_norm - 1.0) > 0.1:
                logger.warning(f"EVA normalization may have failed: norm = {eva_norm:.3f}")
            
            shard_data['eva_blip3o_embeddings'] = eva_normalized
        
        elif self.task_mode == "clip_denoising":
            clip_emb = shard_data['clip_blip3o_embeddings']
            eva_emb = shard_data['eva_blip3o_embeddings']
            
            # Check for NaN/Inf in both
            if torch.isnan(clip_emb).any() or torch.isinf(clip_emb).any():
                logger.warning("Found NaN/Inf in CLIP embeddings")
                clip_emb = torch.nan_to_num(clip_emb, nan=0.0, posinf=1.0, neginf=-1.0)
            
            if torch.isnan(eva_emb).any() or torch.isinf(eva_emb).any():
                logger.warning("Found NaN/Inf in EVA embeddings")
                eva_emb = torch.nan_to_num(eva_emb, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Apply L2 normalization to both
            clip_normalized = F.normalize(clip_emb + eps, p=2, dim=-1)
            eva_normalized = F.normalize(eva_emb + eps, p=2, dim=-1)
            
            # Verify normalization
            clip_norm = torch.norm(clip_normalized, dim=-1).mean().item()
            eva_norm = torch.norm(eva_normalized, dim=-1).mean().item()
            
            if abs(clip_norm - 1.0) > 0.1:
                logger.warning(f"CLIP normalization may have failed: norm = {clip_norm:.3f}")
            if abs(eva_norm - 1.0) > 0.1:
                logger.warning(f"EVA normalization may have failed: norm = {eva_norm:.3f}")
            
            shard_data['clip_blip3o_embeddings'] = clip_normalized
            shard_data['eva_blip3o_embeddings'] = eva_normalized
        
        shard_data['normalization_applied'] = True
        return shard_data

    def _add_spherical_noise(self, clean_embeddings: torch.Tensor, noise_level: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add spherical noise to embeddings using slerp"""
        device = clean_embeddings.device
        dtype = clean_embeddings.dtype
        
        # Generate random noise on sphere
        noise = torch.randn_like(clean_embeddings, device=device, dtype=dtype)
        noise = F.normalize(noise, p=2, dim=-1)
        
        # Spherical linear interpolation (slerp)
        cos_angle = torch.sum(clean_embeddings * noise, dim=-1, keepdim=True)
        cos_angle = torch.clamp(cos_angle, -1 + 1e-7, 1 - 1e-7)
        angle = torch.acos(cos_angle)
        
        # Avoid division by zero
        sin_angle = torch.sin(angle)
        sin_angle = torch.clamp(sin_angle, min=1e-7)
        
        # Slerp formula
        clean_weight = torch.sin((1 - noise_level) * angle) / sin_angle
        noise_weight = torch.sin(noise_level * angle) / sin_angle
        
        noisy_embeddings = clean_weight * clean_embeddings + noise_weight * noise
        
        # Ensure result is on unit sphere
        noisy_embeddings = F.normalize(noisy_embeddings, p=2, dim=-1)
        
        return noisy_embeddings, noise

    def _sample_noise_level(self) -> float:
        """Sample noise level based on schedule"""
        if self.noise_schedule == "uniform":
            return self.rng.uniform(self.min_noise_level, self.max_noise_level)
        elif self.noise_schedule == "cosine":
            u = self.rng.uniform(0, 1)
            t = 0.5 * (1 + math.cos(u * math.pi))
            return self.min_noise_level + t * (self.max_noise_level - self.min_noise_level)
        else:
            raise ValueError(f"Unknown noise schedule: {self.noise_schedule}")

    def _load_next_shard(self) -> bool:
        """Load next shard"""
        # Cleanup previous shard
        if self.current_shard_data is not None:
            del self.current_shard_data
            gc.collect()
        
        # Check if more shards available
        if self.current_shard_idx >= len(self.shard_files):
            self.current_shard_data = None
            return False
        
        # Try to load next shard
        while self.current_shard_idx < len(self.shard_files):
            shard_path = self.shard_files[self.current_shard_idx]
            
            self.current_shard_data = self._load_shard(shard_path)
            
            if self.current_shard_data is not None:
                # Prepare samples
                if self.task_mode == "eva_denoising":
                    num_samples = self.current_shard_data['eva_blip3o_embeddings'].shape[0]
                elif self.task_mode == "clip_denoising":
                    num_samples = self.current_shard_data['clip_blip3o_embeddings'].shape[0]
                
                self.current_samples = list(range(num_samples))
                
                if self.shuffle_within_shard:
                    self.rng.shuffle(self.current_samples)
                
                self.current_sample_idx = 0
                
                logger.info(f"Loaded shard {self.current_shard_idx + 1}/{len(self.shard_files)}: {num_samples} samples")
                self.current_shard_idx += 1
                return True
            else:
                self.current_shard_idx += 1
                continue
        
        self.current_shard_data = None
        return False

    def __len__(self) -> int:
        """Estimate total number of samples"""
        if hasattr(self, '_estimated_length'):
            return self._estimated_length
        
        # Try to estimate from manifest
        if hasattr(self, 'manifest') and 'total_samples' in self.manifest:
            manifest_samples = self.manifest['total_samples']
            if self.max_shards is not None:
                total_shards = self.manifest.get('total_shards', len(self.shard_files))
                if total_shards > 0:
                    estimated_samples = int(manifest_samples * self.max_shards / total_shards)
                    self._estimated_length = estimated_samples
                    return estimated_samples
            else:
                self._estimated_length = manifest_samples
                return manifest_samples
        
        # Fallback estimate
        num_shards = len(self.shard_files) if hasattr(self, 'shard_files') else 1
        avg_samples_per_shard = 1000
        
        estimated_samples = num_shards * avg_samples_per_shard
        self._estimated_length = estimated_samples
        
        logger.debug(f"Estimated dataset length: {estimated_samples} samples from {num_shards} shards")
        return estimated_samples

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate through all samples"""
        self.current_shard_idx = 0
        self.current_shard_data = None
        self.current_sample_idx = 0
        self.total_samples_processed = 0
        
        logger.debug(f"Starting iteration over {len(self.shard_files)} shards for {self.task_mode}")
        
        if not self._load_next_shard():
            return
        
        while self.current_shard_data is not None:
            while self.current_sample_idx < len(self.current_samples):
                try:
                    sample_idx = self.current_samples[self.current_sample_idx]
                    
                    # Extract embeddings based on task mode
                    if self.task_mode == "eva_denoising":
                        clean_eva = self.current_shard_data['eva_blip3o_embeddings'][sample_idx]
                        caption = self.current_shard_data['captions'][sample_idx]
                        
                        # Validation
                        if self.validate_shapes:
                            if clean_eva.shape != (self.expected_tokens, 4096):
                                raise ValueError(f"Invalid EVA shape: {clean_eva.shape}")
                        
                        # Check for NaN/Inf
                        if torch.isnan(clean_eva).any():
                            if self.skip_corrupted:
                                self.current_sample_idx += 1
                                continue
                            else:
                                raise ValueError("NaN detected in EVA embeddings")
                        
                        # Sample noise level and add noise
                        noise_level = self._sample_noise_level()
                        noisy_eva, noise = self._add_spherical_noise(clean_eva, noise_level)
                        
                        # Create sample item for EVA denoising
                        item = {
                            # Model inputs
                            'input_embeddings': noisy_eva,         # [N, 4096] - Noisy EVA input
                            'conditioning_embeddings': clean_eva,  # [N, 4096] - Clean EVA conditioning
                            'target_embeddings': clean_eva,        # [N, 4096] - Clean EVA target
                            'noise': noise,                        # [N, 4096] - Pure noise used
                            'noise_level': noise_level,            # scalar - Noise mixing ratio
                            'caption': caption,
                            
                            # Metadata
                            'task_mode': 'eva_denoising',
                            'key': f"shard_{self.current_shard_idx-1}_sample_{sample_idx}",
                            'sample_idx': sample_idx,
                            'training_mode': self.training_mode,
                            'num_tokens': self.expected_tokens,
                            'input_dim': 4096,
                            'output_dim': 4096,
                            'conditioning_dim': 4096,
                        }
                    
                    elif self.task_mode == "clip_denoising":
                        clean_clip = self.current_shard_data['clip_blip3o_embeddings'][sample_idx]
                        clean_eva = self.current_shard_data['eva_blip3o_embeddings'][sample_idx]
                        caption = self.current_shard_data['captions'][sample_idx]
                        
                        # Validation
                        if self.validate_shapes:
                            if clean_clip.shape != (self.expected_tokens, 1024):
                                raise ValueError(f"Invalid CLIP shape: {clean_clip.shape}")
                            if clean_eva.shape != (self.expected_tokens, 4096):
                                raise ValueError(f"Invalid EVA shape: {clean_eva.shape}")
                        
                        # Check for NaN/Inf
                        if torch.isnan(clean_clip).any() or torch.isnan(clean_eva).any():
                            if self.skip_corrupted:
                                self.current_sample_idx += 1
                                continue
                            else:
                                raise ValueError("NaN detected in embeddings")
                        
                        # Sample noise level and add noise to CLIP
                        noise_level = self._sample_noise_level()
                        noisy_clip, noise = self._add_spherical_noise(clean_clip, noise_level)
                        
                        # Create sample item for CLIP denoising
                        item = {
                            # Model inputs
                            'input_embeddings': noisy_clip,        # [N, 1024] - Noisy CLIP input
                            'conditioning_embeddings': clean_eva,  # [N, 4096] - Clean EVA conditioning
                            'target_embeddings': clean_clip,       # [N, 1024] - Clean CLIP target
                            'noise': noise,                        # [N, 1024] - Pure noise used
                            'noise_level': noise_level,            # scalar - Noise mixing ratio
                            'caption': caption,
                            
                            # Metadata
                            'task_mode': 'clip_denoising',
                            'key': f"shard_{self.current_shard_idx-1}_sample_{sample_idx}",
                            'sample_idx': sample_idx,
                            'training_mode': self.training_mode,
                            'num_tokens': self.expected_tokens,
                            'input_dim': 1024,
                            'output_dim': 1024,
                            'conditioning_dim': 4096,
                        }
                    
                    self.current_sample_idx += 1
                    self.total_samples_processed += 1
                    
                    yield item
                    
                except Exception as e:
                    if self.skip_corrupted:
                        logger.warning(f"Skipping corrupted sample {sample_idx}: {e}")
                        self.current_sample_idx += 1
                        continue
                    else:
                        raise
            
            if not self._load_next_shard():
                break
        
        logger.info(f"Iteration completed: {self.total_samples_processed} samples processed")


def universal_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Universal collate function for both EVA and CLIP denoising
    """
    if not batch:
        raise ValueError("Empty batch")
    
    # Filter valid items
    valid_batch = [item for item in batch if item is not None]
    if not valid_batch:
        raise ValueError("No valid items in batch")
    
    try:
        # Get task mode from first item
        task_mode = valid_batch[0]['task_mode']
        
        # Stack embeddings
        input_embeddings = torch.stack([item['input_embeddings'] for item in valid_batch])
        conditioning_embeddings = torch.stack([item['conditioning_embeddings'] for item in valid_batch])
        target_embeddings = torch.stack([item['target_embeddings'] for item in valid_batch])
        noise = torch.stack([item['noise'] for item in valid_batch])
        noise_levels = torch.tensor([item['noise_level'] for item in valid_batch])
        
        # Collect metadata
        captions = [item['caption'] for item in valid_batch]
        keys = [item['key'] for item in valid_batch]
        
        batch_size, seq_len = input_embeddings.shape[:2]
        device = input_embeddings.device
        dtype = input_embeddings.dtype
        
        # Ensure float32 for stability
        input_embeddings = input_embeddings.float()
        conditioning_embeddings = conditioning_embeddings.float()
        target_embeddings = target_embeddings.float()
        noise = noise.float()
        noise_levels = noise_levels.float()
        
        # Ensure L2 normalization
        eps = 1e-8
        input_embeddings = F.normalize(input_embeddings + eps, p=2, dim=-1)
        conditioning_embeddings = F.normalize(conditioning_embeddings + eps, p=2, dim=-1)
        target_embeddings = F.normalize(target_embeddings + eps, p=2, dim=-1)
        noise = F.normalize(noise + eps, p=2, dim=-1)
        
        # SPHERICAL FLOW MATCHING SETUP
        timesteps = torch.rand(batch_size, device=device, dtype=dtype)
        t_expanded = timesteps.view(batch_size, 1, 1)
        
        # Compute angles between target and noise
        cos_angles = torch.sum(target_embeddings * noise, dim=-1, keepdim=True)
        cos_angles = torch.clamp(cos_angles, -1 + 1e-7, 1 - 1e-7)
        angles = torch.acos(cos_angles)
        
        # Avoid division by zero
        sin_angles = torch.sin(angles)
        sin_angles = torch.clamp(sin_angles, min=1e-7)
        
        # Spherical interpolation: x_t = slerp(noise, target, t)
        target_weight = torch.sin(t_expanded * angles) / sin_angles
        noise_weight = torch.sin((1 - t_expanded) * angles) / sin_angles
        
        x_t = noise_weight * noise + target_weight * target_embeddings
        x_t = F.normalize(x_t + eps, p=2, dim=-1)
        
        # Spherical velocity (for velocity prediction)
        velocity_target = target_embeddings - noise
        
        # Get dimensions for validation
        input_dim = valid_batch[0]['input_dim']
        output_dim = valid_batch[0]['output_dim']
        conditioning_dim = valid_batch[0]['conditioning_dim']
        
        # Validation
        assert input_embeddings.shape == (batch_size, seq_len, input_dim)
        assert conditioning_embeddings.shape == (batch_size, seq_len, conditioning_dim)
        assert target_embeddings.shape == (batch_size, seq_len, output_dim)
        assert x_t.shape == (batch_size, seq_len, input_dim)
        assert velocity_target.shape == (batch_size, seq_len, input_dim)
        assert timesteps.shape == (batch_size,)
        
        return {
            # Model inputs (universal interface)
            'hidden_states': x_t,                            # [B, N, input_dim] - Interpolated state
            'encoder_hidden_states': conditioning_embeddings, # [B, N, conditioning_dim] - Conditioning
            'timestep': timesteps,                           # [B] - Flow matching timesteps
            
            # Training targets
            'target_embeddings': target_embeddings,          # [B, N, output_dim] - Clean target
            'velocity_target': velocity_target,              # [B, N, input_dim] - Velocity for flow matching
            'noise': noise,                                  # [B, N, input_dim] - Pure noise
            'input_embeddings': input_embeddings,            # [B, N, input_dim] - Original input
            
            # Flow matching state
            'x_t': x_t,                                      # [B, N, input_dim] - Current flow state
            'noise_levels': noise_levels,                    # [B] - Original noise levels
            
            # Metadata
            'task_mode': task_mode,
            'captions': captions,
            'keys': keys,
            'batch_size': batch_size,
            'training_mode': valid_batch[0]['training_mode'],
            'num_tokens': valid_batch[0]['num_tokens'],
            'seq_len': seq_len,
            'input_dim': input_dim,
            'output_dim': output_dim,
            'conditioning_dim': conditioning_dim,
            
            # Normalization status
            'embeddings_normalized': True,
            'input_norm_mean': torch.norm(input_embeddings, dim=-1).mean().item(),
            'conditioning_norm_mean': torch.norm(conditioning_embeddings, dim=-1).mean().item(),
            'target_norm_mean': torch.norm(target_embeddings, dim=-1).mean().item(),
        }
        
    except Exception as e:
        logger.error(f"Error in universal collate function: {e}")
        logger.error(f"Batch size: {len(batch)}")
        if batch:
            try:
                logger.error(f"First item keys: {list(batch[0].keys())}")
                logger.error(f"Task mode: {batch[0].get('task_mode', 'unknown')}")
            except:
                pass
        raise


def create_universal_dataloaders(
    chunked_embeddings_dir: Union[str, Path],
    task_mode: str = "eva_denoising",  # NEW: "eva_denoising" or "clip_denoising"
    batch_size: int = 16,
    eval_batch_size: Optional[int] = None,
    training_mode: str = "patch_only",
    max_shards: Optional[int] = None,
    noise_schedule: str = "uniform",
    max_noise_level: float = 0.9,
    min_noise_level: float = 0.1,
    num_workers: int = 0,
    pin_memory: bool = False,
    **kwargs
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Create universal dataloaders for EVA or CLIP denoising"""
    
    if eval_batch_size is None:
        eval_batch_size = batch_size
    
    # Task info
    if task_mode == "eva_denoising":
        logger.info(f"Creating EVA denoising dataloaders:")
        logger.info(f"  INPUT: Noisy EVA embeddings [B, N, 4096]")
        logger.info(f"  CONDITIONING: Clean EVA embeddings [B, N, 4096]")
        logger.info(f"  TARGET: Clean EVA embeddings [B, N, 4096]")
    elif task_mode == "clip_denoising":
        logger.info(f"Creating CLIP denoising dataloaders:")
        logger.info(f"  INPUT: Noisy CLIP embeddings [B, N, 1024]")
        logger.info(f"  CONDITIONING: Clean EVA embeddings [B, N, 4096]")
        logger.info(f"  TARGET: Clean CLIP embeddings [B, N, 1024]")
    else:
        raise ValueError(f"Unknown task_mode: {task_mode}")
    
    logger.info(f"  Noise schedule: {noise_schedule}")
    logger.info(f"  Noise range: [{min_noise_level}, {max_noise_level}]")
    
    # Create training dataset
    train_dataset = UniversalDenoisingDataset(
        chunked_embeddings_dir=chunked_embeddings_dir,
        task_mode=task_mode,
        split="train",
        training_mode=training_mode,
        max_shards=max_shards,
        shuffle_shards=True,
        shuffle_within_shard=True,
        noise_schedule=noise_schedule,
        max_noise_level=max_noise_level,
        min_noise_level=min_noise_level,
        **kwargs
    )
    
    # Create training dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=universal_collate_fn,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    
    # Create evaluation dataset
    eval_dataset = UniversalDenoisingDataset(
        chunked_embeddings_dir=chunked_embeddings_dir,
        task_mode=task_mode,
        split="eval",
        training_mode=training_mode,
        max_shards=max_shards,
        shuffle_shards=False,
        shuffle_within_shard=False,
        noise_schedule="uniform",
        max_noise_level=0.7,
        min_noise_level=0.3,
        **kwargs
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=eval_batch_size,
        num_workers=min(num_workers, 1),
        collate_fn=universal_collate_fn,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=min(num_workers, 1) > 0,
    )
    
    logger.info(f"Universal dataloaders created successfully for {task_mode}")
    
    return train_dataloader, eval_dataloader


# Backward compatibility aliases
def create_eva_denoising_dataloaders(*args, **kwargs):
    """Backward compatibility: create EVA denoising dataloaders"""
    kwargs['task_mode'] = 'eva_denoising'
    return create_universal_dataloaders(*args, **kwargs)

def create_clip_denoising_dataloaders(*args, **kwargs):
    """NEW: Create CLIP denoising dataloaders"""
    kwargs['task_mode'] = 'clip_denoising'
    return create_universal_dataloaders(*args, **kwargs)

# Legacy alias
eva_denoising_collate_fn = universal_collate_fn
BLIP3oEVADenoisingDataset = UniversalDenoisingDataset