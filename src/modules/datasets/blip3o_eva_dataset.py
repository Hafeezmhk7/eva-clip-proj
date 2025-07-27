#!/usr/bin/env python3
"""
Fixed BLIP3-o Dataset for EVA-CLIP Reproduction
Key fixes:
1. Proper data loading and validation
2. Correct collate function for flow matching
3. Better error handling and normalization
4. Clear input/output handling
5. Fixed tensor naming for compatibility
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

logger = logging.getLogger(__name__)


class BLIP3oEVAReproductionDataset(IterableDataset):
    """
    Fixed dataset for EVA-CLIP reproduction with proper data handling
    
    This dataset loads:
    - EVA embeddings [B, N, 4096] as TARGET (what we want to reproduce)
    - CLIP embeddings [B, N, 1024] as CONDITIONING (guidance)
    """
    
    def __init__(
        self,
        chunked_embeddings_dir: Union[str, Path],
        split: str = "train",
        training_mode: str = "patch_only",
        normalize_embeddings: bool = True,
        max_shards: Optional[int] = None,
        shuffle_shards: bool = True,
        shuffle_within_shard: bool = True,
        expected_tokens: Optional[int] = None,
        # Error handling
        skip_corrupted: bool = True,
        validate_shapes: bool = True,
        max_retries: int = 3,
    ):
        super().__init__()
        
        self.chunked_embeddings_dir = Path(chunked_embeddings_dir)
        self.split = split
        self.training_mode = training_mode
        self.normalize_embeddings = normalize_embeddings
        self.max_shards = max_shards
        self.shuffle_shards = shuffle_shards
        self.shuffle_within_shard = shuffle_within_shard
        self.skip_corrupted = skip_corrupted
        self.validate_shapes = validate_shapes
        self.max_retries = max_retries
        
        # Determine expected tokens
        if expected_tokens is None:
            self.expected_tokens = 257 if training_mode == "cls_patch" else 256
        else:
            self.expected_tokens = expected_tokens
        
        # Setup random state
        self.rng = random.Random(42)
        
        # Load manifest and prepare shards
        self._load_manifest()
        self._prepare_shard_list()
        
        # Current state
        self.current_shard_idx = 0
        self.current_shard_data = None
        self.current_sample_idx = 0
        self.total_samples_processed = 0
        
        logger.info(f"EVA Reproduction Dataset initialized:")
        logger.info(f"  Directory: {self.chunked_embeddings_dir}")
        logger.info(f"  Mode: {self.training_mode} ({self.expected_tokens} tokens)")
        logger.info(f"  TARGET: EVA embeddings [B, N, 4096]")
        logger.info(f"  CONDITIONING: CLIP embeddings [B, N, 1024]")
        logger.info(f"  Normalize: {self.normalize_embeddings}")
        logger.info(f"  Shards: {len(self.shard_files)}")

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
        # Look for shard files
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
        """Validate and process shard data"""
        # Check required keys
        required_keys = ['clip_blip3o_embeddings', 'eva_blip3o_embeddings', 'captions']
        for key in required_keys:
            if key not in shard_data:
                raise ValueError(f"Missing key '{key}' in shard {shard_path}")
        
        # Get embeddings
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
            if clip_emb.dim() != 3 or eva_emb.dim() != 3:
                raise ValueError(f"Expected 3D tensors, got CLIP: {clip_emb.shape}, EVA: {eva_emb.shape}")
            
            if clip_emb.shape[0] != eva_emb.shape[0]:
                raise ValueError(f"Batch size mismatch: CLIP {clip_emb.shape[0]} vs EVA {eva_emb.shape[0]}")
            
            clip_tokens, eva_tokens = clip_emb.shape[1], eva_emb.shape[1]
            if clip_tokens != eva_tokens:
                raise ValueError(f"Token count mismatch: CLIP {clip_tokens} vs EVA {eva_tokens}")
        
        # Handle token count adaptation
        current_tokens = clip_emb.shape[1]
        if current_tokens != self.expected_tokens:
            logger.debug(f"Adapting from {current_tokens} to {self.expected_tokens} tokens")
            
            if current_tokens == 256 and self.expected_tokens == 257:
                # Add CLS token (average of patches)
                clip_cls = clip_emb.mean(dim=1, keepdim=True)
                eva_cls = eva_emb.mean(dim=1, keepdim=True)
                shard_data['clip_blip3o_embeddings'] = torch.cat([clip_cls, clip_emb], dim=1)
                shard_data['eva_blip3o_embeddings'] = torch.cat([eva_cls, eva_emb], dim=1)
            elif current_tokens == 257 and self.expected_tokens == 256:
                # Remove CLS token
                shard_data['clip_blip3o_embeddings'] = clip_emb[:, 1:, :]
                shard_data['eva_blip3o_embeddings'] = eva_emb[:, 1:, :]
            else:
                raise ValueError(f"Cannot adapt from {current_tokens} to {self.expected_tokens} tokens")
        
        # Apply normalization if requested
        if self.normalize_embeddings:
            shard_data = self._normalize_embeddings(shard_data)

    def _normalize_embeddings(self, shard_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply L2 normalization to embeddings"""
        clip_emb = shard_data['clip_blip3o_embeddings']
        eva_emb = shard_data['eva_blip3o_embeddings']
        
        # Check for NaN/Inf
        if torch.isnan(clip_emb).any() or torch.isinf(clip_emb).any():
            logger.warning("Found NaN/Inf in CLIP embeddings")
            clip_emb = torch.nan_to_num(clip_emb, nan=0.0, posinf=1.0, neginf=-1.0)
        
        if torch.isnan(eva_emb).any() or torch.isinf(eva_emb).any():
            logger.warning("Found NaN/Inf in EVA embeddings")
            eva_emb = torch.nan_to_num(eva_emb, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Apply L2 normalization
        eps = 1e-8
        clip_normalized = F.normalize(clip_emb + eps, p=2, dim=-1)
        eva_normalized = F.normalize(eva_emb + eps, p=2, dim=-1)
        
        # Check normalization
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

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate through all samples"""
        self.current_shard_idx = 0
        self.current_shard_data = None
        self.current_sample_idx = 0
        self.total_samples_processed = 0
        
        logger.debug(f"Starting iteration over {len(self.shard_files)} shards")
        
        if not self._load_next_shard():
            return
        
        while self.current_shard_data is not None:
            while self.current_sample_idx < len(self.current_samples):
                try:
                    sample_idx = self.current_samples[self.current_sample_idx]
                    
                    # Extract sample data
                    clip_emb = self.current_shard_data['clip_blip3o_embeddings'][sample_idx]
                    eva_emb = self.current_shard_data['eva_blip3o_embeddings'][sample_idx]
                    caption = self.current_shard_data['captions'][sample_idx]
                    
                    # Final validation
                    if self.validate_shapes:
                        if clip_emb.shape != (self.expected_tokens, 1024):
                            raise ValueError(f"Invalid CLIP shape: {clip_emb.shape}")
                        if eva_emb.shape != (self.expected_tokens, 4096):
                            raise ValueError(f"Invalid EVA shape: {eva_emb.shape}")
                    
                    # Check for NaN/Inf
                    if torch.isnan(clip_emb).any() or torch.isnan(eva_emb).any():
                        if self.skip_corrupted:
                            self.current_sample_idx += 1
                            continue
                        else:
                            raise ValueError("NaN detected in embeddings")
                    
                    # Create sample item for EVA reproduction
                    item = {
                        'clip_embeddings': clip_emb,  # [N, 1024] - CONDITIONING
                        'eva_embeddings': eva_emb,    # [N, 4096] - TARGET to reproduce
                        'caption': caption,
                        'key': f"shard_{self.current_shard_idx-1}_sample_{sample_idx}",
                        'sample_idx': sample_idx,
                        'training_mode': self.training_mode,
                        'num_tokens': self.expected_tokens,
                        'normalized': self.current_shard_data.get('normalization_applied', False),
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


def eva_reproduction_collate_fn_fixed(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Fixed collate function for EVA reproduction with proper tensor naming
    
    This function:
    1. Takes clean EVA embeddings as targets
    2. Creates noisy versions for input
    3. Uses CLIP embeddings for conditioning
    4. Sets up flow matching with proper timesteps
    5. Uses correct tensor names for trainer compatibility
    """
    if not batch:
        raise ValueError("Empty batch")
    
    # Filter valid items
    valid_batch = [item for item in batch if item is not None]
    if not valid_batch:
        raise ValueError("No valid items in batch")
    
    try:
        # Stack embeddings
        clip_embeddings = torch.stack([item['clip_embeddings'] for item in valid_batch])  # [B, N, 1024]
        eva_embeddings = torch.stack([item['eva_embeddings'] for item in valid_batch])    # [B, N, 4096]
        
        # Collect metadata
        captions = [item['caption'] for item in valid_batch]
        keys = [item['key'] for item in valid_batch]
        
        batch_size, seq_len, eva_dim = eva_embeddings.shape
        device = eva_embeddings.device
        dtype = eva_embeddings.dtype
        
        # Ensure float32 for stability
        clip_embeddings = clip_embeddings.float()
        eva_embeddings = eva_embeddings.float()
        
        # Apply final normalization to ensure L2 norm = 1
        eps = 1e-8
        clip_embeddings = F.normalize(clip_embeddings + eps, p=2, dim=-1)
        eva_embeddings = F.normalize(eva_embeddings + eps, p=2, dim=-1)
        
        # FLOW MATCHING SETUP for EVA reproduction
        # Sample random timesteps for each sample in batch
        timesteps = torch.rand(batch_size, device=device, dtype=dtype)
        
        # Create noise (source distribution) - same dimension as EVA
        noise = torch.randn_like(eva_embeddings, device=device, dtype=dtype)
        noise = F.normalize(noise + eps, p=2, dim=-1)  # Normalize noise
        
        # Linear interpolation for rectified flow: x_t = (1-t) * noise + t * eva_clean
        t_expanded = timesteps.view(batch_size, 1, 1)  # [B, 1, 1]
        noisy_eva = (1 - t_expanded) * noise + t_expanded * eva_embeddings
        
        # Velocity target: v = eva_clean - noise (for rectified flow)
        velocity_target = eva_embeddings - noise
        
        # Validation
        assert clip_embeddings.shape == (batch_size, seq_len, 1024), f"CLIP shape: {clip_embeddings.shape}"
        assert eva_embeddings.shape == (batch_size, seq_len, 4096), f"EVA shape: {eva_embeddings.shape}"
        assert noisy_eva.shape == (batch_size, seq_len, 4096), f"Noisy EVA shape: {noisy_eva.shape}"
        assert velocity_target.shape == (batch_size, seq_len, 4096), f"Velocity target shape: {velocity_target.shape}"
        assert timesteps.shape == (batch_size,), f"Timesteps shape: {timesteps.shape}"
        
        # Return with correct tensor names for trainer compatibility
        return {
            # Model inputs (using trainer-expected names)
            'encoder_hidden_states': clip_embeddings,    # [B, N, 1024] - CLIP conditioning
            'hidden_states': noisy_eva,                  # [B, N, 4096] - Noisy EVA input
            'timestep': timesteps,                       # [B] - Flow matching timesteps
            
            # Training targets
            'eva_embeddings': eva_embeddings,            # [B, N, 4096] - Clean EVA (target)
            'velocity_target': velocity_target,          # [B, N, 4096] - Velocity for flow matching
            'noise': noise,                              # [B, N, 4096] - Original noise
            
            # Metadata
            'captions': captions,
            'keys': keys,
            'batch_size': batch_size,
            'training_mode': valid_batch[0]['training_mode'],
            'num_tokens': valid_batch[0]['num_tokens'],
            'seq_len': seq_len,
            
            # Normalization status
            'clip_embeddings_normalized': True,
            'eva_embeddings_normalized': True,
            'clip_norm_mean': torch.norm(clip_embeddings, dim=-1).mean().item(),
            'eva_norm_mean': torch.norm(eva_embeddings, dim=-1).mean().item(),
        }
        
    except Exception as e:
        logger.error(f"Error in collate function: {e}")
        logger.error(f"Batch size: {len(batch)}")
        if batch:
            try:
                logger.error(f"First item keys: {list(batch[0].keys())}")
                for key, value in batch[0].items():
                    if torch.is_tensor(value):
                        logger.error(f"  {key}: {value.shape} {value.dtype}")
            except:
                pass
        raise


def create_eva_reproduction_dataloaders(
    chunked_embeddings_dir: Union[str, Path],
    batch_size: int = 16,
    eval_batch_size: Optional[int] = None,
    training_mode: str = "patch_only",
    max_shards: Optional[int] = None,
    normalize_embeddings: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    **kwargs
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Create dataloaders for EVA reproduction with fixed collate function"""
    
    if eval_batch_size is None:
        eval_batch_size = batch_size
    
    logger.info(f"Creating EVA reproduction dataloaders:")
    logger.info(f"  Target: EVA embeddings [B, N, 4096]")
    logger.info(f"  Conditioning: CLIP embeddings [B, N, 1024]")
    logger.info(f"  Normalize: {normalize_embeddings}")
    
    # Create training dataset
    train_dataset = BLIP3oEVAReproductionDataset(
        chunked_embeddings_dir=chunked_embeddings_dir,
        split="train",
        training_mode=training_mode,
        normalize_embeddings=normalize_embeddings,
        max_shards=max_shards,
        shuffle_shards=True,
        shuffle_within_shard=True,
        **kwargs
    )
    
    # Create training dataloader with fixed collate function
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=eva_reproduction_collate_fn_fixed,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    
    # Create evaluation dataset (same data for now)
    eval_dataset = BLIP3oEVAReproductionDataset(
        chunked_embeddings_dir=chunked_embeddings_dir,
        split="eval",
        training_mode=training_mode,
        normalize_embeddings=normalize_embeddings,
        max_shards=max_shards,
        shuffle_shards=False,
        shuffle_within_shard=False,
        **kwargs
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=eval_batch_size,
        num_workers=min(num_workers, 1),
        collate_fn=eva_reproduction_collate_fn_fixed,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=min(num_workers, 1) > 0,
    )
    
    logger.info(f"EVA reproduction dataloaders created successfully")
    
    return train_dataloader, eval_dataloader