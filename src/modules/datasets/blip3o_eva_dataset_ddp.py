#!/usr/bin/env python3
"""
Complete DDP-Aware Dataset for BLIP3-o Universal Denoising
Handles both EVA and CLIP denoising with proper distributed data loading
"""

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import numpy as np
import pickle
import random
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
import glob
import json
import gc
import time
import math

logger = logging.getLogger(__name__)


class DDPUniversalDenoisingDataset(Dataset):
    """
    DDP-aware universal dataset for both EVA and CLIP denoising
    """
    
    def __init__(
        self,
        chunked_embeddings_dir: str,
        task_mode: str = "clip_denoising",
        training_mode: str = "patch_only",
        max_shards: int = 10,
        max_shard_cache: int = 3,
        samples_per_shard_load: int = 1000,
        noise_schedule: str = "uniform",
        max_noise_level: float = 0.9,
        min_noise_level: float = 0.1,
        split: str = "train",
        rank: int = 0,
        world_size: int = 1,
        debug_mode: bool = False,
        seed: int = 42,
    ):
        self.embeddings_dir = Path(chunked_embeddings_dir)
        self.task_mode = task_mode
        self.training_mode = training_mode
        self.max_shards = max_shards
        self.max_shard_cache = max_shard_cache
        self.samples_per_shard_load = samples_per_shard_load
        self.noise_schedule = noise_schedule
        self.max_noise_level = max_noise_level
        self.min_noise_level = min_noise_level
        self.split = split
        self.rank = rank
        self.world_size = world_size
        self.debug_mode = debug_mode
        self.seed = seed
        
        # Set random seed for reproducibility
        random.seed(seed + rank)
        np.random.seed(seed + rank)
        torch.manual_seed(seed + rank)
        
        # Initialize dataset
        self._load_manifest()
        self._setup_shards()
        self._init_cache()
        
        # Task-specific configuration
        self._setup_task_config()
        
        if rank == 0:
            logger.info(f"DDP Dataset initialized for {task_mode}")
            logger.info(f"  Rank: {rank}/{world_size}")
            logger.info(f"  Shards: {len(self.shard_files)}")
            logger.info(f"  Total samples: {self.total_samples}")
            logger.info(f"  Samples per rank: {len(self)}")
    
    def _load_manifest(self):
        """Load dataset manifest"""
        manifest_path = self.embeddings_dir / "embeddings_manifest.json"
        
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                self.manifest = json.load(f)
            
            self.total_samples = self.manifest.get('total_samples', 0)
            if self.rank == 0:
                logger.info(f"Loaded manifest: {self.total_samples} total samples")
        else:
            if self.rank == 0:
                logger.warning(f"No manifest found at {manifest_path}")
            self.manifest = {}
            self.total_samples = 0
    
    def _setup_shards(self):
        """Setup shard files for this dataset"""
        # Find shard files
        shard_pattern = f"*_{self.training_mode}.pkl"
        all_shard_files = list(self.embeddings_dir.glob(shard_pattern))
        all_shard_files.sort()
        
        # Limit shards if specified
        if self.max_shards is not None:
            all_shard_files = all_shard_files[:self.max_shards]
        
        # Split shards across ranks for DDP
        shards_per_rank = len(all_shard_files) // self.world_size
        start_idx = self.rank * shards_per_rank
        
        if self.rank == self.world_size - 1:
            # Last rank gets remaining shards
            end_idx = len(all_shard_files)
        else:
            end_idx = start_idx + shards_per_rank
        
        self.shard_files = all_shard_files[start_idx:end_idx]
        
        if self.rank == 0:
            logger.info(f"Shard distribution:")
            logger.info(f"  Total shards: {len(all_shard_files)}")
            logger.info(f"  Shards per rank: {shards_per_rank}")
            logger.info(f"  Rank {self.rank} shards: {len(self.shard_files)}")
        
        # Load shard metadata
        self.shard_sample_counts = []
        self.shard_sample_offsets = []
        current_offset = 0
        
        for shard_file in self.shard_files:
            try:
                with open(shard_file, 'rb') as f:
                    shard_data = pickle.load(f)
                sample_count = len(shard_data.get('captions', []))
                self.shard_sample_counts.append(sample_count)
                self.shard_sample_offsets.append(current_offset)
                current_offset += sample_count
            except Exception as e:
                if self.rank == 0:
                    logger.error(f"Error loading shard {shard_file}: {e}")
                self.shard_sample_counts.append(0)
                self.shard_sample_offsets.append(current_offset)
        
        self.rank_total_samples = sum(self.shard_sample_counts)
        
        if self.rank == 0:
            logger.info(f"Rank {self.rank} has {self.rank_total_samples} samples across {len(self.shard_files)} shards")
    
    def _setup_task_config(self):
        """Setup task-specific configuration"""
        if self.task_mode == "eva_denoising":
            self.input_key = 'eva_blip3o_embeddings'
            self.conditioning_key = 'eva_blip3o_embeddings'
            self.target_key = 'eva_blip3o_embeddings'
            self.input_dim = 4096
            self.conditioning_dim = 4096
            self.output_dim = 4096
        elif self.task_mode == "clip_denoising":
            self.input_key = 'clip_blip3o_embeddings'
            self.conditioning_key = 'eva_blip3o_embeddings'
            self.target_key = 'clip_blip3o_embeddings'
            self.input_dim = 1024
            self.conditioning_dim = 4096
            self.output_dim = 1024
        else:
            raise ValueError(f"Unknown task mode: {self.task_mode}")
        
        # Token configuration
        if self.training_mode == "cls_patch":
            self.num_tokens = 257
        else:
            self.num_tokens = 256
    
    def _init_cache(self):
        """Initialize shard cache"""
        self.shard_cache = {}
        self.cache_order = []
        self.current_samples = []
        self.current_sample_indices = []
    
    def _load_shard(self, shard_idx: int) -> Dict[str, Any]:
        """Load a shard into cache"""
        if shard_idx in self.shard_cache:
            # Move to end of cache order (LRU)
            self.cache_order.remove(shard_idx)
            self.cache_order.append(shard_idx)
            return self.shard_cache[shard_idx]
        
        # Load shard from disk
        shard_file = self.shard_files[shard_idx]
        
        try:
            with open(shard_file, 'rb') as f:
                shard_data = pickle.load(f)
            
            # Cache management
            if len(self.shard_cache) >= self.max_shard_cache:
                # Remove oldest shard
                oldest_shard = self.cache_order.pop(0)
                del self.shard_cache[oldest_shard]
                gc.collect()
            
            self.shard_cache[shard_idx] = shard_data
            self.cache_order.append(shard_idx)
            
            if self.debug_mode and self.rank == 0:
                logger.debug(f"Loaded shard {shard_idx} with {len(shard_data.get('captions', []))} samples")
            
            return shard_data
            
        except Exception as e:
            if self.rank == 0:
                logger.error(f"Failed to load shard {shard_idx}: {e}")
            return {'captions': [], self.input_key: torch.empty(0), self.conditioning_key: torch.empty(0)}
    
    def _get_sample_from_shard(self, shard_idx: int, sample_idx: int) -> Dict[str, Any]:
        """Get a specific sample from a shard"""
        shard_data = self._load_shard(shard_idx)
        
        try:
            # Extract sample data
            caption = shard_data.get('captions', [''])[sample_idx]
            key = shard_data.get('keys', [f'sample_{sample_idx}'])[sample_idx]
            
            # Get embeddings
            input_embeddings = shard_data[self.input_key][sample_idx]  # [N, input_dim]
            conditioning_embeddings = shard_data[self.conditioning_key][sample_idx]  # [N, conditioning_dim]
            target_embeddings = shard_data[self.target_key][sample_idx]  # [N, target_dim]
            
            # Ensure correct token count
            if input_embeddings.shape[0] != self.num_tokens:
                if self.rank == 0:
                    logger.warning(f"Token count mismatch: expected {self.num_tokens}, got {input_embeddings.shape[0]}")
                # Truncate or pad if necessary
                if input_embeddings.shape[0] > self.num_tokens:
                    input_embeddings = input_embeddings[:self.num_tokens]
                    conditioning_embeddings = conditioning_embeddings[:self.num_tokens]
                    target_embeddings = target_embeddings[:self.num_tokens]
            
            return {
                'input_embeddings': input_embeddings,
                'conditioning_embeddings': conditioning_embeddings,
                'target_embeddings': target_embeddings,
                'caption': caption,
                'key': key,
                'shard_idx': shard_idx,
                'sample_idx': sample_idx,
            }
            
        except (IndexError, KeyError) as e:
            if self.rank == 0:
                logger.warning(f"Error extracting sample {sample_idx} from shard {shard_idx}: {e}")
            
            # Return dummy sample
            dummy_input = torch.randn(self.num_tokens, self.input_dim)
            dummy_conditioning = torch.randn(self.num_tokens, self.conditioning_dim)
            dummy_target = torch.randn(self.num_tokens, self.output_dim)
            
            return {
                'input_embeddings': dummy_input,
                'conditioning_embeddings': dummy_conditioning,
                'target_embeddings': dummy_target,
                'caption': 'dummy',
                'key': 'dummy',
                'shard_idx': shard_idx,
                'sample_idx': sample_idx,
            }
    
    def _add_noise_and_flow(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Add noise and prepare for flow matching"""
        target = sample['target_embeddings']  # [N, D]
        conditioning = sample['conditioning_embeddings']  # [N, D_cond]
        
        # Normalize to unit sphere
        target_normalized = torch.nn.functional.normalize(target, p=2, dim=-1)
        conditioning_normalized = torch.nn.functional.normalize(conditioning, p=2, dim=-1)
        
        # Sample noise level
        if self.noise_schedule == "uniform":
            noise_level = torch.rand(1).item() * (self.max_noise_level - self.min_noise_level) + self.min_noise_level
        elif self.noise_schedule == "cosine":
            # Cosine schedule tends to sample more from middle range
            u = torch.rand(1).item()
            noise_level = (1 - torch.cos(u * math.pi)) / 2
            noise_level = noise_level * (self.max_noise_level - self.min_noise_level) + self.min_noise_level
        else:
            noise_level = 0.5  # Fixed noise level for debugging
        
        # Create noise on same device and same shape as target
        noise = torch.randn_like(target_normalized)
        noise_normalized = torch.nn.functional.normalize(noise, p=2, dim=-1)
        
        # Spherical linear interpolation (SLERP)
        t = noise_level
        
        # Compute angle between target and noise
        cos_angle = torch.sum(target_normalized * noise_normalized, dim=-1, keepdim=True)
        cos_angle = torch.clamp(cos_angle, -1 + 1e-7, 1 - 1e-7)
        angle = torch.acos(cos_angle)
        
        # SLERP interpolation
        sin_angle = torch.sin(angle)
        sin_angle = torch.clamp(sin_angle, min=1e-7)
        
        w0 = torch.sin((1 - t) * angle) / sin_angle
        w1 = torch.sin(t * angle) / sin_angle
        
        # Handle small angles with linear interpolation
        small_angle_mask = angle < 1e-6
        linear_interp = (1 - t) * target_normalized + t * noise_normalized
        linear_interp = torch.nn.functional.normalize(linear_interp, p=2, dim=-1)
        
        slerp_result = w0 * target_normalized + w1 * noise_normalized
        slerp_result = torch.nn.functional.normalize(slerp_result, p=2, dim=-1)
        
        # Use linear interpolation for small angles
        interpolated = torch.where(small_angle_mask, linear_interp, slerp_result)
        
        # Velocity target for flow matching
        velocity_target = (target_normalized - interpolated) / (1e-3 + 1 - t)  # Simple velocity approximation
        
        return {
            **sample,
            'noise': noise_normalized,
            'noise_level': torch.tensor(noise_level),
            'input_embeddings': interpolated,  # Noisy version at timestep t
            'velocity_target': velocity_target,
            'task_mode': self.task_mode,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'conditioning_dim': self.conditioning_dim,
            'num_tokens': self.num_tokens,
        }
    
    def __len__(self) -> int:
        """Return total samples for this rank"""
        return self.rank_total_samples
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get item by global index"""
        # Find which shard this index belongs to
        shard_idx = 0
        sample_idx = idx
        
        for i, count in enumerate(self.shard_sample_counts):
            if sample_idx < count:
                shard_idx = i
                break
            sample_idx -= count
        else:
            # Index out of range, return dummy sample
            if self.rank == 0:
                logger.warning(f"Index {idx} out of range (max: {self.rank_total_samples})")
            
            dummy_input = torch.randn(self.num_tokens, self.input_dim)
            dummy_conditioning = torch.randn(self.num_tokens, self.conditioning_dim)
            dummy_target = torch.randn(self.num_tokens, self.output_dim)
            
            sample = {
                'input_embeddings': dummy_input,
                'conditioning_embeddings': dummy_conditioning,
                'target_embeddings': dummy_target,
                'caption': 'dummy',
                'key': 'dummy',
                'shard_idx': -1,
                'sample_idx': -1,
            }
            return self._add_noise_and_flow(sample)
        
        # Get sample from shard
        sample = self._get_sample_from_shard(shard_idx, sample_idx)
        
        # Add noise and prepare for flow matching
        return self._add_noise_and_flow(sample)


def universal_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Universal collate function for both EVA and CLIP denoising"""
    if not batch:
        raise ValueError("Empty batch")
    
    # Filter valid items
    valid_batch = [item for item in batch if item is not None]
    if not valid_batch:
        raise ValueError("No valid items in batch")
    
    # Get first item to determine structure
    first_item = valid_batch[0]
    task_mode = first_item.get('task_mode', 'unknown')
    
    # Stack tensors
    collated = {}
    
    # Main tensors
    tensor_keys = ['input_embeddings', 'conditioning_embeddings', 'target_embeddings', 
                   'noise', 'noise_level', 'velocity_target']
    
    for key in tensor_keys:
        if key in first_item:
            if key == 'noise_level':
                # Scalar values
                values = torch.tensor([item[key] for item in valid_batch])
            else:
                # Tensor values
                values = torch.stack([item[key] for item in valid_batch])
            collated[key] = values
    
    # Create flow matching inputs (universal interface)
    if 'input_embeddings' in collated and 'noise_level' in collated:
        # These are the noisy embeddings at timestep t
        collated['hidden_states'] = collated['input_embeddings']
        collated['timestep'] = collated['noise_level']
        collated['encoder_hidden_states'] = collated['conditioning_embeddings']
    
    # Metadata
    collated['batch_size'] = len(valid_batch)
    collated['task_mode'] = task_mode
    
    # Lists
    if 'caption' in first_item:
        collated['captions'] = [item['caption'] for item in valid_batch]
    if 'key' in first_item:
        collated['keys'] = [item['key'] for item in valid_batch]
    
    # Dimensions info
    for key in ['input_dim', 'output_dim', 'conditioning_dim', 'num_tokens']:
        if key in first_item:
            collated[key] = first_item[key]
    
    return collated


def create_ddp_dataloaders(
    chunked_embeddings_dir: str,
    task_mode: str = "clip_denoising",
    batch_size: int = 4,
    training_mode: str = "patch_only",
    max_shards: int = 10,
    max_shard_cache: int = 3,
    samples_per_shard_load: int = 1000,
    noise_schedule: str = "uniform",
    max_noise_level: float = 0.9,
    min_noise_level: float = 0.1,
    num_workers: int = 2,
    rank: int = 0,
    world_size: int = 1,
    pin_memory: bool = True,
    debug_mode: bool = False,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create DDP-aware dataloaders for training and evaluation
    """
    
    # Create datasets
    train_dataset = DDPUniversalDenoisingDataset(
        chunked_embeddings_dir=chunked_embeddings_dir,
        task_mode=task_mode,
        training_mode=training_mode,
        max_shards=max_shards,
        max_shard_cache=max_shard_cache,
        samples_per_shard_load=samples_per_shard_load,
        noise_schedule=noise_schedule,
        max_noise_level=max_noise_level,
        min_noise_level=min_noise_level,
        split="train",
        rank=rank,
        world_size=world_size,
        debug_mode=debug_mode,
        seed=seed,
    )
    
    # For evaluation, use a smaller subset and fixed noise
    eval_dataset = DDPUniversalDenoisingDataset(
        chunked_embeddings_dir=chunked_embeddings_dir,
        task_mode=task_mode,
        training_mode=training_mode,
        max_shards=min(3, max_shards),  # Use fewer shards for eval
        max_shard_cache=2,
        samples_per_shard_load=samples_per_shard_load // 4,
        noise_schedule="uniform",
        max_noise_level=0.5,  # Fixed noise level for eval
        min_noise_level=0.5,
        split="eval",
        rank=rank,
        world_size=world_size,
        debug_mode=debug_mode,
        seed=seed + 1000,  # Different seed for eval
    )
    
    # Create samplers for DDP
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=seed
    ) if world_size > 1 else None
    
    eval_sampler = DistributedSampler(
        eval_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        seed=seed
    ) if world_size > 1 else None
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=universal_collate_fn,
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        sampler=eval_sampler,
        shuffle=False,
        num_workers=min(2, num_workers),
        pin_memory=pin_memory,
        collate_fn=universal_collate_fn,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )
    
    if rank == 0:
        logger.info("DDP Dataloaders created successfully")
        logger.info(f"  Train dataset: {len(train_dataset)} samples")
        logger.info(f"  Eval dataset: {len(eval_dataset)} samples")
        logger.info(f"  Train batches per rank: {len(train_dataloader)}")
        logger.info(f"  Eval batches per rank: {len(eval_dataloader)}")
        logger.info(f"  Batch size per rank: {batch_size}")
        logger.info(f"  Total effective batch size: {batch_size * world_size}")
    
    return train_dataloader, eval_dataloader


# Backward compatibility
def create_universal_dataloaders(*args, **kwargs):
    """Backward compatibility wrapper"""
    return create_ddp_dataloaders(*args, **kwargs)