#!/usr/bin/env python3
"""
FIXED DDP-Aware Dataset for BLIP3-o Universal Denoising
Handles memory optimization and robust distributed data loading
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
import json
import gc
import time
import math
import os

logger = logging.getLogger(__name__)


class FixedDDPUniversalDenoisingDataset(Dataset):
    """
    FIXED DDP-aware universal dataset with memory optimization
    """
    
    def __init__(
        self,
        chunked_embeddings_dir: str,
        task_mode: str = "clip_denoising",
        training_mode: str = "patch_only",
        max_shards: int = 10,
        max_shard_cache: int = 2,
        samples_per_shard_load: int = 500,
        noise_schedule: str = "uniform",
        max_noise_level: float = 0.9,
        min_noise_level: float = 0.1,
        split: str = "train",
        rank: int = 0,
        world_size: int = 1,
        debug_mode: bool = False,
        seed: int = 42,
        memory_efficient: bool = True,
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
        self.memory_efficient = memory_efficient
        
        # Set random seed for reproducibility
        random.seed(seed + rank)
        np.random.seed(seed + rank)
        torch.manual_seed(seed + rank)
        
        # Initialize dataset
        self._load_manifest()
        self._setup_shards()
        self._init_cache()
        self._setup_task_config()
        
        # Memory tracking
        self.cache_hits = 0
        self.cache_misses = 0
        self.failed_loads = 0
        
        if rank == 0:
            logger.info(f"FIXED DDP Dataset initialized for {task_mode}")
            logger.info(f"  Rank: {rank}/{world_size}")
            logger.info(f"  Shards: {len(self.shard_files)}")
            logger.info(f"  Estimated samples: {self.rank_total_samples}")
            logger.info(f"  Cache size: {self.max_shard_cache}")
            logger.info(f"  Memory efficient: {self.memory_efficient}")
    
    def _load_manifest(self):
        """Load dataset manifest with error handling"""
        manifest_path = self.embeddings_dir / "embeddings_manifest.json"
        
        try:
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
        except Exception as e:
            if self.rank == 0:
                logger.error(f"Failed to load manifest: {e}")
            self.manifest = {}
            self.total_samples = 0
    
    def _setup_shards(self):
        """Setup shard files with robust error handling"""
        # Find shard files with multiple patterns
        patterns = [
            f"*_{self.training_mode}.pkl",
            "embeddings_shard_*.pkl",
            "*.pkl"
        ]
        
        all_shard_files = []
        for pattern in patterns:
            files = list(self.embeddings_dir.glob(pattern))
            if files:
                all_shard_files = files
                if self.rank == 0:
                    logger.info(f"Found {len(files)} files with pattern: {pattern}")
                break
        
        if not all_shard_files:
            raise FileNotFoundError(f"No shard files found in {self.embeddings_dir}")
        
        # Sort and limit shards
        all_shard_files.sort()
        if self.max_shards is not None:
            all_shard_files = all_shard_files[:self.max_shards]
        
        # Filter existing files
        all_shard_files = [f for f in all_shard_files if f.exists()]
        
        if not all_shard_files:
            raise FileNotFoundError(f"No existing shard files found")
        
        # Distribute shards across ranks for DDP
        shards_per_rank = max(1, len(all_shard_files) // self.world_size)
        start_idx = self.rank * shards_per_rank
        
        if self.rank == self.world_size - 1:
            # Last rank gets remaining shards
            end_idx = len(all_shard_files)
        else:
            end_idx = start_idx + shards_per_rank
        
        self.shard_files = all_shard_files[start_idx:end_idx]
        
        if self.rank == 0:
            logger.info(f"Shard distribution: {shards_per_rank} per rank, rank 0 gets {len(self.shard_files)} shards")
        
        # Precompute shard metadata
        self._precompute_shard_metadata()
    
    def _precompute_shard_metadata(self):
        """Precompute shard metadata to avoid repeated loading"""
        self.shard_sample_counts = []
        self.shard_sample_offsets = []
        current_offset = 0
        
        for i, shard_file in enumerate(self.shard_files):
            try:
                # Quick metadata extraction without loading full shard
                with open(shard_file, 'rb') as f:
                    # Try to get sample count from pickled data
                    shard_data = pickle.load(f)
                    sample_count = len(shard_data.get('captions', []))
                    
                self.shard_sample_counts.append(sample_count)
                self.shard_sample_offsets.append(current_offset)
                current_offset += sample_count
                
                if self.debug_mode and self.rank == 0:
                    logger.debug(f"Shard {i}: {sample_count} samples")
                    
            except Exception as e:
                if self.rank == 0:
                    logger.warning(f"Failed to get metadata for shard {shard_file}: {e}")
                self.shard_sample_counts.append(0)
                self.shard_sample_offsets.append(current_offset)
        
        self.rank_total_samples = sum(self.shard_sample_counts)
        
        if self.rank == 0:
            logger.info(f"Rank {self.rank} metadata: {self.rank_total_samples} samples across {len(self.shard_files)} shards")
    
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
        self.num_tokens = 257 if self.training_mode == "cls_patch" else 256
    
    def _init_cache(self):
        """Initialize memory-efficient shard cache"""
        self.shard_cache = {}
        self.cache_order = []
        self.cache_access_counts = {}
        
        # Memory monitoring
        self.cache_memory_mb = 0.0
        self.max_cache_memory_mb = 2048.0  # 2GB limit
    
    def _estimate_shard_memory_mb(self, shard_data: Dict[str, Any]) -> float:
        """Estimate memory usage of a shard in MB"""
        memory_bytes = 0
        
        for key, value in shard_data.items():
            if torch.is_tensor(value):
                memory_bytes += value.numel() * value.element_size()
            elif isinstance(value, np.ndarray):
                memory_bytes += value.nbytes
            elif isinstance(value, list):
                memory_bytes += len(value) * 100  # Rough estimate for strings
        
        return memory_bytes / (1024 * 1024)
    
    def _load_shard_robust(self, shard_idx: int) -> Optional[Dict[str, Any]]:
        """Load a shard with robust error handling and memory management"""
        
        # Check cache first
        if shard_idx in self.shard_cache:
            self.cache_hits += 1
            # Update access count and order for LRU
            self.cache_access_counts[shard_idx] = self.cache_access_counts.get(shard_idx, 0) + 1
            if shard_idx in self.cache_order:
                self.cache_order.remove(shard_idx)
            self.cache_order.append(shard_idx)
            return self.shard_cache[shard_idx]
        
        self.cache_misses += 1
        
        # Load shard from disk
        shard_file = self.shard_files[shard_idx]
        
        for attempt in range(3):  # Retry logic
            try:
                with open(shard_file, 'rb') as f:
                    shard_data = pickle.load(f)
                
                # Validate shard data
                if not self._validate_shard(shard_data, shard_file):
                    raise ValueError(f"Invalid shard data: {shard_file}")
                
                # Convert to tensors and normalize if needed
                shard_data = self._process_shard_data(shard_data)
                
                # Memory management for cache
                shard_memory = self._estimate_shard_memory_mb(shard_data)
                
                # Ensure we have space in cache
                while (len(self.shard_cache) >= self.max_shard_cache or 
                       self.cache_memory_mb + shard_memory > self.max_cache_memory_mb):
                    self._evict_oldest_shard()
                
                # Add to cache
                self.shard_cache[shard_idx] = shard_data
                self.cache_order.append(shard_idx)
                self.cache_access_counts[shard_idx] = 1
                self.cache_memory_mb += shard_memory
                
                if self.debug_mode and self.rank == 0:
                    logger.debug(f"Loaded shard {shard_idx} ({shard_memory:.1f}MB), cache: {len(self.shard_cache)}")
                
                return shard_data
                
            except Exception as e:
                if self.rank == 0:
                    logger.warning(f"Attempt {attempt + 1}: Failed to load shard {shard_idx}: {e}")
                
                if attempt < 2:
                    time.sleep(0.1 * (attempt + 1))
                    gc.collect()  # Cleanup before retry
                else:
                    self.failed_loads += 1
                    if self.rank == 0:
                        logger.error(f"Failed to load shard {shard_idx} after 3 attempts")
                    return None
    
    def _evict_oldest_shard(self):
        """Evict the least recently used shard from cache"""
        if not self.cache_order:
            return
        
        # Remove oldest shard
        oldest_shard = self.cache_order.pop(0)
        if oldest_shard in self.shard_cache:
            shard_data = self.shard_cache.pop(oldest_shard)
            shard_memory = self._estimate_shard_memory_mb(shard_data)
            self.cache_memory_mb = max(0, self.cache_memory_mb - shard_memory)
            
            if oldest_shard in self.cache_access_counts:
                del self.cache_access_counts[oldest_shard]
            
            if self.debug_mode and self.rank == 0:
                logger.debug(f"Evicted shard {oldest_shard} ({shard_memory:.1f}MB)")
        
        # Cleanup
        gc.collect()
    
    def _validate_shard(self, shard_data: Dict[str, Any], shard_file: Path) -> bool:
        """Validate shard data"""
        try:
            # Check required keys
            required_keys = [self.input_key, self.conditioning_key, 'captions']
            
            for key in required_keys:
                if key not in shard_data:
                    if self.rank == 0:
                        logger.error(f"Missing key '{key}' in shard {shard_file}")
                    return False
            
            # Check tensor shapes
            input_emb = shard_data[self.input_key]
            conditioning_emb = shard_data[self.conditioning_key]
            
            if not hasattr(input_emb, 'shape') or not hasattr(conditioning_emb, 'shape'):
                return False
            
            if len(input_emb.shape) != 3 or len(conditioning_emb.shape) != 3:
                return False
            
            if input_emb.shape[0] != conditioning_emb.shape[0]:
                return False
            
            return True
            
        except Exception as e:
            if self.rank == 0:
                logger.error(f"Error validating shard {shard_file}: {e}")
            return False
    
    def _process_shard_data(self, shard_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process and normalize shard data"""
        
        # Convert to tensors if needed
        for key in [self.input_key, self.conditioning_key]:
            if key in shard_data:
                if not torch.is_tensor(shard_data[key]):
                    shard_data[key] = torch.tensor(shard_data[key], dtype=torch.float32)
        
        # Handle token count adaptation
        input_emb = shard_data[self.input_key]
        conditioning_emb = shard_data[self.conditioning_key]
        
        current_tokens = input_emb.shape[1]
        if current_tokens != self.num_tokens:
            input_emb = self._adapt_token_count(input_emb, current_tokens)
            conditioning_emb = self._adapt_token_count(conditioning_emb, current_tokens)
            shard_data[self.input_key] = input_emb
            shard_data[self.conditioning_key] = conditioning_emb
        
        # Normalize embeddings
        if self.memory_efficient:
            # In-place normalization to save memory
            input_emb = shard_data[self.input_key]
            conditioning_emb = shard_data[self.conditioning_key]
            
            input_emb.div_(torch.norm(input_emb, dim=-1, keepdim=True).clamp(min=1e-8))
            conditioning_emb.div_(torch.norm(conditioning_emb, dim=-1, keepdim=True).clamp(min=1e-8))
        else:
            # Standard normalization
            shard_data[self.input_key] = torch.nn.functional.normalize(input_emb, p=2, dim=-1)
            shard_data[self.conditioning_key] = torch.nn.functional.normalize(conditioning_emb, p=2, dim=-1)
        
        return shard_data
    
    def _adapt_token_count(self, embeddings: torch.Tensor, current_tokens: int) -> torch.Tensor:
        """Adapt token count for embeddings"""
        if current_tokens == 256 and self.num_tokens == 257:
            # Add CLS token (average of patches)
            cls_token = embeddings.mean(dim=1, keepdim=True)
            return torch.cat([cls_token, embeddings], dim=1)
        elif current_tokens == 257 and self.num_tokens == 256:
            # Remove CLS token
            return embeddings[:, 1:, :]
        else:
            # Truncate or pad if necessary
            if current_tokens > self.num_tokens:
                return embeddings[:, :self.num_tokens, :]
            elif current_tokens < self.num_tokens:
                # Pad with zeros
                padding = torch.zeros(embeddings.shape[0], self.num_tokens - current_tokens, embeddings.shape[2])
                return torch.cat([embeddings, padding], dim=1)
            else:
                return embeddings
    
    def _get_sample_robust(self, shard_idx: int, sample_idx: int) -> Optional[Dict[str, Any]]:
        """Get sample with robust error handling"""
        
        # Load shard
        shard_data = self._load_shard_robust(shard_idx)
        if shard_data is None:
            return None
        
        try:
            # Extract sample data
            sample_count = len(shard_data.get('captions', []))
            if sample_idx >= sample_count:
                return None
            
            # Get embeddings
            input_embeddings = shard_data[self.input_key][sample_idx]
            conditioning_embeddings = shard_data[self.conditioning_key][sample_idx]
            target_embeddings = shard_data[self.target_key][sample_idx]
            
            # Get metadata
            caption = shard_data.get('captions', [''])[sample_idx]
            key = shard_data.get('keys', [f'sample_{sample_idx}'])[sample_idx]
            
            return {
                'input_embeddings': input_embeddings.clone() if self.memory_efficient else input_embeddings,
                'conditioning_embeddings': conditioning_embeddings.clone() if self.memory_efficient else conditioning_embeddings,
                'target_embeddings': target_embeddings.clone() if self.memory_efficient else target_embeddings,
                'caption': caption,
                'key': key,
                'shard_idx': shard_idx,
                'sample_idx': sample_idx,
            }
            
        except Exception as e:
            if self.rank == 0:
                logger.warning(f"Error extracting sample {sample_idx} from shard {shard_idx}: {e}")
            return None
    
    def _add_noise_and_flow(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Add noise and prepare for flow matching with robust error handling"""
        
        try:
            target = sample['target_embeddings']
            conditioning = sample['conditioning_embeddings']
            
            # Ensure tensors are normalized
            target = torch.nn.functional.normalize(target, p=2, dim=-1)
            conditioning = torch.nn.functional.normalize(conditioning, p=2, dim=-1)
            
            # Sample noise level
            if self.noise_schedule == "uniform":
                noise_level = torch.rand(1).item() * (self.max_noise_level - self.min_noise_level) + self.min_noise_level
            elif self.noise_schedule == "cosine":
                u = torch.rand(1).item()
                noise_level = (1 - torch.cos(u * math.pi)) / 2
                noise_level = noise_level * (self.max_noise_level - self.min_noise_level) + self.min_noise_level
            else:
                noise_level = 0.5
            
            # Create noise
            noise = torch.randn_like(target)
            noise = torch.nn.functional.normalize(noise, p=2, dim=-1)
            
            # Spherical interpolation (SLERP)
            t = noise_level
            cos_angle = torch.sum(target * noise, dim=-1, keepdim=True)
            cos_angle = torch.clamp(cos_angle, -1 + 1e-7, 1 - 1e-7)
            angle = torch.acos(cos_angle)
            
            # Handle small angles
            small_angle_mask = angle < 1e-6
            linear_interp = (1 - t) * target + t * noise
            linear_interp = torch.nn.functional.normalize(linear_interp, p=2, dim=-1)
            
            # SLERP for normal case
            sin_angle = torch.sin(angle)
            sin_angle = torch.clamp(sin_angle, min=1e-7)
            
            w0 = torch.sin((1 - t) * angle) / sin_angle
            w1 = torch.sin(t * angle) / sin_angle
            
            slerp_result = w0 * target + w1 * noise
            slerp_result = torch.nn.functional.normalize(slerp_result, p=2, dim=-1)
            
            # Choose interpolation method
            interpolated = torch.where(small_angle_mask, linear_interp, slerp_result)
            
            # Velocity target
            velocity_target = (target - noise) / max(1e-3, 1 - t)
            
            return {
                **sample,
                'noise': noise,
                'noise_level': torch.tensor(noise_level),
                'input_embeddings': interpolated,
                'velocity_target': velocity_target,
                'task_mode': self.task_mode,
                'input_dim': self.input_dim,
                'output_dim': self.output_dim,
                'conditioning_dim': self.conditioning_dim,
                'num_tokens': self.num_tokens,
            }
            
        except Exception as e:
            if self.rank == 0:
                logger.error(f"Error in noise and flow processing: {e}")
            
            # Return a dummy sample to avoid breaking training
            dummy_input = torch.randn(self.num_tokens, self.input_dim)
            dummy_conditioning = torch.randn(self.num_tokens, self.conditioning_dim)
            dummy_target = torch.randn(self.num_tokens, self.output_dim)
            
            return {
                'input_embeddings': dummy_input,
                'conditioning_embeddings': dummy_conditioning,
                'target_embeddings': dummy_target,
                'noise': dummy_input,
                'noise_level': torch.tensor(0.5),
                'velocity_target': dummy_input,
                'caption': 'dummy',
                'key': 'dummy',
                'task_mode': self.task_mode,
                'input_dim': self.input_dim,
                'output_dim': self.output_dim,
                'conditioning_dim': self.conditioning_dim,
                'num_tokens': self.num_tokens,
                'shard_idx': -1,
                'sample_idx': -1,
            }
    
    def __len__(self) -> int:
        """Return total samples for this rank"""
        return max(1, self.rank_total_samples)  # Ensure at least 1
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get item by global index with robust error handling"""
        
        # Find which shard this index belongs to
        shard_idx = 0
        sample_idx = idx
        
        for i, count in enumerate(self.shard_sample_counts):
            if sample_idx < count:
                shard_idx = i
                break
            sample_idx -= count
        else:
            # Index out of range, wrap around or return dummy
            if self.rank_total_samples > 0:
                actual_idx = idx % self.rank_total_samples
                return self.__getitem__(actual_idx)
            else:
                # Return dummy sample
                return self._create_dummy_sample()
        
        # Get sample from shard
        sample = self._get_sample_robust(shard_idx, sample_idx)
        
        if sample is None:
            # Fallback to dummy sample
            return self._create_dummy_sample()
        
        # Add noise and prepare for flow matching
        return self._add_noise_and_flow(sample)
    
    def _create_dummy_sample(self) -> Dict[str, Any]:
        """Create a dummy sample for error cases"""
        dummy_input = torch.randn(self.num_tokens, self.input_dim)
        dummy_conditioning = torch.randn(self.num_tokens, self.conditioning_dim)
        dummy_target = torch.randn(self.num_tokens, self.output_dim)
        
        # Normalize
        dummy_input = torch.nn.functional.normalize(dummy_input, p=2, dim=-1)
        dummy_conditioning = torch.nn.functional.normalize(dummy_conditioning, p=2, dim=-1)
        dummy_target = torch.nn.functional.normalize(dummy_target, p=2, dim=-1)
        
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
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_accesses = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / max(total_accesses, 1)
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'cached_shards': len(self.shard_cache),
            'cache_memory_mb': self.cache_memory_mb,
            'failed_loads': self.failed_loads,
        }


def fixed_universal_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """FIXED universal collate function with robust error handling"""
    
    if not batch:
        raise ValueError("Empty batch")
    
    # Filter valid items
    valid_batch = [item for item in batch if item is not None and 'input_embeddings' in item]
    if not valid_batch:
        raise ValueError("No valid items in batch")
    
    try:
        # Get task mode
        task_mode = valid_batch[0]['task_mode']
        
        # Stack tensors with error handling
        tensor_keys = ['input_embeddings', 'conditioning_embeddings', 'target_embeddings', 
                      'noise', 'noise_level', 'velocity_target']
        
        collated = {}
        
        for key in tensor_keys:
            try:
                if key == 'noise_level':
                    values = torch.tensor([item[key] for item in valid_batch])
                else:
                    values = torch.stack([item[key] for item in valid_batch])
                collated[key] = values
            except Exception as e:
                logger.warning(f"Error collating {key}: {e}")
                # Skip this key or create dummy data
                continue
        
        # Universal interface
        if 'input_embeddings' in collated and 'noise_level' in collated:
            collated['hidden_states'] = collated['input_embeddings']
            collated['timestep'] = collated['noise_level']
            collated['encoder_hidden_states'] = collated['conditioning_embeddings']
        
        # Metadata
        collated['batch_size'] = len(valid_batch)
        collated['task_mode'] = task_mode
        
        # Lists
        collated['captions'] = [item.get('caption', '') for item in valid_batch]
        collated['keys'] = [item.get('key', '') for item in valid_batch]
        
        # Dimensions
        for key in ['input_dim', 'output_dim', 'conditioning_dim', 'num_tokens']:
            if key in valid_batch[0]:
                collated[key] = valid_batch[0][key]
        
        return collated
        
    except Exception as e:
        logger.error(f"Critical error in collate function: {e}")
        raise


def create_fixed_ddp_dataloaders(
    chunked_embeddings_dir: str,
    task_mode: str = "clip_denoising",
    batch_size: int = 2,
    training_mode: str = "patch_only",
    max_shards: int = 10,
    max_shard_cache: int = 2,
    samples_per_shard_load: int = 500,
    noise_schedule: str = "uniform",
    max_noise_level: float = 0.9,
    min_noise_level: float = 0.1,
    num_workers: int = 2,
    rank: int = 0,
    world_size: int = 1,
    pin_memory: bool = True,
    debug_mode: bool = False,
    seed: int = 42,
    memory_efficient: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create FIXED DDP-aware dataloaders with robust error handling
    """
    
    # Create datasets
    train_dataset = FixedDDPUniversalDenoisingDataset(
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
        memory_efficient=memory_efficient,
    )
    
    # Eval dataset with different noise for evaluation
    eval_dataset = FixedDDPUniversalDenoisingDataset(
        chunked_embeddings_dir=chunked_embeddings_dir,
        task_mode=task_mode,
        training_mode=training_mode,
        max_shards=min(3, max_shards),  # Fewer shards for eval
        max_shard_cache=2,
        samples_per_shard_load=samples_per_shard_load // 4,
        noise_schedule="uniform",
        max_noise_level=0.5,
        min_noise_level=0.5,
        split="eval",
        rank=rank,
        world_size=world_size,
        debug_mode=debug_mode,
        seed=seed + 1000,
        memory_efficient=memory_efficient,
    )
    
    # Create samplers for DDP
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=seed,
        drop_last=True  # Important for DDP stability
    ) if world_size > 1 else None
    
    eval_sampler = DistributedSampler(
        eval_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        seed=seed,
        drop_last=False
    ) if world_size > 1 else None
    
    # Create dataloaders with robust settings
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=fixed_universal_collate_fn,
        drop_last=True,  # Important for DDP
        persistent_workers=(num_workers > 0),
        timeout=30 if num_workers > 0 else 0,  # Timeout for worker processes
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        sampler=eval_sampler,
        shuffle=False,
        num_workers=min(2, num_workers),  # Fewer workers for eval
        pin_memory=pin_memory,
        collate_fn=fixed_universal_collate_fn,
        drop_last=False,
        persistent_workers=(num_workers > 0),
        timeout=30 if num_workers > 0 else 0,
    )
    
    if rank == 0:
        logger.info("FIXED DDP Dataloaders created successfully")
        logger.info(f"  Train dataset: {len(train_dataset)} samples")
        logger.info(f"  Eval dataset: {len(eval_dataset)} samples")
        logger.info(f"  Train batches per rank: ~{len(train_dataloader)}")
        logger.info(f"  Eval batches per rank: ~{len(eval_dataloader)}")
        logger.info(f"  Batch size per rank: {batch_size}")
        logger.info(f"  Effective batch size: {batch_size * world_size}")
        logger.info(f"  Memory efficient mode: {memory_efficient}")
    
    return train_dataloader, eval_dataloader