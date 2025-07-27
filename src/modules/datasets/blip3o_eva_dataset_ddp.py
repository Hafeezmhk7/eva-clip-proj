#!/usr/bin/env python3
"""
Memory-Optimized BLIP3-o Dataset with DDP Support
Fixes OOM issues by streaming data and limiting memory usage

Key improvements:
1. Streaming data loading instead of loading entire shards
2. LRU cache for shard management
3. DDP-aware data distribution
4. Memory usage monitoring and cleanup
5. Gradient accumulation support
"""

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
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
import psutil
from collections import OrderedDict
import threading

logger = logging.getLogger(__name__)


class MemoryOptimizedShardCache:
    """LRU cache for shard data with memory monitoring"""
    
    def __init__(self, max_cache_size: int = 3, max_memory_gb: float = 16.0):
        self.max_cache_size = max_cache_size
        self.max_memory_gb = max_memory_gb
        self.cache = OrderedDict()
        self.cache_lock = threading.Lock()
        self.memory_threshold = max_memory_gb * 0.8  # Use 80% of limit
        
    def get_memory_usage_gb(self) -> float:
        """Get current process memory usage in GB"""
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
    
    def _evict_oldest(self):
        """Evict oldest cached shard"""
        if self.cache:
            oldest_key = next(iter(self.cache))
            evicted_data = self.cache.pop(oldest_key)
            del evicted_data
            self._cleanup_memory()
            logger.debug(f"Evicted shard {oldest_key} from cache")
    
    def _check_memory_pressure(self) -> bool:
        """Check if we're under memory pressure"""
        current_memory = self.get_memory_usage_gb()
        return current_memory > self.memory_threshold
    
    def put(self, key: str, data: Dict[str, Any]):
        """Add data to cache with memory management"""
        with self.cache_lock:
            # Remove if already exists
            if key in self.cache:
                del self.cache[key]
            
            # Evict based on memory pressure
            while (len(self.cache) >= self.max_cache_size or 
                   self._check_memory_pressure()) and self.cache:
                self._evict_oldest()
            
            # Add new data
            self.cache[key] = data
            logger.debug(f"Cached shard {key} (cache size: {len(self.cache)})")
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get data from cache (LRU)"""
        with self.cache_lock:
            if key in self.cache:
                # Move to end (most recently used)
                data = self.cache.pop(key)
                self.cache[key] = data
                return data
            return None
    
    def clear(self):
        """Clear all cached data"""
        with self.cache_lock:
            self.cache.clear()
            self._cleanup_memory()
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'cache_size': len(self.cache),
            'max_cache_size': self.max_cache_size,
            'memory_usage_gb': self.get_memory_usage_gb(),
            'memory_threshold_gb': self.memory_threshold,
            'cached_shards': list(self.cache.keys())
        }


class DDPDenoisingDataset(IterableDataset):
    """
    Memory-optimized dataset for DDP training with streaming support
    """
    
    def __init__(
        self,
        chunked_embeddings_dir: Union[str, Path],
        task_mode: str = "clip_denoising",
        split: str = "train",
        training_mode: str = "patch_only",
        max_shards: Optional[int] = None,
        max_shard_cache: int = 3,
        samples_per_shard_load: int = 1000,
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
        # DDP parameters
        rank: int = 0,
        world_size: int = 1,
    ):
        super().__init__()
        
        self.chunked_embeddings_dir = Path(chunked_embeddings_dir)
        self.task_mode = task_mode
        self.split = split
        self.training_mode = training_mode
        self.max_shards = max_shards
        self.max_shard_cache = max_shard_cache
        self.samples_per_shard_load = samples_per_shard_load
        self.shuffle_shards = shuffle_shards
        self.shuffle_within_shard = shuffle_within_shard
        self.skip_corrupted = skip_corrupted
        self.validate_shapes = validate_shapes
        self.max_retries = max_retries
        
        # DDP configuration
        self.rank = rank
        self.world_size = world_size
        
        # Spherical noise parameters
        self.noise_schedule = noise_schedule
        self.max_noise_level = max_noise_level
        self.min_noise_level = min_noise_level
        
        # Determine expected tokens
        if expected_tokens is None:
            self.expected_tokens = 257 if training_mode == "cls_patch" else 256
        else:
            self.expected_tokens = expected_tokens
        
        # Setup memory-optimized shard cache
        self.shard_cache = MemoryOptimizedShardCache(
            max_cache_size=max_shard_cache,
            max_memory_gb=16.0  # Conservative limit
        )
        
        # Setup random state with rank-specific seed
        base_seed = 42
        self.rng = random.Random(base_seed + rank)
        
        # Validate task mode
        if task_mode not in ["eva_denoising", "clip_denoising"]:
            raise ValueError(f"task_mode must be 'eva_denoising' or 'clip_denoising', got {task_mode}")
        
        # Load manifest and prepare shards
        self._load_manifest()
        self._prepare_shard_list()
        
        # Current state
        self.current_shard_idx = 0
        self.current_samples = []
        self.current_sample_idx = 0
        self.total_samples_processed = 0
        
        # Log task configuration
        if rank == 0:
            self._log_initialization()

    def _log_initialization(self):
        """Log initialization info (rank 0 only)"""
        if self.task_mode == "eva_denoising":
            logger.info(f"DDP EVA Denoising Dataset initialized:")
            logger.info(f"  INPUT: Noisy EVA embeddings [B, N, 4096]")
            logger.info(f"  CONDITIONING: Clean EVA embeddings [B, N, 4096]")
            logger.info(f"  TARGET: Clean EVA embeddings [B, N, 4096]")
        elif self.task_mode == "clip_denoising":
            logger.info(f"DDP CLIP Denoising Dataset initialized:")
            logger.info(f"  INPUT: Noisy CLIP embeddings [B, N, 1024]")
            logger.info(f"  CONDITIONING: Clean EVA embeddings [B, N, 4096]")
            logger.info(f"  TARGET: Clean CLIP embeddings [B, N, 1024]")
        
        logger.info(f"  Directory: {self.chunked_embeddings_dir}")
        logger.info(f"  Mode: {self.training_mode} ({self.expected_tokens} tokens)")
        logger.info(f"  Noise schedule: {self.noise_schedule}")
        logger.info(f"  Noise range: [{self.min_noise_level}, {self.max_noise_level}]")
        logger.info(f"  Max shards: {self.max_shards}")
        logger.info(f"  Shard cache: {self.max_shard_cache}")
        logger.info(f"  Samples per load: {self.samples_per_shard_load}")
        logger.info(f"  DDP: rank {self.rank}/{self.world_size}")

    def _load_manifest(self):
        """Load embeddings manifest"""
        manifest_path = self.chunked_embeddings_dir / "embeddings_manifest.json"
        
        try:
            if manifest_path.exists():
                with open(manifest_path, 'r') as f:
                    self.manifest = json.load(f)
                if self.rank == 0:
                    logger.info(f"Loaded manifest: {self.manifest.get('total_shards', 0)} shards, {self.manifest.get('total_samples', 0):,} samples")
            else:
                self.manifest = {"total_shards": 0, "total_samples": 0}
                if self.rank == 0:
                    logger.warning(f"No manifest found at {manifest_path}")
        except Exception as e:
            if self.rank == 0:
                logger.warning(f"Failed to load manifest: {e}")
            self.manifest = {"total_shards": 0, "total_samples": 0}

    def _prepare_shard_list(self):
        """Prepare list of shard files with DDP distribution"""
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
                if self.rank == 0:
                    logger.info(f"Found {len(shard_files)} files with pattern: {pattern}")
                break
        
        if not shard_files:
            raise FileNotFoundError(f"No shard files found in {self.chunked_embeddings_dir}")
        
        # Sort files numerically
        shard_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x.stem))) if any(c.isdigit() for c in x.stem) else 0)
        
        # Apply max shards limit
        if self.max_shards is not None:
            shard_files = shard_files[:self.max_shards]
        
        # Filter existing files
        shard_files = [f for f in shard_files if f.exists()]
        
        # DDP: Distribute shards among ranks
        if self.world_size > 1:
            # Each rank gets a subset of shards
            shards_per_rank = len(shard_files) // self.world_size
            remainder = len(shard_files) % self.world_size
            
            start_idx = self.rank * shards_per_rank + min(self.rank, remainder)
            end_idx = start_idx + shards_per_rank + (1 if self.rank < remainder else 0)
            
            self.shard_files = shard_files[start_idx:end_idx]
            
            if self.rank == 0:
                logger.info(f"DDP shard distribution:")
                for r in range(self.world_size):
                    r_start = r * shards_per_rank + min(r, remainder)
                    r_end = r_start + shards_per_rank + (1 if r < remainder else 0)
                    logger.info(f"  Rank {r}: shards {r_start}-{r_end-1} ({r_end-r_start} shards)")
        else:
            self.shard_files = shard_files
        
        if self.shuffle_shards:
            self.rng.shuffle(self.shard_files)
        
        logger.info(f"Rank {self.rank}: Prepared {len(self.shard_files)} shard files")

    def _load_shard_streaming(self, shard_path: Path, start_idx: int = 0, count: int = None) -> Optional[Dict[str, Any]]:
        """Load a portion of a shard for streaming"""
        cache_key = f"{shard_path.name}_{start_idx}_{count}"
        
        # Check cache first
        cached_data = self.shard_cache.get(cache_key)
        if cached_data is not None:
            return cached_data
        
        # Load from disk
        for attempt in range(self.max_retries):
            try:
                with open(shard_path, 'rb') as f:
                    full_shard_data = pickle.load(f)
                
                # Extract subset of data
                if self.task_mode == "eva_denoising":
                    eva_emb = full_shard_data['eva_blip3o_embeddings']
                    total_samples = eva_emb.shape[0]
                elif self.task_mode == "clip_denoising":
                    clip_emb = full_shard_data['clip_blip3o_embeddings']
                    total_samples = clip_emb.shape[0]
                else:
                    raise ValueError(f"Unknown task mode: {self.task_mode}")
                
                # Determine slice
                if count is None:
                    count = min(self.samples_per_shard_load, total_samples - start_idx)
                
                end_idx = min(start_idx + count, total_samples)
                actual_count = end_idx - start_idx
                
                if actual_count <= 0:
                    return None
                
                # Extract subset
                subset_data = {
                    'captions': full_shard_data['captions'][start_idx:end_idx],
                    'keys': full_shard_data.get('keys', [f"key_{i}" for i in range(start_idx, end_idx)])[start_idx:end_idx] if 'keys' in full_shard_data else [f"key_{i}" for i in range(start_idx, end_idx)],
                    'total_samples': actual_count,
                    'shard_idx': start_idx,
                    'source_shard': str(shard_path),
                    'original_total_samples': total_samples,
                }
                
                if self.task_mode == "eva_denoising":
                    subset_data['eva_blip3o_embeddings'] = eva_emb[start_idx:end_idx]
                elif self.task_mode == "clip_denoising":
                    subset_data['clip_blip3o_embeddings'] = clip_emb[start_idx:end_idx]
                    subset_data['eva_blip3o_embeddings'] = full_shard_data['eva_blip3o_embeddings'][start_idx:end_idx]
                
                # Validate and process
                self._validate_and_process_shard(subset_data, shard_path)
                
                # Cache the subset
                self.shard_cache.put(cache_key, subset_data)
                
                # Clean up full shard data
                del full_shard_data
                gc.collect()
                
                return subset_data
                
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
        """Validate and process shard data (same as original but for subset)"""
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
        
        # Convert to tensors and validate
        if self.task_mode == "eva_denoising":
            eva_emb = shard_data['eva_blip3o_embeddings']
            if not torch.is_tensor(eva_emb):
                eva_emb = torch.tensor(eva_emb, dtype=torch.float32)
                shard_data['eva_blip3o_embeddings'] = eva_emb
            
            if self.validate_shapes:
                if eva_emb.dim() != 3:
                    raise ValueError(f"Expected 3D tensor for EVA, got: {eva_emb.shape}")
                if eva_emb.shape[2] != 4096:
                    raise ValueError(f"Expected EVA dim 4096, got: {eva_emb.shape[2]}")
        
        elif self.task_mode == "clip_denoising":
            clip_emb = shard_data['clip_blip3o_embeddings']
            eva_emb = shard_data['eva_blip3o_embeddings']
            
            if not torch.is_tensor(clip_emb):
                clip_emb = torch.tensor(clip_emb, dtype=torch.float32)
                shard_data['clip_blip3o_embeddings'] = clip_emb
            if not torch.is_tensor(eva_emb):
                eva_emb = torch.tensor(eva_emb, dtype=torch.float32)
                shard_data['eva_blip3o_embeddings'] = eva_emb
            
            if self.validate_shapes:
                if clip_emb.dim() != 3:
                    raise ValueError(f"Expected 3D tensor for CLIP, got: {clip_emb.shape}")
                if eva_emb.dim() != 3:
                    raise ValueError(f"Expected 3D tensor for EVA, got: {eva_emb.shape}")
                if clip_emb.shape[2] != 1024:
                    raise ValueError(f"Expected CLIP dim 1024, got: {clip_emb.shape[2]}")
                if eva_emb.shape[2] != 4096:
                    raise ValueError(f"Expected EVA dim 4096, got: {eva_emb.shape[2]}")
        
        # Handle token count adaptation
        if self.task_mode == "eva_denoising":
            eva_emb = shard_data['eva_blip3o_embeddings']
            current_tokens = eva_emb.shape[1]
            if current_tokens != self.expected_tokens:
                shard_data['eva_blip3o_embeddings'] = self._adapt_token_count(eva_emb, current_tokens)
        elif self.task_mode == "clip_denoising":
            clip_emb = shard_data['clip_blip3o_embeddings']
            eva_emb = shard_data['eva_blip3o_embeddings']
            current_tokens = clip_emb.shape[1]
            if current_tokens != self.expected_tokens:
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
            if torch.isnan(eva_emb).any() or torch.isinf(eva_emb).any():
                logger.warning("Found NaN/Inf in EVA embeddings")
                eva_emb = torch.nan_to_num(eva_emb, nan=0.0, posinf=1.0, neginf=-1.0)
            eva_normalized = F.normalize(eva_emb + eps, p=2, dim=-1)
            shard_data['eva_blip3o_embeddings'] = eva_normalized
        
        elif self.task_mode == "clip_denoising":
            clip_emb = shard_data['clip_blip3o_embeddings']
            eva_emb = shard_data['eva_blip3o_embeddings']
            
            if torch.isnan(clip_emb).any() or torch.isinf(clip_emb).any():
                logger.warning("Found NaN/Inf in CLIP embeddings")
                clip_emb = torch.nan_to_num(clip_emb, nan=0.0, posinf=1.0, neginf=-1.0)
            if torch.isnan(eva_emb).any() or torch.isinf(eva_emb).any():
                logger.warning("Found NaN/Inf in EVA embeddings")
                eva_emb = torch.nan_to_num(eva_emb, nan=0.0, posinf=1.0, neginf=-1.0)
            
            clip_normalized = F.normalize(clip_emb + eps, p=2, dim=-1)
            eva_normalized = F.normalize(eva_emb + eps, p=2, dim=-1)
            
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

    def _load_next_samples(self) -> bool:
        """Load next batch of samples from current or next shard"""
        # Clear previous samples
        if self.current_samples:
            del self.current_samples
            gc.collect()
        
        self.current_samples = []
        self.current_sample_idx = 0
        
        # Try to load from current shards
        while self.current_shard_idx < len(self.shard_files):
            shard_path = self.shard_files[self.current_shard_idx]
            
            # Try to load next chunk from current shard
            # We track position within shard using a simple scheme
            shard_position = getattr(self, f'_shard_position_{self.current_shard_idx}', 0)
            
            shard_data = self._load_shard_streaming(
                shard_path, 
                start_idx=shard_position, 
                count=self.samples_per_shard_load
            )
            
            if shard_data is not None:
                num_samples = shard_data['total_samples']
                
                if num_samples > 0:
                    # Process samples from this chunk
                    for i in range(num_samples):
                        try:
                            if self.task_mode == "eva_denoising":
                                clean_eva = shard_data['eva_blip3o_embeddings'][i]
                                caption = shard_data['captions'][i]
                                
                                # Validation
                                if self.validate_shapes and clean_eva.shape != (self.expected_tokens, 4096):
                                    continue
                                
                                if torch.isnan(clean_eva).any():
                                    if self.skip_corrupted:
                                        continue
                                    else:
                                        raise ValueError("NaN detected in EVA embeddings")
                                
                                # Sample noise and create item
                                noise_level = self._sample_noise_level()
                                noisy_eva, noise = self._add_spherical_noise(clean_eva, noise_level)
                                
                                item = {
                                    'input_embeddings': noisy_eva,
                                    'conditioning_embeddings': clean_eva,
                                    'target_embeddings': clean_eva,
                                    'noise': noise,
                                    'noise_level': noise_level,
                                    'caption': caption,
                                    'task_mode': 'eva_denoising',
                                    'key': f"rank{self.rank}_shard{self.current_shard_idx}_pos{shard_position}_sample{i}",
                                    'sample_idx': i,
                                    'training_mode': self.training_mode,
                                    'num_tokens': self.expected_tokens,
                                    'input_dim': 4096,
                                    'output_dim': 4096,
                                    'conditioning_dim': 4096,
                                }
                                
                            elif self.task_mode == "clip_denoising":
                                clean_clip = shard_data['clip_blip3o_embeddings'][i]
                                clean_eva = shard_data['eva_blip3o_embeddings'][i]
                                caption = shard_data['captions'][i]
                                
                                # Validation
                                if self.validate_shapes:
                                    if clean_clip.shape != (self.expected_tokens, 1024):
                                        continue
                                    if clean_eva.shape != (self.expected_tokens, 4096):
                                        continue
                                
                                if torch.isnan(clean_clip).any() or torch.isnan(clean_eva).any():
                                    if self.skip_corrupted:
                                        continue
                                    else:
                                        raise ValueError("NaN detected in embeddings")
                                
                                # Sample noise and create item
                                noise_level = self._sample_noise_level()
                                noisy_clip, noise = self._add_spherical_noise(clean_clip, noise_level)
                                
                                item = {
                                    'input_embeddings': noisy_clip,
                                    'conditioning_embeddings': clean_eva,
                                    'target_embeddings': clean_clip,
                                    'noise': noise,
                                    'noise_level': noise_level,
                                    'caption': caption,
                                    'task_mode': 'clip_denoising',
                                    'key': f"rank{self.rank}_shard{self.current_shard_idx}_pos{shard_position}_sample{i}",
                                    'sample_idx': i,
                                    'training_mode': self.training_mode,
                                    'num_tokens': self.expected_tokens,
                                    'input_dim': 1024,
                                    'output_dim': 1024,
                                    'conditioning_dim': 4096,
                                }
                            
                            self.current_samples.append(item)
                            
                        except Exception as e:
                            if self.skip_corrupted:
                                logger.warning(f"Skipping corrupted sample {i}: {e}")
                                continue
                            else:
                                raise
                    
                    # Update shard position
                    setattr(self, f'_shard_position_{self.current_shard_idx}', shard_position + num_samples)
                    
                    # If we got samples, shuffle and return
                    if self.current_samples:
                        if self.shuffle_within_shard:
                            self.rng.shuffle(self.current_samples)
                        
                        logger.debug(f"Rank {self.rank}: Loaded {len(self.current_samples)} samples from shard {self.current_shard_idx} (pos {shard_position})")
                        return True
                    
                    # If no samples but more data in shard, continue with next chunk
                    if shard_position + num_samples < shard_data.get('original_total_samples', num_samples):
                        continue
                
            # Move to next shard
            self.current_shard_idx += 1
            setattr(self, f'_shard_position_{self.current_shard_idx}', 0)  # Reset position for new shard
        
        # No more data
        return False

    def __len__(self) -> int:
        """Estimate total number of samples for this rank"""
        if hasattr(self, '_estimated_length'):
            return self._estimated_length
        
        # Estimate based on manifest and rank
        total_samples = self.manifest.get('total_samples', 0)
        if total_samples > 0 and self.world_size > 1:
            # Divide by world size for DDP
            estimated_samples = total_samples // self.world_size
        else:
            # Fallback estimate
            num_shards = len(self.shard_files)
            avg_samples_per_shard = 1000
            estimated_samples = num_shards * avg_samples_per_shard
        
        self._estimated_length = estimated_samples
        return estimated_samples

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate through all samples with memory optimization"""
        self.current_shard_idx = 0
        self.current_sample_idx = 0
        self.total_samples_processed = 0
        
        # Clear any existing samples
        if hasattr(self, 'current_samples'):
            del self.current_samples
        self.current_samples = []
        
        # Reset shard positions
        for i in range(len(self.shard_files)):
            setattr(self, f'_shard_position_{i}', 0)
        
        logger.debug(f"Rank {self.rank}: Starting iteration over {len(self.shard_files)} shards for {self.task_mode}")
        
        while self._load_next_samples():
            while self.current_sample_idx < len(self.current_samples):
                item = self.current_samples[self.current_sample_idx]
                self.current_sample_idx += 1
                self.total_samples_processed += 1
                yield item
        
        logger.info(f"Rank {self.rank}: Iteration completed: {self.total_samples_processed} samples processed")
        
        # Final cleanup
        self.shard_cache.clear()


def ddp_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """DDP-aware collate function (same as universal but with DDP awareness)"""
    if not batch:
        raise ValueError("Empty batch")
    
    # Filter valid items
    valid_batch = [item for item in batch if item is not None]
    if not valid_batch:
        raise ValueError("No valid items in batch")
    
    try:
        # Use the same logic as universal collate function
        from src.modules.datasets.blip3o_eva_dataset import universal_collate_fn
        return universal_collate_fn(valid_batch)
        
    except Exception as e:
        logger.error(f"Error in DDP collate function: {e}")
        raise


def create_ddp_dataloaders(
    chunked_embeddings_dir: Union[str, Path],
    task_mode: str = "clip_denoising",
    batch_size: int = 4,
    eval_batch_size: Optional[int] = None,
    training_mode: str = "patch_only",
    max_shards: Optional[int] = None,
    max_shard_cache: int = 3,
    samples_per_shard_load: int = 1000,
    noise_schedule: str = "uniform",
    max_noise_level: float = 0.9,
    min_noise_level: float = 0.1,
    num_workers: int = 2,
    pin_memory: bool = False,
    rank: int = 0,
    world_size: int = 1,
    **kwargs
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Create DDP-aware dataloaders"""
    
    if eval_batch_size is None:
        eval_batch_size = batch_size
    
    # Task info logging (rank 0 only)
    if rank == 0:
        if task_mode == "eva_denoising":
            logger.info(f"Creating DDP EVA denoising dataloaders:")
            logger.info(f"  INPUT: Noisy EVA embeddings [B, N, 4096]")
            logger.info(f"  CONDITIONING: Clean EVA embeddings [B, N, 4096]")
            logger.info(f"  TARGET: Clean EVA embeddings [B, N, 4096]")
        elif task_mode == "clip_denoising":
            logger.info(f"Creating DDP CLIP denoising dataloaders:")
            logger.info(f"  INPUT: Noisy CLIP embeddings [B, N, 1024]")
            logger.info(f"  CONDITIONING: Clean EVA embeddings [B, N, 4096]")
            logger.info(f"  TARGET: Clean CLIP embeddings [B, N, 1024]")
        
        logger.info(f"  Batch size per GPU: {batch_size}")
        logger.info(f"  Effective batch size: {batch_size * world_size}")
        logger.info(f"  Max shards: {max_shards}")
        logger.info(f"  Shard cache: {max_shard_cache}")
        logger.info(f"  DDP: {world_size} processes")
    
    # Create training dataset
    train_dataset = DDPDenoisingDataset(
        chunked_embeddings_dir=chunked_embeddings_dir,
        task_mode=task_mode,
        split="train",
        training_mode=training_mode,
        max_shards=max_shards,
        max_shard_cache=max_shard_cache,
        samples_per_shard_load=samples_per_shard_load,
        shuffle_shards=True,
        shuffle_within_shard=True,
        noise_schedule=noise_schedule,
        max_noise_level=max_noise_level,
        min_noise_level=min_noise_level,
        rank=rank,
        world_size=world_size,
        **kwargs
    )
    
    # Create training dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=ddp_collate_fn,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    
    # Create evaluation dataset (only for rank 0 to avoid duplicated evaluation)
    if rank == 0:
        eval_dataset = DDPDenoisingDataset(
            chunked_embeddings_dir=chunked_embeddings_dir,
            task_mode=task_mode,
            split="eval",
            training_mode=training_mode,
            max_shards=min(5, max_shards) if max_shards else 5,  # Smaller eval set
            max_shard_cache=2,
            samples_per_shard_load=500,
            shuffle_shards=False,
            shuffle_within_shard=False,
            noise_schedule="uniform",
            max_noise_level=0.7,
            min_noise_level=0.3,
            rank=0,
            world_size=1,  # Eval only on rank 0
            **kwargs
        )
        
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=eval_batch_size,
            num_workers=min(num_workers, 1),
            collate_fn=ddp_collate_fn,
            pin_memory=pin_memory,
            drop_last=False,
            persistent_workers=min(num_workers, 1) > 0,
        )
    else:
        eval_dataloader = None
    
    if rank == 0:
        logger.info(f"DDP dataloaders created successfully for {task_mode}")
    
    return train_dataloader, eval_dataloader