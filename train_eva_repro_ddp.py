#!/usr/bin/env python3
"""
Fixed Multi-GPU BLIP3-o Training Script with Proper DDP Support
"""

import os
import sys
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import json
import logging
from pathlib import Path
from datetime import datetime
import traceback
import socket

# Setup paths
sys.path.insert(0, str(Path(__file__).parent))

def find_free_port():
    """Find a free port for DDP communication"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

def setup_logging(output_dir: str, rank: int):
    """Setup logging configuration for DDP"""
    log_file = Path(output_dir) / f'training_rank_{rank}.log'
    
    # Only log to console on rank 0
    handlers = [logging.FileHandler(log_file, mode='w')]
    if rank == 0:
        handlers.append(logging.StreamHandler(sys.stdout))
    
    logging.basicConfig(
        level=logging.INFO,
        format=f'[Rank {rank}] %(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    return logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments for DDP training"""
    parser = argparse.ArgumentParser(description="Multi-GPU BLIP3-o Denoising Training")
    
    # Required arguments
    parser.add_argument("--chunked_embeddings_dir", type=str, required=True,
                       help="Path to chunked embeddings directory")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for checkpoints")
    
    # Task configuration
    parser.add_argument("--task_mode", type=str, default="clip_denoising",
                       choices=["eva_denoising", "clip_denoising"],
                       help="Task mode: eva_denoising or clip_denoising")
    
    # Model configuration
    parser.add_argument("--model_size", type=str, default="base",
                       choices=["tiny", "small", "base", "large"],
                       help="Model size")
    parser.add_argument("--training_mode", type=str, default="patch_only",
                       choices=["patch_only", "cls_patch"],
                       help="Training mode")
    parser.add_argument("--prediction_type", type=str, default="velocity",
                       choices=["velocity", "target", "noise"],
                       help="Flow matching prediction type")
    
    # Training hyperparameters
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size PER GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                       help="Gradient accumulation steps")
    parser.add_argument("--num_epochs", type=int, default=10,
                       help="Number of epochs")
    parser.add_argument("--warmup_steps", type=int, default=100,
                       help="Warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                       help="Max gradient norm")
    
    # Memory optimization
    parser.add_argument("--max_shard_cache", type=int, default=3,
                       help="Maximum number of shards to cache in memory")
    parser.add_argument("--samples_per_shard_load", type=int, default=1000,
                       help="Number of samples to load from each shard at once")
    parser.add_argument("--max_shards", type=int, default=35,
                       help="Maximum number of shards to use")
    
    # Spherical flow matching parameters
    parser.add_argument("--sphere_constraint_weight", type=float, default=0.1,
                       help="Spherical constraint loss weight")
    parser.add_argument("--noise_schedule", type=str, default="uniform",
                       choices=["uniform", "cosine"],
                       help="Noise sampling schedule")
    parser.add_argument("--max_noise_level", type=float, default=0.9,
                       help="Maximum noise level")
    parser.add_argument("--min_noise_level", type=float, default=0.1,
                       help="Minimum noise level")
    
    # Evaluation
    parser.add_argument("--eval_every_n_steps", type=int, default=200,
                       help="Evaluate every N steps")
    parser.add_argument("--eval_num_samples", type=int, default=500,
                       help="Number of samples for evaluation")
    parser.add_argument("--eval_inference_steps", type=int, default=50,
                       help="Number of denoising steps during evaluation")
    
    # System
    parser.add_argument("--fp16", action="store_true", default=True,
                       help="Use mixed precision")
    parser.add_argument("--num_workers", type=int, default=2,
                       help="Number of dataloader workers per GPU")
    
    # Debugging
    parser.add_argument("--debug_mode", action="store_true",
                       help="Enable debug mode")
    parser.add_argument("--overfit_test_size", type=int, default=None,
                       help="Size for overfitting test")
    
    return parser.parse_args()

def setup_ddp():
    """Setup distributed training with proper SLURM handling"""
    # Check if we're in SLURM environment
    if "SLURM_PROCID" in os.environ:
        # SLURM environment
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        local_rank = int(os.environ["SLURM_LOCALID"])
        
        # Get master address from SLURM
        node_list = os.environ.get("SLURM_NODELIST", "localhost")
        master_addr = os.environ.get("MASTER_ADDR", node_list.split('[')[0])
        master_port = os.environ.get("MASTER_PORT", "29500")
        
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = master_port
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(local_rank)
        
        print(f"SLURM DDP Setup - Rank: {rank}/{world_size}, Local rank: {local_rank}")
        print(f"Master: {master_addr}:{master_port}")
        
    elif "LOCAL_RANK" in os.environ:
        # torchrun environment
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        # Single GPU fallback
        print("No distributed environment detected, running on single GPU")
        local_rank = 0
        rank = 0
        world_size = 1
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["LOCAL_RANK"] = "0"
    
    # Set device
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    
    # Initialize process group
    if world_size > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        
        # Set timeout for debugging
        import datetime
        timeout = datetime.timedelta(minutes=30)
        
        try:
            dist.init_process_group(
                backend=backend,
                rank=rank,
                world_size=world_size,
                timeout=timeout
            )
            print(f"Process group initialized successfully on rank {rank}")
        except Exception as e:
            print(f"Failed to initialize process group on rank {rank}: {e}")
            raise
    
    return rank, world_size, local_rank, device

def cleanup_ddp():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()

def create_ddp_model(args, device, rank, logger):
    """Create model and wrap with DDP"""
    try:
        from src.modules.models.blip3o_eva_dit import create_universal_model
    except ImportError:
        logger.error("Could not import universal model")
        raise
    
    if rank == 0:
        logger.info(f"Creating {args.model_size} universal model for {args.task_mode}...")
    
    model = create_universal_model(
        model_size=args.model_size,
        training_mode=args.training_mode,
        task_mode=args.task_mode,
        prediction_type=args.prediction_type
    )
    
    model = model.to(device)
    
    # Wrap with DDP if multi-GPU
    if world_size > 1:
        model = DDP(
            model, 
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False  # Set to True if model has unused parameters
        )
        if rank == 0:
            logger.info(f"Model wrapped with DDP on {world_size} GPUs")
    
    if rank == 0:
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model created with {param_count:,} parameters")
    
    return model

def main():
    """Main DDP training function"""
    args = parse_arguments()
    
    # Setup DDP
    try:
        rank, world_size, local_rank, device = setup_ddp()
    except Exception as e:
        print(f"Failed to setup DDP: {e}")
        return 1
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(args.output_dir, rank)
    
    try:
        if rank == 0:
            logger.info("🚀 Multi-GPU BLIP3-o Denoising Training with DDP")
            logger.info("=" * 80)
            logger.info(f"Task: {args.task_mode}")
            logger.info(f"World size: {world_size}")
            logger.info(f"Device: {device}")
            logger.info(f"Master: {os.environ.get('MASTER_ADDR')}:{os.environ.get('MASTER_PORT')}")
            logger.info("=" * 80)
        
        # Create model
        model = create_ddp_model(args, device, rank, logger)
        
        # Create loss function
        from src.modules.losses.blip3o_eva_loss import create_universal_flow_loss
        loss_fn = create_universal_flow_loss(
            prediction_type=args.prediction_type,
            sphere_constraint_weight=args.sphere_constraint_weight,
            debug_mode=args.debug_mode
        )
        
        # Create dataloaders
        from src.modules.datasets.blip3o_eva_dataset import create_ddp_dataloaders
        train_dataloader, eval_dataloader = create_ddp_dataloaders(
            chunked_embeddings_dir=args.chunked_embeddings_dir,
            task_mode=args.task_mode,
            batch_size=args.batch_size,
            training_mode=args.training_mode,
            max_shards=args.max_shards,
            max_shard_cache=args.max_shard_cache,
            samples_per_shard_load=args.samples_per_shard_load,
            noise_schedule=args.noise_schedule,
            max_noise_level=args.max_noise_level,
            min_noise_level=args.min_noise_level,
            num_workers=args.num_workers,
            rank=rank,
            world_size=world_size,
            pin_memory=torch.cuda.is_available()
        )
        
        # Create trainer
        from src.modules.trainers.blip3o_eva_trainer import DDPDenoisingTrainer
        trainer = DDPDenoisingTrainer(
            model=model,
            loss_fn=loss_fn,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            num_epochs=args.num_epochs,
            warmup_steps=args.warmup_steps,
            max_grad_norm=args.max_grad_norm,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            fp16=args.fp16,
            eval_every_n_steps=args.eval_every_n_steps,
            eval_num_samples=args.eval_num_samples,
            eval_inference_steps=args.eval_inference_steps,
            debug_mode=args.debug_mode,
            overfit_test_size=args.overfit_test_size,
            output_dir=args.output_dir,
            task_mode=args.task_mode,
            device=device,
            rank=rank,
            world_size=world_size
        )
        
        # Synchronize before training
        if dist.is_initialized():
            dist.barrier()
        
        # Start training
        summary = trainer.train()
        
        # Synchronize after training
        if dist.is_initialized():
            dist.barrier()
        
        if rank == 0:
            logger.info("Training completed successfully!")
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        traceback.print_exc()
        return 1
    
    finally:
        cleanup_ddp()

if __name__ == "__main__":
    # Set up environment variables if not already set
    if "OMP_NUM_THREADS" not in os.environ:
        os.environ["OMP_NUM_THREADS"] = "1"
    
    exit_code = main()
    sys.exit(exit_code)