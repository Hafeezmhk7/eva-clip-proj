#!/usr/bin/env python3
"""
Multi-GPU BLIP3-o Training Script with DDP Support and Memory Optimization
Fixes OOM issues and adds distributed training for scaling to multiple GPUs

Usage:
  # Single GPU (fallback)
  python train_eva_repro_ddp.py --task_mode clip_denoising --chunked_embeddings_dir /path --output_dir ./checkpoints

  # Multi-GPU with torchrun
  torchrun --nproc_per_node=4 train_eva_repro_ddp.py --task_mode clip_denoising --chunked_embeddings_dir /path --output_dir ./checkpoints

  # SLURM multi-node (in job script)
  srun python train_eva_repro_ddp.py --task_mode clip_denoising --chunked_embeddings_dir /path --output_dir ./checkpoints
"""

import os
import sys
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import json
import logging
from pathlib import Path
from datetime import datetime
import traceback

# Setup paths
sys.path.insert(0, str(Path(__file__).parent))

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
                       help="Batch size PER GPU (will be scaled by number of GPUs)")
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
    
    # Debugging and testing
    parser.add_argument("--overfit_test_size", type=int, default=None,
                       help="Size for overfitting test (None to disable)")
    parser.add_argument("--debug_mode", action="store_true",
                       help="Enable debug mode")
    parser.add_argument("--max_shards", type=int, default=35,
                       help="Maximum number of shards to use")
    
    # WandB integration
    parser.add_argument("--use_wandb", action="store_true",
                       help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="blip3o-ddp-training",
                       help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                       help="WandB run name (auto-generated if not provided)")
    parser.add_argument("--wandb_tags", type=str, nargs="*", default=[],
                       help="WandB tags for the run")
    
    # System
    parser.add_argument("--fp16", action="store_true", default=True,
                       help="Use mixed precision")
    parser.add_argument("--num_workers", type=int, default=2,
                       help="Number of dataloader workers per GPU")
    
    # DDP specific
    parser.add_argument("--find_unused_parameters", action="store_true",
                       help="Find unused parameters in DDP (slower but more robust)")
    
    return parser.parse_args()

def setup_ddp(rank: int, world_size: int):
    """Setup distributed training"""
    # Initialize process group
    if "SLURM_PROCID" in os.environ:
        # SLURM environment
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        local_rank = int(os.environ["SLURM_LOCALID"])
        
        # Master address and port
        master_addr = os.environ.get("SLURM_LAUNCH_NODE_IPADDR", "localhost")
        master_port = os.environ.get("MASTER_PORT", "12355")
        
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = master_port
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(local_rank)
    
    elif "LOCAL_RANK" in os.environ:
        # torchrun environment
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        # Single GPU fallback
        local_rank = 0
        rank = 0
        world_size = 1
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
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
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            rank=rank,
            world_size=world_size
        )
    
    return rank, world_size, local_rank, device

def cleanup_ddp():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()

def setup_wandb(args, config, logger, rank):
    """Setup WandB logging (only on rank 0)"""
    if not args.use_wandb or rank != 0:
        return None
    
    try:
        import wandb
        logger.info("Setting up WandB logging...")
        
        # Generate run name if not provided
        if args.wandb_run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.wandb_run_name = f"{args.task_mode}_{args.model_size}_ddp_{timestamp}"
        
        # Initialize WandB
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=config,
            tags=args.wandb_tags + [args.task_mode, args.model_size, args.training_mode, "ddp"],
            notes=f"Multi-GPU BLIP3-o {args.task_mode} training with DDP",
            dir=args.output_dir
        )
        
        logger.info(f"✅ WandB initialized on rank 0:")
        logger.info(f"  Project: {args.wandb_project}")
        logger.info(f"  Run name: {args.wandb_run_name}")
        logger.info(f"  URL: {run.url}")
        
        return run
        
    except ImportError:
        logger.error("❌ WandB not installed but --use_wandb specified")
        return None
    except Exception as e:
        logger.error(f"❌ Failed to initialize WandB: {e}")
        return None

def print_ddp_banner(args, logger, rank, world_size):
    """Print DDP-specific banner"""
    if rank == 0:
        logger.info("🚀 Multi-GPU BLIP3-o Denoising Training with DDP")
        logger.info("=" * 80)
        logger.info(f"🎯 Task: {args.task_mode.upper()}")
        logger.info(f"🏗️  Model: {args.model_size} Universal DiT")
        logger.info(f"🔢 GPUs: {world_size}")
        logger.info(f"📦 Batch size per GPU: {args.batch_size}")
        logger.info(f"📦 Effective batch size: {args.batch_size * world_size * args.gradient_accumulation_steps}")
        logger.info(f"🔄 Gradient accumulation: {args.gradient_accumulation_steps}")
        logger.info(f"📊 Max shards: {args.max_shards}")
        logger.info(f"💾 Memory optimization: {args.max_shard_cache} shard cache")
        if args.use_wandb:
            logger.info(f"📈 WandB: {args.wandb_project}")
        logger.info("=" * 80)

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
    if torch.cuda.device_count() > 1 and dist.is_initialized():
        model = DDP(
            model, 
            device_ids=[device.index] if device.type == 'cuda' else None,
            find_unused_parameters=args.find_unused_parameters
        )
        if rank == 0:
            logger.info(f"Model wrapped with DDP (find_unused_parameters={args.find_unused_parameters})")
    
    if rank == 0:
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model created with {param_count:,} parameters on {torch.cuda.device_count()} GPUs")
    
    return model

def create_ddp_dataloaders(args, rank, world_size, logger):
    """Create data loaders with DDP support"""
    try:
        from src.modules.datasets.blip3o_eva_dataset_ddp import create_ddp_dataloaders
    except ImportError:
        logger.error("Could not import DDP dataset")
        raise
    
    if rank == 0:
        logger.info(f"Creating DDP dataloaders for {args.task_mode}...")
    
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
    
    if rank == 0:
        logger.info(f"DDP dataloaders created for {args.task_mode}")
    
    return train_dataloader, eval_dataloader

def create_ddp_trainer(model, loss_fn, train_dataloader, eval_dataloader, args, device, rank, world_size, logger, wandb_instance=None):
    """Create DDP-aware trainer"""
    try:
        from src.modules.trainers.blip3o_eva_trainer_ddp import create_ddp_trainer
    except ImportError:
        logger.error("Could not import DDP trainer")
        raise
    
    if rank == 0:
        logger.info("Creating DDP trainer...")
    
    trainer = create_ddp_trainer(
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
        world_size=world_size,
        wandb_instance=wandb_instance
    )
    
    if rank == 0:
        logger.info("DDP trainer created")
    
    return trainer

def main():
    """Main DDP training function"""
    args = parse_arguments()
    
    # Setup DDP
    rank, world_size, local_rank, device = setup_ddp(0, 1)
    
    # Create output directory early
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(args.output_dir, rank)
    
    try:
        # Print banner (only on rank 0)
        print_ddp_banner(args, logger, rank, world_size)
        
        # Adjust batch size and learning rate for multiple GPUs
        effective_batch_size = args.batch_size * world_size * args.gradient_accumulation_steps
        adjusted_lr = args.learning_rate * (effective_batch_size / 8)  # Scale LR with batch size
        
        if rank == 0:
            logger.info(f"📊 Training Configuration:")
            logger.info(f"  Batch size per GPU: {args.batch_size}")
            logger.info(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
            logger.info(f"  Effective batch size: {effective_batch_size}")
            logger.info(f"  Base learning rate: {args.learning_rate}")
            logger.info(f"  Adjusted learning rate: {adjusted_lr}")
            logger.info(f"  World size: {world_size}")
            logger.info(f"  Device: {device}")
        
        # Create model
        model = create_ddp_model(args, device, rank, logger)
        
        # Create loss function
        try:
            from src.modules.losses.blip3o_eva_loss import create_universal_flow_loss
        except ImportError:
            logger.error("Could not import loss function")
            raise
        
        loss_fn = create_universal_flow_loss(
            prediction_type=args.prediction_type,
            loss_weight=1.0,
            sphere_constraint_weight=args.sphere_constraint_weight,
            debug_mode=args.debug_mode
        )
        
        # Create dataloaders
        train_dataloader, eval_dataloader = create_ddp_dataloaders(args, rank, world_size, logger)
        
        # Save configuration (only on rank 0)
        config = None
        wandb_instance = None
        if rank == 0:
            config = {
                'args': vars(args),
                'model_config': model.module.config.to_dict() if hasattr(model, 'module') else model.config.to_dict(),
                'ddp_config': {
                    'world_size': world_size,
                    'effective_batch_size': effective_batch_size,
                    'adjusted_learning_rate': adjusted_lr,
                    'gradient_accumulation_steps': args.gradient_accumulation_steps,
                },
                'timestamp': datetime.now().isoformat(),
                'experiment_type': f'ddp_{args.task_mode}',
            }
            
            # Setup WandB
            wandb_instance = setup_wandb(args, config, logger, rank)
            
            # Save config
            config_path = output_dir / 'ddp_experiment_config.json'
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"Configuration saved to {config_path}")
        
        # Create trainer
        trainer = create_ddp_trainer(
            model, loss_fn, train_dataloader, eval_dataloader, 
            args, device, rank, world_size, logger, wandb_instance
        )
        
        # Synchronize before training
        if dist.is_initialized():
            dist.barrier()
        
        if rank == 0:
            logger.info(f"\n🚀 Starting DDP {args.task_mode} training...")
            logger.info(f"Expected memory usage reduction: ~{world_size}x due to DDP")
            logger.info(f"Effective training speed increase: ~{world_size}x")
        
        # Start training
        summary = trainer.train()
        
        # Synchronize after training
        if dist.is_initialized():
            dist.barrier()
        
        # Final summary (only on rank 0)
        if rank == 0:
            logger.info("\n" + "=" * 80)
            logger.info(f"🎉 DDP {args.task_mode.upper()} TRAINING COMPLETED!")
            logger.info("=" * 80)
            logger.info(f"📊 Final Results:")
            logger.info(f"  GPUs used: {world_size}")
            logger.info(f"  Total steps: {summary.get('total_steps', 0)}")
            logger.info(f"  Best loss: {summary.get('best_loss', float('inf')):.6f}")
            logger.info(f"  Best similarity: {summary.get('best_eval_similarity', 0):.4f}")
            
            # Save final summary
            summary['ddp_info'] = {
                'world_size': world_size,
                'effective_batch_size': effective_batch_size,
                'adjusted_learning_rate': adjusted_lr,
            }
            
            summary_path = output_dir / 'ddp_final_summary.json'
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Final summary saved to {summary_path}")
            
            if wandb_instance:
                logger.info(f"📊 WandB run: {wandb_instance.url}")
                wandb_instance.finish()
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ DDP training failed: {e}")
        traceback.print_exc()
        return 1
    
    except KeyboardInterrupt:
        logger.info("DDP training interrupted by user")
        return 1
    
    finally:
        # Cleanup
        cleanup_ddp()

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)