#!/usr/bin/env python3
"""
FIXED Multi-GPU BLIP3-o Training Script with Robust DDP Support
Addresses NCCL communication issues and network interface problems
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
import time
import subprocess
import psutil

# Setup paths
sys.path.insert(0, str(Path(__file__).parent))

def get_network_interfaces():
    """Get available network interfaces"""
    interfaces = []
    try:
        import netifaces
        for interface in netifaces.interfaces():
            if interface != 'lo':  # Skip loopback
                interfaces.append(interface)
    except ImportError:
        # Fallback method
        try:
            result = subprocess.run(['ip', 'route'], capture_output=True, text=True)
            for line in result.stdout.split('\n'):
                if 'default' in line:
                    parts = line.split()
                    if len(parts) > 4:
                        interfaces.append(parts[4])
                        break
        except:
            interfaces = ['eth0', 'ib0']  # Common defaults
    
    return list(set(interfaces))  # Remove duplicates

def find_free_port():
    """Find a free port for DDP communication"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

def setup_nccl_environment():
    """FIXED: Setup NCCL environment with proper network detection"""
    
    # Get available interfaces
    interfaces = get_network_interfaces()
    print(f"Available network interfaces: {interfaces}")
    
    # Priority order for interface selection
    preferred_interfaces = ['ib0', 'eth0', 'enp0s3', 'enp0s8']
    
    selected_interface = None
    for preferred in preferred_interfaces:
        if preferred in interfaces:
            selected_interface = preferred
            break
    
    if not selected_interface and interfaces:
        selected_interface = interfaces[0]
    
    if selected_interface:
        os.environ["NCCL_SOCKET_IFNAME"] = selected_interface
        print(f"Selected network interface: {selected_interface}")
        
        # Configure NCCL based on interface type
        if 'ib' in selected_interface:
            # InfiniBand configuration
            os.environ["NCCL_IB_DISABLE"] = "0"
            os.environ["NCCL_NET_GDR_LEVEL"] = "2"
            os.environ["NCCL_P2P_LEVEL"] = "NVL"
            print("Configured for InfiniBand")
        else:
            # Ethernet configuration
            os.environ["NCCL_IB_DISABLE"] = "1"
            print("Configured for Ethernet")
    else:
        print("Warning: No suitable network interface found")
        os.environ["NCCL_SOCKET_IFNAME"] = "lo"
        os.environ["NCCL_IB_DISABLE"] = "1"
    
    # Common NCCL settings for stability
    nccl_env = {
        "NCCL_DEBUG": "WARN",  # Reduced from INFO to avoid spam
        "NCCL_ASYNC_ERROR_HANDLING": "1",
        "NCCL_TIMEOUT": "600",
        "NCCL_BUFFSIZE": "8388608",
        "NCCL_ALGO": "Ring",
        "NCCL_MIN_NCHANNELS": "2",
        "NCCL_MAX_NCHANNELS": "16",
    }
    
    for key, value in nccl_env.items():
        os.environ[key] = value
    
    print("NCCL environment configured")

def setup_logging(output_dir: str, rank: int):
    """Setup logging configuration for DDP"""
    log_file = Path(output_dir) / f'training_rank_{rank}.log'
    
    # Create formatter
    formatter = logging.Formatter(
        f'[Rank {rank}] %(asctime)s - %(levelname)s - %(message)s'
    )
    
    # File handler (all ranks)
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(formatter)
    
    # Setup logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    
    # Console handler (only rank 0)
    if rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger

def parse_arguments():
    """Parse command line arguments for DDP training"""
    parser = argparse.ArgumentParser(description="FIXED Multi-GPU BLIP3-o Denoising Training")
    
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
    parser.add_argument("--batch_size", type=int, default=2,
                       help="Batch size PER GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                       help="Gradient accumulation steps")
    parser.add_argument("--num_epochs", type=int, default=5,
                       help="Number of epochs")
    parser.add_argument("--warmup_steps", type=int, default=100,
                       help="Warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                       help="Max gradient norm")
    
    # Memory optimization
    parser.add_argument("--max_shard_cache", type=int, default=2,
                       help="Maximum number of shards to cache in memory")
    parser.add_argument("--samples_per_shard_load", type=int, default=500,
                       help="Number of samples to load from each shard at once")
    parser.add_argument("--max_shards", type=int, default=10,
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
    parser.add_argument("--eval_every_n_steps", type=int, default=250,
                       help="Evaluate every N steps")
    parser.add_argument("--eval_num_samples", type=int, default=300,
                       help="Number of samples for evaluation")
    parser.add_argument("--eval_inference_steps", type=int, default=25,
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

def setup_ddp_robust():
    """FIXED: Robust DDP setup with multiple fallback strategies"""
    
    print("🔧 Setting up robust DDP environment...")
    
    # Setup NCCL first
    setup_nccl_environment()
    
    # Strategy 1: SLURM environment (most common on clusters)
    if "SLURM_PROCID" in os.environ:
        print("📊 Detected SLURM environment")
        
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        local_rank = int(os.environ.get("SLURM_LOCALID", "0"))
        
        # Get node list and determine master
        node_list = os.environ.get("SLURM_NODELIST", "")
        if node_list:
            # Parse node list (handle formats like "node[001-003]" or "node001")
            if '[' in node_list:
                master_node = node_list.split('[')[0] + node_list.split('[')[1].split('-')[0].split(',')[0].replace(']', '')
            else:
                master_node = node_list.split(',')[0]
        else:
            master_node = socket.gethostname()
        
        # Use provided master address or detected node
        master_addr = os.environ.get("MASTER_ADDR", master_node)
        master_port = os.environ.get("MASTER_PORT", str(find_free_port()))
        
        # Set environment variables
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = master_port
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(local_rank)
        
        print(f"📍 SLURM setup: Rank {rank}/{world_size}, Local rank {local_rank}")
        print(f"📡 Master: {master_addr}:{master_port}")
        
    # Strategy 2: torchrun environment
    elif "LOCAL_RANK" in os.environ and "RANK" in os.environ:
        print("📊 Detected torchrun environment")
        
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        
        master_addr = os.environ.get("MASTER_ADDR", "localhost")
        master_port = os.environ.get("MASTER_PORT", str(find_free_port()))
        
        print(f"📍 torchrun setup: Rank {rank}/{world_size}, Local rank {local_rank}")
        print(f"📡 Master: {master_addr}:{master_port}")
        
    # Strategy 3: Single GPU fallback
    else:
        print("📊 No distributed environment detected, using single GPU")
        
        local_rank = 0
        rank = 0
        world_size = 1
        
        master_addr = "localhost"
        master_port = str(find_free_port())
        
        # Set environment variables for consistency
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = master_port
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(local_rank)
        
        print(f"📍 Single GPU setup: Rank {rank}/{world_size}")
    
    # Set CUDA device
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if local_rank < device_count:
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
            print(f"🎮 Using GPU {local_rank}: {torch.cuda.get_device_name(local_rank)}")
        else:
            print(f"⚠️ Local rank {local_rank} >= device count {device_count}, using CPU")
            device = torch.device("cpu")
    else:
        print("⚠️ CUDA not available, using CPU")
        device = torch.device("cpu")
    
    return rank, world_size, local_rank, device

def init_process_group_robust(rank, world_size):
    """FIXED: Robust process group initialization with retries"""
    
    if world_size <= 1:
        print("🔧 Single process, skipping process group initialization")
        return True
    
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    
    print(f"🔗 Initializing process group with backend: {backend}")
    print(f"📊 Rank: {rank}, World size: {world_size}")
    print(f"📡 Master: {os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}")
    
    # Multiple initialization attempts with increasing timeouts
    max_retries = 3
    base_timeout = 30  # seconds
    
    for attempt in range(max_retries):
        try:
            timeout = base_timeout * (2 ** attempt)  # Exponential backoff
            print(f"🔄 Attempt {attempt + 1}/{max_retries} with timeout {timeout}s...")
            
            import datetime
            timeout_delta = datetime.timedelta(seconds=timeout)
            
            dist.init_process_group(
                backend=backend,
                rank=rank,
                world_size=world_size,
                timeout=timeout_delta,
                init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"
            )
            
            print(f"✅ Process group initialized successfully on rank {rank}")
            
            # Test communication
            if torch.cuda.is_available():
                test_tensor = torch.tensor([rank], dtype=torch.float32, device=f"cuda:{rank}")
                dist.all_reduce(test_tensor)
                print(f"🧪 Communication test passed on rank {rank}: {test_tensor.item()}")
            
            return True
            
        except Exception as e:
            print(f"❌ Attempt {attempt + 1} failed on rank {rank}: {e}")
            
            if attempt < max_retries - 1:
                wait_time = 5 * (attempt + 1)
                print(f"⏳ Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
                
                # Try a different port for next attempt
                if rank == 0:
                    new_port = str(find_free_port())
                    os.environ["MASTER_PORT"] = new_port
                    print(f"🔄 Trying new port: {new_port}")
            else:
                print(f"💥 All initialization attempts failed on rank {rank}")
                return False
    
    return False

def cleanup_ddp():
    """Cleanup distributed training"""
    if dist.is_initialized():
        try:
            dist.destroy_process_group()
            print("🧹 Process group destroyed successfully")
        except Exception as e:
            print(f"⚠️ Error destroying process group: {e}")

def create_ddp_model(args, device, rank, world_size, local_rank, logger):
    """FIXED: Create model and wrap with DDP using proper settings"""
    
    try:
        from src.modules.models.blip3o_eva_dit import create_universal_model
    except ImportError as e:
        logger.error(f"Could not import universal model: {e}")
        raise
    
    if rank == 0:
        logger.info(f"Creating {args.model_size} universal model for {args.task_mode}...")
    
    # Create model
    model = create_universal_model(
        model_size=args.model_size,
        training_mode=args.training_mode,
        task_mode=args.task_mode,
        prediction_type=args.prediction_type
    )
    
    # Move to device
    model = model.to(device)
    
    if rank == 0:
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model created with {param_count:,} parameters")
    
    # Wrap with DDP if multi-GPU
    if world_size > 1 and torch.cuda.is_available():
        
        # FIXED: Proper DDP configuration
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,  # Critical: All parameters must be used
            broadcast_buffers=True,
            gradient_as_bucket_view=True,  # Memory optimization
            static_graph=False  # Allow dynamic graphs
        )
        
        if rank == 0:
            logger.info(f"Model wrapped with DDP on {world_size} GPUs")
    
    return model

def test_ddp_communication(rank, world_size, device):
    """Test DDP communication before training"""
    if world_size <= 1:
        return True
    
    try:
        print(f"🧪 Testing DDP communication on rank {rank}...")
        
        # Test tensor creation and communication
        if torch.cuda.is_available():
            test_tensor = torch.tensor([float(rank)], device=device)
        else:
            test_tensor = torch.tensor([float(rank)])
        
        # All-reduce test
        dist.all_reduce(test_tensor, op=dist.ReduceOp.SUM)
        
        expected_sum = sum(range(world_size))
        actual_sum = test_tensor.item()
        
        if abs(actual_sum - expected_sum) < 1e-6:
            print(f"✅ Communication test passed on rank {rank}: {actual_sum}")
            return True
        else:
            print(f"❌ Communication test failed on rank {rank}: expected {expected_sum}, got {actual_sum}")
            return False
            
    except Exception as e:
        print(f"❌ Communication test error on rank {rank}: {e}")
        return False

def main():
    """FIXED: Main DDP training function with robust error handling"""
    
    # Set environment optimizations early
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")
    
    args = parse_arguments()
    
    # Setup DDP with robust error handling
    try:
        rank, world_size, local_rank, device = setup_ddp_robust()
    except Exception as e:
        print(f"💥 Failed to setup DDP: {e}")
        traceback.print_exc()
        return 1
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(args.output_dir, rank)
    
    try:
        if rank == 0:
            logger.info("🚀 FIXED Multi-GPU BLIP3-o Denoising Training with Robust DDP")
            logger.info("=" * 80)
            logger.info(f"Task: {args.task_mode}")
            logger.info(f"World size: {world_size}")
            logger.info(f"Device: {device}")
            logger.info(f"Network interface: {os.environ.get('NCCL_SOCKET_IFNAME', 'auto')}")
            logger.info(f"Master: {os.environ.get('MASTER_ADDR')}:{os.environ.get('MASTER_PORT')}")
            logger.info("=" * 80)
        
        # Initialize process group
        if not init_process_group_robust(rank, world_size):
            logger.error("Failed to initialize process group")
            return 1
        
        # Test communication
        if not test_ddp_communication(rank, world_size, device):
            logger.error("DDP communication test failed")
            return 1
        
        # Create model
        model = create_ddp_model(args, device, rank, world_size, local_rank, logger)
        
        # Create loss function
        from src.modules.losses.blip3o_eva_loss import create_universal_flow_loss
        loss_fn = create_universal_flow_loss(
            prediction_type=args.prediction_type,
            sphere_constraint_weight=args.sphere_constraint_weight,
            debug_mode=args.debug_mode
        )
        
        # Create dataloaders
        from src.modules.datasets.blip3o_eva_dataset_ddp import create_ddp_dataloaders
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
            pin_memory=torch.cuda.is_available(),
            debug_mode=args.debug_mode
        )
        
        # Create trainer
        from src.modules.trainers.blip3o_eva_trainer_ddp import DDPDenoisingTrainer
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
        if world_size > 1:
            dist.barrier()
            if rank == 0:
                logger.info("✅ All ranks synchronized, starting training...")
        
        # Start training
        summary = trainer.train()
        
        # Final synchronization
        if world_size > 1:
            dist.barrier()
        
        if rank == 0:
            logger.info("🎉 Training completed successfully!")
            
            # Print summary
            if summary:
                logger.info("📊 Training Summary:")
                logger.info(f"  Total steps: {summary.get('total_steps', 'unknown')}")
                logger.info(f"  Best loss: {summary.get('best_loss', 'unknown'):.6f}")
                logger.info(f"  Best similarity: {summary.get('best_eval_similarity', 'unknown'):.4f}")
                logger.info(f"  Total time: {summary.get('total_time_seconds', 0):.1f}s")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("🛑 Training interrupted by user")
        return 130
        
    except Exception as e:
        logger.error(f"💥 Training failed: {e}")
        traceback.print_exc()
        return 1
    
    finally:
        # Cleanup
        try:
            if world_size > 1:
                dist.barrier()  # Final sync
            cleanup_ddp()
        except Exception as e:
            if rank == 0:
                print(f"⚠️ Error during cleanup: {e}")

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)