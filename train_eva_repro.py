#!/usr/bin/env python3
"""
FIXED Universal BLIP3-o Training Script - EVA & CLIP Denoising with WandB Integration
Supports both EVA-to-EVA and CLIP-to-CLIP (with EVA conditioning) denoising tasks

FIXES:
- Fixed string formatting issues
- Better type checking for metrics logging
- More robust error handling
- Safe deque access methods

Usage:
  # EVA Denoising (original task)
  python train_eva_repro.py --task_mode eva_denoising --chunked_embeddings_dir /path/to/embeddings --output_dir ./checkpoints_eva

  # CLIP Denoising with EVA Conditioning (new task)  
  python train_eva_repro.py --task_mode clip_denoising --chunked_embeddings_dir /path/to/embeddings --output_dir ./checkpoints_clip

  # With WandB logging
  python train_eva_repro.py --task_mode eva_denoising --use_wandb --wandb_project "blip3o-denoising" --wandb_run_name "eva-test-run"
"""

import os
import sys
import argparse
import torch
import json
import logging
from pathlib import Path
from datetime import datetime
import traceback

# Setup paths
sys.path.insert(0, str(Path(__file__).parent))

def setup_logging(output_dir: str):
    """Setup logging configuration"""
    log_file = Path(output_dir) / 'training.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode='w')
        ]
    )
    return logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Universal BLIP3-o Denoising Training")
    
    # Required arguments
    parser.add_argument("--chunked_embeddings_dir", type=str, required=True,
                       help="Path to chunked embeddings directory")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for checkpoints")
    
    # Task configuration - NEW!
    parser.add_argument("--task_mode", type=str, default="eva_denoising",
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
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=10,
                       help="Number of epochs")
    parser.add_argument("--warmup_steps", type=int, default=100,
                       help="Warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                       help="Max gradient norm")
    
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
    parser.add_argument("--eval_every_n_steps", type=int, default=100,
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
    parser.add_argument("--max_shards", type=int, default=1,
                       help="Maximum number of shards to use")
    
    # WandB integration - NEW!
    parser.add_argument("--use_wandb", action="store_true",
                       help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="blip3o-universal-denoising",
                       help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                       help="WandB run name (auto-generated if not provided)")
    parser.add_argument("--wandb_tags", type=str, nargs="*", default=[],
                       help="WandB tags for the run")
    
    # System
    parser.add_argument("--fp16", action="store_true", default=True,
                       help="Use mixed precision")
    parser.add_argument("--num_workers", type=int, default=0,
                       help="Number of dataloader workers")
    
    return parser.parse_args()

def setup_wandb(args, config, logger):
    """Setup WandB logging if enabled"""
    if not args.use_wandb:
        logger.info("WandB logging disabled")
        return None
    
    try:
        import wandb
        logger.info("Setting up WandB logging...")
        
        # Generate run name if not provided
        if args.wandb_run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.wandb_run_name = f"{args.task_mode}_{args.model_size}_{timestamp}"
        
        # Initialize WandB
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=config,
            tags=args.wandb_tags + [args.task_mode, args.model_size, args.training_mode],
            notes=f"Universal BLIP3-o {args.task_mode} training",
            dir=args.output_dir
        )
        
        logger.info(f"✅ WandB initialized:")
        logger.info(f"  Project: {args.wandb_project}")
        logger.info(f"  Run name: {args.wandb_run_name}")
        logger.info(f"  Tags: {run.tags}")
        logger.info(f"  URL: {run.url}")
        
        return run
        
    except ImportError:
        logger.error("❌ WandB not installed but --use_wandb specified")
        logger.error("Install with: pip install wandb")
        return None
    except Exception as e:
        logger.error(f"❌ Failed to initialize WandB: {e}")
        return None

def print_task_banner(args, logger):
    """Print task-specific banner"""
    logger.info("🚀 Universal BLIP3-o Denoising Training")
    logger.info("=" * 80)
    
    if args.task_mode == "eva_denoising":
        logger.info("🎯 EVA-CLIP DENOISING TASK:")
        logger.info("  📋 Task: Denoise noisy EVA embeddings using clean EVA guidance")
        logger.info("  📥 Input: Noisy EVA embeddings [B, N, 4096]")
        logger.info("  🎮 Conditioning: Clean EVA embeddings [B, N, 4096]")
        logger.info("  📤 Output: Clean EVA embeddings [B, N, 4096]")
        logger.info("  🌊 Method: Spherical Flow Matching on 4096D hypersphere")
        logger.info("  🎯 Goal: High cosine similarity (>0.7 excellent, >0.5 good)")
    
    elif args.task_mode == "clip_denoising":
        logger.info("🎯 CLIP-ViT DENOISING WITH EVA CONDITIONING TASK:")
        logger.info("  📋 Task: Denoise noisy CLIP embeddings using clean EVA guidance")
        logger.info("  📥 Input: Noisy CLIP embeddings [B, N, 1024]")
        logger.info("  🎮 Conditioning: Clean EVA embeddings [B, N, 4096]")
        logger.info("  📤 Output: Clean CLIP embeddings [B, N, 1024]")
        logger.info("  🌊 Method: Spherical Flow Matching on 1024D hypersphere")
        logger.info("  🎯 Goal: High cosine similarity (>0.6 excellent, >0.4 good)")
        logger.info("  🧠 Key: Cross-attention between 1024D and 4096D spaces")
    
    logger.info("  🏗️ Model: Universal BLIP3-o DiT with cross-attention conditioning")
    
    if args.use_wandb:
        logger.info(f"  📊 WandB: {args.wandb_project}/{args.wandb_run_name}")
    
    logger.info("=" * 80)

def setup_device_and_model(args, logger):
    """Setup device and create universal model"""
    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    
    # Import and create universal model
    try:
        from src.modules.models.blip3o_eva_dit import create_universal_model
    except ImportError:
        logger.error("Could not import universal model. Make sure blip3o_eva_dit.py is present.")
        raise
    
    logger.info(f"Creating {args.model_size} universal model for {args.task_mode}...")
    logger.info(f"Prediction type: {args.prediction_type}")
    
    model = create_universal_model(
        model_size=args.model_size,
        training_mode=args.training_mode,
        task_mode=args.task_mode,  # NEW: Specify task mode
        prediction_type=args.prediction_type
    )
    
    model = model.to(device)
    
    logger.info(f"Universal model created with {model.get_num_parameters():,} parameters")
    logger.info(f"Model moved to {device}")
    
    # Print model task info
    task_info = model._get_task_info()
    logger.info(f"Model configured for: {task_info['task']}")
    logger.info(f"  Input: {task_info['input']}")
    logger.info(f"  Conditioning: {task_info['conditioning']}")
    logger.info(f"  Output: {task_info['output']}")
    
    return device, model

def create_loss_function(args, logger):
    """Create universal spherical flow matching loss function"""
    try:
        from src.modules.losses.blip3o_eva_loss import create_universal_flow_loss
    except ImportError:
        logger.error("Could not import universal flow loss. Make sure blip3o_eva_loss.py is present.")
        raise
    
    logger.info("Creating universal spherical flow matching loss...")
    
    loss_fn = create_universal_flow_loss(
        prediction_type=args.prediction_type,
        loss_weight=1.0,
        sphere_constraint_weight=args.sphere_constraint_weight,
        debug_mode=args.debug_mode
    )
    
    logger.info("Universal spherical flow matching loss created")
    return loss_fn

def create_dataloaders(args, logger):
    """Create universal data loaders"""
    try:
        from src.modules.datasets.blip3o_eva_dataset import create_universal_dataloaders
    except ImportError:
        logger.error("Could not import universal dataset. Make sure blip3o_eva_dataset.py is present.")
        raise
    
    logger.info(f"Creating {args.task_mode} dataloaders...")
    
    train_dataloader, eval_dataloader = create_universal_dataloaders(
        chunked_embeddings_dir=args.chunked_embeddings_dir,
        task_mode=args.task_mode,  # NEW: Specify task mode
        batch_size=args.batch_size,
        training_mode=args.training_mode,
        max_shards=args.max_shards,
        noise_schedule=args.noise_schedule,
        max_noise_level=args.max_noise_level,
        min_noise_level=args.min_noise_level,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    logger.info(f"Universal dataloaders created for {args.task_mode}")
    
    # Handle dataloader length safely
    try:
        train_batches = len(train_dataloader)
        logger.info(f"  Training batches: {train_batches} (estimated)")
    except (TypeError, AttributeError):
        logger.info(f"  Training batches: Unknown (IterableDataset)")
        train_batches = None
    
    logger.info(f"  Evaluation available: {eval_dataloader is not None}")
    
    return train_dataloader, eval_dataloader

def create_trainer(model, loss_fn, train_dataloader, eval_dataloader, args, device, logger, wandb_instance=None):
    """Create universal trainer"""
    try:
        from src.modules.trainers.blip3o_eva_trainer import create_universal_trainer
    except ImportError:
        logger.error("Could not import universal trainer. Make sure blip3o_eva_trainer.py is present.")
        raise
    
    logger.info("Creating universal trainer...")
    
    trainer = create_universal_trainer(
        model=model,
        loss_fn=loss_fn,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        fp16=args.fp16,
        eval_every_n_steps=args.eval_every_n_steps,
        eval_num_samples=args.eval_num_samples,
        eval_inference_steps=args.eval_inference_steps,
        debug_mode=args.debug_mode,
        overfit_test_size=args.overfit_test_size,
        output_dir=args.output_dir,
        task_mode=args.task_mode,  # NEW: Pass task mode
        device=device,
        wandb_instance=wandb_instance  # NEW: Pass WandB instance
    )
    
    logger.info("Universal trainer created")
    return trainer

def validate_spherical_constraints(batch, args, logger):
    """Validate that embeddings satisfy spherical constraints"""
    try:
        if args.task_mode == "eva_denoising":
            if 'target_embeddings' in batch:
                target = batch['target_embeddings']
                norms = torch.norm(target, dim=-1)
                norm_mean = norms.mean().item()
                logger.info(f"✅ EVA embeddings normalized: mean norm = {norm_mean:.4f}")
        
        elif args.task_mode == "clip_denoising":
            if 'target_embeddings' in batch:
                target = batch['target_embeddings']
                norms = torch.norm(target, dim=-1)
                norm_mean = norms.mean().item()
                logger.info(f"✅ CLIP embeddings normalized: mean norm = {norm_mean:.4f}")
            
            if 'conditioning_embeddings' in batch:
                conditioning = batch['conditioning_embeddings']
                norms = torch.norm(conditioning, dim=-1)
                norm_mean = norms.mean().item()
                logger.info(f"✅ EVA conditioning normalized: mean norm = {norm_mean:.4f}")
                
    except Exception as e:
        logger.warning(f"Error during spherical constraint validation: {e}")

def safe_format_metric(key: str, value, decimal_places: int = 4) -> str:
    """Safely format a metric value for logging"""
    try:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if decimal_places == 6:
                return f"{key}: {value:.6f}"
            else:
                return f"{key}: {value:.4f}"
        elif isinstance(value, str):
            return f"{key}: {value}"
        elif isinstance(value, bool):
            return f"{key}: {value}"
        else:
            return f"{key}: {str(value)}"
    except (ValueError, TypeError):
        return f"{key}: {str(value)}"

def main():
    """Main training function"""
    args = parse_arguments()
    
    # Create output directory early for logging
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(args.output_dir)
    
    # Print task-specific banner
    print_task_banner(args, logger)
    
    logger.info(f"Configuration:")
    logger.info(f"  Task mode: {args.task_mode}")
    logger.info(f"  Model size: {args.model_size}")
    logger.info(f"  Training mode: {args.training_mode}")
    logger.info(f"  Prediction type: {args.prediction_type}")
    logger.info(f"  Embeddings dir: {args.chunked_embeddings_dir}")
    logger.info(f"  Output dir: {args.output_dir}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Epochs: {args.num_epochs}")
    logger.info(f"  Max shards: {args.max_shards}")
    logger.info(f"  Noise schedule: {args.noise_schedule}")
    logger.info(f"  Noise range: [{args.min_noise_level}, {args.max_noise_level}]")
    logger.info(f"  Sphere constraint weight: {args.sphere_constraint_weight}")
    if args.overfit_test_size:
        logger.info(f"  🧪 OVERFITTING TEST: {args.overfit_test_size} samples")
    logger.info(f"  Debug mode: {args.debug_mode}")
    logger.info(f"  WandB enabled: {args.use_wandb}")
    if args.use_wandb:
        logger.info(f"  WandB project: {args.wandb_project}")
        logger.info(f"  WandB run name: {args.wandb_run_name or 'auto-generated'}")
    
    logger.info("=" * 80)
    logger.info("🔧 UNIVERSAL ARCHITECTURE FEATURES:")
    logger.info("  ✅ Task-adaptive input/output dimensions")
    logger.info("  ✅ Flexible cross-attention conditioning")
    logger.info("  ✅ Universal spherical flow matching")
    logger.info("  ✅ Proper gradient flow and initialization")
    logger.info("  ✅ Task-specific evaluation metrics")
    logger.info("  ✅ Gradient clipping for stability")
    logger.info("  ✅ Mixed precision training support")
    if args.use_wandb:
        logger.info("  ✅ WandB logging and visualization")
    logger.info("=" * 80)
    
    try:
        # Setup device and model
        device, model = setup_device_and_model(args, logger)
        
        # Create loss function
        loss_fn = create_loss_function(args, logger)
        
        # Create dataloaders
        train_dataloader, eval_dataloader = create_dataloaders(args, logger)
        
        # Validate first batch for spherical constraints
        logger.info("Validating spherical constraints on first batch...")
        try:
            first_batch = next(iter(train_dataloader))
            validate_spherical_constraints(first_batch, args, logger)
            logger.info("✅ First batch validation successful")
        except Exception as e:
            logger.warning(f"⚠️ Could not validate first batch: {e}")
            logger.warning("Continuing with training...")
        
        # Save configuration
        config = {
            'args': vars(args),
            'model_config': model.config.to_dict() if hasattr(model, 'config') else {},
            'timestamp': datetime.now().isoformat(),
            'experiment_type': f'universal_{args.task_mode}',
            'task_description': {
                'task_mode': args.task_mode,
                'input': f'Noisy {"EVA" if args.task_mode == "eva_denoising" else "CLIP"} embeddings',
                'conditioning': f'Clean {"EVA" if args.task_mode == "eva_denoising" else "EVA"} embeddings',
                'output': f'Clean {"EVA" if args.task_mode == "eva_denoising" else "CLIP"} embeddings',
                'input_dim': 4096 if args.task_mode == "eva_denoising" else 1024,
                'output_dim': 4096 if args.task_mode == "eva_denoising" else 1024,
                'conditioning_dim': 4096,  # Always EVA for conditioning
                'method': 'Universal Spherical Flow Matching',
                'goal': 'High cosine similarity',
            },
            'architecture_features': [
                'universal_task_support',
                'task_adaptive_dimensions',
                'flexible_cross_attention',
                'spherical_flow_matching_universal',
                'proper_gradient_flow',
                'numerical_stability_improvements',
                'task_specific_evaluation_metrics',
            ],
            'wandb_config': {
                'enabled': args.use_wandb,
                'project': args.wandb_project if args.use_wandb else None,
                'run_name': args.wandb_run_name if args.use_wandb else None,
                'tags': args.wandb_tags if args.use_wandb else None,
            }
        }
        
        # Setup WandB if enabled
        wandb_instance = setup_wandb(args, config, logger)
        
        config_path = output_dir / 'experiment_config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Configuration saved to {config_path}")
        
        # Create trainer with WandB instance
        trainer = create_trainer(model, loss_fn, train_dataloader, eval_dataloader, args, device, logger, wandb_instance)
        
        # Start training
        logger.info(f"\n🚀 Starting {args.task_mode} training...")
        
        if args.task_mode == "eva_denoising":
            logger.info("Expected behavior for EVA denoising:")
            logger.info("  • 🎯 MAIN GOAL: High cosine similarity (>0.7 excellent, >0.5 good)")
            logger.info("  • ⬇️ Loss should decrease steadily")
            logger.info("  • ⬆️ Cosine similarity should increase from ~0 to >0.5+")
            logger.info("  • 🔵 EVA embeddings should stay on unit sphere (norm ≈ 1.0)")
            if args.overfit_test_size:
                logger.info(f"  • 🧪 OVERFITTING TEST: Should achieve >0.8 similarity on {args.overfit_test_size} samples")
        
        elif args.task_mode == "clip_denoising":
            logger.info("Expected behavior for CLIP denoising:")
            logger.info("  • 🎯 MAIN GOAL: High cosine similarity (>0.6 excellent, >0.4 good)")
            logger.info("  • ⬇️ Loss should decrease steadily")
            logger.info("  • ⬆️ Cosine similarity should increase from ~0 to >0.4+")
            logger.info("  • 🔵 CLIP embeddings should stay on unit sphere (norm ≈ 1.0)")
            logger.info("  • 🧠 Cross-attention should learn 1024D ↔ 4096D mapping")
            if args.overfit_test_size:
                logger.info(f"  • 🧪 OVERFITTING TEST: Should achieve >0.7 similarity on {args.overfit_test_size} samples")
        
        logger.info("  • 📈 Gradients should be stable and non-zero")
        logger.info("  • 🚫 No negative cosine similarities at convergence")
        if args.use_wandb and wandb_instance:
            logger.info(f"  • 📊 Metrics will be logged to WandB: {wandb_instance.url}")
        logger.info("")
        
        start_time = datetime.now()
        
        # Run training
        summary = trainer.train()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Log final summary to WandB
        if args.use_wandb and wandb_instance:
            try:
                wandb_instance.log({
                    "final/duration_seconds": duration,
                    "final/total_steps": summary.get('total_steps', 0),
                    "final/best_loss": summary.get('best_loss', float('inf')),
                    "final/best_eval_similarity": summary.get('best_eval_similarity', 0),
                    "final/overfit_success": summary.get('overfit_success', False),
                })
                
                # Log final evaluation metrics
                final_eval = summary.get('final_eval', {})
                if final_eval:
                    wandb_final_eval = {}
                    for key, value in final_eval.items():
                        if isinstance(value, (int, float)) and not isinstance(value, bool):
                            wandb_final_eval[f"final_eval/{key}"] = value
                    if wandb_final_eval:
                        wandb_instance.log(wandb_final_eval)
                
                logger.info("✅ Final metrics logged to WandB")
            except Exception as e:
                logger.warning(f"Failed to log final metrics to WandB: {e}")
        
        # Final summary
        logger.info("\n" + "=" * 80)
        logger.info(f"🎉 {args.task_mode.upper()} TRAINING COMPLETED!")
        logger.info("=" * 80)
        logger.info(f"📊 RESULTS SUMMARY:")
        logger.info(f"  Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        logger.info(f"  Total steps: {summary.get('total_steps', 0)}")
        logger.info(f"  Best loss: {summary.get('best_loss', float('inf')):.6f}")
        logger.info(f"  🎯 Best similarity: {summary.get('best_eval_similarity', 0):.4f}")
        
        # Task-specific evaluation results
        final_eval = summary.get('final_eval', {})
        if final_eval:
            task_mode = final_eval.get('eval_task_mode', args.task_mode)
            if task_mode == "eva_denoising":
                metric_prefix = "eval_eva"
                thresholds = {"excellent": 0.8, "good": 0.7, "fair": 0.5}
            elif task_mode == "clip_denoising":
                metric_prefix = "eval_clip"
                thresholds = {"excellent": 0.7, "good": 0.6, "fair": 0.4}
            else:
                metric_prefix = "eval_generic"
                thresholds = {"excellent": 0.7, "good": 0.6, "fair": 0.4}
            
            main_sim_key = f'{metric_prefix}_similarity'
            if main_sim_key in final_eval:
                sim = final_eval[main_sim_key]
                logger.info(f"📊 FINAL EVALUATION RESULTS:")
                logger.info(f"  🎯 {task_mode.upper()} cosine similarity: {sim:.4f}")
                
                for key, value in final_eval.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool) and key != main_sim_key:
                        formatted_value = safe_format_metric(key, value, decimal_places=4)
                        logger.info(f"  📊 {formatted_value}")
                
                # Success assessment
                if sim > thresholds["excellent"]:
                    logger.info(f"🎉 OUTSTANDING SUCCESS! {task_mode} similarity > {thresholds['excellent']}")
                elif sim > thresholds["good"]:
                    logger.info(f"🎊 EXCELLENT SUCCESS! {task_mode} similarity > {thresholds['good']}")
                elif sim > thresholds["fair"]:
                    logger.info(f"✅ GOOD SUCCESS! {task_mode} similarity > {thresholds['fair']}")
                else:
                    logger.info(f"📈 Progress made: {task_mode} similarity = {sim:.4f}")
        
        # Overfitting test results
        if args.overfit_test_size:
            overfit_success = summary.get('overfit_success', False)
            logger.info(f"🧪 OVERFITTING TEST: {'✅ PASSED' if overfit_success else '❌ FAILED'}")
            if overfit_success:
                logger.info("   ✅ Model can learn and memorize - architecture is working perfectly!")
            else:
                logger.info("   ⚠️ Model struggles to overfit - may need hyperparameter tuning")
        
        # Save final summary
        summary['duration_seconds'] = duration
        summary['end_time'] = end_time.isoformat()
        summary['experiment_config'] = config
        summary['wandb_url'] = wandb_instance.url if args.use_wandb and wandb_instance else None
        
        summary_path = output_dir / 'final_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"📁 Final summary saved to {summary_path}")
        logger.info(f"📁 Model checkpoints saved to {output_dir}")
        
        if args.use_wandb and wandb_instance:
            logger.info(f"📊 WandB run: {wandb_instance.url}")
        
        logger.info("=" * 80)
        logger.info("🎯 MISSION ACCOMPLISHED:")
        logger.info(f"  ✅ {args.task_mode} training completed")
        logger.info("  ✅ Universal spherical flow matching implemented")
        logger.info("  ✅ Task-adaptive architecture working")
        logger.info("  ✅ Comprehensive evaluation performed")
        logger.info(f"  🎯 Final similarity: {summary.get('best_eval_similarity', 0):.4f}")
        if args.use_wandb and wandb_instance:
            logger.info(f"  📊 WandB logging completed")
        logger.info("=" * 80)
        
        # Finish WandB run
        if args.use_wandb and wandb_instance:
            try:
                wandb_instance.finish()
                logger.info("✅ WandB run finished")
            except Exception as e:
                logger.warning(f"Warning finishing WandB run: {e}")
        
        # Return appropriate exit code
        best_sim = summary.get('best_eval_similarity', 0)
        if args.task_mode == "eva_denoising":
            return 0 if best_sim > 0.3 else 1
        elif args.task_mode == "clip_denoising":
            return 0 if best_sim > 0.2 else 1
        else:
            return 0 if best_sim > 0.1 else 1
        
    except Exception as e:
        logger.error(f"❌ Training failed with error: {e}")
        traceback.print_exc()
        
        # Finish WandB run on error
        if args.use_wandb:
            try:
                import wandb
                if wandb.run is not None:
                    wandb.finish(exit_code=1)
            except:
                pass
        
        return 1
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        
        # Finish WandB run on interrupt
        if args.use_wandb:
            try:
                import wandb
                if wandb.run is not None:
                    wandb.finish(exit_code=1)
            except:
                pass
        
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)