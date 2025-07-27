#!/usr/bin/env python3
"""
Fixed EVA-CLIP Reproduction Training Script with WandB Integration
Main training script with comprehensive fixes for gradient flow and WandB logging
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

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('eva_reproduction_training.log', mode='w')
        ]
    )
    return logging.getLogger(__name__)

def setup_wandb(args, logger):
    """Setup WandB with proper error handling"""
    try:
        import wandb
        
        # Create WandB cache directory
        wandb_cache_dir = None
        if "TMPDIR" in os.environ:
            wandb_cache_dir = Path(os.environ["TMPDIR"]) / "wandb_cache"
            wandb_cache_dir.mkdir(exist_ok=True)
            os.environ["WANDB_CACHE_DIR"] = str(wandb_cache_dir)
        
        logger.info("📊 Setting up WandB monitoring...")
        
        # Setup WandB configuration
        job_id = os.environ.get("SLURM_JOB_ID", "local")
        run_name = f"clip_pred_eva_guid_1shard_{args.model_size}_{job_id}"
        
        # Prepare config for WandB (ensure all values are JSON serializable)
        wandb_config = {
            # Training configuration
            "model_size": args.model_size,
            "training_mode": args.training_mode,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "num_epochs": args.num_epochs,
            "warmup_steps": args.warmup_steps,
            "weight_decay": args.weight_decay,
            "max_grad_norm": args.max_grad_norm,
            "fp16": args.fp16,
            
            # Evaluation config
            "eval_every_n_steps": args.eval_every_n_steps,
            "eval_num_samples": args.eval_num_samples,
            
            # System config
            "max_shards": args.max_shards,
            "num_workers": args.num_workers,
            "job_id": job_id,
            
            # Experiment details
            "experiment_type": "eva_clip_reproduction",
            "task": "reproduce_clean_eva_from_noisy",
            "conditioning": "clip_embeddings",
            "target": "eva_embeddings",
            "method": "rectified_flow_matching",
            "architecture": "blip3o_dit",
        }
        
        # Add overfitting test config if enabled
        if args.overfit_test_size:
            wandb_config["overfit_test_size"] = args.overfit_test_size
            wandb_config["experiment_mode"] = "overfitting_test"
        else:
            wandb_config["experiment_mode"] = "full_training"
        
        # Define tags
        tags = [
            "eva_clip_reproduction",
            args.model_size,
            args.training_mode,
            "blip3o_dit",
            "rectified_flow"
        ]
        
        if args.overfit_test_size:
            tags.append("overfitting_test")
        
        if job_id != "local":
            tags.extend(["slurm", f"job_{job_id}"])
        
        # Initialize WandB with proper configuration
        logger.info("🔑 Initializing WandB...")
        
        wandb.init(
            project="eva-clip",
            name=run_name,
            config=wandb_config,  # Pass config directly, not as config_paths
            tags=tags,
            group=f"eva_repro_{args.model_size}_{args.max_shards}shard",
            job_type="training",
            notes=f"EVA-CLIP reproduction using BLIP3-o DiT architecture. Target: reproduce clean EVA embeddings from noisy versions with CLIP conditioning.",
            save_code=True,
            dir=str(wandb_cache_dir) if wandb_cache_dir else None
        )
        
        logger.info("✅ WandB initialization successful")
        logger.info(f"🎯 WandB Project: eva-clip-reproduction")
        logger.info(f"👤 WandB Run: {run_name}")
        logger.info(f"🏷️ WandB Tags: {', '.join(tags)}")
        if wandb_cache_dir:
            logger.info(f"💾 WandB Cache: {wandb_cache_dir}")
        
        return wandb
        
    except ImportError:
        logger.warning("⚠️ WandB not installed. Continuing without logging...")
        return None
    except Exception as e:
        logger.error(f"❌ Failed to initialize WandB: {e}")
        logger.error("Continuing without WandB logging...")
        return None

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="EVA-CLIP Reproduction with BLIP3-o DiT")
    
    # Required arguments
    parser.add_argument("--chunked_embeddings_dir", type=str, required=True,
                       help="Path to chunked embeddings directory")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for checkpoints")
    
    # Model configuration
    parser.add_argument("--model_size", type=str, default="base",
                       choices=["tiny", "small", "base", "large"],
                       help="Model size")
    parser.add_argument("--training_mode", type=str, default="patch_only",
                       choices=["patch_only", "cls_patch"],
                       help="Training mode")
    
    # Training hyperparameters
    parser.add_argument("--learning_rate", type=float, default=5e-4,
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
    
    # Evaluation
    parser.add_argument("--eval_every_n_steps", type=int, default=100,
                       help="Evaluate every N steps")
    parser.add_argument("--eval_num_samples", type=int, default=500,
                       help="Number of samples for evaluation")
    
    # Debugging and testing
    parser.add_argument("--overfit_test_size", type=int, default=None,
                       help="Size for overfitting test (None to disable)")
    parser.add_argument("--debug_mode", action="store_true",
                       help="Enable debug mode")
    parser.add_argument("--max_shards", type=int, default=1,
                       help="Maximum number of shards to use")
    
    # System
    parser.add_argument("--fp16", action="store_true", default=True,
                       help="Use mixed precision")
    parser.add_argument("--num_workers", type=int, default=0,
                       help="Number of dataloader workers")
    
    # WandB configuration
    parser.add_argument("--disable_wandb", action="store_true",
                       help="Disable WandB logging")
    parser.add_argument("--wandb_project", type=str, default="eva-clip-reproduction",
                       help="WandB project name")
    
    return parser.parse_args()

def setup_device_and_model(args, logger):
    """Setup device and create model"""
    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Clear any cached memory
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    
    # Import and create model
    try:
        from src.modules.models.blip3o_eva_dit import create_universal_model, UniversalDiTConfig
    except ImportError:
        logger.error("Could not import fixed model. Make sure the model module is available.")
        raise
    
    logger.info(f"Creating {args.model_size} model for {args.training_mode} mode...")
    
    model = create_universal_model(
        model_size=args.model_size,
        training_mode=args.training_mode
    )
    
    model = model.to(device)
    
    logger.info(f"Model created with {model.get_num_parameters():,} parameters")
    logger.info(f"Model moved to {device}")
    
    return device, model

def create_loss_function(args, logger):
    """Create loss function"""
    try:
        from src.modules.losses.blip3o_eva_loss import create_universal_flow_loss
    except ImportError:
        logger.error("Could not import fixed loss. Make sure the loss module is available.")
        raise
    
    logger.info("Creating flow matching loss...")
    
    loss_fn = create_universal_flow_loss(
        prediction_type="velocity",
        # flow_type="rectified",
        loss_weight=1.0,
        debug_mode=args.debug_mode
    )
    
    logger.info("Flow matching loss created")
    return loss_fn

def create_dataloaders(args, logger):
    """Create data loaders"""
    try:
        from src.modules.datasets.blip3o_eva_dataset import create_eva_reproduction_dataloaders
    except ImportError:
        logger.error("Could not import fixed dataset. Make sure the dataset module is available.")
        raise
    
    logger.info("Creating dataloaders...")
    
    train_dataloader, eval_dataloader = create_eva_reproduction_dataloaders(
        chunked_embeddings_dir=args.chunked_embeddings_dir,
        batch_size=args.batch_size,
        training_mode=args.training_mode,
        max_shards=args.max_shards,
        normalize_embeddings=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    logger.info(f"Dataloaders created")
    logger.info(f"  Training batches: {len(train_dataloader)}")
    logger.info(f"  Evaluation available: {eval_dataloader is not None}")
    
    return train_dataloader, eval_dataloader

def create_trainer(model, loss_fn, train_dataloader, eval_dataloader, args, device, logger, wandb_instance=None):
    """Create trainer"""
    try:
        from src.modules.trainers.blip3o_eva_trainer_wandb import create_eva_trainer_with_wandb
    except ImportError:
        # Fallback to regular trainer if WandB trainer not available
        try:
            from src.modules.trainers.blip3o_eva_trainer import create_eva_trainer_with_wandb
            logger.warning("Using trainer with WandB integration")
            return create_eva_trainer(
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
                debug_mode=args.debug_mode,
                overfit_test_size=args.overfit_test_size,
                output_dir=args.output_dir,
                device=device
            )
        except ImportError:
            logger.error("Could not import any trainer. Make sure the trainer module is available.")
            raise
    
    logger.info("Creating trainer with WandB integration...")
    
    trainer = create_eva_trainer_with_wandb(
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
        debug_mode=args.debug_mode,
        overfit_test_size=args.overfit_test_size,
        output_dir=args.output_dir,
        device=device,
        wandb_instance=wandb_instance
    )
    
    logger.info("Trainer with WandB integration created")
    return trainer

def main():
    """Main training function"""
    args = parse_arguments()
    logger = setup_logging()
    
    logger.info("🚀 Starting EVA-CLIP Reproduction with Fixed BLIP3-o DiT")
    logger.info("=" * 80)
    logger.info("EXPERIMENT DETAILS:")
    logger.info("  📋 Task: Reproduce clean EVA embeddings from noisy EVA embeddings")
    logger.info("  🧠 Model: BLIP3-o DiT with 3D RoPE and Grouped-Query Attention")
    logger.info("  🎯 Target: EVA embeddings [B, N, 4096]")
    logger.info("  🎮 Conditioning: CLIP embeddings [B, N, 1024]")
    logger.info("  🌊 Method: Rectified Flow Matching")
    logger.info("=" * 80)
    
    # Setup WandB if not disabled
    wandb_instance = None
    if not args.disable_wandb:
        wandb_instance = setup_wandb(args, logger)
    else:
        logger.info("📊 WandB logging disabled by user")
    
    logger.info(f"Configuration:")
    logger.info(f"  Model size: {args.model_size}")
    logger.info(f"  Training mode: {args.training_mode}")
    logger.info(f"  Embeddings dir: {args.chunked_embeddings_dir}")
    logger.info(f"  Output dir: {args.output_dir}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Epochs: {args.num_epochs}")
    logger.info(f"  Max shards: {args.max_shards}")
    if args.overfit_test_size:
        logger.info(f"  🧪 OVERFITTING TEST: {args.overfit_test_size} samples")
    logger.info(f"  Debug mode: {args.debug_mode}")
    logger.info(f"  WandB enabled: {wandb_instance is not None}")
    logger.info("=" * 80)
    
    try:
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup device and model
        device, model = setup_device_and_model(args, logger)
        
        # Create loss function
        loss_fn = create_loss_function(args, logger)
        
        # Create dataloaders
        train_dataloader, eval_dataloader = create_dataloaders(args, logger)
        
        # Create trainer
        trainer = create_trainer(model, loss_fn, train_dataloader, eval_dataloader, args, device, logger, wandb_instance)
        
        # Save configuration
        config = {
            'args': vars(args),
            'model_config': model.config.to_dict() if hasattr(model, 'config') else {},
            'timestamp': datetime.now().isoformat(),
            'experiment_type': 'eva_reproduction',
            'wandb_enabled': wandb_instance is not None,
            'fixes_applied': [
                'gradient_flow_improvements',
                'proper_initialization',
                'correct_data_flow',
                'numerical_stability',
                'overfitting_test_capability',
                'wandb_integration_fixed'
            ]
        }
        
        config_path = output_dir / 'experiment_config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Configuration saved to {config_path}")
        
        # Log configuration to WandB if available
        if wandb_instance:
            wandb_instance.config.update({
                "config_file": str(config_path),
                "total_model_parameters": model.get_num_parameters(),
                "training_steps_per_epoch": len(train_dataloader),
                "total_training_steps": len(train_dataloader) * args.num_epochs
            })
        
        # Start training
        logger.info("\n🚀 Starting training...")
        logger.info("Expected behavior:")
        logger.info("  • Loss should decrease steadily")
        logger.info("  • Velocity similarity should increase")
        logger.info("  • EVA similarity should improve during evaluation")
        logger.info("  • Gradients should be non-zero and stable")
        
        if args.overfit_test_size:
            logger.info(f"  • OVERFITTING TEST: Should achieve >0.8 similarity on {args.overfit_test_size} samples")
        
        if wandb_instance:
            logger.info("  • Real-time metrics will be logged to WandB")
        
        logger.info("")
        
        start_time = datetime.now()
        
        # Run training
        summary = trainer.train()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Final summary
        logger.info("\n" + "=" * 80)
        logger.info("🎉 TRAINING COMPLETED!")
        logger.info("=" * 80)
        logger.info(f"📊 RESULTS SUMMARY:")
        logger.info(f"  Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        logger.info(f"  Total steps: {summary.get('total_steps', 0)}")
        logger.info(f"  Best loss: {summary.get('best_loss', float('inf')):.6f}")
        logger.info(f"  Best EVA similarity: {summary.get('best_eval_similarity', 0):.4f}")
        
        # Log final metrics to WandB
        if wandb_instance:
            wandb_instance.log({
                "training/final_duration_minutes": duration / 60,
                "training/total_steps": summary.get('total_steps', 0),
                "training/best_loss": summary.get('best_loss', float('inf')),
                "training/best_eva_similarity": summary.get('best_eval_similarity', 0),
            })
        
        # Evaluation results
        final_eval = summary.get('final_eval', {})
        if final_eval:
            logger.info(f"📊 FINAL EVALUATION:")
            logger.info(f"  EVA similarity: {final_eval.get('eval_eva_similarity', 0):.4f}")
            logger.info(f"  High quality (>0.7): {final_eval.get('eval_high_quality', 0)*100:.1f}%")
            logger.info(f"  Very high quality (>0.8): {final_eval.get('eval_very_high_quality', 0)*100:.1f}%")
            logger.info(f"  Excellent quality (>0.9): {final_eval.get('eval_excellent_quality', 0)*100:.1f}%")
            logger.info(f"  Samples evaluated: {final_eval.get('eval_samples', 0)}")
            
            # Log final evaluation to WandB
            if wandb_instance:
                for key, value in final_eval.items():
                    wandb_instance.log({f"final_eval/{key}": value})
        
        # Overfitting test results
        if args.overfit_test_size:
            overfit_success = summary.get('overfit_success', False)
            logger.info(f"🧪 OVERFITTING TEST: {'✅ PASSED' if overfit_success else '❌ FAILED'}")
            if overfit_success:
                logger.info("   ✅ Model can learn and memorize - architecture is working!")
            else:
                logger.info("   ⚠️  Model struggles to overfit - check architecture/loss")
            
            # Log overfitting test results to WandB
            if wandb_instance:
                wandb_instance.log({
                    "overfit_test/success": overfit_success,
                    "overfit_test/max_similarity": max(summary.get('similarity_history', [0])) if summary.get('similarity_history') else 0
                })
        
        # Architecture assessment
        best_sim = summary.get('best_eval_similarity', 0)
        logger.info(f"🏗️  ARCHITECTURE ASSESSMENT:")
        if best_sim > 0.7:
            assessment = "EXCELLENT: BLIP3-o DiT architecture working perfectly!"
            logger.info(f"   🎉 {assessment}")
        elif best_sim > 0.4:
            assessment = "GOOD: BLIP3-o DiT architecture shows strong capability!"
            logger.info(f"   ✅ {assessment}")
        elif best_sim > 0.1:
            assessment = "FAIR: BLIP3-o DiT architecture is functional!"
            logger.info(f"   📈 {assessment}")
        else:
            assessment = "NEEDS WORK: Architecture may need tuning!"
            logger.info(f"   ⚠️  {assessment}")
        
        # Log architecture assessment to WandB
        if wandb_instance:
            wandb_instance.log({
                "final_assessment/similarity_score": best_sim,
                "final_assessment/category": assessment.split(":")[0],
            })
            wandb_instance.summary["final_similarity"] = best_sim
            wandb_instance.summary["assessment"] = assessment
        
        # Save final summary
        summary['duration_seconds'] = duration
        summary['end_time'] = end_time.isoformat()
        summary['experiment_config'] = config
        summary['assessment'] = assessment
        
        summary_path = output_dir / 'final_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"📁 Final summary saved to {summary_path}")
        logger.info(f"📁 Model checkpoints saved to {output_dir}")
        
        if wandb_instance:
            logger.info(f"📊 WandB run: {wandb_instance.url}")
            # Save summary as artifact
            artifact = wandb_instance.Artifact(f"training_summary_{wandb_instance.id}", type="summary")
            artifact.add_file(str(summary_path))
            wandb_instance.log_artifact(artifact)
            
            # Finish WandB run
            wandb_instance.finish()
        
        logger.info("=" * 80)
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Training failed with error: {e}")
        traceback.print_exc()
        
        # Log error to WandB if available
        if wandb_instance:
            wandb_instance.log({"error": str(e)})
            wandb_instance.finish(exit_code=1)
        
        return 1
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        
        # Gracefully finish WandB if available
        if wandb_instance:
            wandb_instance.finish(exit_code=2)
        
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)