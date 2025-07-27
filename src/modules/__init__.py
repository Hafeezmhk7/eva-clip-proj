"""
BLIP3-o Modules - Updated for Universal Denoising (EVA & CLIP)
src/modules/__init__.py

Main entry point for all BLIP3-o modules including universal denoising components
"""

import logging

logger = logging.getLogger(__name__)

# Import availability flags for UNIVERSAL denoising components
UNIVERSAL_MODEL_AVAILABLE = False
UNIVERSAL_LOSS_AVAILABLE = False
UNIVERSAL_TRAINER_AVAILABLE = False
UNIVERSAL_DATASET_AVAILABLE = False
UNIVERSAL_CONFIG_AVAILABLE = False

# Try importing UNIVERSAL denoising components (MAIN COMPONENTS)
try:
    from .models.blip3o_eva_dit import (
        UniversalDiTModel,
        UniversalDiTConfig,
        create_universal_model,
        create_clip_denoising_model # This was commented out in your original `__all__.extend` but seemed intended
    )
    UNIVERSAL_MODEL_AVAILABLE = True
    logger.info("✅ Universal denoising model loaded successfully")
    logger.info("   ✅ Supports: EVA denoising (4096D) & CLIP denoising (1024D)")
except ImportError as e:
    logger.error(f"❌ Universal model import failed: {e}")

try:
    from .losses.blip3o_eva_loss import (
        UniversalSphericalFlowMatchingLoss,
        create_universal_flow_loss,
        create_clip_flow_loss # This was commented out in your original `__all__.extend` but seemed intended
    )
    UNIVERSAL_LOSS_AVAILABLE = True
    logger.info("✅ Universal spherical flow matching loss loaded successfully")
    logger.info("   ✅ Supports: Universal spherical flow for any embedding dimension")
except ImportError as e:
    logger.error(f"❌ Universal flow loss import failed: {e}")

try:
    from .trainers.blip3o_eva_trainer import (
        UniversalDenoisingTrainer,
        create_universal_trainer,
        create_clip_denoising_trainer # This was commented out in your original `__all__.extend` but seemed intended
    )
    UNIVERSAL_TRAINER_AVAILABLE = True
    logger.info("✅ Universal denoising trainer loaded successfully")
    logger.info("   ✅ Supports: Task-adaptive training for EVA & CLIP denoising")
except ImportError as e:
    logger.error(f"❌ Universal trainer import failed: {e}")

try:
    from .datasets.blip3o_eva_dataset import (
        UniversalDenoisingDataset,
        create_universal_dataloaders,
        universal_collate_fn,
        create_clip_denoising_dataloaders # This was commented out in your original `__all__.extend` but seemed intended
    )
    UNIVERSAL_DATASET_AVAILABLE = True
    logger.info("✅ Universal denoising dataset loaded successfully")
    logger.info("   ✅ Supports: Flexible data loading for both EVA & CLIP tasks")
except ImportError as e:
    logger.error(f"❌ Universal dataset import failed: {e}")

try:
    from .config.blip3o_config import (
        BLIP3oDiTConfig,
        get_blip3o_config,
        create_config_from_args,
        # Task-specific configs
        DEFAULT_EVA_CONFIG,
        DEFAULT_CLIP_CONFIG,
        EVA_DENOISING_CONFIGS,
        CLIP_DENOISING_CONFIGS
    )
    UNIVERSAL_CONFIG_AVAILABLE = True
    logger.info("✅ Universal configuration loaded successfully")
    logger.info("   ✅ Supports: Task-adaptive configs for EVA & CLIP denoising")
except ImportError as e:
    logger.error(f"❌ Universal config import failed: {e}")

# Export main components
__all__ = [
    # Universal component flags
    "UNIVERSAL_MODEL_AVAILABLE",
    "UNIVERSAL_LOSS_AVAILABLE",
    "UNIVERSAL_TRAINER_AVAILABLE",
    "UNIVERSAL_DATASET_AVAILABLE",
    "UNIVERSAL_CONFIG_AVAILABLE"
]

# UNIVERSAL denoising components (MAIN EXPORTS)
if UNIVERSAL_MODEL_AVAILABLE:
    __all__.extend([
        "UniversalDiTModel", "UniversalDiTConfig", "create_universal_model",
        "create_clip_denoising_model"
    ])

if UNIVERSAL_LOSS_AVAILABLE:
    __all__.extend([
        "UniversalSphericalFlowMatchingLoss", "create_universal_flow_loss",
        "create_clip_flow_loss"
    ])

if UNIVERSAL_TRAINER_AVAILABLE:
    __all__.extend([
        "UniversalDenoisingTrainer", "create_universal_trainer",
        "create_clip_denoising_trainer"
    ])

if UNIVERSAL_DATASET_AVAILABLE:
    __all__.extend([
        "UniversalDenoisingDataset", "create_universal_dataloaders", "universal_collate_fn",
        "create_clip_denoising_dataloaders"
    ])

if UNIVERSAL_CONFIG_AVAILABLE:
    __all__.extend([
        "BLIP3oDiTConfig", "get_blip3o_config", "create_config_from_args",
        "DEFAULT_EVA_CONFIG", "DEFAULT_CLIP_CONFIG",
        "EVA_DENOISING_CONFIGS", "CLIP_DENOISING_CONFIGS"
    ])

def check_environment():
    """Check if all required components are available"""
    universal_status = {
        'universal_model': UNIVERSAL_MODEL_AVAILABLE,
        'universal_loss': UNIVERSAL_LOSS_AVAILABLE,
        'universal_trainer': UNIVERSAL_TRAINER_AVAILABLE,
        'universal_dataset': UNIVERSAL_DATASET_AVAILABLE,
        'universal_config': UNIVERSAL_CONFIG_AVAILABLE,
    }

    all_universal_available = all(universal_status.values())

    if all_universal_available:
        logger.info("🎉 All UNIVERSAL denoising components loaded successfully!")
    else:
        missing = [name for name, available in universal_status.items() if not available]
        logger.error(f"❌ Missing universal components: {missing}")

    return {
        'universal_denoising': universal_status,
        'all_universal_available': all_universal_available,
    }

def get_version_info():
    """Get version and component information"""
    return {
        'blip3o_implementation': 'universal_denoising_v2',
        'main_task': 'universal_denoising',
        'supported_tasks': ['eva_denoising', 'clip_denoising'],
        'universal_components': {
            'model': UNIVERSAL_MODEL_AVAILABLE,
            'loss': UNIVERSAL_LOSS_AVAILABLE,
            'trainer': UNIVERSAL_TRAINER_AVAILABLE,
            'dataset': UNIVERSAL_DATASET_AVAILABLE,
            'config': UNIVERSAL_CONFIG_AVAILABLE,
        },
        'features': [
            'universal_task_support',
            'eva_denoising_4096d',
            'clip_denoising_1024d',
            'eva_conditioning_4096d',
            'spherical_flow_matching',
            'task_adaptive_architecture',
            'flexible_cross_attention',
            'proper_gradient_flow',
            'comprehensive_evaluation_metrics',
            # 'backward_compatibility', # Removed as it's no longer supported
        ]
    }

def get_recommended_components():
    """Get recommended components for different tasks"""
    return {
        'eva_denoising': {
            'description': 'EVA-CLIP denoising with spherical flow matching',
            'input': 'Noisy EVA embeddings [B, N, 4096]',
            'conditioning': 'Clean EVA embeddings [B, N, 4096]',
            'output': 'Clean EVA embeddings [B, N, 4096]',
            'components': {
                'model': 'create_universal_model(task_mode="eva_denoising")',
                'loss': 'create_universal_flow_loss()',
                'trainer': 'create_universal_trainer(task_mode="eva_denoising")',
                'dataset': 'create_universal_dataloaders(task_mode="eva_denoising")',
                'config': 'get_blip3o_config(task_mode="eva_denoising")',
            },
            'available': all([
                UNIVERSAL_MODEL_AVAILABLE,
                UNIVERSAL_LOSS_AVAILABLE,
                UNIVERSAL_TRAINER_AVAILABLE,
                UNIVERSAL_DATASET_AVAILABLE,
                UNIVERSAL_CONFIG_AVAILABLE
            ]),
            'recommended': True,
            'performance_target': 'Cosine similarity > 0.7 (excellent), > 0.5 (good)',
        },
        'clip_denoising': {
            'description': 'NEW: CLIP-ViT denoising with EVA conditioning',
            'input': 'Noisy CLIP embeddings [B, N, 1024]',
            'conditioning': 'Clean EVA embeddings [B, N, 4096]',
            'output': 'Clean CLIP embeddings [B, N, 1024]',
            'components': {
                'model': 'create_universal_model(task_mode="clip_denoising")',
                'loss': 'create_universal_flow_loss()',
                'trainer': 'create_universal_trainer(task_mode="clip_denoising")',
                'dataset': 'create_universal_dataloaders(task_mode="clip_denoising")',
                'config': 'get_blip3o_config(task_mode="clip_denoising")',
            },
            'available': all([
                UNIVERSAL_MODEL_AVAILABLE,
                UNIVERSAL_LOSS_AVAILABLE,
                UNIVERSAL_TRAINER_AVAILABLE,
                UNIVERSAL_DATASET_AVAILABLE,
                UNIVERSAL_CONFIG_AVAILABLE
            ]),
            'recommended': True,
            'performance_target': 'Cosine similarity > 0.6 (excellent), > 0.4 (good)',
        },
    }

def get_task_usage_examples():
    """Get usage examples for different tasks"""
    return {
        'eva_denoising': {
            'training_command': '''
# EVA Denoising Training
python train_universal_denoising.py \\
    --task_mode eva_denoising \\
    --chunked_embeddings_dir /path/to/embeddings \\
    --output_dir ./checkpoints_eva \\
    --model_size base \\
    --batch_size 8 \\
    --num_epochs 10
            ''',
            'python_code': '''
# EVA Denoising in Python
from src.modules import (
    create_universal_model, create_universal_flow_loss,
    create_universal_trainer, create_universal_dataloaders
)

model = create_universal_model(task_mode="eva_denoising", model_size="base")
loss_fn = create_universal_flow_loss()
train_dl, eval_dl = create_universal_dataloaders(
    "/path/to/embeddings", task_mode="eva_denoising"
)
trainer = create_universal_trainer(model, loss_fn, train_dl, eval_dl)
summary = trainer.train()
            '''
        },
        'clip_denoising': {
            'training_command': '''
# CLIP Denoising Training
python train_universal_denoising.py \\
    --task_mode clip_denoising \\
    --chunked_embeddings_dir /path/to/embeddings \\
    --output_dir ./checkpoints_clip \\
    --model_size base \\
    --batch_size 8 \\
    --num_epochs 10
            ''',
            'python_code': '''
# CLIP Denoising in Python
from src.modules import (
    create_universal_model, create_universal_flow_loss,
    create_universal_trainer, create_universal_dataloaders
)

model = create_universal_model(task_mode="clip_denoising", model_size="base")
loss_fn = create_universal_flow_loss()
train_dl, eval_dl = create_universal_dataloaders(
    "/path/to/embeddings", task_mode="clip_denoising"
)
trainer = create_universal_trainer(model, loss_fn, train_dl, eval_dl)
summary = trainer.train()
            '''
        }
    }

# Initialize on import
_status = check_environment()

# Priority messaging
if _status['all_universal_available']:
    logger.info("🎉 UNIVERSAL DENOISING components ready! (RECOMMENDED)")
    logger.info("   ✅ EVA Denoising: Noisy EVA [4096] → Clean EVA [4096]")
    logger.info("   ✅ CLIP Denoising: Noisy CLIP [1024] → Clean CLIP [1024] (EVA conditioning)")
    logger.info("   ✅ Features: Task-adaptive architecture, flexible cross-attention")
    logger.info("   ✅ Usage: create_universal_model(task_mode='eva_denoising|clip_denoising')")
else:
    logger.error("❌ UNIVERSAL DENOISING components missing!")
    logger.error("   Please ensure the following files are present:")
    logger.error("     - src/modules/models/blip3o_eva_dit.py")
    logger.error("     - src/modules/losses/blip3o_eva_loss.py")
    logger.error("     - src/modules/trainers/blip3o_eva_trainer.py")
    logger.error("     - src/modules/datasets/blip3o_eva_dataset.py")
    logger.error("     - src/modules/config/blip3o_config.py")

# Final status
if _status['all_universal_available']:
    logger.info("🎯 READY: Use universal denoising for both EVA and CLIP tasks!")
    logger.info("   🎯 EVA Denoising: --task_mode eva_denoising")
    logger.info("   🎯 CLIP Denoising: --task_mode clip_denoising")
else:
    logger.error("❌ CRITICAL: No denoising components available!")

# Print usage examples if available
if _status['all_universal_available']:
    examples = get_task_usage_examples()
    logger.info("\n📚 QUICK START EXAMPLES:")
    logger.info("   EVA Denoising: python train_universal_denoising.py --task_mode eva_denoising ...")
    logger.info("   CLIP Denoising: python train_universal_denoising.py --task_mode clip_denoising ...")