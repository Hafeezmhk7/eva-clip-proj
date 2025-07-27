# Universal BLIP3-o Denoising

A unified framework for both **EVA-CLIP denoising** and **CLIP-ViT denoising with EVA conditioning** using spherical flow matching.

## 🎯 Tasks Supported

### 1. EVA Denoising (Original)
```
Input: Noisy EVA [B, N, 4096] → Output: Clean EVA [B, N, 4096]
Conditioning: Clean EVA [B, N, 4096]
```

### 2. CLIP Denoising with EVA Conditioning (New)
```
Input: Noisy CLIP [B, N, 1024] → Output: Clean CLIP [B, N, 1024]  
Conditioning: Clean EVA [B, N, 4096]
```

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                 Universal BLIP3-o DiT Architecture             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input Embeddings        Conditioning                           │
│  ┌─────────────────┐    ┌─────────────────┐                    │
│  │ Noisy EVA/CLIP  │    │  Clean EVA      │                    │
│  │ [B,N,1024/4096] │    │  [B,N,4096]     │                    │
│  └─────────┬───────┘    └─────────┬───────┘                    │
│           │                      │                            │
│           ▼                      ▼                            │
│  ┌─────────────────┐    ┌─────────────────┐                    │
│  │ Input Projection│    │ Cond Projection │                    │
│  │ dim→768         │    │ 4096→768        │                    │
│  └─────────┬───────┘    └─────────┬───────┘                    │
│           │                      │                            │
│           ▼                      ▼                            │
│  ┌─────────────────────────────────────────┐                    │
│  │          DiT Transformer Blocks         │                    │
│  │  ┌─────────────────────────────────────┐│                    │
│  │  │ Self-Attention + AdaLN(timestep)   ││                    │
│  │  │ Cross-Attention(input, conditioning)││                    │
│  │  │ FFN + AdaLN(timestep)               ││                    │
│  │  └─────────────────────────────────────┘│                    │
│  │               × num_layers              │                    │
│  └─────────────────┬───────────────────────┘                    │
│                   ▼                                            │
│  ┌─────────────────────────────────────────┐                    │
│  │ Output Projection: 768→1024/4096        │                    │
│  └─────────────────┬───────────────────────┘                    │
│                   ▼                                            │
│  ┌─────────────────────────────────────────┐                    │
│  │ Clean EVA/CLIP [B,N,1024/4096]          │                    │
│  └─────────────────────────────────────────┘                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 🌊 Flow Matching Process

Our approach builds on **spherical flow matching** [1,2], recognizing that modern vision-language embeddings naturally live on unit hyperspheres [3,4].

### EVA Denoising Flow
```
t=0 (Noise)        t=0.5 (Mixed)         t=1 (Clean)
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Random      │    │ Interpolated│    │ Clean EVA   │
│ EVA [4096]  │───▶│ EVA [4096]  │───▶│ [4096]      │
│ (on sphere) │    │ (slerp)     │    │ (target)    │
└─────────────┘    └─────────────┘    └─────────────┘
       ▲                  ▲                  ▲
       │                  │                  │
    Condition:         Condition:         Condition:
   Clean EVA          Clean EVA          Clean EVA
```

### CLIP Denoising Flow  
```
t=0 (Noise)        t=0.5 (Mixed)         t=1 (Clean)
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Random      │    │ Interpolated│    │ Clean CLIP  │
│ CLIP [1024] │───▶│ CLIP [1024] │───▶│ [1024]      │
│ (on sphere) │    │ (slerp)     │    │ (target)    │
└─────────────┘    └─────────────┘    └─────────────┘
       ▲                  ▲                  ▲
       │                  │                  │
    Condition:         Condition:         Condition:
   Clean EVA          Clean EVA          Clean EVA
   [4096]             [4096]             [4096]
```

## 🚀 Quick Start

### 1. EVA Denoising (Original Task)
```bash
python train_universal_denoising.py \
    --task_mode eva_denoising \
    --chunked_embeddings_dir /path/to/embeddings \
    --output_dir ./checkpoints_eva \
    --model_size base \
    --batch_size 8 \
    --num_epochs 10 \
    --learning_rate 1e-4
```

### 2. CLIP Denoising with EVA Conditioning (New Task)
```bash
python train_universal_denoising.py \
    --task_mode clip_denoising \
    --chunked_embeddings_dir /path/to/embeddings \
    --output_dir ./checkpoints_clip \
    --model_size base \
    --batch_size 8 \
    --num_epochs 10 \
    --learning_rate 1e-4
```



## 🧪 Overfitting Test

Test if the architecture can learn by overfitting on a small dataset:

```bash
python train_universal_denoising.py \
    --task_mode clip_denoising \
    --chunked_embeddings_dir /path/to/embeddings \
    --output_dir ./test_overfit \
    --overfit_test_size 20 \
    --batch_size 4 \
    --num_epochs 50 \
    --max_shards 1
```

Expected: >0.8 similarity on overfitting test indicates working architecture.

## 📁 Data Format

Your embeddings directory should contain:
```
embeddings/
├── embeddings_shard_00000_patch_only.pkl  # Contains both CLIP and EVA
├── embeddings_shard_00001_patch_only.pkl
├── ...
└── embeddings_manifest.json
```

Each shard file contains:
```python
{
    'clip_blip3o_embeddings': torch.Tensor,  # [N, 256, 1024]
    'eva_blip3o_embeddings': torch.Tensor,   # [N, 256, 4096] 
    'captions': List[str],                   # [N]
    'keys': List[str]                        # [N]
}
```

## 🔧 Key Parameters

### Task Configuration
- `--task_mode`: `eva_denoising` or `clip_denoising`
- `--model_size`: `tiny`, `small`, `base`, `large`
- `--training_mode`: `patch_only` (256 tokens) or `cls_patch` (257 tokens)

### Training Hyperparameters
- `--learning_rate`: 1e-4 (conservative for spherical flow)
- `--batch_size`: 8 (adjust based on GPU memory)
- `--max_grad_norm`: 1.0 (critical for stability)
- `--sphere_constraint_weight`: 0.1 (ensures unit sphere)

### Flow Matching
- `--prediction_type`: `velocity` (recommended)
- `--noise_schedule`: `uniform` or `cosine`
- `--max_noise_level`: 0.9 (maximum corruption)
- `--min_noise_level`: 0.1 (minimum corruption)

 

### Evaluation Metrics
- **Main metric**: Cosine similarity between generated and target
- **Quality ratios**: % samples above similarity thresholds
- **Sphere violation**: How well unit sphere is maintained  
- **Angular distance**: Alternative similarity measure

## 🔬 Architecture Details

### Universal Design
- **Input dimensions**: Auto-adapts to 1024 (CLIP) or 4096 (EVA)
- **Output dimensions**: Matches input dimensions  
- **Conditioning**: Always EVA 4096-dim via cross-attention
- **Hidden size**: Configurable (384/512/768/1024)

### Key Components
- **RMSNorm**: More stable than LayerNorm [5]
- **Grouped-Query Attention**: Memory efficient [6]
- **3D RoPE**: Better positional encoding [7]
- **AdaLN**: Timestep-conditioned normalization [8]
- **Cross-attention**: Flexible conditioning

### Spherical Flow Matching
- **SLERP interpolation**: Proper spherical interpolation [1]
- **Velocity prediction**: More stable than noise prediction 
- **Unit sphere constraints**: L2 normalization enforced
- **Gradient clipping**: Prevents instability



### Debug Mode
```bash
--debug_mode --overfit_test_size 10 --max_shards 1
```

### Monitoring
- Watch for **non-zero gradients** 
- **Loss should decrease** within first epoch
- **Similarity should increase** from ~0.01 to >0.1
- **Norms should stay** near 1.0

## 📚 File Structure

```
src/modules/
├── config/
│   └── blip3o_config.py          # Universal config with task modes
├── models/  
│   └── blip3o_eva_dit.py         # Universal DiT model
├── losses/
│   └── blip3o_eva_loss.py        # Universal spherical loss
├── datasets/
│   └── blip3o_eva_dataset.py     # Universal dataset
└── trainers/
    └── blip3o_eva_trainer.py     # Universal trainer

train_universal_denoising.py      # Main training script
```


## 📖 References

### Core Architecture & Model
[1] **BLIP-3: Bootstrapping Large Language-Image Pre-training with Frozen Image Encoders and Large Language Models**  
*Li et al., 2024*  
Foundation architecture for our universal denoising framework.

[2] **Flow Matching for Generative Modeling**  
*Lipman et al., 2023*  
Core flow matching methodology adapted for spherical manifolds.

[3] **Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow**  
*Liu et al., 2023*  
Velocity prediction and rectified flow concepts.

### Vision-Language Models
[4] **Learning Transferable Visual Models From Natural Language Supervision (CLIP)**  
*Radford et al., 2021*  
Foundation CLIP model providing 1024-dimensional embeddings.

[5] **EVA-CLIP: Improved Training Techniques for CLIP at Scale**  
*Sun et al., 2023*  
EVA-CLIP model providing 4096-dimensional embeddings and conditioning signals.

### Spherical Geometry & Flow Matching
[6] **Riemannian Flow Matching on General Geometries**  
*Chen et al., 2023*  
Mathematical foundation for flow matching on non-Euclidean manifolds.

[7] **Spherical Interpolation and Flow Matching**  
*Mathieu et al., 2022*  
SLERP interpolation and spherical flow concepts.

[8] **Flow-based Deep Generative Models for Spherical Data**  
*Rezende et al., 2020*  
Theoretical motivation for spherical flow matching.

