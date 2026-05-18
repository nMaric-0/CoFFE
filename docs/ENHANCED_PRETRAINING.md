# Enhanced Masked Modeling for MFT-CPEA

## Overview

The **Enhanced Masked Spectral-Spatial Model** extends the original MFT-CPEA pretraining with comprehensive multi-task reconstruction objectives. This approach combines multiple self-supervised learning strategies to create more robust and generalizable representations.

## Features

The enhanced pretraining includes four complementary reconstruction tasks:

### 1. **Spatial Token Masking** (Weight: 1.0)
- **Purpose**: Learn spatial context and relationships
- **Method**: Randomly mask 60% of spatial tokens (patches)
- **Target**: Reconstruct HSI pixel values at masked positions
- **Implementation**: Only visible tokens pass through encoder (efficient!)

### 2. **Spectral Band Masking** (Weight: 0.5)
- **Purpose**: Learn spectral correlations across bands
- **Method**: Randomly mask 50% of HSI spectral bands
- **Target**: Reconstruct missing bands from visible bands
- **Benefit**: Forces model to understand spectral signatures

### 3. **LiDAR Reconstruction** (Weight: 0.3)
- **Purpose**: Learn elevation-spectral relationships
- **Method**: Mask 50% of LiDAR spatial tokens
- **Target**: Reconstruct LiDAR values from HSI context
- **Innovation**: Multimodal reconstruction (not just using LiDAR as context)

### 4. **Denoising Autoencoding** (Weight: 0.2)
- **Purpose**: Learn robust features invariant to noise
- **Method**: Add Gaussian noise (σ=0.1) to 50% of samples
- **Target**: Reconstruct clean HSI from noisy input
- **Benefit**: Improves robustness and generalization

## Architecture

```
Input: HSI [B, 144, 11, 11] + LiDAR [B, 1, 11, 11]
  │
  ├─► Gaussian Noise (σ=0.1, p=0.5)
  │
  ├─► Spectral Masking (50% bands → 0)
  │
  ├─► LiDAR Masking (50% tokens → mask_value)
  │
  ├─► Tokenization → [B, 121, D]
  │
  ├─► Spatial Masking (60% tokens masked)
  │
  ├─► Encoder: [CLS, AUX, 48 visible tokens]
  │     └─► Transformer + Projection Head
  │
  ├─► Decoder Branch 1: Spatial Decoder
  │     └─► Reconstruct HSI @ masked positions
  │
  ├─► Decoder Branch 2: Spectral Decoder
  │     └─► Reconstruct missing bands
  │
  ├─► Decoder Branch 3: LiDAR Decoder
  │     └─► Reconstruct LiDAR from AUX token
  │
  └─► Decoder Branch 4: Denoise Decoder
        └─► Reconstruct clean HSI from noisy

Loss = 1.0 * L_spatial + 0.5 * L_spectral + 0.3 * L_lidar + 0.2 * L_denoise
```

## Key Components

### GaussianNoiseAugmentation
```python
GaussianNoiseAugmentation(
    noise_std=0.1,           # Standard deviation
    apply_probability=0.5    # Apply to 50% of samples
)
```

### LiDARMasking
```python
LiDARMasking(
    num_tokens=121,          # 11x11 spatial tokens
    aux_channels=1,          # LiDAR channels
    patch_size=11,
    mask_ratio=0.5           # Mask 50% of tokens
)
```

### EnhancedMaskedSpectralSpatialModel
```python
EnhancedMaskedSpectralSpatialModel(
    encoder=encoder,                    # MFT-CPEA encoder
    hsi_channels=144,
    aux_channels=1,
    spatial_mask_ratio=0.6,
    spectral_mask_ratio=0.5,
    lidar_mask_ratio=0.5,
    noise_std=0.1,
    noise_probability=0.5,
    loss_weights={                      # Multi-task weights
        "spatial": 1.0,
        "spectral": 0.5,
        "lidar": 0.3,
        "denoise": 0.2
    }
)
```

## Usage

### 1. Training

```bash
# Single dataset (Houston)
python scripts/pretrain_enhanced.py \
    --config configs/pretrain/houston_pretrain_enhanced.yaml

# Resume from checkpoint
python scripts/pretrain_enhanced.py \
    --config configs/pretrain/houston_pretrain_enhanced.yaml \
    --resume checkpoints/pretrained/houston_enhanced/checkpoint_epoch_400.pth

# Combined datasets
python scripts/pretrain_enhanced.py \
    --config configs/pretrain/houston_trento_pretrain_enhanced.yaml
```

### 2. Configuration

Key parameters in config file:

```yaml
pretrain:
  # Masking ratios
  spatial_mask_ratio: 0.6      # 60% spatial tokens
  spectral_mask_ratio: 0.5     # 50% spectral bands
  lidar_mask_ratio: 0.5        # 50% LiDAR tokens

  # Noise augmentation
  noise_std: 0.1               # Gaussian noise std
  noise_probability: 0.5       # Apply to 50% samples

  # Loss weights (tune for your dataset!)
  loss_weights:
    spatial: 1.0
    spectral: 0.5
    lidar: 0.3
    denoise: 0.2
```

### 3. Loading Pretrained Encoder

```python
import torch
from models import MFTCPEACosine

# Load pretrained checkpoint
checkpoint = torch.load("checkpoints/pretrained/houston_enhanced/encoder_final.pth")

# Create encoder
encoder = MFTCPEACosine(
    hsi_channels=144,
    aux_channels=1,
    embed_dim=128,
    # ... other params
)

# Load weights
encoder.load_state_dict(checkpoint["encoder_state_dict"])

# Use for few-shot learning or fine-tuning
```

## Advantages Over Standard Pretraining

| Aspect | Standard | Enhanced |
|--------|----------|----------|
| **Reconstruction Tasks** | 1 (spatial only) | 4 (spatial + spectral + LiDAR + denoise) |
| **LiDAR Usage** | Context only | Context + reconstruction |
| **Noise Robustness** | None | Denoising objective |
| **Spectral Learning** | Implicit | Explicit via band masking |
| **Multimodal Fusion** | One-way (LiDAR→HSI) | Bidirectional |
| **Training Signal** | Single loss | Multi-task losses |

## Expected Benefits

1. **Better Spectral Understanding**: Explicit band reconstruction forces model to learn spectral correlations
2. **Stronger Multimodal Fusion**: LiDAR reconstruction creates bidirectional learning
3. **Improved Robustness**: Denoising augmentation reduces overfitting and improves generalization
4. **Richer Representations**: Multi-task learning creates more diverse feature spaces
5. **Transfer Learning**: Better pretraining → better few-shot performance

## Loss Weight Tuning

The loss weights can be adjusted based on your priorities:

```python
# Emphasize spatial reconstruction (default)
loss_weights = {"spatial": 1.0, "spectral": 0.5, "lidar": 0.3, "denoise": 0.2}

# Emphasize spectral learning (for hyperspectral classification)
loss_weights = {"spatial": 0.8, "spectral": 1.0, "lidar": 0.3, "denoise": 0.2}

# Emphasize multimodal fusion (for LiDAR-rich datasets)
loss_weights = {"spatial": 1.0, "spectral": 0.4, "lidar": 0.8, "denoise": 0.2}

# Balance all tasks equally
loss_weights = {"spatial": 1.0, "spectral": 1.0, "lidar": 1.0, "denoise": 1.0}
```

### Dynamic Weight Adjustment

You can also adjust weights during training:

```python
# Start with spatial emphasis
pretrain_model.set_loss_weights({"spatial": 1.0, "spectral": 0.3})

# After 200 epochs, emphasize spectral
pretrain_model.set_loss_weights({"spectral": 1.0, "spatial": 0.5})
```

## Training Schedule

Recommended schedule (800 epochs):

- **Epochs 1-40**: Warmup (linear LR increase)
- **Epochs 40-600**: Main training (cosine decay)
- **Epochs 600-800**: Fine-tuning (low LR plateau)

Monitor validation losses for all tasks:
- If `loss_spatial` plateaus early → increase `spatial_mask_ratio`
- If `loss_spectral` too high → increase weight or decrease mask ratio
- If `loss_lidar` dominates → decrease weight
- If `loss_denoise` near zero → increase `noise_std`

## Comparison with Original

```python
# Original: Single task
loss = MSE(predicted_pixels, target_pixels)  # Only masked positions

# Enhanced: Multi-task
loss = (1.0 * MSE_spatial +      # Masked pixel reconstruction
        0.5 * MSE_spectral +     # Masked band reconstruction
        0.3 * MSE_lidar +        # LiDAR reconstruction
        0.2 * MSE_denoise)       # Denoising reconstruction
```

## Files

- **Model**: `pretrain/masked_modeling_enhanced.py`
- **Training Script**: `scripts/pretrain_enhanced.py`
- **Config**: `configs/pretrain/houston_pretrain_enhanced.yaml`
- **Tests**: `tests/test_pretrain_enhanced.py`

## Testing

```bash
# Run unit tests
python tests/test_pretrain_enhanced.py

# Quick training test (10 epochs)
python scripts/pretrain_enhanced.py \
    --config configs/pretrain/houston_pretrain_enhanced.yaml \
    --epochs 10
```

## Citation

If you use this enhanced pretraining approach, please cite:

```bibtex
@article{mft-cpea-enhanced,
  title={Enhanced Masked Modeling for Multimodal Remote Sensing},
  year={2025},
  note={Extension of MFT-CPEA with multi-task reconstruction}
}
```

## References

- **MAE**: He et al., "Masked Autoencoders Are Scalable Vision Learners", CVPR 2022
- **SS-MAE**: Lin et al., "Spatial-Spectral Masked Autoencoder", NeurIPS 2023
- **Denoising AE**: Vincent et al., "Extracting and Composing Robust Features", ICML 2008
- **Multi-Task Learning**: Caruana, "Multitask Learning", Machine Learning 1997
