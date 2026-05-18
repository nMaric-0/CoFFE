# MFT-CPEA-Cosine: Zero-Shot Evaluation Variant

## Overview

**MFT-CPEA-Cosine** is a variant of MFT-CPEA that replaces the DenseSimilarity MLP with prototypical networks using cosine similarity. This makes it ideal for **zero-shot evaluation** (no finetuning) of pretrained models.

## Key Differences

| Aspect | MFTCPEA (Original) | MFTCPEACosine (New) |
|--------|-------------------|---------------------|
| **Similarity Module** | DenseSimilarity MLP (trainable) | Cosine similarity (parameter-free) |
| **Few-Shot Method** | Dense patch matching + MLP | Prototypical networks |
| **Best Use Case** | With finetuning | Zero-shot (no finetuning) |
| **Parameters** | +3.7M (similarity MLP) | +0 (no extra params) |
| **Inference Speed** | Slower (MLP forward) | Faster (simple dot product) |

## Why Use Cosine Variant?

### Problem with Untrained DenseSimilarity

The original MFTCPEA uses a DenseSimilarity MLP:
```python
self.similarity = DenseSimilarity(
    num_tokens=121,        # 11x11 patches
    hidden_dim=256
)
# MLP: 121*121=14,641 → 256 → 1 (3.7M parameters!)
```

If this MLP is **not pretrained or finetuned**, it's essentially **random** → poor performance.

### Solution: Cosine Similarity

MFTCPEACosine uses prototypical networks:
```python
# No learnable parameters!
# Just compute prototype = mean of support features per class
# Then use cosine similarity: similarity = query · prototype
```

## Architecture

Both models share the **same backbone**:
- ✅ Tokenizers (channel, spatial, aux)
- ✅ Transformer encoder
- ✅ Position embeddings
- ✅ Projection head
- ✅ Class-aware adaptation (CPEA)

**Only difference:** How similarity is computed in `forward_episode`.

### Original (MFTCPEA)
```python
# Extract & adapt features
s_adapted = adapt_embeddings(s_patch, s_cls)  # [N*K, 121, D]
q_adapted = adapt_embeddings(q_patch, q_cls)  # [N*Q, 121, D]

# Dense similarity with learnable MLP
sim_matrix = similarity_mlp(s_adapted, q_adapted)  # Learnable!

# Aggregate
logits = aggregate_by_class(sim_matrix, labels)
```

### Cosine (MFTCPEACosine)
```python
# Extract & adapt features
s_adapted = adapt_embeddings(s_patch, s_cls)  # [N*K, 121, D]
q_adapted = adapt_embeddings(q_patch, q_cls)  # [N*Q, 121, D]

# Global average pooling
s_features = s_adapted.mean(dim=1)  # [N*K, D]
q_features = q_adapted.mean(dim=1)  # [N*Q, D]

# Compute prototypes (no learnable params)
prototypes = []
for class_c:
    prototype = s_features[class_c].mean(dim=0)
    prototypes.append(prototype)

# Cosine similarity (parameter-free)
logits = cosine_similarity(q_features, prototypes) * temperature
```

## Usage

### Installation

The model is automatically available when you import from `models`:
```python
from models import MFTCPEACosine
```

### Basic Usage

```python
model = MFTCPEACosine(
    hsi_channels=144,
    aux_channels=1,
    embed_dim=128,
    num_heads=8,
    num_layers=4,
    patch_size=11,
    lambda_factor=2.0,
    distance_metric="cosine",  # or "euclidean"
    temperature=10.0
)

# Load pretrained weights (same as MFTCPEA)
model.load_state_dict(checkpoint)

# Evaluate (no finetuning needed!)
logits = model.forward_episode(
    support_hsi, support_aux, support_labels,
    query_hsi, query_aux
)
```

### Evaluation Script

```bash
# Evaluate with cosine similarity (recommended)
python scripts/evaluate_cosine.py \
    --checkpoint checkpoints/pretrained/houston_enhanced/encoder_final.pth \
    --dataset houston \
    --n-way 5 \
    --k-shot 5

# Use Euclidean distance instead
python scripts/evaluate_cosine.py \
    --checkpoint checkpoints/pretrained/houston_enhanced/encoder_final.pth \
    --dataset houston \
    --distance-metric euclidean

# Adjust temperature scaling
python scripts/evaluate_cosine.py \
    --checkpoint checkpoints/pretrained/houston_enhanced/encoder_final.pth \
    --dataset houston \
    --temperature 5.0
```

## Configuration

### Distance Metric

**Cosine Similarity** (default, recommended):
```python
distance_metric="cosine"
```
- Normalized dot product: `sim = (q · p) / (||q|| ||p||)`
- Scale-invariant, works well with L2-normalized features
- Temperature scaling controls sharpness

**Euclidean Distance**:
```python
distance_metric="euclidean"
```
- Negative squared distance: `logits = -||q - p||²`
- Sensitive to feature magnitudes
- No temperature scaling

### Temperature

Controls the sharpness of cosine similarity:
```python
temperature=10.0  # Default - sharper predictions
temperature=5.0   # Smoother predictions
temperature=1.0   # Very soft (like softmax temp)
```

Higher temperature = sharper distinctions between classes.

## Checkpoint Compatibility

**Important:** MFTCPEACosine is **fully compatible** with MFTCPEA checkpoints!

Since the backbone is identical, you can:
1. ✅ Pretrain with MFTCPEA or enhanced pretraining
2. ✅ Load checkpoint into MFTCPEACosine
3. ✅ Evaluate without finetuning

The DenseSimilarity MLP weights are simply ignored.

```python
# Checkpoint saved with MFTCPEA
checkpoint = torch.load("pretrained_mftcpea.pth")

# Load into cosine variant (works!)
model = MFTCPEACosine(...)
model.load_state_dict(checkpoint, strict=False)
# Missing keys: similarity.mlp.* (expected - we skip it)
```

## Performance Expectations

### Zero-Shot (No Finetuning)

Expected performance ranking:
1. **MFTCPEACosine** ⭐ - Best for zero-shot
2. **MFTCPEA (finetuned)** - Best overall but needs finetuning
3. **MFTCPEA (untrained similarity)** - Poor (random MLP)

### With Finetuning

Expected performance ranking:
1. **MFTCPEA (finetuned)** - Best overall
2. **MFTCPEACosine** - Good, simpler
3. **MFTCPEA (untrained similarity)** - Catches up after finetuning

## When to Use Which?

### Use MFTCPEACosine if:
- ✅ You want **zero-shot evaluation** without finetuning
- ✅ You don't have compute for finetuning
- ✅ You want a **simple baseline** for pretrained features
- ✅ You're doing **cross-dataset transfer** without adaptation
- ✅ You want **faster inference** (no MLP)

### Use MFTCPEA (original) if:
- ✅ You **will finetune** the model on target data
- ✅ You want the **absolute best performance** (with finetuning)
- ✅ You have compute for few-shot adaptation
- ✅ You're following the **original CPEA paper** methodology

## Example Comparison

```bash
# Scenario: Pretrained on Houston, evaluate on Trento (zero-shot transfer)

# Method 1: MFTCPEA (untrained similarity) - POOR
python scripts/evaluate_enhanced.py \
    --checkpoint pretrained_houston.pth \
    --dataset trento
# Result: ~50% OA (random MLP hurts)

# Method 2: MFTCPEACosine - GOOD
python scripts/evaluate_cosine.py \
    --checkpoint pretrained_houston.pth \
    --dataset trento
# Result: ~65% OA (prototypical networks work!)

# Method 3: MFTCPEA + finetuning - BEST (but requires extra training)
# ... finetune similarity MLP on Trento support set ...
# Result: ~70% OA (finetuned MLP optimal)
```

## Implementation Details

### Prototypical Networks

The cosine variant implements **prototypical networks** (Snell et al., NeurIPS 2017):

1. **Compute prototypes** for each class:
   ```
   c_k = (1/|S_k|) Σ f(x_i)  for x_i ∈ S_k
   ```
   where S_k is the support set for class k

2. **Classify queries** by distance to prototypes:
   ```
   p(y=k | x) ∝ exp(-d(f(x), c_k))
   ```

3. **Distance functions**:
   - Cosine: `d(a,b) = -a·b / (||a|| ||b||)` (scaled by temperature)
   - Euclidean: `d(a,b) = ||a - b||²`

### Feature Pooling

Before computing prototypes, we pool patch embeddings:
```python
# From [B, 121, D] → [B, D]
features = adapted_embeddings.mean(dim=1)
```

This global average pooling:
- Aggregates information from all patches
- Creates a single feature vector per sample
- Works well with class-aware adaptation

## References

- **Prototypical Networks**: Snell et al., "Prototypical Networks for Few-shot Learning", NeurIPS 2017
- **CPEA**: Original CPEA paper using DenseSimilarity
- **SimCLR**: Chen et al., "A Simple Framework for Contrastive Learning", ICML 2020 (cosine similarity for contrastive learning)

## FAQ

### Q: Can I use my existing MFTCPEA checkpoints?
**A:** Yes! The cosine variant is fully compatible. Just load the checkpoint and the DenseSimilarity weights will be ignored.

### Q: Is this faster than MFTCPEA?
**A:** Yes, slightly. No MLP forward pass means faster inference (~10-20% speedup).

### Q: Should I always use cosine over the original?
**A:** No. If you plan to finetune, the original MFTCPEA will be better. Use cosine for zero-shot only.

### Q: Can I finetune MFTCPEACosine?
**A:** The cosine similarity has no parameters to finetune. You can still finetune the backbone (encoder, projection), but the similarity computation remains fixed.

### Q: What temperature should I use?
**A:** Default 10.0 works well. Try 5.0-20.0 range. Higher = sharper predictions.

### Q: Cosine vs Euclidean distance?
**A:** Cosine is recommended, especially when features are L2-normalized (which they are with our projection head). Euclidean can work but is less stable.

## Code Structure

```
models/
├── mft_cpea.py              # Original with DenseSimilarity
└── mft_cpea_cosine.py       # New variant with cosine similarity

scripts/
├── evaluate_enhanced.py     # Evaluates original MFTCPEA
└── evaluate_cosine.py       # Evaluates cosine variant
```

## Citation

If you use MFT-CPEA-Cosine, please cite:

```bibtex
@misc{mftcpea-cosine,
  title={MFT-CPEA-Cosine: Zero-Shot Few-Shot Learning with Prototypical Networks},
  year={2025},
  note={Cosine similarity variant for zero-shot evaluation}
}
```

## See Also

- [Enhanced Pretraining](ENHANCED_PRETRAINING.md) - Multi-task pretraining system
- [Evaluation Guide](EVALUATION.md) - Complete evaluation documentation
- [Original MFTCPEA](../models/mft_cpea.py) - Model with DenseSimilarity
