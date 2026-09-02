# Pretraining — SimMIM and MAE regimes

Per-scene, label-free pretraining of the **CoFFE** encoder (and of the **MFT**
architectural control), as used for the paper's Table 2.

> **Status.** This file was rewritten in phase 4 of the repository cleanup to
> carry the paper's vocabulary and to drop claims that no longer describe the
> code. It documents *what the code does*; the full reproduction recipe
> (per-cell mask rates, checkpoint epochs, schedules) lands in phase 8.
> `PAPER_CANON.md` is the source of truth for names and protocol constants.

## The two objectives

Both operate on the same token space: an 11×11 patch becomes 121 pixel tokens,
each carrying all HSI bands plus the auxiliary (LiDAR) bands concatenated at the
input, with one class-agnostic token prepended.

| Objective | `pretrain.objective` | What it does | Code |
|---|---|---|---|
| **SimMIM** | `"simmim"` | In-place masking, nothing is dropped: masked entries are replaced (band masking with a learnable per-channel fill, token masking with a learnable mask token) and an MLP decoder reconstructs the full per-pixel band vector. Loss is MSE over the **union** of the two masks. | [`coffe/pretrain/simmim.py`](../coffe/pretrain/simmim.py) |
| **MAE** | `"mae"` | Token-drop recipe (He et al.): 75 % of pixel tokens are removed before encoding, and a transformer decoder reconstructs them. No projection head. | [`coffe/pretrain/mae_pretrain.py`](../coffe/pretrain/mae_pretrain.py) |

The MFT control supports the same two objectives through
[`coffe/pretrain/mft_spatial_mae.py`](../coffe/pretrain/mft_spatial_mae.py) (SimMIM token)
and [`coffe/pretrain/mft_mae.py`](../coffe/pretrain/mft_mae.py) (MAE).

Frozen configs on the authors' machines spell the SimMIM objective
`"enhanced"`; that value is still accepted and mapped by
[`coffe/compat.py`](../coffe/compat.py) (PAPER_CANON §7.3).

## The three SimMIM regimes

A regime is a **pair of mask rates**, not a separate code path:

| Regime | `band_mask_ratio` | `spatial_mask_ratio` | Legacy id |
|---|---|---|---|
| SimMIM band | 0.85 | 0 | `spectral` |
| SimMIM token *(headline)* | 0 | 0.75 | `spatial` |
| SimMIM band+token | 0.85 | 0.75 | `both` |

Two caveats the audit established, both recorded in PAPER_CANON §8:

- **D19** — Table 2's Houston band+token cell was pretrained at
  `band_mask_ratio: 0.75`, not 0.85. The rate that produced a given cell is the
  one in that run's frozen `pretrain_config.yaml`, not the table above.
- The per-scene configs under `configs/coffe/` are **base** configs, not
  per-cell recipes, and they do not all carry the same regime:

  | Config | `band_mask_ratio` | `spatial_mask_ratio` | i.e. |
  |---|---|---|---|
  | `configs/coffe/houston_simmim.yaml` | 0.0 | 0.75 | SimMIM token |
  | every other `configs/coffe/*_simmim*.yaml` (Houston HSI-only, both Trento, both MUUFL) | 0.9 | 0.0 | SimMIM band, at a rate **no Table 2 cell used** |

  The 5-seed significance experiment overrides the pair at runtime from the
  canonical run dir (`scripts/reproduce/sig_significance_config.py`). **Set both rates
  explicitly to reproduce a specific cell** — do not assume a config's name
  implies its regime.

## Other knobs that matter

- `recon_center_sigma` — Gaussian centre weighting of the reconstruction loss,
  normalised to **mean one**, up-weighting the pixel that will be classified.
  `null` disables it (the MAE configs).
- `use_projection` — the projection head is a pretraining-only part. The MAE
  path forces it off; evaluation discards it in every paper run.
- `lambda_factor` — read at *evaluation* time, not during pretraining. It is the
  weight of the class-agnostic token in the eval feature
  (`CoFFE.eval_patch_embeddings`, PAPER_CANON §8 D3); the configs carry it so
  that eval can rebuild the architecture from the pretraining run.

## Running it

```bash
# CoFFE, Houston, SimMIM token regime
python scripts/pretrain.py --config configs/coffe/houston_simmim.yaml

# resume
python scripts/pretrain.py --config configs/coffe/houston_simmim.yaml \
    --resume checkpoint_epoch_400.pth

# the MFT control, same objective
python scripts/pretrain.py --config configs/mft/houston_simmim_token.yaml
```

Configs live under `configs/coffe/` (CoFFE), `configs/mft/` (the control) and
`configs/hypersigma/` (label-free foundation-model adaptation, a different
script: `scripts/adapt_hypersigma.py`).

From a notebook, [`coffe/runners/pretrain_runner.py`](../coffe/runners/pretrain_runner.py) wraps the
same entry point and writes everything under `experiments/<name>/`.

## Loading a pretrained encoder

```python
import torch
from coffe.models import CoFFE

model = CoFFE(
    hsi_channels=144,
    aux_channels=1,
    embed_dim=128,
    num_heads=2,
    num_layers=2,
    patch_size=11,
    use_projection=False,
)  # eval discards the head
ckpt = torch.load("experiments/<run>/checkpoints/checkpoint_epoch_950.pth", map_location="cpu")
# Pretraining wrappers save the encoder under an `encoder.` prefix; the
# evaluator's fix_state_dict_keys strips it. By hand:
state = {
    k[len("encoder.") :]: v for k, v in ckpt["model_state_dict"].items() if k.startswith("encoder.")
}
model.load_state_dict(state, strict=False)
```

`scripts/evaluate.py` does this for you, including the encoder-prefix key
mapping (`fix_state_dict_keys`). Checkpoints written before the phase-4 rename
load unchanged: class and file names changed, `nn.Module` attribute names — the
state_dict keys — did not (PAPER_CANON §7.2).

## See also

- [`docs/EVAL_PROTOCOL.md`](EVAL_PROTOCOL.md) — the evaluation protocol.
- [`PAPER_CANON.md`](../PAPER_CANON.md) §3 (objectives), §8 (known
  discrepancies), §9 (config naming).
- [`tests/equivalence/`](../tests/equivalence/) — the goldens that pin the
  pretraining loss for every regime.
