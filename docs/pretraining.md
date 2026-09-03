# Pretraining — SimMIM and MAE regimes

Per-scene, label-free pretraining of the **CoFFE** encoder (and of the **MFT**
architectural control), as used for the paper's Table 2.

> **Scope.** This file documents *what the code does*.
> [`PAPER_CANON.md`](../PAPER_CANON.md) is the source of truth for names and
> protocol constants; the per-cell recipes (mask rates, schedule, evaluated
> epoch) live in the `configs/{coffe,mft}/` headers, and the
> [README](../README.md#reproduce) maps each paper table to its command.

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

A regime is a **pair of mask rates**, not a separate code path (the table below
is the nominal pair; the 24 per-cell configs carry their own — see
"Reproducing a published cell"):

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
  | the other five **base** configs — `houston_simmim_hsi.yaml`, `{trento,muufl}_simmim{,_hsi}.yaml` | 0.9 | 0.0 | SimMIM band, at a rate **no Table 2 cell used** |

  The 5-seed significance experiment overrides the pair at runtime from the
  canonical run dir (`scripts/reproduce/sig_significance_config.py`). **Use the
  per-cell config to reproduce a specific cell** (below) — do not assume a
  config's name implies its regime.

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
# CoFFE, Houston, SimMIM token regime — the Table 2 cell recipe (OA 75.30)
python scripts/pretrain.py --config configs/coffe/houston_simmim_token.yaml

# resume
python scripts/pretrain.py --config configs/coffe/houston_simmim_token.yaml \
    --resume checkpoint_epoch_400.pth

# the MFT control, same objective
python scripts/pretrain.py --config configs/mft/houston_simmim_token.yaml
```

Configs live under `configs/coffe/` (CoFFE), `configs/mft/` (the control) and
`configs/hypersigma/` (label-free foundation-model adaptation, a different
script: `scripts/adapt_hypersigma.py` — see [`hypersigma.md`](hypersigma.md)).

From a notebook, [`coffe/runners/pretrain_runner.py`](../coffe/runners/pretrain_runner.py) wraps the
same entry point and writes everything under `experiments/<name>/`.

## Reproducing a published cell

Each of the 30 Table 2 cells has a config carrying **the exact recipe of the run
that produced its published mean** — generated from that run's frozen
`pretrain_config.yaml`, and stamped in its header with the cell it reproduces,
that cell's paper OA, the source run, and the **checkpoint epoch that was
evaluated**:

```
configs/coffe/<scene>_simmim_{band,token,band_token}[_hsi].yaml
configs/coffe/<scene>_mae[_hsi].yaml
configs/mft/<scene>_{simmim_token,mae}.yaml
```

Three things not to assume:

1. **The evaluated checkpoint is not the final one.** Schedules are 1500 epochs
   (Houston MAE, Houston MFT: 3000; one Houston SimMIM band run: 2000), but the
   published numbers come from epoch 950 on Houston, 975 on Trento/MUUFL, and
   800 for one cell. That was the tool's choice, not the authors': a checkpoint
   sort ordered filenames as strings, so `"...950"` beat `"...1500"`
   (PAPER_CANON §8 D17; the sort is numeric now). Train the full schedule, then
   evaluate the epoch the config names.
2. **Mask rates are per cell, not global** (D19, D20): read them off the config,
   never off the regime's nominal `(r_b, r_s)`.
3. **The ± column is a different experiment.** It is the across-seed std of
   5-seed runs pretrained fresh to 700 epochs
   (`scripts/reproduce/run_significance_experiment.py`), not a spread around the
   published mean. The six `configs/coffe/<scene>_simmim[_hsi].yaml` files are
   *that* runner's base configs — not cell recipes — and say so in their
   headers.

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

- [`docs/evaluation.md`](evaluation.md) — the evaluation protocol.
- [`docs/hypersigma.md`](hypersigma.md) — the foundation-model route, which
  pretrains differently (HyperSIGMA's own 75 % token masking).
- [`PAPER_CANON.md`](../PAPER_CANON.md) §3 (objectives), §8 (known
  discrepancies), §9 (config naming).
- [`tests/equivalence/`](../tests/equivalence/) — the goldens that pin the
  pretraining loss for every regime.
