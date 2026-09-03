# The HyperSIGMA route — label-free adaptation of a foundation model

The paper's third route: take the released HyperSIGMA ViT-Base checkpoints, put
them in front of the *same* frozen-encoder few-shot protocol as CoFFE, and see
how far a general hyperspectral foundation model gets on these three scenes.
This file documents the mechanics; [`PAPER_CANON.md`](../PAPER_CANON.md) §1 and
§6 are the source of truth for names and for the Table 3 numbers.

Nothing here uses labels for adaptation, and nothing is finetuned on the
few-shot task: adaptation is continued **masked reconstruction** on the target
scene's unlabelled patches, and evaluation is the frozen-encoder
[Euclidean NCM protocol](EVAL_PROTOCOL.md).

## Getting the checkpoints

```bash
bash scripts/download_hypersigma_checkpoints.sh    # -> checkpoints/hypersigma/
```

Two files, ~1.4 GB each, from
[WHU-Sigma/HyperSIGMA](https://huggingface.co/WHU-Sigma/HyperSIGMA): the
SpatViT-B and SpecViT-B ImageClassification MAE checkpoints. The filenames the
script asks for are the ones current at the time the *sources* were vendored
(upstream commit `c5981e0`, recorded in `third_party/HyperSIGMA/NOTICE`); the
script's `SPAT_URL` / `SPEC_URL` environment variables override them if the
HuggingFace tree moves. It prints their SHA256 and skips files that already
exist. They are **not** in git.

The spatial branch expects a 100-channel input. Houston (144 bands) gets a real
PCA; Trento (63) and MUUFL (64) cannot, so the dual encoder spectrally resamples
their raw bands up to 100 (`spat_resample_to: 100`):

```bash
python scripts/fit_pca_hypersigma.py --dataset houston --n-components 100
# -> checkpoints/hypersigma/pca_houston_100band.pkl (+ _stats.pkl)
```

The spectral branch ingests raw bands and handles any band count through its
built-in `AdaptiveAvgPool1d`.

## Two input regimes

The mismatch this route has to solve is geometric: HyperSIGMA was pretrained on
64×64 inputs, and these scenes are 11×11 patches.

| Regime | What it does | Flags |
|---|---|---|
| **`backbone_native`** (64×64) | keeps both encoders at their pretrained geometry (SpatViT 64×64 / patch-8 / 100 ch, SpecViT 64×64), so the pretrained input projections load, and fits the 11×11 patch up to 64×64 — either by bicubic **upscale** or by zero-**pad** into a centred canvas | `--native-geometry --input-fit {upscale,pad}` |
| **`patch_native`** (11×11) | keeps the data's geometry and re-initialises the input projections (SpatViT patch-3), so the pretrained bodies see 11×11 tokens | the default (no `--native-geometry`) |

## Adaptations

`model.adapt_mode` in the config selects what sees gradients. The transformer
bodies are **always frozen**; adaptation touches the pieces that are randomly
initialised for this geometry plus the decoders. Masking is HyperSIGMA's own
75 % token masking with per-patch z-scored reconstruction targets, and the
schedule is 2000 epochs (≈36 h per scene on one RTX 4090 —
[`PAPER_CANON.md`](../PAPER_CANON.md) §3).

| Canon name | `adapt_mode` | Trains | Loss |
|---|---|---|---|
| `frozen` | — (no adaptation) | nothing | — |
| `spatial` | `spatial_only` | SpatViT `patch_embed`, `pos_embed`, deformable `sampling_offsets`, mask tokens, spatial decoder | L_spat |
| `spectral` | `spectral_only` | SpecViT `spat_map`, `pos_embed`, `l1`, deformable `sampling_offsets`, mask tokens, spectral decoder | L_spec |
| `joint_sem` | `joint_sem` | both branches' pieces + the SEM + three decoders | L_spat + L_spec + L_fused (uniform) |
| `sem_only` | `sem_only` | the fusion path only — SEM, fused decoder, spectral `l1`, mask tokens — on top of fully loaded native-geometry encoders | L_fused |

Of HyperSIGMA's ~180M parameters (both ViT-Base bodies plus the SEM), the
label-free adaptation trains 1.13M–8.45M depending on the mode
([`PAPER_CANON.md`](../PAPER_CANON.md) §6; `docs/refactor/LOG.md`'s phase-7
S5 entry records why the `11×11 / spectral` cell falls below that range).

```bash
# 11x11 joint+SEM adaptation, Houston: the 100-band config, which is the one
# the published cell used (the run overrode its schedule - see below)
python scripts/adapt_hypersigma.py \
    --config configs/hypersigma/houston_patchnative_pca100_joint_sem.yaml

# 64x64 pad, SEM-only, all three scenes
bash scripts/reproduce/run_native_sem_pad_experiments.sh
```

Each config's header states which Table 3 cell it reproduces, or says plainly
that it reproduces none as written — `<scene>_patchnative_joint_sem.yaml` is in
the second category: it is the **3-band** spatial front-end, and every published
11×11 cell ran at 100 bands.

## Evaluation

```bash
# 64x64 upscale / frozen / spatial features — Table 3's Houston 61.14 cell
python scripts/evaluate_hypersigma.py \
    --dataset houston --native-geometry --input-fit upscale \
    --mode spat_pool --adapted-checkpoint none \
    --split all --k-query 100 --num-episodes 2000

# fused SEM features from an adapted checkpoint. NOTE this example is the
# 3-band front-end (pca_houston_3band + the houston_k3 adapt dir), which
# reproduces no published cell; the Table 3 11x11 joint+SEM cell needs the
# 100-band run's checkpoint.
python scripts/evaluate_hypersigma.py \
    --dataset houston --mode fused \
    --adapted-checkpoint checkpoints/hypersigma_adapted/houston_k3/checkpoint.pth \
    --pca-spat-path checkpoints/hypersigma/pca_houston_3band.pkl \
    --split all --k-query 100 --num-episodes 2000
```

`--mode` picks the feature: `spat_pool` (spatial branch, 768-d), `spec_pool`
(spectral branch, 768-d) or `fused` (SEM output, 512-d). The literal
`--adapted-checkpoint none` is the unadapted (frozen) ablation.

Three things to know about this evaluator, all of them recorded in
[`PAPER_CANON.md`](../PAPER_CANON.md) §8 D18:

1. **Table 3 used 2000 episodes**, not Table 2's 1000 — and this route's own
   defaults match neither: `--k-query 30 --num-episodes 600 --split test`, which
   mirror `configs/hypersigma/houston_eval.yaml`'s constants and reproduce no
   published cell. Phase 7 aligned `coffe/eval/episodic.py`'s defaults with the
   canon but not this evaluator's, so **every published-cell command must pass
   `--k-query 100 --num-episodes 2000 --split all` explicitly**.
2. **One cell used `k_query=30`** — Houston 11×11 spectral (21.85 ± 0.10). The
   rest used 100.
3. The evaluator accumulates **both** a cosine and a Euclidean confusion matrix
   from the same features, so `results.json` carries parallel blocks. **The
   paper quotes the Euclidean block throughout**, including for the one cell
   whose `eval_config.json` records `distance_metric: cosine`.

`--n-way` defaults to the scene's full class count, which is the paper setting.

## Which command produced which Table 3 cell

**Two** of Table 3's eight rows have a committed driver end to end, **one** is
partial, and **five** have none. The full row-by-row map, including what the
gaps are, is in
[`scripts/reproduce/README.md`](../scripts/reproduce/README.md), and the
cell-by-cell provenance trace (which run, which evaluation directory, which
protocol) is [`docs/refactor/AUDIT.md`](refactor/AUDIT.md) §3 D2. In short:

- **committed** (2 rows): 64×64 pad SEM-only, and 11×11 spatial (a config base
  plus the `spatial_only` / 100-band overrides its driver applies in code).
- **partial** (1 row): 11×11 joint+SEM. All three cells ran at 100 bands.
  Houston has the matching config, `houston_patchnative_pca100_joint_sem.yaml`,
  but the run overrode its schedule (2000 epochs / batch 128 / lr 1e-5 /
  min_lr 5e-7 against the file's 3000 / 64 / 1.5e-4 / 1e-6); Trento and MUUFL
  have only the 3-band config and added `spat_resample_to: 100` plus their own
  schedule at launch. For all three,
  `experiments/hypersigma_adapt_<scene>_pca100_joint_sem_run1/pretrain_overrides.yaml`
  is the authority (PAPER_CANON §8 D13).
- **not committed** (5 rows): the four 64×64 frozen rows (12 cells) and 11×11
  spectral (3 cells), driven from `notebooks/evaluate_hypersigma_native.ipynb`
  and `notebooks/evaluate_hypersigma.ipynb`. The frozen rows involve no
  training, so the command above reproduces them from flags alone; the spectral
  cells' adaptation settings were never reconstructed into a config,
  deliberately — inventing settings for a frozen paper run is worse than naming
  the notebook that produced it.

No `backbone_native` adaptation was ever completed for Trento or MUUFL; that is
a coverage gap in the paper's sweep, not a missing file here.

## See also

- [`docs/EVAL_PROTOCOL.md`](EVAL_PROTOCOL.md) — the shared protocol.
- [`results/README.md`](../results/README.md) — the JSONs behind Table 3.
- [`third_party/HyperSIGMA/`](../third_party/HyperSIGMA) — the vendored upstream
  sources under their own Apache-2.0 LICENSE, plus the `NOTICE` that lists the
  local patches applied to them (an `mmengine` import shim, a `patch_size == 3`
  FPN branch, relative-import fixes).
