# Classification maps — putting the protocol's decisions back on the scene

The evaluation in [`docs/evaluation.md`](evaluation.md) reports numbers, not
pictures, and two facts about this repository stand between those numbers and a
map next to the ground truth:

1. **The packs carry no coordinates.** `data/raw/<Scene>11x11/` holds
   `HSI_Tr.mat` `[N, 11, 11, C]`, `LIDAR_*.mat`, and a flat `[1, N]` label
   vector per split — nothing records which scene pixel sample `i` was cut
   around. The extraction happened upstream in the MFT data preparation, which
   kept only the patch stack.
2. **The protocol gives each pixel many decisions, not one.** PAPER_CANON §4
   draws 1000–2000 episodes, each with its own 5-shot support, so a query pixel
   is classified many times under many different prototypes. That is what the
   OA and its CI average over; a map needs a single decision per pixel.

Two entry points close those gaps, in order.

## 1. Recover the coordinates

```bash
python scripts/recover_patch_coords.py --dataset houston \
    --gt-train /path/TRLabel.mat --gt-test /path/TSLabel.mat \
    --out results/coords_houston.npz --report results/coords_houston.json
```

**You supply the masks.** They are not in the packs and not in this repository:
Houston's are the DFC2013 `TRLabel` / `TSLabel` rasters (349×1905), Trento's its
166×600 ground truth, MUUFL's the 325×220 one — from the owners named in the
[README](../README.md#data).

**How it can be sure.** MFT cut one patch per labelled pixel by scanning the
mask, so the pack's label vector is a fingerprint of that scan. The script
enumerates the mask's labelled pixels in each candidate order — MATLAB's
column-major `find` order, C row-major, and each of those grouped by class — and
keeps the order whose label sequence reproduces the pack's *element for
element*. Over Houston's 2832 + 12197 entries, an accidental match of a
15-valued sequence does not happen, so a match is proof, and the enumeration
that produced it is the missing index → (row, col) table.

**When the labels are not enough.** A label sequence pins which pixels, not
always the walk order: the two class-grouped candidates produce the same
sequence by construction, and a class-contiguous mask can make a raster order
coincide with them. If the matching orders enumerate the same pixels in the same
sequence, that is not an ambiguity and recovery proceeds; if they genuinely
disagree, the script fails (exit 1) rather than guessing. Settle it with the
scene's own cube:

```bash
python scripts/recover_patch_coords.py --dataset houston \
    --gt-train /path/TRLabel.mat --gt-test /path/TSLabel.mat \
    --hsi /path/Houston.mat --max-checks 512        # then --order-train ...
```

Patch `i`'s centre pixel must equal `cube[row, col]`. Border padding never
touches a patch centre, so the check holds at the scene edge too — worth running
once even when a single order matched, since it is the only test that involves
the pixels themselves rather than their labels.

The table is written in the **`split="all"` index space** — train pack first,
then test, exactly the concatenation `PatchedMultimodalDataset` performs — which
is the index space the episode sampler and the feature cache use. A table
recovered for one split cannot be used with another; `predict_map.py` refuses
the mismatch.

## 2. Classify every pixel once, and paint

```bash
python scripts/predict_map.py \
    --checkpoint experiments/<run>/checkpoints/checkpoint_epoch_950.pth \
    --dataset houston --coords results/coords_houston.npz \
    --out-dir results/maps/houston --seeds 42 123 456
```

The encoder is frozen, so the scene is encoded **once**
([`coffe.eval.feature_cache`](../coffe/eval/feature_cache.py)) and the pass then
indexes that matrix: support drawn once at the seed through the episode
sampler's own `fixed_support` path, N = every class in the scene, and every
remaining labelled sample classified exactly once by the same
nearest-class-mean the evaluators use
([`coffe.eval.prediction_map`](../coffe/eval/prediction_map.py)).

Per seed it writes:

| File | What it holds |
|---|---|
| `decisions_<scene>_seed<N>.npz` | support / query dataset indices, truth and prediction per query |
| `maps_<scene>_seed<N>.npz` | `truth`, `prediction`, `support`, `agreement` canvases `[H, W]` |
| `map_<scene>_seed<N>.png` | the three panels with the scene's pinned class palette |
| `summary_<scene>.json` | OA / AA / Kappa of the pass, per-class recall, the config |

`agreement` is 1 correct, 2 wrong, 3 support, 0 unlabelled — the panel that
shows *where* the errors are rather than how many.

Without `--coords` the pass still runs and the decisions are saved; only the
painting is skipped. That is the useful mode until the masks are in hand.

## Two things the map is not

**It is not wall-to-wall.** Only labelled pixels have a patch in the packs —
15,029 of Houston's 664,845 — so every canvas is a masked map. Classifying the
unlabelled remainder would mean cutting patches for every pixel, which needs the
full scene cube, not the packs.

**Its OA is not a paper cell.** One support draw over the whole labelled pool is
a single sample of the distribution Table 2 and Table 3 average over, and a
single pass has no ± column. Expect it inside the published CI, not equal to the
published mean; `--seeds` paints several draws so the spread is visible. The
published numbers stay the ones the evaluators produce.

## See also

- [`docs/evaluation.md`](evaluation.md) — the protocol these decisions come from.
- [`coffe/data/scene_coords.py`](../coffe/data/scene_coords.py) — the recovery,
  the scan orders, and the cube check.
- `tests/unit/test_scene_coords.py`, `tests/integration/test_prediction_map.py`,
  `tests/integration/test_map_cli.py` — all synthetic; none needs `data/raw/`.
