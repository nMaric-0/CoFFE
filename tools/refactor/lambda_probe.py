#!/usr/bin/env python3
"""Does ``lambda_factor`` change the paper's computed numbers?  (phase-1 D3 probe)

Answer, measured 2026-08-31 on the headline Houston checkpoint: **yes.**
Removing it flips 5.4 % of query predictions and moves OA by -1.70 pp.
Requires data/raw/ + a real checkpoint, so it is a `-m data` style check,
not part of the default battery.

Replicates scripts/evaluate_cosine.py's episode path exactly (CPU), for a real
Table 2 checkpoint, and compares lambda=0.5 (as it ran) against lambda=0.0
(as removal would give). Reports:

  * per-query prediction agreement  -- the sensitive test
  * per-episode OA, both settings
  * how much cls_emb actually varies across samples (the mechanism)

Euclidean NCM is translation-invariant, so if cls_emb were constant across
samples, lambda would provably contribute nothing.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.datasets.patched import HoustonPatchedDataset
from data.samplers.patched_episode_sampler import PatchedEpisodeSampler
from scripts.evaluate_cosine import load_model_with_checkpoint
from utils.seed import set_seed

REPO = Path(__file__).resolve().parents[2]
# The run behind Table 2's headline Houston cell (75.30), at its evaluated epoch.
CKPT = str(REPO / "experiments/houston_enhanced_spatial_mask_test_run1_seed52"
                  "/checkpoints/checkpoint_epoch_950.pth")
N_EPISODES = int(sys.argv[1]) if len(sys.argv) > 1 else 40

# Arch + eval params exactly as the run's eval_config.json records them.
MODEL_CFG = dict(name="mft_cpea", embed_dim=128, num_heads=2, num_layers=2,
                 patch_size=11, lambda_factor=0.5, dropout=0.1,
                 use_projection=False, proj_hidden_dim=512, proj_num_layers=1,
                 proj_l2_normalize=True, use_aux=True,
                 distance_metric="euclidean", temperature=10.0,
                 prototype_mode="mean_features", pool_sigma=None)

device = "cpu"
ds = HoustonPatchedDataset(data_root="./data/raw", split="all")
model = load_model_with_checkpoint(CKPT, "houston", MODEL_CFG, device)
model.eval()
print(f"model lambda_factor = {model.lambda_factor}, pool_sigma = {model.pool_sigma}, "
      f"projection = {type(model.projection).__name__}")

set_seed(42, deterministic=True)
sampler = PatchedEpisodeSampler(dataset=ds, n_way=15, k_shot=5, k_query=100,
                                num_episodes=N_EPISODES, seed=42)

agree = tot = 0
oa_l, oa_0 = [], []
cls_stats = []

with torch.no_grad():
    for i, ep in enumerate(sampler):
        if i >= N_EPISODES:
            break
        sh, sa, sl = ep["support_hsi"], ep["support_aux"], ep["support_labels"]
        qh, qa, ql = ep["query_hsi"], ep["query_aux"], ep["query_labels"]

        s_patch, s_cls, _ = model.forward_features(sh, sa)
        q_patch, q_cls, _ = model.forward_features(qh, qa)

        if i == 0:
            allcls = torch.cat([s_cls, q_cls], 0)
            mp = torch.cat([s_patch.mean(1), q_patch.mean(1)], 0)
            cls_stats = [
                allcls.mean(0).norm().item(),
                allcls.std(0).mean().item(),
                (allcls - allcls.mean(0)).norm(dim=1).mean().item(),
                (mp - mp.mean(0)).norm(dim=1).mean().item(),
            ]

        for lam, bag in ((0.5, oa_l), (0.0, oa_0)):
            sf = (s_patch + lam * s_cls.unsqueeze(1)).mean(1)
            qf = (q_patch + lam * q_cls.unsqueeze(1)).mean(1)
            logits = model._forward_mean_features(sf, sl, qf)
            p = logits.argmax(1)
            bag.append((p == ql).float().mean().item() * 100)
            if lam == 0.5:
                pred_l = p
            else:
                pred_0 = p
        agree += (pred_l == pred_0).sum().item()
        tot += ql.numel()

print(f"\nepisodes           : {len(oa_l)}")
print(f"OA  lambda=0.5     : {np.mean(oa_l):.4f}   (per-episode std {np.std(oa_l):.3f})")
print(f"OA  lambda=0.0     : {np.mean(oa_0):.4f}   (per-episode std {np.std(oa_0):.3f})")
print(f"OA  difference     : {np.mean(oa_l) - np.mean(oa_0):+.4f} pp")
print(f"prediction agreement: {agree}/{tot} = {100.0*agree/tot:.4f} %")
print(f"episodes with identical OA: {sum(1 for a,b in zip(oa_l,oa_0) if abs(a-b)<1e-9)}/{len(oa_l)}")
print("\ncls_emb mechanism (episode 0):")
print(f"  ||mean(cls_emb)||           = {cls_stats[0]:.4f}")
print(f"  mean per-dim std of cls_emb = {cls_stats[1]:.6f}")
print(f"  mean ||cls - mean(cls)||    = {cls_stats[2]:.6f}   <- if ~0, lambda is inert")
print(f"  mean ||mp  - mean(mp)||     = {cls_stats[3]:.6f}   (patch-pool spread, for scale)")
