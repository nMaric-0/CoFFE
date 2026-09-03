"""
Few-shot evaluation for CoFFE and the MFT architectural control.

The encoder is frozen and never finetuned: each episode's class prototype is
the mean of its K support features, and queries are assigned by
nearest-class-mean (PAPER_CANON §4). The paper protocol is Euclidean NCM
(``--distance-metric euclidean``); cosine is kept as an option value only.

There are no learnable similarity parameters — any DenseSimilarity weights in
an old checkpoint are ignored on load.

Usage:
    # Paper protocol: N-way (all classes), 5-shot, Euclidean NCM
    python scripts/evaluate.py \
        --checkpoint experiments/<run>/checkpoints/checkpoint_epoch_950.pth \
        --dataset houston \
        --k-shot 5 --distance-metric euclidean --no-projection
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

from coffe.compat import normalize_model_name
from coffe.data.datasets import (
    HoustonPatchedDataset,
    MUUFLPatchedDataset,
    TrentoPatchedDataset,
)
from coffe.data.samplers.patched_episode_sampler import PatchedEpisodeSampler
from coffe.models import CoFFE, MFTOriginal
from coffe.utils.seed import set_seed
from coffe.utils.spatial_weights import center_weighted_pool

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


DATASETS = {
    "houston": HoustonPatchedDataset,
    "trento": TrentoPatchedDataset,
    "muufl": MUUFLPatchedDataset,
}

DATASET_SPECS = {
    "houston": {"hsi_channels": 144, "aux_channels": 1, "num_classes": 15},
    "trento": {"hsi_channels": 63, "aux_channels": 1, "num_classes": 6},
    "muufl": {"hsi_channels": 64, "aux_channels": 2, "num_classes": 11},
}

CLASS_NAMES = {
    "houston": [
        "Healthy grass",
        "Stressed grass",
        "Synthetic grass",
        "Trees",
        "Soil",
        "Water",
        "Residential",
        "Commercial",
        "Road",
        "Highway",
        "Railway",
        "Parking Lot 1",
        "Parking Lot 2",
        "Tennis Court",
        "Running Track",
    ],
    "trento": ["Apple trees", "Buildings", "Ground", "Woods", "Vineyard", "Roads"],
    "muufl": [
        "Trees",
        "Mostly grass",
        "Mixed ground surface",
        "Dirt and sand",
        "Road",
        "Water",
        "Building shadow",
        "Building",
        "Sidewalk",
        "Yellow curb",
        "Cloth panels",
    ],
}


def load_checkpoint_with_key_mapping(checkpoint_path: str, device: str) -> tuple[dict, dict | None]:
    """Load checkpoint and handle various key naming conventions."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)

    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        config = ckpt.get("config")
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        config = ckpt.get("config")
    elif "encoder_state_dict" in ckpt:
        state_dict = ckpt["encoder_state_dict"]
        config = ckpt.get("config")
    else:
        state_dict = ckpt
        config = None

    return state_dict, config


def fix_state_dict_keys(state_dict: dict, model_state: dict) -> dict:
    """Fix key mismatches between checkpoint and model."""
    fixed_state = {}

    for k, v in state_dict.items():
        new_key = k

        if k.startswith("module."):
            new_key = k[7:]

        if k.startswith("encoder.encoder.") or (
            k.startswith("encoder.")
            and not any(mk.startswith("encoder.encoder.") for mk in model_state)
            and k[8:] in model_state
        ):
            new_key = k[8:]

        if k.startswith("layers.") and "encoder.layers.0.norm1.weight" in model_state:
            new_key = "encoder." + k

        skip_prefixes = (
            "spatial_masking",
            "spectral_masking",
            "lidar_masking",
            "spatial_decoder",
            "spectral_decoder",
            "lidar_decoder",
            "denoise_decoder",
            "enc_to_dec",
            "mask_token",
            "null_lidar",
            "noise_augmentation",
            "decoder",
            "contrastive_head",
            "similarity",  # Skip DenseSimilarity too
        )
        if any(k.startswith(prefix) for prefix in skip_prefixes):
            continue

        fixed_state[new_key] = v

    return fixed_state


def load_model_with_checkpoint(
    checkpoint_path: str, dataset_name: str, model_config: dict, device: str
) -> nn.Module:
    """Load the few-shot model (CoFFE, or the MFT control when
    model_config['name'] == 'mft_original') with checkpoint handling.

    Frozen eval/pretrain configs carry the pre-paper model name; it is
    normalised here through ``coffe.compat`` (PAPER_CANON §7.3).
    """
    specs = DATASET_SPECS[dataset_name]
    model_name = normalize_model_name(model_config.get("name", "coffe"), origin="model.name")

    if model_name == "mft_original":
        model = MFTOriginal(
            hsi_channels=specs["hsi_channels"],
            aux_channels=specs["aux_channels"],
            use_aux=model_config.get("use_aux", True),
            embed_dim=model_config.get("embed_dim", 64),
            num_heads=model_config.get("num_heads", 8),
            num_layers=model_config.get("num_layers", 2),
            mlp_dim=model_config.get("mlp_dim", 512),
            patch_size=model_config.get("patch_size", 11),
            dropout=model_config.get("dropout", 0.1),
            attention_type=model_config.get("attention_type", "mcross"),
            distance_metric=model_config.get("distance_metric", "euclidean"),
            temperature=model_config.get("temperature", 10.0),
            prototype_mode=model_config.get("prototype_mode", "mean_features"),
        )
    else:
        model = CoFFE(
            hsi_channels=specs["hsi_channels"],
            aux_channels=specs["aux_channels"],
            use_aux=model_config.get("use_aux", True),
            embed_dim=model_config.get("embed_dim", 128),
            num_heads=model_config.get("num_heads", 2),
            num_layers=model_config.get("num_layers", 2),
            patch_size=model_config.get("patch_size", 11),
            # Config key stays `lambda_factor`: frozen pretrain_config.yaml /
            # eval_config.json record it under that name (PAPER_CANON §7.3).
            cls_token_weight=model_config.get("lambda_factor", 0.5),
            dropout=model_config.get("dropout", 0.1),
            use_projection=model_config.get("use_projection", False),
            proj_hidden_dim=model_config.get("proj_hidden_dim"),
            proj_num_layers=model_config.get("proj_num_layers", 2),
            proj_l2_normalize=model_config.get("proj_l2_normalize", True),
            distance_metric=model_config.get("distance_metric", "euclidean"),
            temperature=model_config.get("temperature", 10.0),
            prototype_mode=model_config.get("prototype_mode", "mean_features"),
            pool_sigma=model_config.get("pool_sigma"),
        )

    if checkpoint_path.lower() in ["none", "null", "random"]:
        logger.info("Using random initialization (no pretrained weights)")
        model = model.to(device)
        model.eval()
        return model

    logger.info(f"Loading checkpoint from {checkpoint_path}")
    state_dict, config = load_checkpoint_with_key_mapping(checkpoint_path, device)

    if config:
        logger.info(
            f"Checkpoint config: hsi={config.get('hsi_channels')}, aux={config.get('aux_channels')}"
        )

    model_state = model.state_dict()
    fixed_state = fix_state_dict_keys(state_dict, model_state)

    # Fail loudly on a checkpoint <-> dataset band-count mismatch, so an input
    # projection can never be silently left randomly initialized (e.g. a 64-band
    # Trento checkpoint evaluated on 145-band Houston). The weight whose shape[1]
    # encodes the input band count differs by architecture:
    #   - CoFFE: channel_tokenizer.conv.0.weight = [embed_dim, HSI+aux, 1, 1].
    #   - original MFT: separate HSI/aux front-ends. hsi_hetconv.gwconv.weight
    #     (grouped conv over the 3D-conv output) has shape[1] = 8*(HSI-8)/groups,
    #     which is monotonic in the HSI band count, and aux_conv.0.weight has
    #     shape[1] == aux bands. Both still detect a checkpoint/dataset mismatch.
    def _assert_band_count(key, expected, what):
        ckpt_w = fixed_state.get(key)
        model_w = model_state.get(key)
        if ckpt_w is None or model_w is None or ckpt_w.shape[1] == model_w.shape[1]:
            return
        ckpt_bands, model_bands = ckpt_w.shape[1], model_w.shape[1]
        # The commonest cause is not a wrong --dataset but a modality mismatch:
        # the HSI-only cells pretrain with use_aux=false, while the CLI defaults
        # to --use-aux, so the model is built one aux-channel-count too wide.
        # Naming the flag that actually fixes it (phase-8 gate) beats sending
        # the reader to --dataset, which is usually already right.
        aux = specs["aux_channels"]
        use_aux = model_config.get("use_aux", True)
        hint = "Set --dataset to the dataset this checkpoint was pretrained on."
        if what == "HSI+aux" and use_aux and ckpt_bands == model_bands - aux:
            hint = (
                f"The difference is exactly this scene's {aux} aux channel(s), so the "
                f"checkpoint was almost certainly pretrained HSI-only (use_aux=false, "
                "the configs/coffe/*_hsi.yaml cells): re-run with --no-aux. If the "
                "checkpoint really is HSI+LiDAR, check --dataset instead."
            )
        elif what == "HSI+aux" and not use_aux and ckpt_bands == model_bands + aux:
            hint = (
                f"The difference is exactly this scene's {aux} aux channel(s), so the "
                "checkpoint was pretrained with LiDAR fused in: drop --no-aux."
            )
        raise ValueError(
            f"Band-count mismatch ({what}): checkpoint expects {ckpt_bands} "
            f"but dataset '{dataset_name}' provides {model_bands} ({expected}). "
            f"The input projection cannot load and would be left randomly "
            f"initialized. {hint}"
        )

    if model_name == "mft_original":
        _assert_band_count("hsi_hetconv.gwconv.weight", f"{specs['hsi_channels']} HSI", "HSI")
        _assert_band_count("aux_conv.0.weight", f"{specs['aux_channels']} aux", "aux")
    else:
        _assert_band_count(
            "channel_tokenizer.conv.0.weight",
            f"{specs['hsi_channels']} HSI + {specs['aux_channels']} aux",
            "HSI+aux",
        )

    compatible_state = {}
    shape_mismatches = []

    channel_keys = {
        "channel_tokenizer.conv.0.weight",
        "channel_tokenizer.conv.1.weight",
        "channel_tokenizer.conv.1.bias",
        "channel_tokenizer.conv.1.running_mean",
        "channel_tokenizer.conv.1.running_var",
        "aux_tokenizer.mlp.0.weight",
        "aux_tokenizer.mlp.0.bias",
    }

    for k, v in fixed_state.items():
        if k in model_state:
            if v.shape == model_state[k].shape:
                compatible_state[k] = v
            else:
                shape_mismatches.append((k, v.shape, model_state[k].shape))
                if k in channel_keys:
                    logger.debug(f"Skipping channel-dependent key {k}")
                else:
                    logger.warning(f"Shape mismatch for {k}: {v.shape} vs {model_state[k].shape}")

    missing, unexpected = model.load_state_dict(compatible_state, strict=False)

    truly_missing = [k for k in missing if k not in channel_keys]

    logger.info(f"Loaded {len(compatible_state)}/{len(fixed_state)} keys from checkpoint")
    logger.info(
        "Note: DenseSimilarity parameters intentionally skipped "
        "(the evaluator is a nearest-class-mean classifier, no learnable similarity)"
    )

    if truly_missing:
        logger.warning(f"Missing keys (not channel-dependent): {truly_missing[:5]}")
    if unexpected:
        logger.warning(f"Unexpected keys: {unexpected[:5]}")

    model = model.to(device)
    model.eval()

    return model


def compute_kappa(confusion_matrix: np.ndarray) -> float:
    """Compute Cohen's Kappa coefficient."""
    n = confusion_matrix.sum()
    if n == 0:
        return 0.0

    p_o = np.diag(confusion_matrix).sum() / n
    row_sums = confusion_matrix.sum(axis=1)
    col_sums = confusion_matrix.sum(axis=0)
    p_e = (row_sums * col_sums).sum() / (n * n)

    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1 - p_e)


def compute_metrics_from_confusion_matrix(confusion_matrix: np.ndarray) -> dict:
    """Compute OA, AA, Kappa, and per-class accuracy."""
    n_classes = confusion_matrix.shape[0]

    per_class_acc = []
    for i in range(n_classes):
        class_total = confusion_matrix[i, :].sum()
        acc = confusion_matrix[i, i] / class_total * 100 if class_total > 0 else 0.0
        per_class_acc.append(acc)

    total = confusion_matrix.sum()
    correct = np.diag(confusion_matrix).sum()
    oa = correct / total * 100 if total > 0 else 0.0

    valid_classes = [acc for i, acc in enumerate(per_class_acc) if confusion_matrix[i, :].sum() > 0]
    aa = np.mean(valid_classes) if valid_classes else 0.0

    kappa = compute_kappa(confusion_matrix) * 100

    return {
        "per_class_acc": per_class_acc,
        "OA": oa,
        "AA": aa,
        "Kappa": kappa,
    }


@torch.no_grad()
def evaluate(
    model: CoFFE,
    sampler: PatchedEpisodeSampler,
    device: str,
    num_episodes: int = 600,
    n_way: int = 5,
    num_total_classes: int = 15,
    num_example_episodes: int = 3,
    max_tsne_samples: int = 0,
) -> dict:
    """Evaluate model and compute metrics with confidence intervals.

    Uses manual feature extraction (instead of model.forward_episode) to
    capture intermediate embeddings and prototypes for visualization.

    Args:
        model: The CoFFE model.
        sampler: Episode sampler.
        device: Torch device string.
        num_episodes: Number of episodes to evaluate.
        n_way: N-way classification.
        num_total_classes: Total classes in the dataset (for global conf matrix).
        num_example_episodes: Number of episodes to save full features for
            per-episode t-SNE plots. Set to 0 to skip.
        max_tsne_samples: Max samples per class for aggregated t-SNE.
            0 = keep all samples (no subsampling).
    """
    model.eval()

    episode_oas = []
    episode_aas = []
    episode_kappas = []

    class_results = defaultdict(lambda: {"correct": 0, "total": 0})
    class_episode_accs = defaultdict(list)

    # Global confusion matrix in original-class space (1-indexed labels,
    # so allocate num_total_classes+1 to use label values directly as indices)
    global_conf_matrix = np.zeros((num_total_classes + 1, num_total_classes + 1), dtype=np.int64)
    # Pairwise confusion tracking for co-occurrence analysis
    pairwise_confusion = defaultdict(lambda: defaultdict(int))
    pairwise_totals = defaultdict(lambda: defaultdict(int))

    # Feature space data collection
    example_episodes_data = []  # per-episode feature snapshots
    aggregated_features = defaultdict(list)  # {orig_class: [feature vectors]}

    for episode in tqdm(sampler, total=min(num_episodes, len(sampler)), desc="Evaluating"):
        if len(episode_oas) >= num_episodes:
            break

        support_hsi = episode["support_hsi"].to(device)
        support_aux = episode["support_aux"].to(device)
        support_labels = episode["support_labels"].to(device)
        query_hsi = episode["query_hsi"].to(device)
        query_aux = episode["query_aux"].to(device)
        query_labels = episode["query_labels"].to(device)

        original_classes = episode["original_classes"].tolist()

        # Manual feature extraction (replicates model.forward_episode internals)
        # to capture intermediate embeddings and prototypes for visualization
        s_patch, s_cls, _ = model.forward_features(support_hsi, support_aux)
        q_patch, q_cls, _ = model.forward_features(query_hsi, query_aux)

        s_adapted = model.eval_patch_embeddings(s_patch, s_cls)
        q_adapted = model.eval_patch_embeddings(q_patch, q_cls)

        if model.pool_sigma is not None:
            s_features = center_weighted_pool(s_adapted, model._center_pool_weights)  # [N*K, D]
            q_features = center_weighted_pool(q_adapted, model._center_pool_weights)  # [N*Q, D]
        else:
            s_features = s_adapted.mean(dim=1)  # [N*K, D]
            q_features = q_adapted.mean(dim=1)  # [N*Q, D]

        if model.distance_metric == "cosine":
            s_features = F.normalize(s_features, p=2, dim=-1)
            q_features = F.normalize(q_features, p=2, dim=-1)

        prototypes = model.compute_prototypes(s_features, support_labels)  # [N, D]

        if model.prototype_mode == "mean_distances":
            logits = model._forward_mean_distances(s_features, support_labels, q_features)
        else:
            logits = model._forward_mean_features(s_features, support_labels, q_features)

        preds = logits.argmax(dim=1)
        query_labels_cpu = query_labels.cpu().numpy()
        preds_cpu = preds.cpu().numpy()

        conf_matrix = np.zeros((n_way, n_way), dtype=np.int64)
        for true_label, pred_label in zip(query_labels_cpu, preds_cpu):
            conf_matrix[true_label, pred_label] += 1

        metrics = compute_metrics_from_confusion_matrix(conf_matrix)
        episode_oas.append(metrics["OA"])
        episode_aas.append(metrics["AA"])
        episode_kappas.append(metrics["Kappa"])

        for way_idx, orig_class in enumerate(original_classes):
            class_total = conf_matrix[way_idx, :].sum()
            class_correct = conf_matrix[way_idx, way_idx]

            class_results[orig_class]["correct"] += class_correct
            class_results[orig_class]["total"] += class_total

            if class_total > 0:
                class_episode_accs[orig_class].append(class_correct / class_total * 100)

        # Accumulate into global confusion matrix (map local -> original indices)
        for way_i, orig_i in enumerate(original_classes):
            for way_j, orig_j in enumerate(original_classes):
                global_conf_matrix[orig_i, orig_j] += conf_matrix[way_i, way_j]
                pairwise_confusion[orig_i][orig_j] += conf_matrix[way_i, way_j]
                pairwise_totals[orig_i][orig_j] += conf_matrix[way_i, :].sum()

        # Collect feature data for visualizations
        episode_idx = len(episode_oas) - 1

        # Per-episode snapshot for example episodes
        if episode_idx < num_example_episodes:
            example_episodes_data.append(
                {
                    "s_features": s_features.cpu().numpy(),
                    "q_features": q_features.cpu().numpy(),
                    "prototypes": prototypes.cpu().numpy(),
                    "s_labels": support_labels.cpu().numpy(),
                    "q_labels": query_labels_cpu,
                    "q_preds": preds_cpu,
                    "original_classes": original_classes,
                }
            )

        # Accumulate query features for aggregated t-SNE (mapped to orig class)
        q_feats_np = q_features.cpu().numpy()
        for way_idx, orig_class in enumerate(original_classes):
            mask = query_labels_cpu == way_idx
            feats_for_class = q_feats_np[mask]
            cur_count = len(aggregated_features[orig_class])
            if max_tsne_samples > 0 and cur_count >= max_tsne_samples:
                continue
            if max_tsne_samples > 0:
                remaining = max_tsne_samples - cur_count
                feats_for_class = feats_for_class[:remaining]
            aggregated_features[orig_class].append(feats_for_class)

    episode_oas = np.array(episode_oas)
    episode_aas = np.array(episode_aas)
    episode_kappas = np.array(episode_kappas)

    # Concatenate aggregated feature lists into arrays
    for cls in aggregated_features:
        aggregated_features[cls] = np.concatenate(aggregated_features[cls], axis=0)

    def compute_ci(values):
        from scipy.stats import t as t_dist

        mean = np.mean(values)
        std = np.std(values)
        ci_95 = t_dist.ppf(0.975, df=len(values) - 1) * std / np.sqrt(len(values))
        return mean, std, ci_95

    oa_mean, oa_std, oa_ci = compute_ci(episode_oas)
    aa_mean, aa_std, aa_ci = compute_ci(episode_aas)
    kappa_mean, kappa_std, kappa_ci = compute_ci(episode_kappas)

    per_class_results = {}
    for orig_class in sorted(class_results.keys()):
        data = class_results[orig_class]
        acc = data["correct"] / data["total"] * 100 if data["total"] > 0 else 0.0

        if class_episode_accs[orig_class]:
            accs = np.array(class_episode_accs[orig_class])
            mean, std, ci = compute_ci(accs)
        else:
            mean, std, ci = acc, 0.0, 0.0

        pooled_acc = data["correct"] / data["total"] * 100 if data["total"] > 0 else 0.0
        per_class_results[orig_class] = {
            "accuracy": float(mean),
            "pooled_accuracy": float(pooled_acc),
            "std": float(std),
            "ci_95": float(ci),
            "total_samples": int(data["total"]),
            "correct_samples": int(data["correct"]),
        }

    # NOTE: With balanced episodes (same k_query per class), OA and AA are
    # identical by construction since each class contributes equally.
    return {
        "per_class": per_class_results,
        "OA": {"mean": float(oa_mean), "std": float(oa_std), "ci_95": float(oa_ci)},
        "AA": {"mean": float(aa_mean), "std": float(aa_std), "ci_95": float(aa_ci)},
        "Kappa": {"mean": float(kappa_mean), "std": float(kappa_std), "ci_95": float(kappa_ci)},
        "num_episodes": len(episode_oas),
        # Visualization data
        "global_conf_matrix": global_conf_matrix,
        "episode_oas": episode_oas,
        "episode_aas": episode_aas,
        "episode_kappas": episode_kappas,
        "class_episode_accs": dict(class_episode_accs),
        "pairwise_confusion": {k: dict(v) for k, v in pairwise_confusion.items()},
        "pairwise_totals": {k: dict(v) for k, v in pairwise_totals.items()},
        # Feature space data for t-SNE plots
        "example_episodes_data": example_episodes_data,
        "aggregated_features": dict(aggregated_features),
    }


def print_results_table(
    results: dict,
    dataset_name: str,
    n_way: int,
    k_shot: int,
    distance_metric: str,
    prototype_mode: str = "mean_features",
) -> None:
    """Print the per-class and summary metric tables for one evaluation.

    Deliberately ``print`` rather than ``logging``: this is the CLI's
    user-facing report, and a timestamped log prefix on every row would
    break the table. Everything else in the package logs.
    """
    class_names = CLASS_NAMES.get(dataset_name, [f"Class {i}" for i in range(20)])

    print("\n" + "=" * 80)
    print(f"EVALUATION RESULTS - {dataset_name.upper()} ({n_way}-way {k_shot}-shot)")
    print(f"Method: nearest-class-mean (distance={distance_metric}, mode={prototype_mode})")
    print("=" * 80)

    print("\n" + "-" * 80)
    print(f"{'Class':<25} {'Accuracy':>12} {'± 95% CI':>12} {'Samples':>12}")
    print("-" * 80)

    for class_idx in sorted(results["per_class"].keys()):
        class_data = results["per_class"][class_idx]
        # Class labels are 1-indexed (background=0 is skipped), but CLASS_NAMES is 0-indexed
        name_idx = class_idx - 1  # Convert 1-indexed to 0-indexed
        class_name = (
            class_names[name_idx] if 0 <= name_idx < len(class_names) else f"Class {class_idx}"
        )

        if len(class_name) > 23:
            class_name = class_name[:20] + "..."

        print(
            f"{class_name:<25} {class_data['accuracy']:>11.2f}% "
            f"{class_data['ci_95']:>11.2f}% {class_data['total_samples']:>12d}"
        )

    print("-" * 80)

    print("\n" + "-" * 80)
    print("SUMMARY METRICS")
    print("-" * 80)
    print(
        f"{'Overall Accuracy (OA)':<25} {results['OA']['mean']:>11.2f}% "
        f"± {results['OA']['ci_95']:.2f}%"
    )
    print(
        f"{'Average Accuracy (AA)':<25} {results['AA']['mean']:>11.2f}% "
        f"± {results['AA']['ci_95']:.2f}%"
    )
    print(
        f"{'Kappa (×100)':<25} {results['Kappa']['mean']:>11.2f}  ± {results['Kappa']['ci_95']:.2f}"
    )
    print("-" * 80)
    print(f"Episodes evaluated: {results['num_episodes']}")
    print("=" * 80 + "\n")


def run_single_seed(
    args: argparse.Namespace,
    seed: int,
    device: str,
    dataset: Dataset,
    model: nn.Module,
    dataset_name: str,
) -> dict:
    """Run evaluation for a single seed."""
    set_seed(seed, deterministic=True)

    sampler = PatchedEpisodeSampler(
        dataset=dataset,
        n_way=args.n_way,
        k_shot=args.k_shot,
        k_query=args.k_query,
        num_episodes=args.num_episodes,
        seed=seed,
    )
    args.n_way = sampler.n_way

    logger.info(
        f"[Seed {seed}] Sampler: {args.n_way}-way {args.k_shot}-shot, {args.num_episodes} episodes"
    )

    num_total_classes = DATASET_SPECS[dataset_name]["num_classes"]
    results = evaluate(
        model,
        sampler,
        device,
        args.num_episodes,
        args.n_way,
        num_total_classes=num_total_classes,
        num_example_episodes=args.num_example_episodes,
        max_tsne_samples=args.max_tsne_samples,
    )
    return results


def generate_plots(
    results: dict,
    dataset_name: str,
    n_way: int,
    k_shot: int,
    output_dir: str | Path,
    distance_metric: str,
    prototype_mode: str,
    max_tsne_samples: int = 0,
) -> None:
    """Generate and save all visualization plots for few-shot evaluation."""
    from coffe.utils.visualization import (
        plot_aggregated_feature_space,
        plot_confusion_matrix,
        plot_episode_distributions,
        plot_episode_feature_space,
        plot_pairwise_confusion_rate,
        plot_per_class_accuracy,
        plot_per_class_boxplots,
    )

    class_names_full = CLASS_NAMES.get(dataset_name, [])
    active_classes = sorted(results["per_class"].keys())

    display_names = []
    for cls in active_classes:
        idx = cls - 1  # 1-indexed to 0-indexed
        if 0 <= idx < len(class_names_full):
            display_names.append(class_names_full[idx])
        else:
            display_names.append(f"Class {cls}")

    tag = f"{dataset_name.upper()} ({n_way}-way {k_shot}-shot, {distance_metric})"

    # 1. Aggregated confusion matrix
    cm = results["global_conf_matrix"]
    cm_sub = cm[np.ix_(active_classes, active_classes)]
    plot_confusion_matrix(
        cm_sub,
        display_names,
        title=f"Aggregated Confusion Matrix \u2013 {tag}",
        save_path=str(output_dir / f"{dataset_name}_confusion_matrix.png"),
    )

    # 2. Per-class accuracy bar chart with CI error bars
    accs = [results["per_class"][cls]["accuracy"] for cls in active_classes]
    cis = [results["per_class"][cls]["ci_95"] for cls in active_classes]
    plot_per_class_accuracy(
        display_names,
        accs,
        cis,
        title=f"Per-Class Accuracy \u2013 {tag}",
        save_path=str(output_dir / f"{dataset_name}_per_class_accuracy.png"),
    )

    # 3. Episode metric distributions (OA, AA, Kappa histograms)
    plot_episode_distributions(
        results["episode_oas"],
        results["episode_aas"],
        results["episode_kappas"],
        title=f"Episode Distributions \u2013 {tag}",
        save_path=str(output_dir / f"{dataset_name}_episode_distributions.png"),
    )

    # 4. Per-class accuracy box/violin plots
    plot_per_class_boxplots(
        display_names,
        results["class_episode_accs"],
        active_classes,
        title=f"Per-Class Accuracy Across Episodes \u2013 {tag}",
        save_path=str(output_dir / f"{dataset_name}_per_class_boxplots.png"),
    )

    # 5. Pairwise confusion rates heatmap
    plot_pairwise_confusion_rate(
        results["pairwise_confusion"],
        results["pairwise_totals"],
        display_names,
        active_classes,
        title=f"Pairwise Confusion Rates \u2013 {tag}",
        save_path=str(output_dir / f"{dataset_name}_pairwise_confusion_rates.png"),
    )

    # 6. Per-episode feature space t-SNE plots
    for i, ep_data in enumerate(results.get("example_episodes_data", [])):
        orig_classes = ep_data["original_classes"]
        ep_class_names = []
        for cls in orig_classes:
            idx = cls - 1
            if 0 <= idx < len(class_names_full):
                ep_class_names.append(class_names_full[idx])
            else:
                ep_class_names.append(f"Class {cls}")

        plot_episode_feature_space(
            s_features=ep_data["s_features"],
            q_features=ep_data["q_features"],
            prototypes=ep_data["prototypes"],
            s_labels=ep_data["s_labels"],
            q_labels=ep_data["q_labels"],
            q_preds=ep_data["q_preds"],
            class_names=ep_class_names,
            title=f"Episode {i + 1}",
            save_path=str(output_dir / f"{dataset_name}_episode_{i + 1:03d}_features.svg"),
            legend_save_path=str(output_dir / f"{dataset_name}_episode_{i + 1:03d}_legend.svg"),
            dataset_label=dataset_name,
        )

    # 7. Aggregated feature space t-SNE (across all episodes)
    agg_feats = results.get("aggregated_features", {})
    if agg_feats:
        plot_aggregated_feature_space(
            class_features=agg_feats,
            class_names=display_names,
            class_order=active_classes,
            title=f"Aggregated Feature Space \u2013 {tag}",
            save_path=str(output_dir / f"{dataset_name}_aggregated_features.png"),
            max_samples_per_class=max_tsne_samples,
        )

    logger.info(f"All plots saved to {output_dir}")


def _resolve_eval_device(requested: str, cpu_flag: bool) -> str:
    """Resolve the evaluation device. Mirrors the pretrain contract:
      - cpu_flag True or requested 'cpu'  -> 'cpu'
      - 'auto'                            -> 'cuda:0' if CUDA, else 'cpu'
      - 'cuda'                            -> 'cuda:0' (must be available)
      - 'cuda:N'                          -> that index (must be valid)
    Raises if CUDA is requested but unavailable / out of range.
    """
    if cpu_flag or requested == "cpu":
        return "cpu"
    if requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device='cuda' but CUDA is not available.")
        return "cuda:0"
    if requested.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"device={requested!r} but CUDA is not available.")
        idx = int(requested.split(":", 1)[1])
        count = torch.cuda.device_count()
        if idx < 0 or idx >= count:
            raise RuntimeError(
                f"device={requested!r} but only {count} CUDA device(s) visible "
                f"(valid indices: 0..{count - 1})."
            )
        return requested
    raise RuntimeError(
        f"Unrecognised device={requested!r}. Expected 'cuda', 'cuda:N', 'auto', or 'cpu'."
    )


def main(args: argparse.Namespace) -> dict | list[dict]:
    """Evaluate one frozen checkpoint under the paper protocol.

    N-way (every class in the scene), 5-shot, 1000 episodes, Euclidean
    nearest-class-mean over the frozen encoder (PAPER_CANON §4). Returns the
    single result dict, or the list of per-seed dicts when several seeds are
    requested.
    """
    requested = getattr(args, "device", "auto") or "auto"
    device = _resolve_eval_device(requested, getattr(args, "cpu", False))
    if device.startswith("cuda"):
        torch.cuda.set_device(device)
        logger.info(
            f"Using device: {device} ({torch.cuda.get_device_name(int(device.split(':', 1)[1]))})"
        )
    else:
        logger.info(f"Using device: {device}")

    # Determine seeds to use
    seeds = args.seeds if args.seeds else [args.seed]

    dataset_name = args.dataset.lower()
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    DatasetClass = DATASETS[dataset_name]
    dataset = DatasetClass(
        data_root=args.data_root, patch_size=args.patch_size, split=args.split, normalize=True
    )

    logger.info(f"Loaded {dataset_name} ({args.split}): {len(dataset)} samples")

    model_config = {
        "name": normalize_model_name(getattr(args, "name", "coffe"), origin="--name"),
        "attention_type": getattr(args, "attention_type", "mcross"),
        "mlp_dim": getattr(args, "mlp_dim", 512),
        "embed_dim": args.embed_dim,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "patch_size": args.patch_size,
        "lambda_factor": args.lambda_factor,
        "dropout": args.dropout,
        "distance_metric": args.distance_metric,
        "temperature": args.temperature,
        "prototype_mode": args.prototype_mode,
        "use_projection": args.use_projection,
        "use_aux": getattr(args, "use_aux", True),
        # Projection-head shape: read from the pretrain experiment so the
        # checkpoint's projection weights actually load (otherwise the head
        # silently runs with random weights). See coffe/runners/eval_runner.py.
        "proj_hidden_dim": getattr(args, "proj_hidden_dim", None),
        "proj_num_layers": getattr(args, "proj_num_layers", 2),
        "proj_l2_normalize": getattr(args, "proj_l2_normalize", True),
        "pool_sigma": args.pool_sigma,
    }

    _model_label = "MFT (original)" if model_config["name"] == "mft_original" else "CoFFE"
    logger.info(
        f"Loading {_model_label} (distance={args.distance_metric}, "
        f"temp={args.temperature}, mode={args.prototype_mode})"
    )
    model = load_model_with_checkpoint(args.checkpoint, dataset_name, model_config, device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")
    logger.info("Note: No DenseSimilarity MLP - nearest-class-mean on frozen features")

    # Run evaluation for each seed
    all_results = []
    for seed in seeds:
        results = run_single_seed(args, seed, device, dataset, model, dataset_name)
        all_results.append(results)
        print_results_table(
            results,
            dataset_name,
            args.n_way,
            args.k_shot,
            args.distance_metric,
            args.prototype_mode,
        )

    # Aggregate across seeds if multiple
    if len(seeds) > 1:
        oa_means = [r["OA"]["mean"] for r in all_results]
        aa_means = [r["AA"]["mean"] for r in all_results]
        kappa_means = [r["Kappa"]["mean"] for r in all_results]

        print("\n" + "=" * 80)
        print(f"AGGREGATE RESULTS ACROSS {len(seeds)} SEEDS: {seeds}")
        print("=" * 80)
        print(f"{'OA':<25} {np.mean(oa_means):>11.2f}% ± {np.std(oa_means):.2f}%")
        print(f"{'AA':<25} {np.mean(aa_means):>11.2f}% ± {np.std(aa_means):.2f}%")
        print(f"{'Kappa (×100)':<25} {np.mean(kappa_means):>11.2f}  ± {np.std(kappa_means):.2f}")
        print("=" * 80 + "\n")

    # Use last seed's results as primary (or only) result
    results = all_results[-1]

    if args.output:
        output_data = {
            # Canonical writer values (PAPER_CANON §8 D16); readers accept the
            # pre-paper class names via coffe.compat.LEGACY_MODEL_TYPES.
            "model_type": "MFTOriginal" if model_config["name"] == "mft_original" else "CoFFE",
            "distance_metric": args.distance_metric,
            "temperature": args.temperature,
            "prototype_mode": args.prototype_mode,
            "checkpoint": str(args.checkpoint),
            "dataset": dataset_name,
            "split": args.split,
            "n_way": args.n_way,
            "k_shot": args.k_shot,
            "k_query": args.k_query,
            "seeds": seeds,
            "num_episodes": results["num_episodes"],
        }

        if len(seeds) > 1:
            output_data["per_seed_results"] = [
                {"seed": s, "OA": r["OA"], "AA": r["AA"], "Kappa": r["Kappa"]}
                for s, r in zip(seeds, all_results)
            ]
            output_data["aggregate"] = {
                "OA_mean": float(np.mean(oa_means)),
                "OA_std": float(np.std(oa_means)),
                "AA_mean": float(np.mean(aa_means)),
                "AA_std": float(np.std(aa_means)),
                "Kappa_mean": float(np.mean(kappa_means)),
                "Kappa_std": float(np.std(kappa_means)),
            }
        else:
            output_data["seed"] = seeds[0]
            output_data["OA"] = results["OA"]
            output_data["AA"] = results["AA"]
            output_data["Kappa"] = results["Kappa"]
            output_data["per_class"] = {str(k): v for k, v in results["per_class"].items()}
            output_data["class_names"] = CLASS_NAMES.get(dataset_name, [])

        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Results saved to {args.output}")

    # Generate visualization plots
    if not args.no_plots:
        output_dir = (
            Path(args.output_dir)
            if args.output_dir
            else Path(f"./outputs/eval/{dataset_name}_{args.n_way}way_{args.k_shot}shot")
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        generate_plots(
            results,
            dataset_name,
            args.n_way,
            args.k_shot,
            output_dir,
            args.distance_metric,
            args.prototype_mode,
            args.max_tsne_samples,
        )

    return all_results if len(seeds) > 1 else results


# Defaults for the programmatic entry point. Aligned with the paper protocol at
# the phase-7 gate (2026-09-03): before then this table carried 8 heads / 4
# layers / lambda 2.0 / projection on / k_query 15 / split "test", none of which
# any published run used. No paper number depends on it either way — on the
# paper path `coffe.runners.eval_runner` seeds the architecture from the run's
# frozen `pretrain_config.yaml` and the reproduce pipeline passes the protocol
# keys explicitly, so no default here is ever consulted (which is why the
# equivalence goldens are unchanged). The alignment is so that a bare
# `run_evaluation(checkpoint, dataset)` is the paper's protocol rather than a
# configuration nothing ran.
_DEFAULT_ARGS = {
    "n_way": None,  # N-way: every class in the scene (PAPER_CANON §4)
    "k_shot": 5,  # §4
    "k_query": 100,  # §4 (Table 2; Table 3 varies - §8 D18)
    # 1000 is Table 2's count; Table 3 used 2000 (§8 D18), so this is a choice
    # rather than a constant. Table 2 is the CoFFE/MFT table this entry point
    # serves.
    "num_episodes": 1000,
    "data_root": "./data/raw",
    "split": "all",  # what every paper eval passed
    "patch_size": 11,  # §2
    # Model selection: "coffe" (default) or "mft_original" (the MFT control).
    "name": "coffe",
    "attention_type": "mcross",
    "mlp_dim": 512,
    "embed_dim": 128,  # §2
    "num_heads": 2,  # §2
    "num_layers": 2,  # §2
    "lambda_factor": 0.5,  # §8 D3: the value every paper CoFFE run records
    "dropout": 0.1,
    "use_projection": False,  # §4: the head is discarded at eval
    "use_aux": True,
    "proj_hidden_dim": None,
    "proj_num_layers": 2,
    "proj_l2_normalize": True,
    "distance_metric": "euclidean",
    "temperature": 10.0,
    "prototype_mode": "mean_features",
    "pool_sigma": None,
    "seed": 42,
    "seeds": None,
    "cpu": False,
    "device": "auto",
    "output": None,
    "output_dir": None,
    "no_plots": False,
    "num_example_episodes": 3,
    "max_tsne_samples": 0,
}


def run_evaluation(checkpoint: str, dataset: str, **overrides: Any) -> dict | list[dict]:
    """Programmatic entry point used by notebooks.

    Keyword arguments mirror the CLI flags (with underscores instead of
    dashes). Unspecified values fall back to the defaults in `_DEFAULT_ARGS`.
    Returns the same value as `main(args)`.
    """
    cfg = {**_DEFAULT_ARGS, "checkpoint": checkpoint, "dataset": dataset, **overrides}
    return main(argparse.Namespace(**cfg))
