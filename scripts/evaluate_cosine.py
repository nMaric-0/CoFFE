#!/usr/bin/env python
"""
Few-shot evaluation script for MFT-CPEA-Cosine.

This script evaluates the cosine similarity variant of MFT-CPEA which uses
prototypical networks instead of DenseSimilarity. This version is ideal for
zero-shot evaluation (no finetuning) as it has no learnable similarity parameters.

Key differences from evaluate_enhanced.py:
- Uses MFTCPEACosine instead of MFTCPEA
- No DenseSimilarity MLP (no learnable params)
- Uses cosine similarity to class prototypes
- Better for pretrained models without finetuning

Usage:
    # Evaluate pretrained model (zero-shot)
    python scripts/evaluate_cosine.py \
        --checkpoint checkpoints/pretrained/houston_enhanced/encoder_final.pth \
        --dataset houston \
        --n-way 5 --k-shot 5

    # Use Euclidean distance instead of cosine
    python scripts/evaluate_cosine.py \
        --checkpoint checkpoints/pretrained/houston_enhanced/encoder_final.pth \
        --dataset houston \
        --distance-metric euclidean
"""

import argparse
import logging
import sys
from pathlib import Path
from collections import defaultdict

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import json

from utils.spatial_weights import center_weighted_pool

from data.datasets import (
    HoustonPatchedDataset,
    TrentoPatchedDataset,
    MUUFLPatchedDataset,
)
from data.samplers.patched_episode_sampler import PatchedEpisodeSampler
from models import MFTCPEACosine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
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
        "Healthy grass", "Stressed grass", "Synthetic grass", "Trees",
        "Soil", "Water", "Residential", "Commercial", "Road", "Highway",
        "Railway", "Parking Lot 1", "Parking Lot 2", "Tennis Court", "Running Track"
    ],
    "trento": [
        "Apple trees", "Buildings", "Ground", "Woods", "Vineyard", "Roads"
    ],
    "muufl": [
        "Trees", "Mostly grass", "Mixed ground surface", "Dirt and sand",
        "Road", "Water", "Building shadow", "Building", "Sidewalk",
        "Yellow curb", "Cloth panels"
    ],
}


def load_checkpoint_with_key_mapping(checkpoint_path: str, device: str):
    """Load checkpoint and handle various key naming conventions."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)

    if 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
        config = ckpt.get('config')
    elif 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
        config = ckpt.get('config')
    elif 'encoder_state_dict' in ckpt:
        state_dict = ckpt['encoder_state_dict']
        config = ckpt.get('config')
    else:
        state_dict = ckpt
        config = None

    return state_dict, config


def fix_state_dict_keys(state_dict: dict, model_state: dict) -> dict:
    """Fix key mismatches between checkpoint and model."""
    fixed_state = {}

    for k, v in state_dict.items():
        new_key = k

        if k.startswith('module.'):
            new_key = k[7:]

        if k.startswith('encoder.encoder.'):
            new_key = k[8:]
        elif k.startswith('encoder.') and not any(mk.startswith('encoder.encoder.') for mk in model_state.keys()):
            if k[8:] in model_state:
                new_key = k[8:]

        if k.startswith('layers.') and 'encoder.layers.0.norm1.weight' in model_state:
            new_key = 'encoder.' + k

        skip_prefixes = (
            'spatial_masking', 'spectral_masking', 'lidar_masking',
            'spatial_decoder', 'spectral_decoder', 'lidar_decoder', 'denoise_decoder',
            'enc_to_dec', 'mask_token', 'null_lidar', 'noise_augmentation',
            'decoder', 'contrastive_head', 'similarity'  # Skip DenseSimilarity too
        )
        if any(k.startswith(prefix) for prefix in skip_prefixes):
            continue

        fixed_state[new_key] = v

    return fixed_state


def load_model_with_checkpoint(
    checkpoint_path: str,
    dataset_name: str,
    model_config: dict,
    device: str
):
    """Load MFTCPEACosine model with proper checkpoint handling."""
    specs = DATASET_SPECS[dataset_name]

    model = MFTCPEACosine(
        hsi_channels=specs["hsi_channels"],
        aux_channels=specs["aux_channels"],
        embed_dim=model_config.get("embed_dim", 128),
        num_heads=model_config.get("num_heads", 8),
        num_layers=model_config.get("num_layers", 4),
        patch_size=model_config.get("patch_size", 11),
        lambda_factor=model_config.get("lambda_factor", 2.0),
        dropout=model_config.get("dropout", 0.1),
        use_projection=model_config.get("use_projection", False),
        distance_metric=model_config.get("distance_metric", "cosine"),
        temperature=model_config.get("temperature", 10.0),
        prototype_mode=model_config.get("prototype_mode", "mean_features"),
        pool_sigma=model_config.get("pool_sigma", None)
    )

    if checkpoint_path.lower() in ['none', 'null', 'random']:
        logger.info("Using random initialization (no pretrained weights)")
        model = model.to(device)
        model.eval()
        return model

    logger.info(f"Loading checkpoint from {checkpoint_path}")
    state_dict, config = load_checkpoint_with_key_mapping(checkpoint_path, device)

    if config:
        logger.info(f"Checkpoint config: hsi={config.get('hsi_channels')}, "
                   f"aux={config.get('aux_channels')}")

    model_state = model.state_dict()
    fixed_state = fix_state_dict_keys(state_dict, model_state)

    compatible_state = {}
    shape_mismatches = []

    channel_keys = {
        'channel_tokenizer.conv.0.weight',
        'channel_tokenizer.conv.1.weight',
        'channel_tokenizer.conv.1.bias',
        'channel_tokenizer.conv.1.running_mean',
        'channel_tokenizer.conv.1.running_var',
        'aux_tokenizer.mlp.0.weight',
        'aux_tokenizer.mlp.0.bias',
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
    logger.info(f"Note: DenseSimilarity parameters intentionally skipped (using cosine similarity)")

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
        if class_total > 0:
            acc = confusion_matrix[i, i] / class_total * 100
        else:
            acc = 0.0
        per_class_acc.append(acc)

    total = confusion_matrix.sum()
    correct = np.diag(confusion_matrix).sum()
    oa = correct / total * 100 if total > 0 else 0.0

    valid_classes = [acc for i, acc in enumerate(per_class_acc)
                     if confusion_matrix[i, :].sum() > 0]
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
    model: MFTCPEACosine,
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
        model: The MFTCPEACosine model.
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
    global_conf_matrix = np.zeros(
        (num_total_classes + 1, num_total_classes + 1), dtype=np.int64
    )
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

        s_adapted = model.adapt_embeddings(s_patch, s_cls)
        q_adapted = model.adapt_embeddings(q_patch, q_cls)

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
            example_episodes_data.append({
                "s_features": s_features.cpu().numpy(),
                "q_features": q_features.cpu().numpy(),
                "prototypes": prototypes.cpu().numpy(),
                "s_labels": support_labels.cpu().numpy(),
                "q_labels": query_labels_cpu,
                "q_preds": preds_cpu,
                "original_classes": original_classes,
            })

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
        if data["total"] > 0:
            acc = data["correct"] / data["total"] * 100
        else:
            acc = 0.0

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


def print_results_table(results: dict, dataset_name: str, n_way: int, k_shot: int, distance_metric: str, prototype_mode: str = "mean_features"):
    """Print results in formatted table."""
    class_names = CLASS_NAMES.get(dataset_name, [f"Class {i}" for i in range(20)])

    print("\n" + "=" * 80)
    print(f"EVALUATION RESULTS - {dataset_name.upper()} ({n_way}-way {k_shot}-shot)")
    print(f"Method: MFT-CPEA-Cosine (distance={distance_metric}, mode={prototype_mode})")
    print("=" * 80)

    print("\n" + "-" * 80)
    print(f"{'Class':<25} {'Accuracy':>12} {'± 95% CI':>12} {'Samples':>12}")
    print("-" * 80)

    for class_idx in sorted(results["per_class"].keys()):
        class_data = results["per_class"][class_idx]
        # Class labels are 1-indexed (background=0 is skipped), but CLASS_NAMES is 0-indexed
        name_idx = class_idx - 1  # Convert 1-indexed to 0-indexed
        class_name = class_names[name_idx] if 0 <= name_idx < len(class_names) else f"Class {class_idx}"

        if len(class_name) > 23:
            class_name = class_name[:20] + "..."

        print(f"{class_name:<25} {class_data['accuracy']:>11.2f}% "
              f"{class_data['ci_95']:>11.2f}% {class_data['total_samples']:>12d}")

    print("-" * 80)

    print("\n" + "-" * 80)
    print("SUMMARY METRICS")
    print("-" * 80)
    print(f"{'Overall Accuracy (OA)':<25} {results['OA']['mean']:>11.2f}% ± {results['OA']['ci_95']:.2f}%")
    print(f"{'Average Accuracy (AA)':<25} {results['AA']['mean']:>11.2f}% ± {results['AA']['ci_95']:.2f}%")
    print(f"{'Kappa (×100)':<25} {results['Kappa']['mean']:>11.2f}  ± {results['Kappa']['ci_95']:.2f}")
    print("-" * 80)
    print(f"Episodes evaluated: {results['num_episodes']}")
    print("=" * 80 + "\n")


def run_single_seed(args, seed, device, dataset, model, dataset_name):
    """Run evaluation for a single seed."""
    from utils.seed import set_seed
    set_seed(seed, deterministic=True)

    sampler = PatchedEpisodeSampler(
        dataset=dataset,
        n_way=args.n_way,
        k_shot=args.k_shot,
        k_query=args.k_query,
        num_episodes=args.num_episodes,
        seed=seed
    )

    logger.info(f"[Seed {seed}] Sampler: {args.n_way}-way {args.k_shot}-shot, {args.num_episodes} episodes")

    num_total_classes = DATASET_SPECS[dataset_name]["num_classes"]
    results = evaluate(
        model, sampler, device, args.num_episodes, args.n_way,
        num_total_classes=num_total_classes,
        num_example_episodes=args.num_example_episodes,
        max_tsne_samples=args.max_tsne_samples,
    )
    return results


def generate_plots(results, dataset_name, n_way, k_shot, output_dir,
                    distance_metric, prototype_mode, max_tsne_samples=0):
    """Generate and save all visualization plots for few-shot evaluation."""
    from utils.visualization import (
        plot_confusion_matrix,
        plot_per_class_accuracy,
        plot_episode_distributions,
        plot_per_class_boxplots,
        plot_pairwise_confusion_rate,
        plot_episode_feature_space,
        plot_aggregated_feature_space,
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
        cm_sub, display_names,
        title=f"Aggregated Confusion Matrix \u2013 {tag}",
        save_path=str(output_dir / f"{dataset_name}_confusion_matrix.png"),
    )

    # 2. Per-class accuracy bar chart with CI error bars
    accs = [results["per_class"][cls]["accuracy"] for cls in active_classes]
    cis = [results["per_class"][cls]["ci_95"] for cls in active_classes]
    plot_per_class_accuracy(
        display_names, accs, cis,
        title=f"Per-Class Accuracy \u2013 {tag}",
        save_path=str(output_dir / f"{dataset_name}_per_class_accuracy.png"),
    )

    # 3. Episode metric distributions (OA, AA, Kappa histograms)
    plot_episode_distributions(
        results["episode_oas"], results["episode_aas"], results["episode_kappas"],
        title=f"Episode Distributions \u2013 {tag}",
        save_path=str(output_dir / f"{dataset_name}_episode_distributions.png"),
    )

    # 4. Per-class accuracy box/violin plots
    plot_per_class_boxplots(
        display_names, results["class_episode_accs"], active_classes,
        title=f"Per-Class Accuracy Across Episodes \u2013 {tag}",
        save_path=str(output_dir / f"{dataset_name}_per_class_boxplots.png"),
    )

    # 5. Pairwise confusion rates heatmap
    plot_pairwise_confusion_rate(
        results["pairwise_confusion"], results["pairwise_totals"],
        display_names, active_classes,
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
            title=f"Episode {i+1} Feature Space \u2013 {tag}",
            save_path=str(output_dir / f"{dataset_name}_episode_{i+1:03d}_features.png"),
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


def main(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    logger.info(f"Using device: {device}")

    # Determine seeds to use
    seeds = args.seeds if args.seeds else [args.seed]

    dataset_name = args.dataset.lower()
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    DatasetClass = DATASETS[dataset_name]
    dataset = DatasetClass(
        data_root=args.data_root,
        patch_size=args.patch_size,
        split=args.split,
        normalize=True
    )

    logger.info(f"Loaded {dataset_name} ({args.split}): {len(dataset)} samples")

    model_config = {
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
        "pool_sigma": args.pool_sigma,
    }

    logger.info(f"Loading MFT-CPEA-Cosine (distance={args.distance_metric}, temp={args.temperature}, mode={args.prototype_mode})")
    model = load_model_with_checkpoint(args.checkpoint, dataset_name, model_config, device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")
    logger.info("Note: No DenseSimilarity MLP - using prototypical networks")

    # Run evaluation for each seed
    all_results = []
    for seed in seeds:
        results = run_single_seed(args, seed, device, dataset, model, dataset_name)
        all_results.append(results)
        print_results_table(results, dataset_name, args.n_way, args.k_shot, args.distance_metric, args.prototype_mode)

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
            "model_type": "MFTCPEACosine",
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

        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Results saved to {args.output}")

    # Generate visualization plots
    if not args.no_plots:
        output_dir = Path(args.output_dir) if args.output_dir else Path(
            f"./outputs/cosine_eval/{dataset_name}_{args.n_way}way_{args.k_shot}shot"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        generate_plots(
            results, dataset_name, args.n_way, args.k_shot, output_dir,
            args.distance_metric, args.prototype_mode, args.max_tsne_samples,
        )

    return all_results if len(seeds) > 1 else results


_DEFAULT_ARGS = {
    "n_way": 5,
    "k_shot": 5,
    "k_query": 15,
    "num_episodes": 2000,
    "data_root": "./data/raw",
    "split": "test",
    "patch_size": 11,
    "embed_dim": 128,
    "num_heads": 8,
    "num_layers": 4,
    "lambda_factor": 2.0,
    "dropout": 0.1,
    "use_projection": True,
    "distance_metric": "cosine",
    "temperature": 10.0,
    "prototype_mode": "mean_features",
    "pool_sigma": None,
    "seed": 42,
    "seeds": None,
    "cpu": False,
    "output": None,
    "output_dir": None,
    "no_plots": False,
    "num_example_episodes": 3,
    "max_tsne_samples": 0,
}


def run_evaluation(checkpoint: str, dataset: str, **overrides):
    """Programmatic entry point used by notebooks.

    Keyword arguments mirror the CLI flags (with underscores instead of
    dashes). Unspecified values fall back to the defaults in `_DEFAULT_ARGS`.
    Returns the same value as `main(args)`.
    """
    cfg = {**_DEFAULT_ARGS, "checkpoint": checkpoint, "dataset": dataset, **overrides}
    return main(argparse.Namespace(**cfg))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Few-shot evaluation for MFT-CPEA-Cosine (prototypical networks)"
    )

    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Checkpoint path or 'random' for no pretraining")
    parser.add_argument("--dataset", type=str, required=True,
                       choices=["houston", "trento", "muufl"])

    # Few-shot settings
    parser.add_argument("--n-way", type=int, default=5)
    parser.add_argument("--k-shot", type=int, default=5)
    parser.add_argument("--k-query", type=int, default=15)
    parser.add_argument("--num-episodes", type=int, default=2000)

    # Dataset
    parser.add_argument("--data-root", type=str, default="./data/raw")
    parser.add_argument("--split", type=str, default="test",
                       choices=["train", "test", "all"])
    parser.add_argument("--patch-size", type=int, default=11)

    # Model config
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--lambda-factor", type=float, default=2.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--use-projection", action="store_true", default=True,
                       help="Use projection head (default: True, matching pretraining)")
    parser.add_argument("--no-projection", dest="use_projection", action="store_false",
                       help="Disable projection head")

    # Cosine-specific
    parser.add_argument("--distance-metric", type=str, default="cosine",
                       choices=["cosine", "euclidean"],
                       help="Distance metric for prototypical matching")
    parser.add_argument("--temperature", type=float, default=10.0,
                       help="Temperature scaling for cosine similarity")
    parser.add_argument("--prototype-mode", type=str, default="mean_features",
                       choices=["mean_features", "mean_distances"],
                       help="Prototype computation: 'mean_features' (avg features, then distance) "
                            "or 'mean_distances' (distance to each, then avg)")
    parser.add_argument("--pool-sigma", type=float, default=None,
                       help="Gaussian sigma for center-weighted spatial pooling. "
                            "None = uniform mean pooling (default). Recommended: 2.0")

    # Other
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (used if --seeds not provided)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                       help="Multiple seeds for evaluation. Runs once per seed and reports "
                            "aggregate stats. Example: --seeds 42 123 456")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output", type=str, default=None)

    # Visualization
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Directory for output plots. Auto-generated if not set.")
    parser.add_argument("--no-plots", action="store_true", default=False,
                       help="Disable visualization plot generation")
    parser.add_argument("--num-example-episodes", type=int, default=3,
                       help="Number of episodes to generate per-episode t-SNE "
                            "feature space plots for. 0 to skip.")
    parser.add_argument("--max-tsne-samples", type=int, default=0,
                       help="Max samples per class for aggregated t-SNE plot. "
                            "0 = use all samples (no subsampling).")

    args = parser.parse_args()
    main(args)
