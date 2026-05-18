"""Visualization utilities for MFT-CPEA evaluation."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Optional


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[list] = None,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
):
    """Plot confusion matrix with row-normalized colors and raw count annotations.

    Args:
        cm: Confusion matrix [N, N] (raw counts).
        class_names: List of class name strings.
        title: Plot title.
        save_path: If provided, save figure to this path.
    """
    n = cm.shape[0]
    figsize = max(8, n * 0.8)
    fig, ax = plt.subplots(figsize=(figsize, figsize))

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm.astype(float), row_sums,
        where=row_sums != 0, out=np.zeros_like(cm, dtype=float),
    )

    labels = class_names if class_names else [str(i) for i in range(n)]

    sns.heatmap(
        cm_norm,
        annot=cm,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        cbar_kws={"label": "Recall (row-normalised)"},
        linewidths=0.5,
    )
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    save_path: Optional[str] = None,
):
    """Plot t-SNE visualization of embeddings."""
    from sklearn.manifold import TSNE

    tsne = TSNE(n_components=2, random_state=42)
    emb_2d = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(10, 10))
    scatter = ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels, cmap='tab10')
    ax.legend(*scatter.legend_elements(), title="Classes")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


# ---------------------------------------------------------------------------
# Few-shot episodic evaluation visualizations
# ---------------------------------------------------------------------------


def plot_per_class_accuracy(
    class_names: list,
    accuracies: list,
    ci_95: list,
    title: str = "Per-Class Accuracy",
    save_path: Optional[str] = None,
):
    """Horizontal bar chart of per-class accuracy with 95% CI error bars.

    Args:
        class_names: Display names for each class.
        accuracies: Accuracy (%) per class.
        ci_95: 95% confidence interval half-width per class.
        title: Plot title.
        save_path: If provided, save figure to this path.
    """
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(10, max(4, n * 0.45)))

    y_pos = np.arange(n)
    mean_acc = np.mean(accuracies)

    colors = ["#e74c3c" if a < mean_acc - 10 else "#3498db" for a in accuracies]

    ax.barh(y_pos, accuracies, xerr=ci_95, capsize=3,
            color=colors, edgecolor="white", linewidth=0.5,
            error_kw={"linewidth": 1.0, "capthick": 1.0})

    ax.axvline(mean_acc, color="gray", linestyle="--", linewidth=1.2,
               alpha=0.7, label=f"Mean: {mean_acc:.1f}%")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("Accuracy (%)", fontsize=11)
    ax.set_xlim(0, 105)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()

    for i, (acc, ci) in enumerate(zip(accuracies, ci_95)):
        ax.text(min(acc + ci + 1.5, 104), i, f"{acc:.1f}%",
                va="center", fontsize=8)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_episode_distributions(
    episode_oas: np.ndarray,
    episode_aas: np.ndarray,
    episode_kappas: np.ndarray,
    title: str = "Episode Metric Distributions",
    save_path: Optional[str] = None,
    n_bins: int = 40,
):
    """Histogram + KDE of per-episode OA, AA, and Kappa.

    Args:
        episode_oas: Per-episode overall accuracy array.
        episode_aas: Per-episode average accuracy array.
        episode_kappas: Per-episode kappa (x100) array.
        title: Suptitle for the figure.
        save_path: If provided, save figure to this path.
        n_bins: Number of histogram bins.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    metrics = [
        (episode_oas, "Overall Accuracy (%)", "OA"),
        (episode_aas, "Average Accuracy (%)", "AA"),
        (episode_kappas, "Kappa (\u00d7100)", "Kappa"),
    ]

    for ax, (values, xlabel, label) in zip(axes, metrics):
        mean_val = np.mean(values)
        std_val = np.std(values)

        ax.hist(values, bins=n_bins, color="#3498db", edgecolor="white",
                alpha=0.7, density=True)

        # KDE overlay
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(values)
            x_grid = np.linspace(values.min(), values.max(), 200)
            ax.plot(x_grid, kde(x_grid), color="#2c3e50", linewidth=1.5)
        except Exception:
            pass

        ax.axvline(mean_val, color="#e74c3c", linestyle="--", linewidth=2,
                   label=f"Mean: {mean_val:.2f}")
        ax.axvspan(mean_val - 1.96 * std_val, mean_val + 1.96 * std_val,
                   color="#e74c3c", alpha=0.08, label="95% range")

        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title(f"{label} (n={len(values)})", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_per_class_boxplots(
    class_names: list,
    class_episode_accs: dict,
    class_order: list,
    title: str = "Per-Class Accuracy Across Episodes",
    save_path: Optional[str] = None,
):
    """Violin + box plots of per-class accuracy distributions across episodes.

    Args:
        class_names: Display names (aligned with class_order).
        class_episode_accs: {orig_class_id: [list of per-episode accuracies]}.
        class_order: Sorted list of original class ids to plot.
        title: Plot title.
        save_path: If provided, save figure to this path.
    """
    n = len(class_order)
    fig, ax = plt.subplots(figsize=(max(10, n * 0.8), 6))

    data = []
    for cls in class_order:
        accs = class_episode_accs.get(cls, [])
        data.append(np.array(accs) if len(accs) > 0 else np.array([0.0]))

    parts = ax.violinplot(data, positions=range(n), showmeans=True,
                          showmedians=True, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor("#3498db")
        pc.set_alpha(0.3)
    parts["cmeans"].set_color("#e74c3c")
    parts["cmeans"].set_linewidth(1.5)
    parts["cmedians"].set_color("#2c3e50")

    ax.set_xticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.set_ylim(-5, 105)
    ax.grid(axis="y", alpha=0.3)

    for i, cls in enumerate(class_order):
        n_eps = len(class_episode_accs.get(cls, []))
        ax.text(i, -3, f"n={n_eps}", ha="center", va="top",
                fontsize=7, color="gray")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_pairwise_confusion_rate(
    pairwise_confusion: dict,
    pairwise_totals: dict,
    class_names: list,
    class_order: list,
    title: str = "Pairwise Confusion Rates",
    save_path: Optional[str] = None,
):
    """Heatmap of off-diagonal confusion rates between co-occurring class pairs.

    For each pair (i, j), shows how often class i was predicted as class j
    among episodes where both appeared. Diagonal is zeroed to focus on errors.

    Args:
        pairwise_confusion: {orig_i: {orig_j: count_predicted_as_j}}.
        pairwise_totals: {orig_i: {orig_j: total_query_samples_of_i}}.
        class_names: Display names (aligned with class_order).
        class_order: Sorted list of original class ids.
        title: Plot title.
        save_path: If provided, save figure to this path.
    """
    n = len(class_order)
    rate_matrix = np.zeros((n, n))

    for i, cls_i in enumerate(class_order):
        for j, cls_j in enumerate(class_order):
            total = pairwise_totals.get(cls_i, {}).get(cls_j, 0)
            confused = pairwise_confusion.get(cls_i, {}).get(cls_j, 0)
            rate_matrix[i, j] = (confused / total * 100) if total > 0 else 0.0

    # Zero diagonal to focus on off-diagonal confusion
    off_diag = rate_matrix.copy()
    np.fill_diagonal(off_diag, 0)

    annot = np.empty((n, n), dtype=object)
    for i in range(n):
        for j in range(n):
            if i == j:
                annot[i, j] = ""
            elif rate_matrix[i, j] > 0:
                annot[i, j] = f"{rate_matrix[i, j]:.1f}"
            else:
                annot[i, j] = ""

    figsize = max(8, n * 0.7)
    fig, ax = plt.subplots(figsize=(figsize, figsize))

    vmax = max(30, np.max(off_diag)) if np.max(off_diag) > 0 else 30

    sns.heatmap(
        off_diag,
        annot=annot,
        fmt="",
        cmap="Reds",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={"label": "Confusion Rate (%)"},
        linewidths=0.5,
        vmin=0,
        vmax=vmax,
    )
    ax.set_xlabel("Predicted as", fontsize=12)
    ax.set_ylabel("True class", fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Feature space (t-SNE) visualizations
# ---------------------------------------------------------------------------


def _get_class_colors(n_classes: int):
    """Return a list of distinct colors for up to n_classes."""
    if n_classes <= 10:
        cmap = matplotlib.colormaps.get_cmap("tab10")
    else:
        cmap = matplotlib.colormaps.get_cmap("tab20")
    return [cmap(i / max(n_classes - 1, 1)) for i in range(n_classes)]


def plot_episode_feature_space(
    s_features: np.ndarray,
    q_features: np.ndarray,
    prototypes: np.ndarray,
    s_labels: np.ndarray,
    q_labels: np.ndarray,
    q_preds: np.ndarray,
    class_names: list,
    title: str = "Episode Feature Space",
    save_path: Optional[str] = None,
):
    """t-SNE scatter plot of prototypes, support, and query samples for one episode.

    Args:
        s_features: Support features [N*K, D].
        q_features: Query features [N*Q, D].
        prototypes: Class prototypes [N, D].
        s_labels: Support labels [N*K] (0-indexed within episode).
        q_labels: Query ground-truth labels [N*Q] (0-indexed).
        q_preds: Query predictions [N*Q] (0-indexed).
        class_names: Display names for the N classes in this episode.
        title: Plot title.
        save_path: If provided, save figure to this path.
    """
    from sklearn.manifold import TSNE

    n_classes = prototypes.shape[0]
    n_support = s_features.shape[0]
    n_query = q_features.shape[0]

    # Concatenate all points: [prototypes; support; query]
    all_features = np.concatenate([prototypes, s_features, q_features], axis=0)

    perplexity = min(30, max(5, all_features.shape[0] // 4))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                init="pca", learning_rate="auto")
    coords_2d = tsne.fit_transform(all_features)

    proto_coords = coords_2d[:n_classes]
    support_coords = coords_2d[n_classes:n_classes + n_support]
    query_coords = coords_2d[n_classes + n_support:]

    colors = _get_class_colors(n_classes)
    fig, ax = plt.subplots(figsize=(10, 9))

    # Plot support samples (medium circles)
    for c in range(n_classes):
        mask = s_labels == c
        ax.scatter(
            support_coords[mask, 0], support_coords[mask, 1],
            c=[colors[c]], s=60, alpha=0.6, edgecolors="white", linewidths=0.5,
            marker="o",
        )

    # Plot query samples — correct vs misclassified
    correct_mask = q_labels == q_preds
    for c in range(n_classes):
        c_mask = q_labels == c
        # Correctly classified
        ok = c_mask & correct_mask
        if ok.any():
            ax.scatter(
                query_coords[ok, 0], query_coords[ok, 1],
                c=[colors[c]], s=30, alpha=0.5, edgecolors="gray",
                linewidths=0.3, marker="s",
            )
        # Misclassified
        bad = c_mask & ~correct_mask
        if bad.any():
            ax.scatter(
                query_coords[bad, 0], query_coords[bad, 1],
                c=[colors[c]], s=50, alpha=0.9, edgecolors="red",
                linewidths=1.5, marker="X",
            )

    # Plot prototypes (large stars)
    for c in range(n_classes):
        ax.scatter(
            proto_coords[c, 0], proto_coords[c, 1],
            c=[colors[c]], s=350, marker="*", edgecolors="black",
            linewidths=1.2, zorder=10,
        )

    # Build legend
    from matplotlib.lines import Line2D
    legend_elements = []
    for c in range(n_classes):
        legend_elements.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[c],
                   markersize=8, label=class_names[c])
        )
    legend_elements.append(
        Line2D([0], [0], marker="*", color="w", markerfacecolor="gray",
               markeredgecolor="black", markersize=14, label="Prototype")
    )
    legend_elements.append(
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markersize=7, label="Support")
    )
    legend_elements.append(
        Line2D([0], [0], marker="s", color="w", markerfacecolor="gray",
               markersize=6, label="Query (correct)")
    )
    legend_elements.append(
        Line2D([0], [0], marker="X", color="w", markerfacecolor="gray",
               markeredgecolor="red", markersize=8, label="Query (wrong)")
    )
    ax.legend(handles=legend_elements, fontsize=8, loc="best",
              framealpha=0.8, ncol=1)

    n_correct = int(correct_mask.sum())
    acc = n_correct / n_query * 100 if n_query > 0 else 0
    ax.set_title(f"{title}  (acc={acc:.1f}%, {n_correct}/{n_query})", fontsize=13)
    ax.set_xlabel("t-SNE dim 1", fontsize=10)
    ax.set_ylabel("t-SNE dim 2", fontsize=10)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_aggregated_feature_space(
    class_features: dict,
    class_names: list,
    class_order: list,
    title: str = "Aggregated Feature Space",
    save_path: Optional[str] = None,
    max_samples_per_class: int = 0,
):
    """t-SNE of features accumulated across all episodes, colored by original class.

    Args:
        class_features: {orig_class_id: np.ndarray [M, D]} accumulated features.
        class_names: Display names aligned with class_order.
        class_order: Sorted list of original class ids.
        title: Plot title.
        save_path: If provided, save figure to this path.
        max_samples_per_class: If > 0, randomly subsample each class to this many
            samples before running t-SNE.  0 means use all samples.
    """
    from sklearn.manifold import TSNE

    rng = np.random.RandomState(42)
    all_feats = []
    all_labels = []

    for idx, cls in enumerate(class_order):
        feats = class_features.get(cls)
        if feats is None or len(feats) == 0:
            continue
        feats = np.asarray(feats)
        if max_samples_per_class > 0 and len(feats) > max_samples_per_class:
            chosen = rng.choice(len(feats), size=max_samples_per_class, replace=False)
            feats = feats[chosen]
        all_feats.append(feats)
        all_labels.append(np.full(len(feats), idx, dtype=np.int64))

    if not all_feats:
        return None

    all_feats = np.concatenate(all_feats, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    n_samples = all_feats.shape[0]
    n_classes = len(class_order)

    perplexity = min(50, max(5, n_samples // 4))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                init="pca", learning_rate="auto")
    coords_2d = tsne.fit_transform(all_feats)

    colors = _get_class_colors(n_classes)
    fig, ax = plt.subplots(figsize=(12, 10))

    for idx in range(n_classes):
        mask = all_labels == idx
        if not mask.any():
            continue
        ax.scatter(
            coords_2d[mask, 0], coords_2d[mask, 1],
            c=[colors[idx]], s=12, alpha=0.45, edgecolors="none",
            label=f"{class_names[idx]} ({mask.sum()})",
        )

    # Plot class centroids
    for idx in range(n_classes):
        mask = all_labels == idx
        if not mask.any():
            continue
        cx = coords_2d[mask, 0].mean()
        cy = coords_2d[mask, 1].mean()
        ax.scatter(
            cx, cy, c=[colors[idx]], s=200, marker="*",
            edgecolors="black", linewidths=1.0, zorder=10,
        )
        ax.annotate(
            class_names[idx], (cx, cy), fontsize=7,
            textcoords="offset points", xytext=(6, 6),
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, lw=0),
        )

    ax.legend(fontsize=7, loc="best", framealpha=0.8, ncol=2,
              markerscale=2.5)
    ax.set_title(f"{title}  (n={n_samples})", fontsize=13)
    ax.set_xlabel("t-SNE dim 1", fontsize=10)
    ax.set_ylabel("t-SNE dim 2", fontsize=10)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return fig
