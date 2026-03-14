#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gat_matrix_permutation.py

Compute fold-wise GAT (Generalization Across Time) matrices and perform
cluster-based permutation tests to identify regions significantly different
from chance level (0.5) in both directions.

Requires: numpy, scipy, torch, pandas, sklearn, matplotlib, mne
"""

import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from mne.decoding import LinearModel
from mne.stats import permutation_cluster_1samp_test
from scipy import sparse, stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler

from coefficient_geometry import compute_feature_density, get_rotation_angle_matrix

POU_TOKENS = ["ĠÏĢÎ¿Ïħ", "▁που"]


def load_activations(path):
    """Load all .pt files under `path`, sorted by integer in filename."""
    files = sorted(
        [f for f in os.listdir(path) if not f.startswith(".")],
        key=lambda fn: int(re.search(r"\d+", fn).group()),
    )
    return [torch.load(os.path.join(path, fn)) for fn in files]


def extract_features(activation_files, layers):
    """
    Returns a DataFrame with one row per (sample, layer), containing:
      - file_id: which activation file
      - layer: layer index
      - order: 1 for SVO, 0 for VSO
      - before_*, after_*: 4 features each
    """
    rows = []
    for file_id, f in enumerate(activation_files):
        sentence = f["sentence"]
        tokens = f["tokens"]
        # find the clause boundary token
        clause_tok = next(
            (
                i
                for i, t in enumerate(tokens)
                if any(pou_token in t for pou_token in POU_TOKENS)
            ),
            None,
        )
        if clause_tok is None or clause_tok >= len(tokens) - 1:
            continue

        # determine SVO vs VSO
        words = sentence.split()
        aft_words = words[words.index("που") + 1 :]
        order = (
            1
            if (
                len(aft_words) >= 2
                and aft_words[0].lower()
                in ["ο", "η", "το", "οι", "τα", "των", "της", "του"]
            )
            else 0
        )

        for L in layers:
            # X = np.array(f["hidden_states"][L], dtype=np.float64)
            X = f["hidden_states"][L].detach().cpu().numpy().astype(np.float64)
            bef, aft = X[: clause_tok + 1], X[clause_tok + 1 :]

            def feats(mat):
                flat = mat.ravel()
                # Compute features and handle inf/nan
                mean_val = flat.mean()
                var_val = np.var(flat)
                kurt_val = stats.kurtosis(flat)
                skew_val = stats.skew(flat)

                # Replace inf/nan with finite values
                mean_val = np.nan_to_num(mean_val, nan=0.0, posinf=1e10, neginf=-1e10)
                var_val = np.nan_to_num(var_val, nan=0.0, posinf=1e10, neginf=-1e10)
                kurt_val = np.nan_to_num(kurt_val, nan=0.0, posinf=1e10, neginf=-1e10)
                skew_val = np.nan_to_num(skew_val, nan=0.0, posinf=1e10, neginf=-1e10)

                return [mean_val, var_val, kurt_val, skew_val]

            bef_feats = feats(bef)
            aft_feats = feats(aft)

            rows.append(
                {
                    "file_id": file_id,
                    "layer": L,
                    "order": order,
                    **{f"bef_{i}": v for i, v in enumerate(bef_feats)},
                    **{f"aft_{i}": v for i, v in enumerate(aft_feats)},
                }
            )

    return pd.DataFrame(rows)


def compute_gat_folds(df, layers, region_prefix="aft", n_splits=20):
    """
    Compute GAT matrix for each CV fold separately.
    Returns array of shape (n_folds, n_layers, n_layers)
    """
    # pivot into a 3D array: samples × layers × features(4)
    samples = df["file_id"].unique()
    n_samples = len(samples)
    n_layers = len(layers)
    n_units = 4  # mean, var, kurtosis, skewness
    feats_arr = np.zeros((n_samples, n_layers, 4), dtype=np.float64)
    labels = np.zeros(n_samples, dtype=int)

    # build a map file_id -> row index
    id2idx = {fid: idx for idx, fid in enumerate(samples)}

    for _, row in df.iterrows():
        sidx = id2idx[row["file_id"]]
        L = int(row["layer"])
        labels[sidx] = row["order"]
        # region features named "aft_0".."aft_3" or "bef_0".."bef_3"
        feats_arr[sidx, L, :] = [row[f"{region_prefix}_{k}"] for k in range(4)]

    # set up CV
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_matrices = []
    fold_patterns = np.zeros((n_splits, n_layers, n_units))

    # compute GAT matrix for each fold
    for fold_idx, (train_idx, test_idx) in enumerate(
        skf.split(np.zeros(n_samples), labels)
    ):
        M_fold = np.zeros((n_layers, n_layers), dtype=np.float64)

        # loop over train/test layer pairs
        for i, L_train in enumerate(layers):
            for j, L_test in enumerate(layers):
                Xtr = feats_arr[train_idx, i, :]
                Xt = feats_arr[test_idx, j, :]
                ytr = labels[train_idx]
                yt = labels[test_idx]

                # check if we have both classes in test set
                if len(np.unique(yt)) < 2:
                    M_fold[i, j] = np.nan
                    continue

                # scale & fit
                scaler = RobustScaler().fit(Xtr)
                clf = LogisticRegression(max_iter=1000, random_state=42)
                clf = LinearModel(clf)
                clf.fit(scaler.transform(Xtr), ytr)
                # predict & score
                probs = clf.predict_proba(scaler.transform(Xt))[:, 1]
                M_fold[i, j] = roc_auc_score(yt, probs)
                fold_patterns[fold_idx, i, :] = clf.patterns_

        fold_matrices.append(M_fold)

    return np.array(fold_matrices), fold_patterns.mean(axis=0)


def perform_cluster_permutation_2d(fold_matrices, alpha=0.05, n_permutations=5000):
    """
    Perform cluster-based permutation test on 2D GAT matrices.
    Tests against null hypothesis that AUC = 0.5 (chance level).
    """
    n_folds, n_layers, _ = fold_matrices.shape

    # Center data around 0.5 (chance level)
    data_centered = fold_matrices - 0.5

    # Handle NaNs by setting them to 0
    data_centered = np.nan_to_num(data_centered)

    # Flatten spatial dimensions for cluster test
    # Shape: (n_folds, n_layers * n_layers)
    data_flat = data_centered.reshape(n_folds, -1)

    # Set up adjacency for 2D grid (each pixel connected to its 4 neighbors)
    def get_2d_adjacency(n_rows, n_cols):
        """Create adjacency matrix for 2D grid connectivity."""
        n_points = n_rows * n_cols
        row_indices = []
        col_indices = []

        for i in range(n_rows):
            for j in range(n_cols):
                idx = i * n_cols + j
                # Connect to neighbors (4-connectivity)
                if i > 0:  # up
                    neighbor_idx = (i - 1) * n_cols + j
                    row_indices.extend([idx, neighbor_idx])
                    col_indices.extend([neighbor_idx, idx])
                if i < n_rows - 1:  # down
                    neighbor_idx = (i + 1) * n_cols + j
                    row_indices.extend([idx, neighbor_idx])
                    col_indices.extend([neighbor_idx, idx])
                if j > 0:  # left
                    neighbor_idx = i * n_cols + (j - 1)
                    row_indices.extend([idx, neighbor_idx])
                    col_indices.extend([neighbor_idx, idx])
                if j < n_cols - 1:  # right
                    neighbor_idx = i * n_cols + (j + 1)
                    row_indices.extend([idx, neighbor_idx])
                    col_indices.extend([neighbor_idx, idx])

        # Create sparse matrix
        data = np.ones(len(row_indices), dtype=bool)
        adjacency = sparse.coo_matrix(
            (data, (row_indices, col_indices)),
            shape=(n_points, n_points),
            dtype=bool,
        )
        return adjacency.tocsr()  # Convert to CSR format for efficiency

    adjacency = get_2d_adjacency(n_layers, n_layers)

    # Conservative t-threshold for two-tailed test
    df = n_folds - 1
    t_thresh = stats.t.ppf(1 - alpha / 2, df)

    # Run cluster-based permutation test
    T_obs, clusters, p_values, _ = permutation_cluster_1samp_test(
        data_flat,
        n_permutations=n_permutations,
        threshold=t_thresh,
        tail=0,  # two-tailed
        adjacency=adjacency,
        out_type="indices",
        verbose=True,
    )

    # Reshape T_obs back to 2D
    T_obs_2d = T_obs.reshape(n_layers, n_layers)

    # Process significant clusters
    sig_clusters_positive = []
    sig_clusters_negative = []

    for cluster, p_val in zip(clusters, p_values):
        if p_val <= alpha:
            cluster_indices = cluster[0] if isinstance(cluster, tuple) else cluster
            # Convert flat indices back to 2D coordinates
            coords_2d = [(idx // n_layers, idx % n_layers) for idx in cluster_indices]

            # Determine if cluster is positive or negative based on mean T value
            cluster_t_values = T_obs[cluster_indices]
            if cluster_t_values.mean() > 0:
                sig_clusters_positive.append(
                    (coords_2d, p_val, cluster_t_values.mean())
                )
            else:
                sig_clusters_negative.append(
                    (coords_2d, p_val, cluster_t_values.mean())
                )

    return {
        "T_obs": T_obs_2d,
        "mean_matrix": np.nanmean(fold_matrices, axis=0),
        "std_matrix": np.nanstd(fold_matrices, axis=0),
        "sig_clusters_positive": sig_clusters_positive,
        "sig_clusters_negative": sig_clusters_negative,
        "n_folds": n_folds,
    }


def plot_gat_with_clusters(results, layers, output_prefix):
    """Plot GAT matrix with significant clusters highlighted as contours."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Plot the GAT matrix as background
    im = ax.imshow(
        results["mean_matrix"],
        origin="lower",
        cmap="RdBu_r",
        vmin=0,
        vmax=1,
        interpolation="nearest",
        alpha=0.8,
    )

    # Create cluster masks
    pos_mask = np.zeros_like(results["mean_matrix"])
    neg_mask = np.zeros_like(results["mean_matrix"])

    # Mark positive clusters (above chance)
    for coords, p_val, t_mean in results["sig_clusters_positive"]:
        for i, j in coords:
            pos_mask[i, j] = 1

    # Mark negative clusters (below chance)
    for coords, p_val, t_mean in results["sig_clusters_negative"]:
        for i, j in coords:
            neg_mask[i, j] = 1

    # Draw contours around significant clusters
    if pos_mask.sum() > 0:
        ax.contour(
            pos_mask,
            levels=[0.5],
            colors=["red"],
            linewidths=3,
            linestyles="-",
            alpha=0.9,
        )
        # Optionally add filled contours with low alpha
        ax.contourf(pos_mask, levels=[0.5, 1.5], colors=["red"], alpha=0.2)

    if neg_mask.sum() > 0:
        ax.contour(
            neg_mask,
            levels=[0.5],
            colors=["blue"],
            linewidths=3,
            linestyles="-",
            alpha=0.9,
        )
        # Optionally add filled contours with low alpha
        ax.contourf(neg_mask, levels=[0.5, 1.5], colors=["blue"], alpha=0.2)

    # Add diagonal line for reference
    ax.plot(
        [0, len(layers) - 1],
        [0, len(layers) - 1],
        "k--",
        alpha=0.5,
        linewidth=1,
    )

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("AUC", rotation=270, labelpad=20, fontsize=12)

    # Labels and title
    ax.set_xlabel("Test Layer", fontsize=14, fontweight="bold")
    ax.set_ylabel("Train Layer", fontsize=14, fontweight="bold")
    ax.set_title(
        "Generalization Across Layers",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    # Set ticks
    tick_positions = list(range(0, len(layers), 4))
    tick_labels = [str(layers[i]) for i in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)

    # Create custom legend for clusters
    legend_elements = []
    if results["sig_clusters_positive"]:
        legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                color="red",
                lw=3,
                label="Above chance (p<0.01)",
            )
        )
    if results["sig_clusters_negative"]:
        legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                color="blue",
                lw=3,
                label="Below chance (p<0.01)",
            )
        )

    if legend_elements:
        ax.legend(
            handles=legend_elements,
            loc="upper right",
            frameon=True,
            fancybox=True,
            shadow=True,
            fontsize=10,
        )

    plt.tight_layout()

    # High-quality output
    plt.savefig(
        f"figures/{output_prefix}.pdf",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.savefig(
        f"figures/{output_prefix}.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )

    # plt.show()

    # Print cluster information
    print("\n=== SIGNIFICANT CLUSTERS ===")
    print(f"Total folds: {results['n_folds']}")

    if results["sig_clusters_positive"]:
        print(f"\nPositive clusters (AUC > 0.5):")
        for i, (coords, p_val, t_mean) in enumerate(results["sig_clusters_positive"]):
            print(
                f"  Cluster {i+1}: {len(coords)} pixels, p = {p_val:.4f}, t = {t_mean:.3f}"
            )
            # Show layer ranges
            train_layers = sorted(set(coord[0] for coord in coords))
            test_layers = sorted(set(coord[1] for coord in coords))
            print(
                f"    Train layers: {train_layers[0]}-{train_layers[-1]}, Test layers: {test_layers[0]}-{test_layers[-1]}"
            )
    else:
        print("\nNo significant positive clusters found.")

    if results["sig_clusters_negative"]:
        print(f"\nNegative clusters (AUC < 0.5):")
        for i, (coords, p_val, t_mean) in enumerate(results["sig_clusters_negative"]):
            print(
                f"  Cluster {i+1}: {len(coords)} pixels, p = {p_val:.4f}, t = {t_mean:.3f}"
            )
            # Show layer ranges
            train_layers = sorted(set(coord[0] for coord in coords))
            test_layers = sorted(set(coord[1] for coord in coords))
            print(
                f"    Train layers: {train_layers[0]}-{train_layers[-1]}, Test layers: {test_layers[0]}-{test_layers[-1]}"
            )
    else:
        print("\nNo significant negative clusters found.")


def plot_angle_matrix(angle_matrix, layers, output_prefix):
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    vmin = 0
    vmax = np.pi  # /2

    im = ax.imshow(
        angle_matrix,
        origin="lower",
        cmap="viridis",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Rotation Angle (rad)", rotation=270, labelpad=20, fontsize=12)
    # cbar.set_ticks([0, np.pi/4, np.pi/2])
    # cbar.set_ticklabels(['0', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$'])
    cbar.set_ticks([0, np.pi / 2, np.pi])
    cbar.set_ticklabels(["0", r"$\frac{\pi}{2}$", r"$\pi$"])
    cbar.ax.tick_params(labelsize=18)

    # isoline at np.pi/4
    ax.contour(
        angle_matrix,
        levels=[np.pi / 4],
        colors=["white"],
        linewidths=2,
        linestyles="--",
        alpha=0.9,
    )

    # isoline at np.pi/2
    ax.contour(
        angle_matrix,
        levels=[np.pi / 2],
        colors=["red"],
        linewidths=2,
        linestyles="--",
        alpha=0.9,
    )

    # Labels and title
    ax.set_xlabel("Layer j", fontsize=14, fontweight="bold")
    ax.set_ylabel("Layer i", fontsize=14, fontweight="bold")
    ax.set_title(
        "Rotation Angle Matrix",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    # Set ticks
    tick_positions = list(range(0, len(layers), 4))
    tick_labels = [str(layers[i]) for i in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)

    plt.tight_layout()

    # High-quality output
    plt.savefig(
        f"figures/{output_prefix}_angles.pdf",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.savefig(
        f"figures/{output_prefix}_angles.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )


def plot_density(density, layers, output_prefix):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    ax.plot(layers, density, marker="o", linestyle="-", color="blue")

    ax.set_xlabel("Layer", fontsize=14, fontweight="bold")
    ax.set_ylabel("Feature Density", fontsize=14, fontweight="bold")
    ax.set_title(
        "Feature Density Across Layers",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    ax.set_xticks(layers[::4])
    ax.set_xticklabels([str(l) for l in layers[::4]])

    plt.tight_layout()

    # High-quality output
    plt.savefig(
        f"figures/{output_prefix}_density.pdf",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.savefig(
        f"figures/{output_prefix}_density.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )


def plot_coefficients(coeffs, layers, output_prefix):
    """Imshow of coefficients
    X axis = layers
    Y axis = the 4 coefficients (mean, var, kurtosis, skewness)
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # 2 slopes color map, centered on 0
    cmap = plt.get_cmap("RdBu_r")

    im = ax.imshow(
        coeffs.T,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        interpolation="nearest",
        vmin=-np.max(np.abs(coeffs)),
        vmax=np.max(np.abs(coeffs)),
    )

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Coefficient Value", rotation=270, labelpad=20, fontsize=12)

    # Labels and title
    ax.set_xlabel("Layer", fontsize=14, fontweight="bold")
    ax.set_ylabel("Coefficient", fontsize=14, fontweight="bold")
    ax.set_title(
        "Logistic Regression Coefficients",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    # Set ticks
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([str(l) for l in layers])
    ax.set_yticks(range(4))
    ax.set_yticklabels(["Mean", "Variance", "Kurtosis", "Skewness"])

    plt.tight_layout()

    plt.savefig(
        f"figures/{output_prefix}_coefficients.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate Figure 2: Generalization Across Time (GAT) matrix with cluster-based permutation tests."
    )
    parser.add_argument(
        "--activations_path",
        type=str,
        default="activations",
        help="Path to activation files directory (default: krikri_activations)",
    )
    parser.add_argument(
        "--n_layers",
        type=int,
        default=32,
        help="Number of layers in the model (default: 32)",
    )
    parser.add_argument(
        "--n_folds",
        type=int,
        default=20,
        help="Number of cross-validation folds (default: 20)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for cluster tests (default: 0.05)",
    )
    parser.add_argument(
        "--n_permutations",
        type=int,
        default=5000,
        help="Number of permutations for cluster tests (default: 5000)",
    )
    parser.add_argument(
        "--region",
        type=str,
        default="aft",
        choices=["aft", "bef"],
        help="Region to analyze: aft (post-clause) or bef (pre-clause) (default: aft)",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="figure_2",
        help="Output filename prefix (default: figure_2)",
    )
    args = parser.parse_args()

    # Settings
    path_to_data = args.activations_path
    layers = list(range(args.n_layers))
    n_folds = args.n_folds
    alpha = args.alpha
    n_permutations = args.n_permutations

    print(f"Loading activations from {path_to_data}...")
    # Load data and extract features
    activ_files = load_activations(path_to_data)
    df_feats = extract_features(activ_files, layers)

    print(f"Computing GAT matrices for {n_folds} folds...")
    # Compute fold-wise GAT matrices
    fold_matrices, patterns = compute_gat_folds(
        df_feats, layers, region_prefix="aft", n_splits=n_folds
    )

    print(
        f"Running cluster-based permutation test with {n_permutations} permutations..."
    )
    # Perform cluster-based permutation test
    results = perform_cluster_permutation_2d(
        fold_matrices, alpha=alpha, n_permutations=n_permutations
    )

    print("Plotting results...")
    plot_gat_with_clusters(results, layers, args.output_prefix)

    print("Computing angles and feature density from patterns")
    angles = get_rotation_angle_matrix(patterns)
    plot_angle_matrix(angles, layers, args.output_prefix)

    plot_coefficients(patterns, layers, args.output_prefix)
    print("Done with all plots")


if __name__ == "__main__":
    main()
