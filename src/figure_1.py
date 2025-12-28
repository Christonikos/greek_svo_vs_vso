#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cluster_perm_layers_two_tailed_lines.py

Run two‐tailed cluster‐based permutation tests on “before” / “after” AUC curves per layer,
draw thick horizontal lines above (post‐clause) or below (pre‐clause) the curves where clusters
are significant at α = 0.05 (two‐tailed), using a conservative t‐threshold based on df = n_folds−1.

Requires: numpy, scipy, torch, pandas, sklearn, matplotlib, mne
"""
import os
import re
import numpy as np
import torch
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from mne.stats import permutation_cluster_1samp_test
import seaborn as sns

POU_TOKENS = ["ĠÏĢÎ¿Ïħ", "▁που"]

def extract_layer_data(activation_files, layer_idx):
    results = []
    for f in activation_files:
        tokens = f["tokens"]
        clause_token_pos = next(
            (i for i, t in enumerate(tokens) if any(pou_token in t for pou_token in POU_TOKENS)), 
            None
        )
        if clause_token_pos is None or clause_token_pos >= len(tokens) - 1:
            print("clause_token_pos not found")
            continue

        words = f["sentence"].split()
        after = words[words.index("που") + 1 :]
        order = (
            "SVO"
            if len(after) >= 2
            and after[0].lower()
            in ["ο", "η", "το", "οι", "τα", "των", "της", "του"]
            else "VSO"
        )

        X = np.array(f["hidden_states"][layer_idx], dtype=np.float64)
        bef, aft = X[: clause_token_pos + 1], X[clause_token_pos + 1 :]

        def feats(mat, pfx):
            flat = mat.ravel()
            return {
                f"{pfx}_mean": flat.mean(),
                f"{pfx}_range": np.var(flat),
                f"{pfx}_min": stats.kurtosis(flat),
                f"{pfx}_max": stats.skew(flat),
            }

        row = {"order": order, "layer": layer_idx}
        row.update(feats(bef, "before"))
        row.update(feats(aft, "after"))
        results.append(row)

    if not results:
        return None
    return pd.DataFrame(results)


def stratified_cv_auc_multifeature(df, region, return_folds=False):
    cols = [f"{region}_{stat}" for stat in ("mean", "range", "min", "max")]
    X = df[cols].values

    # Clean any remaining invalid values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    y = (df["order"] == "SVO").astype(int).values
    if len(df) < 10 or len(np.unique(y)) < 2:
        return (np.nan, np.nan, []) if return_folds else (np.nan, np.nan)

    skf = StratifiedKFold(10, shuffle=True, random_state=42)
    aucs = []
    for tr, te in skf.split(X, y):
        Xtr, Xt = X[tr], X[te]
        ytr, yt = y[tr], y[te]
        if len(np.unique(yt)) < 2:
            continue
        scl = RobustScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(scl.transform(Xtr), ytr)
        probs = clf.predict_proba(scl.transform(Xt))[:, 1]
        aucs.append(roc_auc_score(yt, probs))

    if not aucs:
        return (np.nan, np.nan, []) if return_folds else (np.nan, np.nan)
    return (
        (np.mean(aucs), stats.sem(aucs), aucs)
        if return_folds
        else (np.mean(aucs), stats.sem(aucs))
    )


def load_activations(path):
    files = sorted(
        [f for f in os.listdir(path) if not f.startswith(".")],
        key=lambda fn: int(re.search(r"\d+", fn).group()),
    )
    return [torch.load(os.path.join(path, fn)) for fn in files]


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate Figure 1: Sentence type classification per layer with cluster-based permutation tests."
    )
    parser.add_argument(
        "--activations_path",
        type=str,
        default="krikri_activations",
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
        default=0.01,
        help="Significance level for cluster tests (default: 0.01)",
    )
    parser.add_argument(
        "--n_permutations",
        type=int,
        default=5000,
        help="Number of permutations for cluster tests (default: 5000)",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="figure_1",
        help="Output filename prefix (default: figure_1)",
    )
    args = parser.parse_args()

    os.makedirs("figures", exist_ok=True)
    # ─── Settings ───────────────────────────────────────────────────────────────
    path_to_data = args.activations_path
    layers = list(range(args.n_layers))
    regions = ["before", "after"]
    n_folds = args.n_folds
    alpha = args.alpha
    df = n_folds - 1
    # two‐tailed critical t for α=0.01:
    t_thresh = stats.t.ppf(1 - alpha / 2, df)
    # ────────────────────────────────────────────────────────────────────────────

    # Load and compute per‐fold AUCs
    activs = load_activations(path_to_data)
    results = {r: {"mean": [], "sem": [], "folds": []} for r in regions}

    for L in layers:
        df_layer = extract_layer_data(activs, L)
        if df_layer is None or len(df_layer) < n_folds:
            for r in regions:
                results[r]["mean"].append(np.nan)
                results[r]["sem"].append(np.nan)
                results[r]["folds"].append([np.nan] * n_folds)
            continue
        for r in regions:
            mn, se, folds = stratified_cv_auc_multifeature(
                df_layer, r, return_folds=True
            )
            results[r]["mean"].append(mn)
            results[r]["sem"].append(se)
            results[r]["folds"].append(folds)

    # ─── Cluster‐based permutation, two‐tailed ──────────────────────────────────
    sig_clusters = {}
    T_obs_dict = {}

    for r in regions:
        data = np.array(results[r]["folds"])  # (layers, folds)
        data = (np.nan_to_num(data) - 0.5).T  # (folds, layers)

        T_obs, clusters, p_values, _ = permutation_cluster_1samp_test(
            data,
            n_permutations=5000,
            threshold=t_thresh,  # two‐tailed threshold
            tail=0,  # 0 = two‐tailed
            out_type="indices",
            verbose=True,
        )
        T_obs_dict[r] = T_obs

        good = []
        for c, p in zip(clusters, p_values):
            if p <= alpha:
                idxs = c[0] if isinstance(c, tuple) else c
                good.append((idxs, p))
        sig_clusters[r] = good

        print(f"\n=== REGION: {r} ===")
        if not good:
            print("  no significant clusters.")
        for idxs, p in good:
            print(f"  * cluster at layers {idxs.tolist()}  (p = {p:.3f})")

    # ─── Plot with horizontal significance bars ─────────────────────────────────
    plt.style.use("default")

    # Create figure with clean proportions
    fig, ax = plt.subplots(figsize=(8, 6))

    # Set clean white background
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Reversed color scheme - blue for pre-clause, pink for post-clause
    colors = {
        "before": "#FF9999",
        "after": "#6B73FF",
    }  # Pink for pre, blue for post
    labels = {"before": "Pre-clause", "after": "Post-clause"}

    # Plot data with clean, minimalist style
    for r in regions:
        x = np.array(layers)
        y = np.array(results[r]["mean"])
        e = np.array(results[r]["sem"])
        valid = ~np.isnan(y)

        # Elegant shaded confidence intervals first (behind lines)
        ax.fill_between(
            x[valid],
            y[valid] - e[valid],
            y[valid] + e[valid],
            color=colors[r],
            alpha=0.3,
            linewidth=0,
            zorder=1,
        )

        # Clean main lines
        linestyle = "-" if r == "after" else "--"
        ax.plot(
            x[valid],
            y[valid],
            color=colors[r],
            linewidth=2,
            label=labels[r],
            linestyle=linestyle,
            zorder=3,
        )

        # Thick significance bars like reference
        T = T_obs_dict[r]
        for idxs, p_val in sig_clusters[r]:
            cluster_t = T[idxs]
            start, end = idxs.min(), idxs.max()

            if r == "after" and cluster_t.mean() > 0:
                # Thick line above the curve
                y_line = np.max(y[idxs] + e[idxs]) + 0.03
                ax.plot(
                    [start, end],
                    [y_line, y_line],
                    color=colors[r],
                    linewidth=8,
                    solid_capstyle="butt",
                    zorder=4,
                )

            elif r == "before" and cluster_t.mean() < 0:
                # Thick line below the curve
                y_line = np.min(y[idxs] - e[idxs]) - 0.03
                ax.plot(
                    [start, end],
                    [y_line, y_line],
                    color=colors[r],
                    linewidth=8,
                    solid_capstyle="butt",
                    zorder=4,
                )

    # Chance level line
    ax.axhline(
        0.5, linestyle="-", color="black", linewidth=1, alpha=0.5, zorder=2
    )

    # Proper axis settings - y-axis from 0.2 to 1
    ax.set_xlim(0, len(layers) - 1)
    ax.set_ylim(0.3, 1)

    # Clean labels
    ax.set_xlabel(
        "Transformer layer",
        fontsize=12,
        color="black",
        fontweight="bold",
    )
    ax.set_ylabel(
        "Area Under the Curve (AUC)",
        fontsize=12,
        color="black",
        fontweight="bold",
    )

    # Clean ticks
    ax.set_xticks(range(0, len(layers), 4))
    ax.set_xticklabels(range(0, len(layers), 4), fontsize=10, color="black")
    ax.set_yticks([0.3, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(
        ["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=10, color="black"
    )

    # Create custom legend with significance
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            color=colors["before"],
            linestyle="--",
            linewidth=2,
            label="Pre-clause",
        ),
        Line2D(
            [0],
            [0],
            color=colors["after"],
            linestyle="-",
            linewidth=2,
            label="Post-clause",
        ),
        Line2D([0], [0], color="gray", linewidth=8, label="p < 0.01"),
    ]
    ax.legend(
        handles=legend_elements, loc="upper right", frameon=False, fontsize=11
    )

    # Proper despine using seaborn
    sns.despine(ax=ax, trim=True, offset=10)

    # No grid
    ax.grid(False)
    ax.set_title(
        "Sentence type classification.",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    # Perfect spacing
    plt.tight_layout()

    # High-quality output
    plt.savefig(
        f"figures/{args.output_prefix}.pdf",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.savefig(
        f"figures/{args.output_prefix}.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )

    plt.show()


if __name__ == "__main__":
    main()
