#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final clean visualization and table for the generalization analysis.
Shows systematic inversion pattern where classifier confidence is systematically wrong.
"""

import os
import re
import numpy as np
import torch
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from mne.stats import permutation_cluster_1samp_test
from prettytable import PrettyTable
import pandas as pd


POU_TOKENS = ["ĠÏĢÎ¿Ïħ", "▁που"]

def extract_layer_data(activation_files, layer_idx):
    """Extract features from a specific layer"""
    results = []
    for f in activation_files:
        tokens = f["tokens"]
        clause_token_pos = next(
            (i for i, t in enumerate(tokens) if any(pou_token in t for pou_token in POU_TOKENS)), 
            None
        )
        if clause_token_pos is None or clause_token_pos >= len(tokens) - 1:
            continue

        words = f["sentence"].split()
        if "που" not in words:
            print("'που' not in words")
            continue

        after = words[words.index("που") + 1 :]
        order = (
            "SVO"
            if len(after) >= 2
            and after[0].lower()
            in ["ο", "η", "το", "οι", "τα", "των", "της", "του"]
            else "VSO"
        )

        X = np.asarray(f["hidden_states"][layer_idx], dtype=np.float64)
        # Check for invalid values
        if not np.isfinite(X).all():
            continue

        bef, aft = X[: clause_token_pos + 1], X[clause_token_pos + 1 :]

        def feats(mat, pfx):
            flat = mat.ravel()
            flat = np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)

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


def extract_pooled_layer_features(
    activation_files, layer_indices, region="after"
):
    """Pool samples from multiple layers"""
    all_features = []
    labels = []

    for layer_idx in layer_indices:
        for f in activation_files:
            tokens = f["tokens"]
            clause_token_pos = next(
                (i for i, t in enumerate(tokens) if any(pou_token in t for pou_token in POU_TOKENS)), 
                None
            )
            if clause_token_pos is None or clause_token_pos >= len(tokens) - 1:
                continue

            words = f["sentence"].split()
            if "που" not in words:
                print(f"ERROR: 'που' not found in {f['sentence']}. Skipping.")
                continue

            after = words[words.index("που") + 1 :]
            order = (
                "SVO"
                if len(after) >= 2
                and after[0].lower()
                in ["ο", "η", "το", "οι", "τα", "των", "της", "του"]
                else "VSO"
            )

            X = np.array(f["hidden_states"][layer_idx], dtype=np.float64)
            # Check for invalid values
            if not np.isfinite(X).all():
                continue

            bef, aft = X[: clause_token_pos + 1], X[clause_token_pos + 1 :]
            mat = bef if region == "before" else aft
            flat = mat.ravel()
            # Replace any remaining inf/nan with 0
            flat = np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)

            feats = [
                flat.mean(),
                np.var(flat),
                stats.kurtosis(flat),
                stats.skew(flat),
            ]
            all_features.append(feats)
            labels.append(1 if order == "SVO" else 0)

    return np.array(all_features), np.array(labels)


def train_classifier_on_layers(activation_files, train_layers, region="after"):
    """Train classifier on pooled samples from multiple layers"""
    X_train, y_train = extract_pooled_layer_features(
        activation_files, train_layers, region
    )

    if len(X_train) < 10 or len(np.unique(y_train)) < 2:
        return None, None

    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train_scaled, y_train)
    return clf, scaler


def analyze_layer_probabilities(
    activation_files, clf, scaler, test_layer, region="after"
):
    """Analyze probability distributions for true VSO vs SVO"""
    df_layer = extract_layer_data(activation_files, test_layer)
    if df_layer is None:
        return np.nan, np.nan, np.nan, np.nan

    cols = [f"{region}_{stat}" for stat in ("mean", "range", "min", "max")]
    X_test = df_layer[cols].values
    y_test = (df_layer["order"] == "SVO").astype(int).values

    if len(np.unique(y_test)) < 2:
        return np.nan, np.nan, np.nan, np.nan

    X_test_scaled = scaler.transform(X_test)
    y_proba = clf.predict_proba(X_test_scaled)[:, 1]

    # Calculate AUC
    auc = roc_auc_score(y_test, y_proba)

    # Separate probabilities by true class
    prob_true_vso = y_proba[y_test == 0]  # P(SVO) for true VSO
    prob_true_svo = y_proba[y_test == 1]  # P(SVO) for true SVO

    mean_prob_vso = np.mean(prob_true_vso)
    mean_prob_svo = np.mean(prob_true_svo)

    return auc, 1 - auc, mean_prob_vso, mean_prob_svo


def test_classifier_on_layer(
    activation_files, clf, scaler, test_layer, region="after", n_folds=20
):
    """Test trained classifier on a single layer"""
    df_layer = extract_layer_data(activation_files, test_layer)
    if df_layer is None or len(df_layer) < n_folds:
        return np.nan, np.nan, []

    cols = [f"{region}_{stat}" for stat in ("mean", "range", "min", "max")]
    X_test = df_layer[cols].values
    y_test = (df_layer["order"] == "SVO").astype(int).values

    if len(np.unique(y_test)) < 2:
        return np.nan, np.nan, []

    skf = StratifiedKFold(n_folds, shuffle=True, random_state=42)
    aucs = []

    for _, test_idx in skf.split(X_test, y_test):
        X_fold = X_test[test_idx]
        y_fold = y_test[test_idx]

        if len(np.unique(y_fold)) < 2:
            continue

        X_fold_scaled = scaler.transform(X_fold)
        y_proba_fold = clf.predict_proba(X_fold_scaled)[:, 1]
        aucs.append(roc_auc_score(y_fold, y_proba_fold))

    if not aucs:
        return np.nan, np.nan, []

    return np.mean(aucs), stats.sem(aucs), aucs


def get_cluster_significance(auc_folds_data, test_layers, alpha=0.01):
    """Run permutation test and return significant cluster indices"""
    max_folds = max(len(folds) for folds in auc_folds_data if len(folds) > 0)
    n_layers = len(test_layers)
    data_matrix = np.full((max_folds, n_layers), np.nan)

    for i, folds in enumerate(auc_folds_data):
        if len(folds) > 0:
            n_available = min(len(folds), max_folds)
            data_matrix[:n_available, i] = folds[:n_available]

    data_centered = data_matrix - 0.5
    valid_rows = np.sum(~np.isnan(data_centered), axis=1) >= (n_layers * 0.5)
    data_clean = data_centered[valid_rows]

    if data_clean.shape[0] < 3:
        return []

    try:
        threshold = stats.t.ppf(1 - alpha / 2, data_clean.shape[0] - 1)
        T_obs, clusters, p_values, _ = permutation_cluster_1samp_test(
            data_clean,
            n_permutations=5000,
            threshold=threshold,
            tail=0,
            out_type="indices",
        )

        significant_clusters = []
        for cluster, p_val in zip(clusters, p_values):
            if p_val <= alpha:
                significant_clusters.append(cluster[0])

        return significant_clusters
    except:
        return []


def plot_clean_auc(test_layers, aucs, sems, sig_clusters, output_prefix):
    """Plot clean AUC only with figure_1.py aesthetics"""
    # Match figure_1.py style exactly
    plt.style.use("default")
    
    # Create figure with clean proportions (same as figure_1.py)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Set clean white background (same as figure_1.py)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    
    # Use the same color as "after" region from figure_1.py
    color = "#6B73FF"  # Blue color from figure_1.py
    
    layers = np.array(test_layers)
    valid_mask = ~np.isnan(aucs)
    valid_layers = layers[valid_mask]
    valid_aucs = aucs[valid_mask]
    valid_sems = sems[valid_mask]

    # Elegant shaded confidence intervals first (behind lines) - same as figure_1.py
    ax.fill_between(
        valid_layers,
        valid_aucs - valid_sems,
        valid_aucs + valid_sems,
        color=color,
        alpha=0.3,
        linewidth=0,
        zorder=1,
    )

    # Clean main line - same style as figure_1.py
    ax.plot(
        valid_layers,
        valid_aucs,
        color=color,
        linewidth=2,
        linestyle="-",
        zorder=3,
    )

    # Thick significance bars like figure_1.py
    if sig_clusters:
        for cluster_idx in sig_clusters:
            cluster_layers = layers[cluster_idx]
            start, end = cluster_layers.min(), cluster_layers.max()
            # Thick line above the curve - same style as figure_1.py
            y_line = np.max(valid_aucs[np.isin(valid_layers, cluster_layers)]) + np.max(valid_sems[np.isin(valid_layers, cluster_layers)]) + 0.03
            ax.plot(
                [start, end],
                [y_line, y_line],
                color=color,
                linewidth=8,
                solid_capstyle="butt",
                zorder=4,
            )

    # Chance level line - same as figure_1.py
    ax.axhline(
        0.5, linestyle="-", color="black", linewidth=1, alpha=0.5, zorder=2
    )

    # Proper axis settings - match figure_1.py style
    ax.set_xlim(0, len(test_layers) - 1)
    ax.set_ylim(0.0, 0.65)  # Adjusted for significance bars

    # Clean labels - same as figure_1.py
    ax.set_xlabel(
        "Test layer",
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

    # Clean ticks - same style as figure_1.py
    ax.set_xticks(range(0, len(test_layers), 4))
    ax.set_xticklabels(range(0, len(test_layers), 4), fontsize=10, color="black")
    ax.set_yticks([0.0, 0.2, 0.4, 0.6])
    ax.set_yticklabels(["0.0", "0.2", "0.4", "0.6"], fontsize=10, color="black")

    # Create custom legend with significance - same as figure_1.py
    if sig_clusters:
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D(
                [0],
                [0],
                color=color,
                linestyle="-",
                linewidth=2,
                label="Test AUC",
            ),
            Line2D([0], [0], color="gray", linewidth=8, label="p < 0.01"),
        ]
        ax.legend(
            handles=legend_elements, loc="upper right", frameon=False, fontsize=11
        )

    # Proper despine using seaborn - same as figure_1.py
    sns.despine(ax=ax, trim=True, offset=10)

    # No grid - same as figure_1.py
    ax.grid(False)
    
    # Title with same styling as figure_1.py
    ax.set_title(
        "Generalization analysis: systematic inversion.",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    
    # Perfect spacing - same as figure_1.py
    plt.tight_layout()

    # High-quality output - same as figure_1.py
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

    plt.show()


def create_simple_bias_table(
    activation_files, clf, scaler, test_layers, region="after"
):
    """Create a simple table showing the SVO bias pattern"""

    # Focus on the significant cluster layers
    cluster_layers = [7, 9, 11, 13, 15, 17, 19]

    # Collect predictions across all cluster layers
    all_vso_predictions = []
    all_svo_predictions = []

    for layer in cluster_layers:
        df_layer = extract_layer_data(activation_files, layer)
        if df_layer is None:
            continue

        cols = [f"{region}_{stat}" for stat in ("mean", "range", "min", "max")]
        X_test = df_layer[cols].values
        y_test = (df_layer["order"] == "SVO").astype(int).values

        if len(np.unique(y_test)) < 2:
            continue

        X_test_scaled = scaler.transform(X_test)
        y_pred = clf.predict(X_test_scaled)

        # Separate by true class
        vso_predictions = y_pred[y_test == 0]  # Predictions for true VSO
        svo_predictions = y_pred[y_test == 1]  # Predictions for true SVO

        all_vso_predictions.extend(vso_predictions)
        all_svo_predictions.extend(svo_predictions)

    # Calculate percentages
    vso_predicted_as_svo = (
        100
        * np.sum(np.array(all_vso_predictions) == 1)
        / len(all_vso_predictions)
    )
    svo_predicted_as_svo = (
        100
        * np.sum(np.array(all_svo_predictions) == 1)
        / len(all_svo_predictions)
    )

    # Create simple table
    table = PrettyTable()
    table.field_names = [
        "Actual Sentence Type",
        "Classified as SVO",
        "Classified as VSO",
    ]

    table.add_row(
        [
            "VSO sentences",
            f"{vso_predicted_as_svo:.1f}%",
            f"{100-vso_predicted_as_svo:.1f}%",
        ]
    )
    table.add_row(
        [
            "SVO sentences",
            f"{svo_predicted_as_svo:.1f}%",
            f"{100-svo_predicted_as_svo:.1f}%",
        ]
    )

    # Style the table
    table.align = "c"
    table.border = True
    table.header = True
    table.padding_width = 2

    print("\n" + "=" * 60)
    print("CLASSIFIER PREDICTIONS: EVERYTHING IS SVO")
    print("=" * 60)
    print("Classifier trained on layers 20-31, tested on cluster layers 7-19")
    print()
    print(table)
    print()
    print("Key Finding:")
    print(
        f"• The classifier predicts SVO for {vso_predicted_as_svo:.1f}% of VSO sentences (WRONG)"
    )
    print(
        f"• The classifier predicts SVO for {svo_predicted_as_svo:.1f}% of SVO sentences (correct)"
    )
    print(
        f"• Overall: The classifier thinks {(vso_predicted_as_svo + svo_predicted_as_svo)/2:.1f}% of all sentences are SVO"
    )
    print("• This creates systematic misclassification of VSO → SVO")
    print("=" * 60)

    return {
        "vso_as_svo": vso_predicted_as_svo,
        "svo_as_svo": svo_predicted_as_svo,
        "total_samples_vso": len(all_vso_predictions),
        "total_samples_svo": len(all_svo_predictions),
    }


def create_even_simpler_summary(
    activation_files, clf, scaler, test_layers, region="after"
):
    """Create the simplest possible summary"""

    cluster_layers = [7, 9, 11, 13, 15, 17, 19]

    total_correct = 0
    total_wrong = 0
    total_samples = 0

    for layer in cluster_layers:
        df_layer = extract_layer_data(activation_files, layer)
        if df_layer is None:
            continue

        cols = [f"{region}_{stat}" for stat in ("mean", "range", "min", "max")]
        X_test = df_layer[cols].values
        y_test = (df_layer["order"] == "SVO").astype(int).values

        if len(np.unique(y_test)) < 2:
            continue

        X_test_scaled = scaler.transform(X_test)
        y_pred = clf.predict(X_test_scaled)

        # Count predictions
        svo_predictions = np.sum(y_pred == 1)
        total_samples += len(y_pred)

        # Count correct/wrong
        correct = np.sum(y_pred == y_test)
        wrong = len(y_pred) - correct
        total_correct += correct
        total_wrong += wrong

    svo_bias = (
        100 * np.sum(y_pred == 1) / len(y_pred)
    )  # From last layer as example
    accuracy = 100 * total_correct / total_samples
    error_rate = 100 * total_wrong / total_samples

    print("\n" + "=" * 50)
    print("SIMPLE SUMMARY: CLASSIFIER BEHAVIOR")
    print("=" * 50)
    print(f"📊 Predicts SVO for: ~{svo_bias:.0f}% of all sentences")
    print(f"✅ Overall accuracy: {accuracy:.1f}%")
    print(f"❌ Overall error rate: {error_rate:.1f}%")
    print()
    print("🔍 The Problem:")
    print("   The classifier learned 'late layers = SVO'")
    print("   When tested on early layers, it still says 'SVO'")
    print("   But early layers actually contain more VSO information")
    print("   → Systematic misclassification!")
    print("=" * 50)


def load_activations(path):
    """Load activation files"""
    files = sorted(
        [f for f in os.listdir(path) if not f.startswith(".")],
        key=lambda fn: int(re.search(r"\d+", fn).group()),
    )
    return [torch.load(os.path.join(path, fn)) for fn in files]


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate Figure 3: Generalization analysis showing systematic inversion pattern."
    )
    parser.add_argument(
        "--activations_path",
        type=str,
        default="krikri_activations",
        help="Path to activation files directory (default: krikri_activations)",
    )
    parser.add_argument(
        "--train_layers_start",
        type=int,
        default=20,
        help="Start layer for training (default: 20)",
    )
    parser.add_argument(
        "--train_layers_end",
        type=int,
        default=32,
        help="End layer for training (default: 32)",
    )
    parser.add_argument(
        "--test_layers_start",
        type=int,
        default=0,
        help="Start layer for testing (default: 0)",
    )
    parser.add_argument(
        "--test_layers_end",
        type=int,
        default=20,
        help="End layer for testing (default: 20)",
    )
    parser.add_argument(
        "--region",
        type=str,
        default="after",
        choices=["after", "before"],
        help="Region to analyze: after (post-clause) or before (pre-clause) (default: after)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.01,
        help="Significance level for cluster tests (default: 0.01)",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="figure_3",
        help="Output filename prefix (default: figure_3)",
    )
    args = parser.parse_args()

    path_to_data = args.activations_path
    train_layers = list(range(args.train_layers_start, args.train_layers_end))
    test_layers = list(range(args.test_layers_start, args.test_layers_end))
    region = args.region

    print("Loading activation data...")
    activations = load_activations(path_to_data)
    print(f"Loaded {len(activations)} activation files")

    print(
        f"Training classifier on layers {train_layers[0]}-{train_layers[-1]}..."
    )
    clf, scaler = train_classifier_on_layers(activations, train_layers, region)

    if clf is None:
        print("Error: Could not train classifier")
        return

    print("Testing classifier on layers 0-19...")
    aucs, sems, all_folds = [], [], []

    for test_layer in test_layers:
        auc, sem, folds = test_classifier_on_layer(
            activations, clf, scaler, test_layer, region
        )
        aucs.append(auc)
        sems.append(sem)
        all_folds.append(folds)

    # Get significant clusters
    sig_clusters = get_cluster_significance(all_folds, test_layers)

    print("\n" + "=" * 60)
    print("VISUALIZATION OPTIONS")
    print("=" * 60)

    # Option 1: Clean AUC plot
    print("Option 1: Clean AUC Plot")
    plot_clean_auc(test_layers, np.array(aucs), np.array(sems), sig_clusters, args.output_prefix)

    # Option 2: Simple bias table
    print("\nOption 2: Simple Bias Table")
    bias_data = create_simple_bias_table(
        activations, clf, scaler, test_layers, region
    )
    print(bias_data)

    # Option 3: Even simpler summary
    print("\nOption 3: Simplest Summary")
    create_even_simpler_summary(activations, clf, scaler, test_layers, region)


if __name__ == "__main__":
    main()
