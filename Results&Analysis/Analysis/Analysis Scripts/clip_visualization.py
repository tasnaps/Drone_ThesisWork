#!/usr/bin/env python3
"""
Visualize individual audio clips using t-SNE/UMAP.
Color-coded by class (drone/non-drone), symbols by dataset.

Uses the same CSV output from 9_audio_dataset_analyzer.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

# Try to import UMAP (optional)
try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: UMAP not installed. Only t-SNE will be available.")

# Configuration
IN_CSV = "audio_analysis_results/audio_dataset_stats_detailed.csv"
OUT_DIR = "audio_analysis_results/clip_visualizations"
FEATURES = ["length", "rms", "peak", "spec_avg", "crest_factor"]

# Sampling config
SAMPLING_MODE = "weighted_balanced"  # "weighted_balanced" or "uniform"

# Uniform sampling config
SAMPLES_PER_DATASET = 300  # Max samples per dataset (to avoid overcrowding)

# Weighted, class-balanced sampling config
TARGET_TOTAL_SAMPLES = 2500
MIN_SAMPLES_PER_DATASET = 60
MAX_SAMPLES_PER_DATASET = 350

RANDOM_STATE = 42

# Dataset display names (Finnish)
DATASET_DISPLAY_NAMES = {
    "Authors Collection": "Tämä pro gradu",
    "Drone_print": "DronePrint",
    #"FusionDataset": "Yhdistelmäaineisto",
    "Wonjun Yi": "Yi ym.",
    "S&E": "Svanström ym.",
    "Emo Soundscapes Mixes": "Emo Soundscapes",
    #"Calibration Dataset": "Kalibrointiaineisto",
    "ESC-50": "ESC-50",
    "H-2": "H-2",
    "Al-Emadi": "Al-Emadi ym.",
    "UrbanSound8K": "UrbanSound8K",
}

# Datasets included in the visualization
INCLUDED_DATASETS = set(DATASET_DISPLAY_NAMES.keys())

# Define which datasets are drone-only, non-drone-only, or mixed
# For mixed datasets, we'll try to infer from filepath
DRONE_ONLY_DATASETS = {"Wonjun Yi", "H-2", "Drone_print"}
NON_DRONE_ONLY_DATASETS = {"Emo Soundscapes Mixes", "ESC-50", "UrbanSound8K"}
MIXED_DATASETS = {"Al-Emadi", "S&E", "Authors Collection"}

# Path hints for class labeling (applies to all datasets)
POSITIVE_SUBDIRS = ["/yes_drone/", "\\yes_drone\\"]
NEGATIVE_SUBDIRS = ["/unknown/", "\\unknown\\"]

# Colorblind-friendly palette (Okabe-Ito + extensions)
ACCESSIBLE_PALETTE = [
    "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442",
    "#56B4E9", "#E69F00", "#000000", "#999999", "#332288",
    "#88CCEE", "#117733", "#44AA99", "#DDCC77", "#AA4499",
    "#882255", "#661100", "#6699CC", "#AA4466", "#4477AA",
]

CLASS_COLORS = {
    "drone": "#0072B2",   # blue
    "ei-drone": "#D55E00"  # orange
}


def infer_label_from_path(filepath: str, dataset: str) -> str:
    """Infer drone/non-drone label from filepath or dataset."""
    filepath_lower = filepath.lower()

    # Primary rule: explicit class subfolders
    for indicator in POSITIVE_SUBDIRS:
        if indicator in filepath_lower:
            return "drone"

    for indicator in NEGATIVE_SUBDIRS:
        if indicator in filepath_lower:
            return "ei-drone"

    # Drone-only datasets
    if dataset in DRONE_ONLY_DATASETS:
        return "drone"
    
    # Non-drone-only datasets
    if dataset in NON_DRONE_ONLY_DATASETS:
        return "ei-drone"
    
    # Mixed datasets - try to infer from path
    # Common patterns in paths
    drone_indicators = [
        "/drone/", "\\drone\\", "/yes_drone/", "\\yes_drone\\",
        "_drone_", "drone_", "/drones/", "\\drones\\",
        "/positive/", "\\positive\\"
    ]
    non_drone_indicators = [
        "/no_drone/", "\\no_drone\\", "/unknown/", "\\unknown\\",
        "/background/", "\\background\\", "/negative/", "\\negative\\",
        "/other/", "\\other\\"
    ]
    
    for indicator in drone_indicators:
        if indicator in filepath_lower:
            return "drone"
    
    for indicator in non_drone_indicators:
        if indicator in filepath_lower:
            return "ei-drone"
    
    # Al-Emadi specific: check for Drone vs Background folder
    if dataset == "Al-Emadi":
        if "drone" in filepath_lower and "background" not in filepath_lower:
            return "drone"
        else:
            return "ei-drone"
    
    # Default: unknown -> treat as non-drone for safety
    return "ei-drone"


def _display_name(ds: str) -> str:
    """Return the display name for a dataset."""
    return DATASET_DISPLAY_NAMES.get(ds, ds)


def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    """Load CSV and add class labels."""
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Keep only explicitly included datasets
    df = df[df["dataset"].isin(INCLUDED_DATASETS)].copy()

    # Add class label
    df["label"] = df.apply(
        lambda row: infer_label_from_path(row["filepath"], row["dataset"]),
        axis=1
    )
    
    # Filter out rows with NaN in features
    df = df.dropna(subset=FEATURES)
    
    print(f"Loaded {len(df)} valid samples")
    print(f"Class distribution:")
    print(df["label"].value_counts())
    print(f"\nDataset distribution:")
    print(df["dataset"].value_counts())
    
    return df


def stratified_sample(df: pd.DataFrame, n_per_group: int) -> pd.DataFrame:
    """Sample up to n_per_group samples from each dataset."""
    sampled_dfs = []
    
    for dataset in df["dataset"].unique():
        dataset_df = df[df["dataset"] == dataset]
        n_samples = min(len(dataset_df), n_per_group)
        sampled_dfs.append(
            dataset_df.sample(n=n_samples, random_state=RANDOM_STATE)
        )
    
    result = pd.concat(sampled_dfs, ignore_index=True)
    print(f"\nSampled {len(result)} total clips")
    print(f"Class distribution after sampling:")
    print(result["label"].value_counts())
    
    return result


def run_tsne(X: np.ndarray, perplexity: int = 30) -> np.ndarray:
    """Run t-SNE dimensionality reduction."""
    print(f"Running t-SNE (perplexity={perplexity})...")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=RANDOM_STATE,
        max_iter=1000,
        init="pca"
    )
    return tsne.fit_transform(X)


def run_umap(X: np.ndarray, n_neighbors: int = 15) -> np.ndarray:
    """Run UMAP dimensionality reduction."""
    if not UMAP_AVAILABLE:
        raise ImportError("UMAP is not installed")
    
    print(f"Running UMAP (n_neighbors={n_neighbors})...")
    reducer = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        random_state=RANDOM_STATE
    )
    return reducer.fit_transform(X)


def _get_dataset_colors(datasets) -> dict:
    """Return consistent colors for datasets using an accessible palette."""
    ordered = sorted(datasets)
    colors = [ACCESSIBLE_PALETTE[i % len(ACCESSIBLE_PALETTE)] for i in range(len(ordered))]
    return {ds: colors[i] for i, ds in enumerate(ordered)}


def plot_embedding(
    coords: np.ndarray,
    df: pd.DataFrame,
    title: str,
    output_path: str,
    color_by: str = "label"
) -> None:
    """Create scatter plot of embedding."""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Color by dataset, shape by class for better accessibility
    datasets = df["dataset"].unique()
    dataset_colors = _get_dataset_colors(datasets)
    class_markers = {"drone": "o", "ei-drone": "s"}

    # Plot each combination of class and dataset
    for label in ["drone", "ei-drone"]:
        for dataset in datasets:
            mask = (df["label"] == label) & (df["dataset"] == dataset)
            if mask.sum() == 0:
                continue
            
            ax.scatter(
                coords[mask, 1],
                coords[mask, 0],
                c=dataset_colors[dataset],
                marker=class_markers[label],
                s=50,
                alpha=0.7,
                label=f"{_display_name(dataset)} ({label})",
                edgecolors="black",
                linewidths=0.4
            )
    
    ax.set_xlabel("Dimensio 1", fontsize=12)
    ax.set_ylabel("Dimensio 2", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Create legend with smaller font and outside plot
    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc='upper left',
        fontsize=8,
        markerscale=1.2,
        framealpha=0.9
    )
    
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_embedding_simple(
    coords: np.ndarray,
    df: pd.DataFrame,
    title: str,
    output_path: str
) -> None:
    """Create simple scatter plot colored only by class (drone/non-drone)."""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Define colors
    class_colors = CLASS_COLORS
    class_labels = {"drone": "Drooni", "ei-drone": "Ei drooni"}
    
    for label in ["drone", "ei-drone"]:
        mask = df["label"] == label
        if mask.sum() == 0:
            continue
        
        ax.scatter(
            coords[mask, 1],
            coords[mask, 0],
            c=class_colors[label],
            s=30,
            alpha=0.5,
            label=class_labels[label],
            edgecolors='none'
        )
    
    ax.set_xlabel("Dimensio 1", fontsize=14)
    ax.set_ylabel("Dimensio 2", fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    ax.legend(fontsize=12, markerscale=2)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_embedding_by_dataset(
    coords: np.ndarray,
    df: pd.DataFrame,
    title: str,
    output_path: str
) -> None:
    """Create scatter plot colored by dataset, shaped by class."""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Color palette for datasets
    datasets = df["dataset"].unique()
    dataset_colors = _get_dataset_colors(datasets)

    # Markers for class
    class_markers = {"drone": "o", "ei-drone": "s"}
    
    for dataset in datasets:
        for label in ["drone", "ei-drone"]:
            mask = (df["dataset"] == dataset) & (df["label"] == label)
            if mask.sum() == 0:
                continue
            
            ax.scatter(
                coords[mask, 1],
                coords[mask, 0],
                c=dataset_colors[dataset],
                marker=class_markers[label],
                s=40,
                alpha=0.7,
                label=f"{_display_name(dataset)} ({'●' if label == 'drone' else '■'})",
                edgecolors="black",
                linewidths=0.3
            )
    
    ax.set_xlabel("Dimensio 1", fontsize=12)
    ax.set_ylabel("Dimensio 2", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc='upper left',
        fontsize=8,
        markerscale=1.5,
        framealpha=0.9,
        title="Aineisto (● drooni, ■ ei)"
    )
    
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def _adjust_budgets(
    budgets: dict,
    counts: dict,
    target_total: int,
    min_per_dataset: int,
    max_per_dataset: int
) -> dict:
    """Adjust per-dataset budgets to better match the target total."""
    current = sum(budgets.values())

    if current > target_total:
        for ds in sorted(budgets, key=budgets.get, reverse=True):
            if current <= target_total:
                break
            reducible = max(0, budgets[ds] - min_per_dataset)
            if reducible == 0:
                continue
            reduce_by = min(reducible, current - target_total)
            budgets[ds] -= reduce_by
            current -= reduce_by

    elif current < target_total:
        for ds in sorted(budgets, key=lambda d: counts[d] - budgets[d], reverse=True):
            if current >= target_total:
                break
            remaining = min(counts[ds], max_per_dataset) - budgets[ds]
            if remaining <= 0:
                continue
            add_by = min(remaining, target_total - current)
            budgets[ds] += add_by
            current += add_by

    return budgets


def _allocate_dataset_budgets(
    df: pd.DataFrame,
    target_total: int,
    min_per_dataset: int,
    max_per_dataset: int
) -> dict:
    """Allocate per-dataset sample budgets using sqrt weighting."""
    counts = df["dataset"].value_counts().to_dict()
    weights = {ds: np.sqrt(n) for ds, n in counts.items()}
    total_weight = sum(weights.values())
    budgets = {}

    for ds, n in counts.items():
        raw = int(round(target_total * (weights[ds] / total_weight)))
        capped = max(min_per_dataset, min(max_per_dataset, raw))
        budgets[ds] = min(n, capped)

    return _adjust_budgets(budgets, counts, target_total, min_per_dataset, max_per_dataset)


def weighted_class_balanced_sample(
    df: pd.DataFrame,
    target_total: int,
    min_per_dataset: int,
    max_per_dataset: int
) -> pd.DataFrame:
    """Sample with dataset weights and per-dataset class balance when possible."""
    budgets = _allocate_dataset_budgets(df, target_total, min_per_dataset, max_per_dataset)
    sampled_dfs = []
    labels = ["drone", "ei-drone"]

    print("\nSampling budgets per dataset:")
    for ds in sorted(budgets.keys()):
        print(f"  {ds}: {budgets[ds]}")

    for dataset, budget in budgets.items():
        dataset_df = df[df["dataset"] == dataset]
        by_label = {label: dataset_df[dataset_df["label"] == label] for label in labels}

        half = budget // 2
        counts = {label: min(len(by_label[label]), half) for label in labels}
        remaining = budget - sum(counts.values())

        # Fill remaining samples from the label with more capacity
        while remaining > 0:
            capacity = {label: len(by_label[label]) - counts[label] for label in labels}
            best_label = max(capacity, key=capacity.get)
            if capacity[best_label] <= 0:
                break
            counts[best_label] += 1
            remaining -= 1

        for label in labels:
            if counts[label] > 0:
                sampled_dfs.append(
                    by_label[label].sample(n=counts[label], random_state=RANDOM_STATE)
                )

    result = pd.concat(sampled_dfs, ignore_index=True)
    print(f"\nSampled {len(result)} total clips")
    print("Class distribution after sampling:")
    print(result["label"].value_counts())

    return result


def sample_for_visualization(df: pd.DataFrame) -> pd.DataFrame:
    """Select the sampling strategy for visualization."""
    if SAMPLING_MODE == "weighted_balanced":
        return weighted_class_balanced_sample(
            df,
            target_total=TARGET_TOTAL_SAMPLES,
            min_per_dataset=MIN_SAMPLES_PER_DATASET,
            max_per_dataset=MAX_SAMPLES_PER_DATASET
        )
    if SAMPLING_MODE == "uniform":
        return stratified_sample(df, SAMPLES_PER_DATASET)
    raise ValueError(f"Unknown SAMPLING_MODE: {SAMPLING_MODE}")


def main():
    """Main function."""
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # Load and prepare data
    df = load_and_prepare_data(IN_CSV)
    
    # Sample data to avoid overcrowding
    df_sampled = sample_for_visualization(df)

    # Extract features and standardize
    X = df_sampled[FEATURES].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"\nFeature matrix shape: {X_scaled.shape}")
    
    # Run t-SNE
    coords_tsne = run_tsne(X_scaled, perplexity=40)
    
    # Plot t-SNE results
    plot_embedding_simple(
        coords_tsne, df_sampled,
        "t-SNE: Ääniklipit luokittain",
        os.path.join(OUT_DIR, "tsne_by_class.png")
    )
    
    plot_embedding_by_dataset(
        coords_tsne, df_sampled,
        "t-SNE: Ääniklipit aineistoittain",
        os.path.join(OUT_DIR, "tsne_by_dataset.png")
    )
    
    plot_embedding(
        coords_tsne, df_sampled,
        "t-SNE: Ääniklipit (aineisto + luokka)",
        os.path.join(OUT_DIR, "tsne_full.png")
    )
    
    # Run UMAP if available
    if UMAP_AVAILABLE:
        coords_umap = run_umap(X_scaled, n_neighbors=20)
        
        plot_embedding_simple(
            coords_umap, df_sampled,
            "UMAP: Ääniklipit luokittain",
            os.path.join(OUT_DIR, "umap_by_class.png")
        )
        
        plot_embedding_by_dataset(
            coords_umap, df_sampled,
            "UMAP: Ääniklipit aineistoittain",
            os.path.join(OUT_DIR, "umap_by_dataset.png")
        )
        
        plot_embedding(
            coords_umap, df_sampled,
            "UMAP: Ääniklipit (aineisto + luokka)",
            os.path.join(OUT_DIR, "umap_full.png")
        )
    
    print(f"\nDone! Results saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
