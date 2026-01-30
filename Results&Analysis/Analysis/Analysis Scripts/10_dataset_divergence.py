import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import jensenshannon, squareform
from scipy.cluster import hierarchy
from sklearn.manifold import MDS
from typing import Optional

IN_CSV = "audio_analysis_results/audio_dataset_stats_detailed.csv"
OUT_DIR = "audio_analysis_results/divergences"
METRICS = ["length", "rms", "peak", "spec_avg", "crest_factor"]

# Dataset display name mapping (edit these to change how names appear on heatmaps)
# Key = name as it appears in the CSV, Value = name to display
DATASET_DISPLAY_NAMES = {
    "Authors Collection": "Meriläisen aineisto",
    "Drone_print": "Drone Print",
    "FusionDataset": "Yhdistelmäaineisto",
    "Wonjun Yi": "Wonjun Yi ym.",
    "S&E": "Svanström ym.",
    "Emo Soundscapes Mixes": "Emo Soundscapes",
    "Calibration Dataset": "Kalibrointiaineisto",
    "ESC-50": "ESC-50",
    "H-2": "H-2",
    "Al-Emadi": "Al-Emadi ym.",
    "UrbanSound8K": "UrbanSound8K",
}


def _display_name(ds: str) -> str:
    """Return the display name for a dataset, or the original if not mapped."""
    return DATASET_DISPLAY_NAMES.get(ds, ds)

# Configurable parameters
PERCENTILE_CLIP = (1, 99)     # set to None to disable clipping
TARGET_BIN_COUNT = 50
MIN_UNIQUE_FOR_FD = 5         # require this many unique values to apply FD rule
EPS = 1e-12                   # smoothing constant only for empty bins now
MIN_DATASETS_FOR_CLUSTER = 2

# --- Heatmap sizing config ---
CELL_W = 1.0           # width per dataset (inches) - keep boxes compact
CELL_H = 1.0           # height per dataset (inches)
EXTRA_W = 5.0          # space for dendrogram/colorbar (inches)
EXTRA_H = 4.0
FIG_W_MIN, FIG_W_MAX = 12, 50
FIG_H_MIN, FIG_H_MAX = 10, 50

# Font sizes - LARGE for readability
ANNOT_FONT = 22        # numbers inside cells
TICK_FONT = 20         # dataset names on axes
TITLE_FONT = 26
LABEL_FONT = 20

# If True, scale fonts a bit based on number of datasets in each plot
AUTO_SCALE_FONTS = True

# Minimum font size even with many datasets
MIN_FONT = 16


def _scaled_font(base: int, n: int) -> int:
    """Scale font sizes based on number of datasets.

    More datasets -> slightly smaller font to reduce overlap.
    Few datasets -> larger font for readability.
    Always stays above MIN_FONT for readability.
    """
    if not AUTO_SCALE_FONTS:
        return base
    if n <= 6:
        scaled = int(base * 1.30)
    elif n <= 10:
        scaled = int(base * 1.15)
    elif n <= 16:
        scaled = base
    elif n <= 24:
        scaled = int(base * 0.92)
    else:
        scaled = int(base * 0.85)
    return max(MIN_FONT, scaled)


os.makedirs(OUT_DIR, exist_ok=True)


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Return Jensen–Shannon divergence (not the square root)."""
    return jensenshannon(p, q, base=2.0) ** 2


def make_bins(values: np.ndarray) -> Optional[np.ndarray]:
    """Compute shared bin edges for a metric."""
    vals = np.sort(values[np.isfinite(values)])
    if vals.size == 0:
        return None
    if PERCENTILE_CLIP:
        lo, hi = np.percentile(vals, PERCENTILE_CLIP)
        if lo == hi:
            return None  # constant: skip metric
        clipped = vals[(vals >= lo) & (vals <= hi)]
        if clipped.size >= 2:
            vals = clipped
            lo, hi = vals[0], vals[-1]
    else:
        lo, hi = vals[0], vals[-1]
        if lo == hi:
            return None
    # Freedman–Diaconis bin width if data varied enough
    uniq = np.unique(vals)
    if uniq.size >= MIN_UNIQUE_FOR_FD:
        q75, q25 = np.percentile(vals, [75, 25])
        iqr = q75 - q25
        if iqr > 0:
            h = 2 * iqr / (vals.size ** (1 / 3))
            if h > 0:
                fd_bins = int(np.clip(np.ceil((hi - lo) / h), 5, 200))
                if fd_bins >= 5:
                    return np.linspace(lo, hi, fd_bins + 1)
    # Fallback: fixed bin count
    return np.linspace(lo, hi, TARGET_BIN_COUNT + 1)


def histogram(vals: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """Normalized histogram with selective smoothing (only zero bins)."""
    h, _ = np.histogram(vals, bins=bins)
    zero_mask = (h == 0)
    if zero_mask.any():
        h = h.astype(float)
        h[zero_mask] = EPS
    h = h.astype(float)
    total = h.sum()
    if total == 0:
        # Degenerate (all NaN or empty); return uniform
        return np.full(len(bins) - 1, 1.0 / (len(bins) - 1))
    return h / total


# Load data
try:
    df = pd.read_csv(IN_CSV)
except FileNotFoundError:
    print(f"Error: missing input CSV `{IN_CSV}`")
    raise SystemExit(1)

datasets = sorted(df["dataset"].dropna().unique())
if not datasets:
    print("No datasets found; exiting.")
    raise SystemExit(0)

# Gather per-metric values
metric_values = {m: {ds: df.loc[df.dataset == ds, m].dropna().astype(float).values
                     for ds in datasets}
                 for m in METRICS}

divergence_matrices = {}
bin_registry = {}

for metric in METRICS:
    print(f"\nMetric: {metric}")
    arrays = [arr for arr in metric_values[metric].values() if arr.size > 0 and np.isfinite(arr).any()]
    if not arrays:
        print("  No data; skipping.")
        continue

    all_vals = np.concatenate(arrays)
    bins = make_bins(all_vals)
    if bins is None or bins.size < 3:
        print("  Insufficient variance (constant or near-constant); skipping.")
        continue
    bin_registry[metric] = bins

    # Precompute histograms
    hists = {}
    for ds, arr in metric_values[metric].items():
        vals = arr[np.isfinite(arr)]
        if vals.size > 0:
            hists[ds] = histogram(vals, bins)

    present_ds = sorted(hists.keys())
    n_present = len(present_ds)
    if n_present < MIN_DATASETS_FOR_CLUSTER:
        print("  Not enough datasets with data; skipping.")
        continue

    # Initialize full matrix aligned to global dataset order
    D_full = np.full((len(datasets), len(datasets)), np.nan)
    # Compute divergences only for present datasets
    for i, dsi in enumerate(present_ds):
        pi = hists[dsi]
        gi = datasets.index(dsi)
        for j, dsj in enumerate(present_ds):
            gj = datasets.index(dsj)
            if i == j:
                D_full[gi, gj] = 0.0
            elif j > i:
                pj = hists[dsj]
                div = js_divergence(pi, pj)
                D_full[gi, gj] = D_full[gj, gi] = div

    divergence_matrices[metric] = D_full

    # Plot clustermap using only present datasets
    idx = [datasets.index(ds) for ds in present_ds]
    D_plot = D_full[np.ix_(idx, idx)]
    condensed = squareform(D_plot, force='tovector', checks=False)

    linkage = hierarchy.average(condensed)

    # Dynamic figure size based on number of datasets
    fig_w = min(max(EXTRA_W + CELL_W * n_present, FIG_W_MIN), FIG_W_MAX)
    fig_h = min(max(EXTRA_H + CELL_H * n_present, FIG_H_MIN), FIG_H_MAX)

    # Convert dataset names to display names for the heatmap labels
    display_names = [_display_name(ds) for ds in present_ds]

    cg = sns.clustermap(
        D_plot,
        row_cluster=True,
        col_cluster=True,
        row_linkage=linkage,
        col_linkage=linkage,
        xticklabels=display_names,
        yticklabels=display_names,
        cmap="viridis_r",
        annot=True,
        annot_kws={"size": _scaled_font(ANNOT_FONT, n_present)},
        fmt=".2f",
        linewidths=0.6,
        dendrogram_ratio=0.12,
        figsize=(fig_w, fig_h)
    )
    cg.ax_heatmap.set_title(f"JS Divergenssi ({metric})", pad=14, fontsize=_scaled_font(TITLE_FONT, n_present))
    plt.setp(
        cg.ax_heatmap.get_xticklabels(),
        rotation=45,
        ha="right",
        fontsize=_scaled_font(TICK_FONT, n_present)
    )
    plt.setp(
        cg.ax_heatmap.get_yticklabels(),
        rotation=0,
        fontsize=_scaled_font(TICK_FONT, n_present)
    )
    cg.ax_heatmap.tick_params(pad=6)

    # Make colorbar tick labels a bit larger too
    if cg.ax_cbar is not None:
        cg.ax_cbar.tick_params(labelsize=_scaled_font(TICK_FONT, n_present))

    out_path = os.path.join(OUT_DIR, f"js_divergence_{metric}.png")
    cg.figure.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(cg.figure)
    print(f"  Saved {out_path}")

    # Overall average (pairwise mean over metrics with data)
    if not divergence_matrices:
        print("\nNo divergence matrices computed; aborting overall MDS.")
        raise SystemExit(0)

    metric_list = list(divergence_matrices.keys())
    stack = np.stack([divergence_matrices[m] for m in metric_list], axis=0)
    valid_mask = ~np.isnan(stack)
    counts = valid_mask.sum(axis=0)
    with np.errstate(invalid="ignore"):
        summed = np.nansum(stack, axis=0)
    avg_D = np.divide(summed, counts, where=counts > 0)

    # If a pair never had any metric, leave as NaN and exclude from MDS
    row_valid = ~(np.all(np.isnan(avg_D), axis=1))
    used_idx = np.where(row_valid)[0]
    if used_idx.size < 2:
        print("Not enough datasets with overlapping metrics for MDS.")
    else:
        D_mds = avg_D[np.ix_(used_idx, used_idx)]
        # Replace any residual NaNs (should be none) with column means
        col_means = np.nanmean(D_mds, axis=0)
        inds = np.where(np.isnan(D_mds))
        D_mds[inds] = np.take(col_means, inds[1])

        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42, n_init=8, normalized_stress='auto')
        coords = mds.fit_transform(D_mds)
        plt.figure(figsize=(8, 6))
        for i, ds_idx in enumerate(used_idx):
            ds = datasets[ds_idx]
            x, y = coords[i]
            plt.scatter(x, y, s=110)
            plt.text(x + 0.01, y + 0.01, _display_name(ds), fontsize=9)
        plt.title("MDS (Average JS Divergence)")
        plt.xlabel("Dim 1")
        plt.ylabel("Dim 2")
        plt.grid(alpha=0.4, linestyle="--")
        plt.tight_layout()
        mds_path = os.path.join(OUT_DIR, "mds_overall.png")
        plt.savefig(mds_path, dpi=150)
        plt.close()
        print(f"\nSaved overall MDS plot to {mds_path}")


print("\nDone.")
