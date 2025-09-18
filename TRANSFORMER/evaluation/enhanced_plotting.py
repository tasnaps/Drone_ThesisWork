#!/usr/bin/env python3
"""
Enhanced plotting utilities for transformer model evaluation.
This module provides advanced visualization capabilities for probability distributions,
threshold analysis, and detailed performance insights.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, List, Optional, Tuple


def setup_plotting_style():
    """Setup consistent plotting style across all plots."""
    plt.style.use('default')
    sns.set_palette("husl")

    # Set consistent font sizes and styles
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 14,
        'lines.linewidth': 2,
        'grid.alpha': 0.3
    })


def plot_log_scale_probability_distributions(results: Dict, output_dir: str = "./plots",
                                           dataset_filter: Optional[List[str]] = None,
                                           save_individual: bool = True) -> List[str]:
    """
    Create log-scale probability distribution plots similar to your script.

    Args:
        results: Dictionary of evaluation results by dataset
        output_dir: Directory to save plots
        dataset_filter: Optional list of dataset names to plot (None = all)
        save_individual: Whether to save individual dataset plots

    Returns:
        List of saved plot filenames
    """
    setup_plotting_style()
    os.makedirs(output_dir, exist_ok=True)
    saved_plots = []

    datasets_to_plot = dataset_filter or list(results.keys())

    print(f"\n=== Creating Log-Scale Probability Distribution Plots ===")

    # Combined plot for all datasets
    if len(datasets_to_plot) > 1:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Log-Scale Probability Distributions by Dataset', fontsize=16)
        axes = axes.flatten()

        for i, dataset_name in enumerate(datasets_to_plot[:6]):  # Max 6 for subplot
            if dataset_name not in results or 'probabilities' not in results[dataset_name]:
                continue

            ax = axes[i]
            _plot_single_log_distribution(results[dataset_name], dataset_name, ax)

        # Hide unused subplots
        for j in range(len(datasets_to_plot), 6):
            axes[j].set_visible(False)

        plt.tight_layout()
        combined_filename = os.path.join(output_dir, 'log_scale_probability_distributions_combined.png')
        plt.savefig(combined_filename, dpi=300, bbox_inches='tight')
        plt.close()
        saved_plots.append(combined_filename)
        print(f"Saved combined plot: {combined_filename}")

    # Individual plots for each dataset
    if save_individual:
        for dataset_name in datasets_to_plot:
            if dataset_name not in results or 'probabilities' not in results[dataset_name]:
                continue

            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            _plot_single_log_distribution(results[dataset_name], dataset_name, ax)

            individual_filename = os.path.join(output_dir, f'{dataset_name}_log_probability_distribution.png')
            plt.savefig(individual_filename, dpi=300, bbox_inches='tight')
            plt.close()
            saved_plots.append(individual_filename)
            print(f"Saved individual plot for {dataset_name}: {individual_filename}")

    return saved_plots

def _plot_single_log_distribution(dataset_result: Dict, dataset_name: str, ax):
    """Helper function to plot log-scale distribution for a single dataset."""
    probs = np.array(dataset_result['probabilities'])
    labels = np.array(dataset_result['true_labels'])

    # Separate by true class
    prob_0 = probs[labels == 0]  # Non-drone
    prob_1 = probs[labels == 1]  # Drone

    # Handle edge case where all probabilities are 0
    if np.all(probs == 0):
        ax.text(0.5, 0.5, 'All probabilities are zero',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{dataset_name} (No valid probabilities)')
        return

    # Create log-spaced bins
    min_p = probs[probs > 0].min() if np.any(probs > 0) else 1e-10
    max_p = probs.max() if probs.max() > 0 else 1.0

    # Ensure we have a reasonable range for log scale
    if min_p >= max_p:
        min_p = 1e-10
        max_p = 1.0

    bins = np.logspace(np.log10(min_p), np.log10(max_p), 100)

    # Plot histograms
    if len(prob_0) > 0:
        ax.hist(prob_0[prob_0 > 0], bins=bins, alpha=0.7,
                label=f'Non-drone ({len(prob_0)})', color='red', density=True)

    if len(prob_1) > 0:
        ax.hist(prob_1[prob_1 > 0], bins=bins, alpha=0.7,
                label=f'Drone ({len(prob_1)})', color='blue', density=True)

    # Set log scale and formatting
    ax.set_xscale('log')
    ax.set_xlabel('Drone Probability (log scale)')
    ax.set_ylabel('Density')
    ax.set_title(f'{dataset_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add vertical lines for common thresholds
    thresholds = [0.1, 0.2, 0.5]
    colors = ['orange', 'yellow', 'black']
    for thresh, color in zip(thresholds, colors):
        if min_p <= thresh <= max_p:
            ax.axvline(x=thresh, color=color, linestyle='--', alpha=0.6,
                      label=f'Threshold {thresh}')

    # Print statistics
    print(f"\n{dataset_name} Statistics:")
    if len(prob_0) > 0:
        print(f"  Non-drone: mean={np.mean(prob_0):.6f}, median={np.median(prob_0):.6f}, "
              f"min={np.min(prob_0):.6f}, max={np.max(prob_0):.6f}")
    if len(prob_1) > 0:
        print(f"  Drone: mean={np.mean(prob_1):.6f}, median={np.median(prob_1):.6f}, "
              f"min={np.min(prob_1):.6f}, max={np.max(prob_1):.6f}")


def plot_probability_heatmap(results: Dict, output_dir: str = "./plots") -> str:
    """
    Create a heatmap showing probability distributions across datasets.

    Args:
        results: Dictionary of evaluation results by dataset
        output_dir: Directory to save plots

    Returns:
        Filename of saved plot
    """
    setup_plotting_style()
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n=== Creating Probability Heatmap ===")

    # Prepare data for heatmap
    datasets = []
    probability_ranges = {
        '0.0-0.001': [],
        '0.001-0.01': [],
        '0.01-0.1': [],
        '0.1-0.3': [],
        '0.3-0.7': [],
        '0.7-0.9': [],
        '0.9-1.0': []
    }

    for dataset_name, dataset_result in results.items():
        if 'probabilities' not in dataset_result:
            continue

        datasets.append(dataset_name)
        probs = np.array(dataset_result['probabilities'])

        # Calculate percentage in each range
        total = len(probs)
        probability_ranges['0.0-0.001'].append(np.sum((probs >= 0.0) & (probs < 0.001)) / total * 100)
        probability_ranges['0.001-0.01'].append(np.sum((probs >= 0.001) & (probs < 0.01)) / total * 100)
        probability_ranges['0.01-0.1'].append(np.sum((probs >= 0.01) & (probs < 0.1)) / total * 100)
        probability_ranges['0.1-0.3'].append(np.sum((probs >= 0.1) & (probs < 0.3)) / total * 100)
        probability_ranges['0.3-0.7'].append(np.sum((probs >= 0.3) & (probs < 0.7)) / total * 100)
        probability_ranges['0.7-0.9'].append(np.sum((probs >= 0.7) & (probs < 0.9)) / total * 100)
        probability_ranges['0.9-1.0'].append(np.sum((probs >= 0.9) & (probs <= 1.0)) / total * 100)

    # Create heatmap data
    heatmap_data = np.array([probability_ranges[range_name] for range_name in probability_ranges.keys()])

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(12, 8))

    im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')

    # Set labels
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.set_yticks(range(len(probability_ranges)))
    ax.set_yticklabels(list(probability_ranges.keys()))

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Percentage of Samples (%)')

    # Add text annotations
    for i in range(len(probability_ranges)):
        for j in range(len(datasets)):
            text = ax.text(j, i, f'{heatmap_data[i, j]:.1f}%',
                          ha="center", va="center", color="black" if heatmap_data[i, j] < 50 else "white")

    ax.set_title('Probability Distribution Heatmap Across Datasets')
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Probability Range')

    plt.tight_layout()
    filename = os.path.join(output_dir, 'probability_heatmap.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved probability heatmap: {filename}")
    return filename


def plot_threshold_sensitivity_analysis(results: Dict, output_dir: str = "./plots") -> str:
    """
    Create detailed threshold sensitivity analysis plots.

    Args:
        results: Dictionary of evaluation results by dataset
        output_dir: Directory to save plots

    Returns:
        Filename of saved plot
    """
    setup_plotting_style()
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n=== Creating Threshold Sensitivity Analysis ===")

    # Define threshold range
    thresholds = np.logspace(-4, 0, 100)  # From 0.0001 to 1.0

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Threshold Sensitivity Analysis Across All Datasets', fontsize=16)

    # Collect all data
    all_probs = []
    all_labels = []
    dataset_metrics = {}

    for dataset_name, dataset_result in results.items():
        if 'probabilities' not in dataset_result or 'true_labels' not in dataset_result:
            continue

        probs = np.array(dataset_result['probabilities'])
        labels = np.array(dataset_result['true_labels'])

        all_probs.extend(probs)
        all_labels.extend(labels)

        # Calculate metrics for each threshold for this dataset
        if len(np.unique(labels)) > 1:  # Only for datasets with both classes
            dataset_f1s = []
            dataset_precisions = []
            dataset_recalls = []

            for thresh in thresholds:
                preds = (probs >= thresh).astype(int)

                # Calculate metrics
                tp = np.sum((preds == 1) & (labels == 1))
                fp = np.sum((preds == 1) & (labels == 0))
                fn = np.sum((preds == 0) & (labels == 1))

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

                dataset_f1s.append(f1)
                dataset_precisions.append(precision)
                dataset_recalls.append(recall)

            dataset_metrics[dataset_name] = {
                'f1': dataset_f1s,
                'precision': dataset_precisions,
                'recall': dataset_recalls
            }

    # Plot 1: F1 scores by threshold for each dataset
    ax1 = axes[0, 0]
    for dataset_name, metrics in dataset_metrics.items():
        ax1.plot(thresholds, metrics['f1'], label=dataset_name, alpha=0.7)
    ax1.set_xscale('log')
    ax1.set_xlabel('Threshold (log scale)')
    ax1.set_ylabel('F1 Score')
    ax1.set_title('F1 Score vs Threshold by Dataset')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Precision vs Recall curves
    ax2 = axes[0, 1]
    for dataset_name, metrics in dataset_metrics.items():
        ax2.plot(metrics['recall'], metrics['precision'], label=dataset_name, alpha=0.7)
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curves')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Overall performance (combined data)
    ax3 = axes[1, 0]
    if len(all_probs) > 0 and len(np.unique(all_labels)) > 1:
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)

        overall_f1s = []
        overall_precisions = []
        overall_recalls = []

        for thresh in thresholds:
            preds = (all_probs >= thresh).astype(int)

            tp = np.sum((preds == 1) & (all_labels == 1))
            fp = np.sum((preds == 1) & (all_labels == 0))
            fn = np.sum((preds == 0) & (all_labels == 1))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            overall_f1s.append(f1)
            overall_precisions.append(precision)
            overall_recalls.append(recall)

        ax3.plot(thresholds, overall_f1s, 'b-', linewidth=3, label='F1 Score')
        ax3.plot(thresholds, overall_precisions, 'r-', linewidth=3, label='Precision')
        ax3.plot(thresholds, overall_recalls, 'g-', linewidth=3, label='Recall')

        # Find optimal F1 threshold
        optimal_idx = np.argmax(overall_f1s)
        optimal_threshold = thresholds[optimal_idx]
        optimal_f1 = overall_f1s[optimal_idx]

        ax3.axvline(x=optimal_threshold, color='black', linestyle='--', alpha=0.8,
                   label=f'Optimal F1 Threshold: {optimal_threshold:.4f}')
        ax3.axvline(x=0.5, color='orange', linestyle='--', alpha=0.8,
                   label='Current Threshold: 0.5')

        print(f"Optimal F1 threshold: {optimal_threshold:.4f} (F1: {optimal_f1:.3f})")

    ax3.set_xscale('log')
    ax3.set_xlabel('Threshold (log scale)')
    ax3.set_ylabel('Score')
    ax3.set_title('Overall Performance vs Threshold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Distribution of optimal thresholds by dataset
    ax4 = axes[1, 1]
    optimal_thresholds = []
    dataset_names = []

    for dataset_name, metrics in dataset_metrics.items():
        optimal_idx = np.argmax(metrics['f1'])
        optimal_thresh = thresholds[optimal_idx]
        optimal_thresholds.append(optimal_thresh)
        dataset_names.append(dataset_name)

    if optimal_thresholds:
        bars = ax4.bar(range(len(dataset_names)), optimal_thresholds, alpha=0.7)
        ax4.set_yscale('log')
        ax4.set_xlabel('Dataset')
        ax4.set_ylabel('Optimal Threshold (log scale)')
        ax4.set_title('Optimal F1 Thresholds by Dataset')
        ax4.set_xticks(range(len(dataset_names)))
        ax4.set_xticklabels(dataset_names, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)

        # Add value labels on bars
        for i, (bar, thresh) in enumerate(zip(bars, optimal_thresholds)):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                    f'{thresh:.4f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    filename = os.path.join(output_dir, 'threshold_sensitivity_analysis.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved threshold sensitivity analysis: {filename}")
    return filename


def plot_class_separation_analysis(results: Dict, output_dir: str = "./plots") -> str:
    """
    Create plots showing class separation quality across datasets.

    Args:
        results: Dictionary of evaluation results by dataset
        output_dir: Directory to save plots

    Returns:
        Filename of saved plot
    """
    setup_plotting_style()
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n=== Creating Class Separation Analysis ===")

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Class Separation Analysis Across Datasets', fontsize=16)

    separation_data = []

    for dataset_name, dataset_result in results.items():
        if 'probabilities' not in dataset_result or 'true_labels' not in dataset_result:
            continue

        probs = np.array(dataset_result['probabilities'])
        labels = np.array(dataset_result['true_labels'])

        # Separate by class
        drone_probs = probs[labels == 1]
        non_drone_probs = probs[labels == 0]

        if len(drone_probs) == 0 or len(non_drone_probs) == 0:
            continue

        # Calculate separation metrics
        drone_mean = np.mean(drone_probs)
        non_drone_mean = np.mean(non_drone_probs)
        drone_std = np.std(drone_probs)
        non_drone_std = np.std(non_drone_probs)

        separation = drone_mean - non_drone_mean

        # Cohen's d (effect size)
        pooled_std = np.sqrt(((len(drone_probs) - 1) * drone_std**2 +
                             (len(non_drone_probs) - 1) * non_drone_std**2) /
                            (len(drone_probs) + len(non_drone_probs) - 2))
        cohens_d = separation / pooled_std if pooled_std > 0 else 0

        # Overlap coefficient (simplified)
        min_max = min(np.max(non_drone_probs), np.max(drone_probs))
        max_min = max(np.min(non_drone_probs), np.min(drone_probs))
        overlap = max(0, min_max - max_min) / (np.max(probs) - np.min(probs)) if np.max(probs) > np.min(probs) else 1

        separation_data.append({
            'dataset': dataset_name,
            'separation': separation,
            'cohens_d': cohens_d,
            'overlap': overlap,
            'drone_mean': drone_mean,
            'non_drone_mean': non_drone_mean,
            'drone_std': drone_std,
            'non_drone_std': non_drone_std
        })

    if not separation_data:
        print("No valid separation data found")
        return ""

    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(separation_data)

    # Plot 1: Class separation by dataset
    ax1 = axes[0, 0]
    bars = ax1.bar(range(len(df)), df['separation'], alpha=0.7,
                   color=['green' if s > 0 else 'red' for s in df['separation']])
    ax1.set_xlabel('Dataset')
    ax1.set_ylabel('Class Separation (Drone Mean - Non-drone Mean)')
    ax1.set_title('Class Separation by Dataset')
    ax1.set_xticks(range(len(df)))
    ax1.set_xticklabels(df['dataset'], rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)

    # Add value labels
    for i, (bar, sep) in enumerate(zip(bars, df['separation'])):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.01),
                f'{sep:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)

    # Plot 2: Effect size (Cohen's d)
    ax2 = axes[0, 1]
    colors = ['red' if d < 0.2 else 'orange' if d < 0.5 else 'yellow' if d < 0.8 else 'green'
              for d in df['cohens_d']]
    bars2 = ax2.bar(range(len(df)), df['cohens_d'], alpha=0.7, color=colors)
    ax2.set_xlabel('Dataset')
    ax2.set_ylabel("Cohen's d (Effect Size)")
    ax2.set_title('Class Separability (Effect Size)')
    ax2.set_xticks(range(len(df)))
    ax2.set_xticklabels(df['dataset'], rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)

    # Add horizontal lines for interpretation
    ax2.axhline(y=0.2, color='red', linestyle='--', alpha=0.5, label='Small')
    ax2.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Medium')
    ax2.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Large')
    ax2.legend()

    # Plot 3: Mean probabilities by class
    ax3 = axes[1, 0]
    x = np.arange(len(df))
    width = 0.35

    ax3.bar(x - width/2, df['non_drone_mean'], width, label='Non-drone', alpha=0.7, color='red')
    ax3.bar(x + width/2, df['drone_mean'], width, label='Drone', alpha=0.7, color='blue')

    ax3.set_xlabel('Dataset')
    ax3.set_ylabel('Mean Probability')
    ax3.set_title('Mean Probabilities by Class')
    ax3.set_xticks(x)
    ax3.set_xticklabels(df['dataset'], rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Separation vs Performance correlation
    ax4 = axes[1, 1]

    # Get F1 scores if available
    f1_scores = []
    for _, row in df.iterrows():
        dataset_name = row['dataset']
        if dataset_name in results and 'file_level_f1' in results[dataset_name]:
            f1_scores.append(results[dataset_name]['file_level_f1'])
        else:
            f1_scores.append(0)

    if any(f1 > 0 for f1 in f1_scores):
        scatter = ax4.scatter(df['separation'], f1_scores, alpha=0.7, s=100)
        ax4.set_xlabel('Class Separation')
        ax4.set_ylabel('F1 Score')
        ax4.set_title('Class Separation vs Performance')
        ax4.grid(True, alpha=0.3)

        # Add trend line
        if len(df) > 2:
            z = np.polyfit(df['separation'], f1_scores, 1)
            p = np.poly1d(z)
            ax4.plot(df['separation'], p(df['separation']), "r--", alpha=0.8)

        # Add dataset labels
        for i, dataset in enumerate(df['dataset']):
            ax4.annotate(dataset, (df.iloc[i]['separation'], f1_scores[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
    else:
        ax4.text(0.5, 0.5, 'No F1 scores available', ha='center', va='center',
                transform=ax4.transAxes)
        ax4.set_title('Class Separation vs Performance (No Data)')

    plt.tight_layout()
    filename = os.path.join(output_dir, 'class_separation_analysis.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved class separation analysis: {filename}")
    return filename
