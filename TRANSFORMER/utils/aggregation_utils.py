#!/usr/bin/env python3
"""
Chunk aggregation utilities for transformer model evaluation.
This module handles aggregation of predictions from multiple chunks of the same file.
"""

import numpy as np

def analyze_logits_statistics(results, save_path="logits_analysis.png"):
    """
    Analyze and display logits statistics for each dataset.

    Args:
        results: Dictionary of evaluation results by dataset
        save_path: Path to save the analysis plot
    """
    print("\n" + "="*80)
    print("LOGITS STATISTICS ANALYSIS BY DATASET")
    print("="*80)

    import matplotlib.pyplot as plt

    # Create figure for logits analysis
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Logits Analysis by Dataset', fontsize=16)

    all_logits_data = []

    for dataset_name, data in results.items():
        if 'raw_logits' not in data:
            continue

        raw_logits = np.array(data['raw_logits'])  # Shape: (n_samples, 2)
        true_labels = np.array(data['true_labels'])

        # Extract logits for each class
        logit_class0 = raw_logits[:, 0]  # Logits for "unknown" class
        logit_class1 = raw_logits[:, 1]  # Logits for "yes_drone" class

        # Calculate logit differences (decision margin)
        logit_diff = logit_class1 - logit_class0  # Positive = drone prediction

        # Separate by true class
        drone_mask = (true_labels == 1)
        non_drone_mask = (true_labels == 0)

        drone_logit_diff = logit_diff[drone_mask] if np.any(drone_mask) else np.array([])
        non_drone_logit_diff = logit_diff[non_drone_mask] if np.any(non_drone_mask) else np.array([])

        print(f"\n{dataset_name}:")
        print(f"  Total samples: {len(raw_logits)}")
        print(f"  True drone samples: {len(drone_logit_diff)}")
        print(f"  True non-drone samples: {len(non_drone_logit_diff)}")

        # Statistics for logit differences
        print(f"  Logit Difference Statistics:")
        print(f"    Overall mean: {np.mean(logit_diff):.4f}")
        print(f"    Overall std:  {np.std(logit_diff):.4f}")
        print(f"    Overall range: [{np.min(logit_diff):.4f}, {np.max(logit_diff):.4f}]")

        if len(drone_logit_diff) > 0:
            print(f"    Drone samples mean: {np.mean(drone_logit_diff):.4f}")
            print(f"    Drone samples std:  {np.std(drone_logit_diff):.4f}")

        if len(non_drone_logit_diff) > 0:
            print(f"    Non-drone samples mean: {np.mean(non_drone_logit_diff):.4f}")
            print(f"    Non-drone samples std:  {np.std(non_drone_logit_diff):.4f}")

        # Separation analysis
        if len(drone_logit_diff) > 0 and len(non_drone_logit_diff) > 0:
            separation = np.mean(drone_logit_diff) - np.mean(non_drone_logit_diff)
            print(f"    Class separation (drone_mean - non_drone_mean): {separation:.4f}")

            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(drone_logit_diff) - 1) * np.var(drone_logit_diff) +
                                (len(non_drone_logit_diff) - 1) * np.var(non_drone_logit_diff)) /
                               (len(drone_logit_diff) + len(non_drone_logit_diff) - 2))
            cohens_d = separation / pooled_std if pooled_std > 0 else 0
            print(f"    Effect size (Cohen's d): {cohens_d:.4f}")

            if cohens_d < 0.2:
                effect_interpretation = "negligible"
            elif cohens_d < 0.5:
                effect_interpretation = "small"
            elif cohens_d < 0.8:
                effect_interpretation = "medium"
            else:
                effect_interpretation = "large"
            print(f"    Effect interpretation: {effect_interpretation}")

        # Raw logits statistics
        print(f"  Raw Logits Statistics:")
        print(f"    Class 0 (unknown) - mean: {np.mean(logit_class0):.4f}, std: {np.std(logit_class0):.4f}")
        print(f"    Class 1 (drone) - mean: {np.mean(logit_class1):.4f}, std: {np.std(logit_class1):.4f}")

        # Store data for plotting
        all_logits_data.append({
            'dataset': dataset_name,
            'logit_diff': logit_diff,
            'drone_logit_diff': drone_logit_diff,
            'non_drone_logit_diff': non_drone_logit_diff,
            'logit_class0': logit_class0,
            'logit_class1': logit_class1
        })

    # Create plots
    _create_logits_plots(all_logits_data, axes)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nLogits analysis complete. Plot saved as '{save_path}'")


def _create_logits_plots(all_logits_data, axes):
    """Create the individual logits analysis plots."""
    if not all_logits_data:
        return

    # Plot 1: Logit differences distribution
    ax1 = axes[0, 0]
    for data in all_logits_data[:5]:  # Plot first 5 datasets
        ax1.hist(data['logit_diff'], bins=30, alpha=0.6, label=data['dataset'], density=True)
    ax1.axvline(x=0, color='black', linestyle='--', alpha=0.8, label='Decision boundary')
    ax1.set_xlabel('Logit Difference (drone - unknown)')
    ax1.set_ylabel('Density')
    ax1.set_title('Logit Differences Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Logit differences by class
    ax2 = axes[0, 1]
    for i, data in enumerate(all_logits_data[:3]):  # Plot first 3 datasets
        if len(data['drone_logit_diff']) > 0:
            ax2.hist(data['drone_logit_diff'], bins=20, alpha=0.6,
                    label=f"{data['dataset']} (drone)", color=f'C{i}', density=True)
        if len(data['non_drone_logit_diff']) > 0:
            ax2.hist(data['non_drone_logit_diff'], bins=20, alpha=0.6,
                    label=f"{data['dataset']} (non-drone)", color=f'C{i}',
                    linestyle='--', histtype='step', density=True)
    ax2.axvline(x=0, color='black', linestyle='--', alpha=0.8)
    ax2.set_xlabel('Logit Difference')
    ax2.set_ylabel('Density')
    ax2.set_title('Logit Differences by True Class')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Raw logits scatter
    ax3 = axes[0, 2]
    for i, data in enumerate(all_logits_data[:3]):
        ax3.scatter(data['logit_class0'], data['logit_class1'],
                   alpha=0.6, label=data['dataset'], s=20)
    ax3.plot([-10, 10], [-10, 10], 'k--', alpha=0.8, label='Equal logits')
    ax3.set_xlabel('Class 0 (unknown) Logits')
    ax3.set_ylabel('Class 1 (drone) Logits')
    ax3.set_title('Raw Logits Scatter Plot')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Class separation analysis
    ax4 = axes[1, 0]
    dataset_names = []
    separations = []
    effect_sizes = []

    for data in all_logits_data:
        if len(data['drone_logit_diff']) > 0 and len(data['non_drone_logit_diff']) > 0:
            dataset_names.append(data['dataset'])
            sep = np.mean(data['drone_logit_diff']) - np.mean(data['non_drone_logit_diff'])
            separations.append(sep)

            # Calculate Cohen's d
            pooled_std = np.sqrt(((len(data['drone_logit_diff']) - 1) * np.var(data['drone_logit_diff']) +
                                (len(data['non_drone_logit_diff']) - 1) * np.var(data['non_drone_logit_diff'])) /
                               (len(data['drone_logit_diff']) + len(data['non_drone_logit_diff']) - 2))
            cohens_d = sep / pooled_std if pooled_std > 0 else 0
            effect_sizes.append(cohens_d)

    if dataset_names:
        x_pos = np.arange(len(dataset_names))
        ax4.bar(x_pos, separations, alpha=0.7)
        ax4.set_xlabel('Dataset')
        ax4.set_ylabel('Class Separation')
        ax4.set_title('Logit Class Separation by Dataset')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(dataset_names, rotation=45)
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)

    # Plot 5: Effect sizes
    ax5 = axes[1, 1]
    if dataset_names:
        colors = ['red' if d < 0.2 else 'orange' if d < 0.5 else 'yellow' if d < 0.8 else 'green'
                 for d in effect_sizes]
        ax5.bar(x_pos, effect_sizes, alpha=0.7, color=colors)
        ax5.set_xlabel('Dataset')
        ax5.set_ylabel("Cohen's d")
        ax5.set_title('Effect Size (Class Separability)')
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels(dataset_names, rotation=45)
        ax5.grid(True, alpha=0.3)

        # Add horizontal lines for effect size interpretation
        ax5.axhline(y=0.2, color='red', linestyle='--', alpha=0.5, label='Small')
        ax5.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Medium')
        ax5.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Large')
        ax5.legend()

    # Plot 6: Logit variance analysis
    ax6 = axes[1, 2]
    variances = []
    for data in all_logits_data:
        var_class0 = np.var(data['logit_class0'])
        var_class1 = np.var(data['logit_class1'])
        variances.append([var_class0, var_class1])

    if variances:
        variances = np.array(variances)
        x_pos = np.arange(len(all_logits_data))
        width = 0.35
        ax6.bar(x_pos - width/2, variances[:, 0], width, label='Class 0 (unknown)', alpha=0.7)
        ax6.bar(x_pos + width/2, variances[:, 1], width, label='Class 1 (drone)', alpha=0.7)
        ax6.set_xlabel('Dataset')
        ax6.set_ylabel('Logit Variance')
        ax6.set_title('Logit Variance by Class')
        ax6.set_xticks(x_pos)
        ax6.set_xticklabels([data['dataset'] for data in all_logits_data], rotation=45)
        ax6.legend()
        ax6.grid(True, alpha=0.3)
