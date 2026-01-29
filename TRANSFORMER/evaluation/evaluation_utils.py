#!/usr/bin/env python3
"""
Shared utilities for transformer model evaluation scripts.
This module contains common functions used by both clip-based and whole-file evaluation scripts.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Import the new output manager
from .output_manager import EvaluationOutputManager

# Import enhanced plotting functionality and file protection
try:
    from evaluation.enhanced_plotting import (
        plot_log_scale_probability_distributions,
        plot_probability_heatmap,
        plot_threshold_sensitivity_analysis,
        plot_class_separation_analysis,
        generate_comprehensive_plot_suite
    )
    ENHANCED_PLOTTING_AVAILABLE = True
except ImportError:
    ENHANCED_PLOTTING_AVAILABLE = False

try:
    from evaluation.file_protection import file_protector, safe_plt_savefig, safe_df_to_csv
    FILE_PROTECTION_AVAILABLE = True
except ImportError:
    FILE_PROTECTION_AVAILABLE = False


def save_detailed_csv_results(results, output_dir="eval_results_organized", file_suffix="detailed_results",
                              data_type="file", aggregation_info=None, config=None, threshold=None):
    """
    Save detailed results to CSV files for comprehensive analysis.
    Now uses organized output structure with the EvaluationOutputManager.

    Args:
        results: Dictionary of evaluation results by dataset
        output_dir: Directory to save CSV files (now defaults to organized structure)
        file_suffix: Suffix for CSV filenames
        data_type: Type of data ("file" or "clip")
        aggregation_info: Dict with aggregation method details (for file-level)
        config: EvaluationConfig object with threshold settings
        threshold: Calibrated threshold to use (overrides config if provided)
    """
    print(f"\n=== Saving Detailed {data_type.title()}-Level Results to Organized CSV Structure ===")

    # Initialize the output manager for organized structure
    output_manager = EvaluationOutputManager(output_dir)

    # Create a run directory for this evaluation
    run_name = f"{data_type}_evaluation_{output_manager.timestamp}"
    run_dir = output_manager.get_run_directory(run_name)

    print(f"📁 Using organized output directory: {run_dir}")

    # Get threshold: prioritize passed threshold, then config, then defaults
    if threshold is not None:
        final_threshold = threshold
        if final_threshold < 0:
            print(f"  -> INVERTED threshold detected (using < logic)")
    elif config:
        if data_type == "file":
            threshold = config.file.threshold
        else:  # clip
            threshold = config.clip.threshold
        print(f"Using config threshold: {threshold:.6f}")
    else:
        # Fallback to config defaults if no config passed
        from TRANSFORMER.config.config import EvaluationConfig
        default_config = EvaluationConfig()
        if data_type == "file":
            threshold = default_config.file.threshold
        else:  # clip
            threshold = default_config.clip.threshold
        print(f"Using default threshold: {threshold:.6f}")

    # Default aggregation info for file-level data
    if data_type == "file" and aggregation_info is None:
        aggregation_info = {
            'method': 'whole_file',
            'threshold': threshold
        }

    # 1. Save individual dataset detailed results using organized structure
    individual_results = {}
    for ds_name, metrics in results.items():
        if 'probabilities' in metrics and 'true_labels' in metrics:
            true_labels = metrics['true_labels']
            probabilities = metrics['probabilities']

            # Handle different data types
            if data_type == "file":
                # For whole-file evaluation
                file_predictions = metrics.get('file_predictions', [])
                file_labels = metrics.get('file_labels', [])

                dataset_results = []
                for i in range(len(true_labels)):
                    # For file-level, true_labels should align with probabilities
                    pred_label = file_predictions[i] if i < len(file_predictions) else (1 if probabilities[i] > threshold else 0)

                    dataset_results.append({
                        'file_id': i,
                        'true_label': true_labels[i],
                        'predicted_label': pred_label,
                        'drone_probability': probabilities[i],
                        'aggregation_method': aggregation_info.get('method', 'whole_file'),
                        'aggregation_threshold': aggregation_info.get('threshold', threshold),
                        'split': 'whole_file'
                    })

            elif data_type == "clip":
                # For clip-based evaluation
                dataset_results = []
                for i in range(len(true_labels)):
                    dataset_results.append({
                        'clip_index': i,
                        'true_label': true_labels[i],
                        'predicted_label': 1 if probabilities[i] > threshold else 0,  # Use configured threshold
                        'drone_probability': probabilities[i],
                        'original_length': 16000,  # 1-second clips at 16kHz
                        'split': 'clip_based'
                    })

            # Save individual dataset CSV using organized structure
            df_dataset = pd.DataFrame(dataset_results)
            csv_filename = f'{ds_name}_{file_suffix}.csv'

            # Save using output manager
            output_manager.save_csv_results(df_dataset, csv_filename, run_name, dataset_specific=True)
            individual_results[ds_name] = len(dataset_results)

            print(f"✅ Saved {ds_name}: {len(dataset_results)} {data_type} records")

    # 2. Save combined results across all datasets using organized structure
    all_results = []
    for ds_name, metrics in results.items():
        if 'probabilities' in metrics and 'true_labels' in metrics:
            true_labels = metrics['true_labels']
            probabilities = metrics['probabilities']

            for i in range(len(true_labels)):
                result_record = {
                    'dataset': ds_name,
                    f'{data_type}_index': i,
                    'true_label': true_labels[i],
                    'predicted_label': 1 if probabilities[i] > threshold else 0,
                    'probability': probabilities[i],
                }

                # Add aggregation info for file-level data
                if data_type == "file" and aggregation_info:
                    result_record['aggregation_method'] = aggregation_info.get('method', 'whole_file')
                    result_record['aggregation_threshold'] = aggregation_info.get('threshold', threshold)

                all_results.append(result_record)

    # Save combined CSV using organized structure
    if all_results:
        df_combined = pd.DataFrame(all_results)
        combined_filename = f'{data_type}_level_results.csv'

        # Save to both run directory and main datasets directory
        output_manager.save_csv_results(df_combined, combined_filename, run_name)
        output_manager.save_csv_results(df_combined, combined_filename, dataset_specific=True)

        print(f"✅ Saved combined {data_type}-level results: {len(all_results)} records")

    # 3. Save summary statistics using organized structure
    summary_results = []
    for ds_name, metrics in results.items():
        summary_record = {
            'dataset': ds_name,
            'num_files': metrics.get('num_files', 0),
            'num_splits': metrics.get('num_splits', 0),
            'file_level_accuracy': metrics.get('file_level_accuracy', 0),
            'file_level_f1': metrics.get('file_level_f1', 0),
            'file_level_precision': metrics.get('file_level_precision', 0),
            'file_level_recall': metrics.get('file_level_recall', 0),
            'unique_true_labels': metrics.get('unique_true_labels', 0),
            'unique_pred_labels': metrics.get('unique_pred_labels', 0),
            'detection_rate': metrics.get('detection_rate', ''),
            'false_negative_rate': metrics.get('false_negative_rate', ''),
            'specificity': metrics.get('specificity', ''),
            'false_positive_rate': metrics.get('false_positive_rate', ''),
        }

        # Add clip-level metrics if available
        if 'clip_level_accuracy' in metrics:
            summary_record['num_clips'] = metrics.get('num_clips', 0)
            summary_record['clip_level_accuracy'] = metrics.get('clip_level_accuracy', 0)
            summary_record['clip_level_f1'] = metrics.get('clip_level_f1', 0)

        summary_results.append(summary_record)

    # Save summary CSV using organized structure
    df_summary = pd.DataFrame(summary_results)
    summary_filename = f'dataset_summary_{data_type}_results.csv'

    # Save to summaries directory
    summary_path = output_manager.structure['summaries'] / summary_filename
    if FILE_PROTECTION_AVAILABLE:
        safe_df_to_csv(df_summary, str(summary_path), index=False, quoting=1)
    else:
        df_summary.to_csv(summary_path, index=False)

    print(f"✅ Saved dataset summary: {len(summary_results)} datasets")

    # 4. Save evaluation run summary
    run_summary = {
        'evaluation_type': f'{data_type}_based',
        'datasets_processed': len(results),
        'total_records': len(all_results) if all_results else 0,
        'individual_results': individual_results,
        'threshold_used': threshold,
        'aggregation_info': aggregation_info,
        'output_structure': {
            'run_directory': str(run_dir),
            'csv_files': list(individual_results.keys()) + [combined_filename, summary_filename],
            'organized_structure': True
        }
    }

    output_manager.save_summary(run_summary, run_name)

    print(f"\n📁 Organized CSV Files Generated:")
    print(f"   📂 Run directory: {run_dir}")
    print(f"   📊 Individual datasets: {len(individual_results)} files in datasets/")
    print(f"   📈 Combined {data_type}-level: {run_dir}/csv/{combined_filename}")
    print(f"   📋 Dataset summaries: {summary_path}")
    print(f"   🗂️  Run summary: {run_dir}/run_summary.json")

    return output_manager, run_dir
def plot_probability_distributions(results, title_suffix="", save_prefix=""):
    """
    Create probability distribution plots for evaluation results.
    Now includes enhanced log-scale plotting when available.

    Args:
        results: Dictionary of evaluation results by dataset
        title_suffix: Additional text for plot titles
        save_prefix: Prefix for saved plot filenames
    """
    print(f"\n=== Plotting Prediction Probabilities{title_suffix} ===")

    # Use enhanced plotting if available
    if ENHANCED_PLOTTING_AVAILABLE:
        print("🎨 Using enhanced plotting with log-scale distributions...")

        # Generate comprehensive plot suite
        output_dir = "./enhanced_plots"
        try:
            enhanced_plots = generate_comprehensive_plot_suite(results, output_dir)

            print("✅ Enhanced plots generated:")
            for plot_type, filename in enhanced_plots.items():
                print(f"  📊 {plot_type.replace('_', ' ').title()}: {filename}")

            # Also create the traditional plot for backward compatibility
            filename = _create_traditional_probability_plot(results, title_suffix, save_prefix)
            return filename

        except Exception as e:
            print(f"⚠️  Enhanced plotting failed: {e}")
            print("Falling back to traditional plotting...")
            return _create_traditional_probability_plot(results, title_suffix, save_prefix)
    else:
        print("Using traditional plotting (enhanced plotting not available)")
        return _create_traditional_probability_plot(results, title_suffix, save_prefix)


def _create_traditional_probability_plot(results, title_suffix="", save_prefix="", config=None):
    """Create the traditional probability distribution plot."""
    # Get threshold from config
    if config:
        threshold = config.clip.threshold if "clip" in title_suffix.lower() else config.file.threshold
    else:
        from TRANSFORMER.config.config import EvaluationConfig
        default_config = EvaluationConfig()
        threshold = default_config.clip.threshold if "clip" in title_suffix.lower() else default_config.file.threshold

    # Create probability distribution plots
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle(f'Prediction Probability Distributions by Dataset{title_suffix}', fontsize=16)
    axes = fig.subplots(2, 5)

    datasets_to_plot = list(results.keys())[:10]  # Plot first 10 datasets
    for i, dataset_name in enumerate(datasets_to_plot):
        if 'probabilities' not in results[dataset_name]:
            continue

        row = i // 5
        col = i % 5
        ax = axes[row, col]

        probs = np.array(results[dataset_name]['probabilities'])
        labels = np.array(results[dataset_name]['true_labels'])

        # Plot histograms for true drone vs non-drone
        drone_probs = probs[labels == 1]  # True drone samples
        non_drone_probs = probs[labels == 0]  # True non-drone samples

        # Plot overlapping histograms
        ax.hist(non_drone_probs, bins=50, alpha=0.7, label=f'Non-drone ({len(non_drone_probs)})',
                color='blue', density=True)
        if len(drone_probs) > 0:
            ax.hist(drone_probs, bins=50, alpha=0.7, label=f'Drone ({len(drone_probs)})',
                    color='red', density=True)

        # Add vertical line at configured threshold
        ax.axvline(x=threshold, color='black', linestyle='--', alpha=0.8,
                  label=f'Current threshold ({threshold:.3f})')

        ax.set_title(f'{dataset_name}', fontsize=12)
        ax.set_xlabel('P(yes_drone)')
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Print some statistics using configured threshold
        data_type = "files" if title_suffix.lower().find("file") != -1 else "clips"
        print(f"\n{dataset_name}:")
        print(f"  Mean P(drone) for non-drone {data_type}: {np.mean(non_drone_probs):.4f}")
        if len(drone_probs) > 0:
            print(f"  Mean P(drone) for drone {data_type}: {np.mean(drone_probs):.4f}")
        print(f"  % {data_type} with P(drone) > {threshold:.3f}: {np.mean(probs > threshold)*100:.2f}%")
        print(f"  % {data_type} with P(drone) > 0.3: {np.mean(probs > 0.3)*100:.2f}%")
        print(f"  % {data_type} with P(drone) > 0.2: {np.mean(probs > 0.2)*100:.2f}%")

    # Hide empty subplots
    for i in range(len(datasets_to_plot), 10):
        row = i // 5
        col = i % 5
        axes[row, col].set_visible(False)

    plt.tight_layout()
    filename = f'{save_prefix}prediction_probabilities_by_dataset.png'
    safe_plt_savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    return filename


def plot_threshold_analysis(results, title_suffix="", save_prefix=""):
    """
    Create threshold analysis plots for evaluation results.

    Args:
        results: Dictionary of evaluation results by dataset
        title_suffix: Additional text for plot titles
        save_prefix: Prefix for saved plot filenames
    """
    # Create summary probability plot
    plt.figure(figsize=(15, 8))

    # Combine all datasets for overall view
    all_drone_probs = []
    all_non_drone_probs = []
    dataset_stats = []

    for dataset_name, data in results.items():
        if 'probabilities' not in data:
            continue

        probs = np.array(data['probabilities'])
        labels = np.array(data['true_labels'])

        drone_probs = probs[labels == 1]
        non_drone_probs = probs[labels == 0]

        all_drone_probs.extend(drone_probs)
        all_non_drone_probs.extend(non_drone_probs)

        # Calculate potential performance at different thresholds
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        for thresh in thresholds:
            pred_at_thresh = (probs > thresh).astype(int)
            if len(np.unique(labels)) > 1:  # Only for mixed datasets
                acc = accuracy_score(labels, pred_at_thresh)
                precision, recall, f1, _ = precision_recall_fscore_support(labels, pred_at_thresh, average='binary', zero_division=0.0)
                dataset_stats.append({
                    'dataset': dataset_name,
                    'threshold': thresh,
                    'accuracy': acc,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                })

    # Plot combined histograms
    plt.subplot(2, 2, 1)
    plt.hist(all_non_drone_probs, bins=50, alpha=0.7, label=f'Non-drone ({len(all_non_drone_probs)})',
             color='blue', density=True)
    if len(all_drone_probs) > 0:
        plt.hist(all_drone_probs, bins=50, alpha=0.7, label=f'Drone ({len(all_drone_probs)})',
                 color='red', density=True)
    plt.axvline(x=0.5, color='black', linestyle='--', alpha=0.8, label='Current threshold (0.5)')
    plt.xlabel('P(yes_drone)')
    plt.ylabel('Density')
    plt.title(f'Combined Probability Distribution{title_suffix}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot threshold analysis for mixed datasets
    if dataset_stats:
        df_stats = pd.DataFrame(dataset_stats)

        plt.subplot(2, 2, 2)
        for dataset in df_stats['dataset'].unique():
            data = df_stats[df_stats['dataset'] == dataset]
            plt.plot(data['threshold'], data['f1'], marker='o', label=dataset)
        plt.xlabel('Threshold')
        plt.ylabel('F1 Score')
        plt.title('F1 Score vs Threshold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 3)
        for dataset in df_stats['dataset'].unique():
            data = df_stats[df_stats['dataset'] == dataset]
            plt.plot(data['threshold'], data['recall'], marker='s', label=dataset)
        plt.xlabel('Threshold')
        plt.ylabel('Recall')
        plt.title('Recall vs Threshold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 4)
        for dataset in df_stats['dataset'].unique():
            data = df_stats[df_stats['dataset'] == dataset]
            plt.plot(data['threshold'], data['precision'], marker='^', label=dataset)
        plt.xlabel('Threshold')
        plt.ylabel('Precision')
        plt.title('Precision vs Threshold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = f'{save_prefix}threshold_analysis.png'
    safe_plt_savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    return filename


def print_performance_analysis(results, data_type="file"):
    """
    Print detailed performance analysis and recommendations.

    Args:
        results: Dictionary of evaluation results by dataset
        data_type: Type of data being analyzed ("file" or "clip")
    """
    # Combine all datasets for overall analysis
    all_drone_probs = []
    all_non_drone_probs = []

    for dataset_name, data in results.items():
        if 'probabilities' not in data:
            continue

        probs = np.array(data['probabilities'])
        labels = np.array(data['true_labels'])

        drone_probs = probs[labels == 1]
        non_drone_probs = probs[labels == 0]

        all_drone_probs.extend(drone_probs)
        all_non_drone_probs.extend(non_drone_probs)

    print(f"\nOverall Statistics ({data_type}-level):")
    print(f"Total {data_type}s analyzed: {len(all_drone_probs) + len(all_non_drone_probs)}")
    print(f"True drone {data_type}s: {len(all_drone_probs)}")
    print(f"True non-drone {data_type}s: {len(all_non_drone_probs)}")

    if len(all_drone_probs) > 0:
        print(f"Mean P(drone) for true drone {data_type}s: {np.mean(all_drone_probs):.4f}")
        print(f"Max P(drone) for true drone {data_type}s: {np.max(all_drone_probs):.4f}")
        print(f"Min P(drone) for true drone {data_type}s: {np.min(all_drone_probs):.4f}")
        print(f"Std P(drone) for true drone {data_type}s: {np.std(all_drone_probs):.4f}")

    print(f"Mean P(drone) for true non-drone {data_type}s: {np.mean(all_non_drone_probs):.4f}")
    print(f"Max P(drone) for true non-drone {data_type}s: {np.max(all_non_drone_probs):.4f}")
    print(f"Min P(drone) for true non-drone {data_type}s: {np.min(all_non_drone_probs):.4f}")

    print("\nModel Performance Analysis:")
    if len(all_drone_probs) > 0:
        drone_mean = np.mean(all_drone_probs)
        non_drone_mean = np.mean(all_non_drone_probs)
        separation = drone_mean - non_drone_mean
        print(f"Probability separation: {separation:.4f}")

    if len(all_drone_probs) > 0:
        drone_mean = np.mean(all_drone_probs)
        non_drone_mean = np.mean(all_non_drone_probs)

    else:
        print("No drone files found in evaluation set")


def print_evaluation_summary(results, script_type="file"):
    """
    Print a comprehensive evaluation summary.

    Args:
        results: Dictionary of evaluation results by dataset
        script_type: Type of evaluation ("file" or "clip")
    """
    print("\n=== Evaluation Summary ===")

    if script_type == "file":
        header = f"{'Dataset':<15} {'Files':<8} {'Splits':<7} {'File-Acc':<10} {'File-F1':<10} {'File-Prec':<10} {'File-Rec':<10}"
        print(header)
        print("-" * len(header))

        for ds_name, metrics in results.items():
            print(f"{ds_name:<15} {metrics.get('num_files', 0):<8} {metrics.get('num_splits', 0):<7} "
                  f"{metrics.get('file_level_accuracy', 0):<10.3f} {metrics.get('file_level_f1', 0):<10.3f} "
                  f"{metrics.get('file_level_precision', 0):<10.3f} {metrics.get('file_level_recall', 0):<10.3f}")

    elif script_type == "clip":
        header = f"{'Dataset':<15} {'Files':<8} {'Clips':<8} {'Splits':<7} {'File-Acc':<10} {'File-F1':<10} {'Clip-Acc':<10} {'Clip-F1':<10}"
        print(header)
        print("-" * len(header))

        for ds_name, metrics in results.items():
            print(f"{ds_name:<15} {metrics.get('num_files', 0):<8} {metrics.get('num_clips', 0):<8} {metrics.get('num_splits', 0):<7} "
                  f"{metrics.get('file_level_accuracy', 0):<10.3f} {metrics.get('file_level_f1', 0):<10.3f} "
                  f"{metrics.get('clip_level_accuracy', 0):<10.3f} {metrics.get('clip_level_f1', 0):<10.3f}")


def print_detailed_prediction_summary(results):
    """
    Print detailed prediction summary by dataset.

    Args:
        results: Dictionary of evaluation results by dataset
    """
    print("\n" + "="*80)
    print("DETAILED PREDICTION SUMMARY BY DATASET")
    print("="*80)

    total_drone_files = 0
    total_non_drone_files = 0
    total_drone_correct = 0
    total_non_drone_correct = 0

    for ds_name, metrics in results.items():
        if 'file_predictions' in metrics and 'file_labels' in metrics:
            file_preds = metrics['file_predictions']
            file_labels = metrics['file_labels']

            # Count true drone vs non-drone files
            drone_mask = (file_labels == 1)
            non_drone_mask = (file_labels == 0)

            num_drone_files = np.sum(drone_mask)
            num_non_drone_files = np.sum(non_drone_mask)

            # Count correct predictions
            correct_drone = np.sum((file_preds == 1) & (file_labels == 1))
            correct_non_drone = np.sum((file_preds == 0) & (file_labels == 0))

            # Calculate percentages
            drone_accuracy = (correct_drone / num_drone_files * 100) if num_drone_files > 0 else 0
            non_drone_accuracy = (correct_non_drone / num_non_drone_files * 100) if num_non_drone_files > 0 else 0

            print(f"\n{ds_name}:")
            if num_drone_files > 0:
                print(f"  Drone files:     {num_drone_files:,} total → {correct_drone:,} predicted correctly ({drone_accuracy:.1f}%)")
            if num_non_drone_files > 0:
                print(f"  Non-drone files: {num_non_drone_files:,} total → {correct_non_drone:,} predicted correctly ({non_drone_accuracy:.1f}%)")

            # Show misclassifications
            missed_drones = num_drone_files - correct_drone
            false_alarms = num_non_drone_files - correct_non_drone
            if missed_drones > 0:
                print(f"  Missed drones:   {missed_drones:,} files ({missed_drones/num_drone_files*100:.1f}% of drone files)")
            if false_alarms > 0:
                print(f"  False alarms:    {false_alarms:,} files ({false_alarms/num_non_drone_files*100:.1f}% of non-drone files)")

            # Add to totals
            total_drone_files += num_drone_files
            total_non_drone_files += num_non_drone_files
            total_drone_correct += correct_drone
            total_non_drone_correct += correct_non_drone

    # Overall summary
    print("\n" + "-"*50)
    print("OVERALL SUMMARY ACROSS ALL DATASETS:")
    print("-"*50)

    if total_drone_files > 0:
        overall_drone_acc = total_drone_correct / total_drone_files * 100
        print(f"Total drone files:     {total_drone_files:,} → {total_drone_correct:,} correctly identified ({overall_drone_acc:.1f}%)")

    if total_non_drone_files > 0:
        overall_non_drone_acc = total_non_drone_correct / total_non_drone_files * 100
        print(f"Total non-drone files: {total_non_drone_files:,} → {total_non_drone_correct:,} correctly identified ({overall_non_drone_acc:.1f}%)")

    total_files = total_drone_files + total_non_drone_files
    total_correct = total_drone_correct + total_non_drone_correct
    if total_files > 0:
        overall_accuracy = total_correct / total_files * 100
        print(f"Overall file accuracy: {total_correct:,}/{total_files:,} ({overall_accuracy:.1f}%)")

        missed_total = total_drone_files - total_drone_correct
        false_alarms_total = total_non_drone_files - total_non_drone_correct
        print(f"Total missed drones:   {missed_total:,}")
        print(f"Total false alarms:    {false_alarms_total:,}")


def integrate_with_existing_analysis(results, analysis_output_dir="probability_analysis"):
    """
    Integration bridge to work with your existing analyze_probability_distributions.py
    """
    print(f"\n=== INTEGRATING WITH EXISTING ANALYSIS ===")

    # Save results in format compatible with analyze_probability_distributions.py
    compatible_results_dir = os.path.join(analysis_output_dir, "compatible_format")
    os.makedirs(compatible_results_dir, exist_ok=True)

    for dataset_name, metrics in results.items():
        if 'probabilities' in metrics and 'true_labels' in metrics:
            # Create CSV in format expected by your analysis script
            compatible_data = []

            for i, (prob, label) in enumerate(zip(metrics['probabilities'], metrics['true_labels'])):
                compatible_data.append({
                    'file_id': i,
                    'true_label': int(label),
                    'drone_probability': float(prob),
                    'predicted_label': 1 if prob > 0.5 else 0
                })

            # Save in format your existing script expects
            df = pd.DataFrame(compatible_data)
            csv_path = os.path.join(compatible_results_dir, f"{dataset_name}_detailed_results.csv")
            df.to_csv(csv_path, index=False)

            print(f"  Saved {dataset_name}: {len(compatible_data)} records")

    print(f"Compatible data saved to: {compatible_results_dir}")
    print("You can now run analyze_probability_distributions.py on this data")

    return compatible_results_dir

def create_unified_pipeline_runner():
    """
    Create a unified runner that combines all analysis approaches
    """
    def run_complete_analysis_pipeline(strategy_type="file", model_path=None,
                                     output_dir="./unified_results", **kwargs):
        """
        Run the complete analysis pipeline combining all approaches
        """
        print("STARTING UNIFIED ANALYSIS PIPELINE")
        print("="*60)

        # Get configuration
        from TRANSFORMER.config.config import get_config
        config = get_config()

        # Override with provided parameters
        if model_path:
            config.model.model_path = model_path
        if output_dir != "./unified_results":
            config.output.base_output_dir = output_dir

        # 1. Run main evaluation using strategy factory
        print("\n1️⃣  Running enhanced evaluation...")

        try:
            from evaluation_strategy_factory import run_evaluation

            # Prepare strategy-specific parameters
            strategy_params = {}
            if strategy_type == "clip":
                strategy_params = {
                    'clip_duration': config.clip.clip_duration,
                    'batch_size': config.clip.batch_size,
                    'max_clips_per_dataset': config.clip.max_clips_per_dataset,
                }
            else:  # file
                strategy_params = {
                    'batch_size': config.file.batch_size,
                    'large_file_threshold': config.file.large_file_threshold,
                    'very_large_threshold': config.file.very_large_threshold,
                    'max_file_length': config.file.max_file_length,
                    'max_file_size_mb': config.file.max_file_size_mb,
                }

            # Override with any additional kwargs
            strategy_params.update(kwargs)

            # Run evaluation
            results = run_evaluation(
                strategy_type=strategy_type,
                model_path=config.model.model_path,
                output_dir=config.output.base_output_dir,
                **strategy_params
            )

        except Exception as e:
            print(f"❌ Enhanced evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return None

        if not results:
            print("❌ No results obtained from evaluation")
            return None

        # 2. Create compatible data for existing analysis
        print("\n2️⃣  Preparing data for existing analysis...")
        try:
            compatible_dir = integrate_with_existing_analysis(results,
                                                           f"{config.output.base_output_dir}/compatible_format")
            print(f"✅ Compatible data prepared in: {compatible_dir}")
        except Exception as e:
            print(f"⚠️  Failed to prepare compatible data: {e}")
            compatible_dir = None

        # 3. Run advanced analysis if available
        print("\n3️⃣  Running advanced analysis...")
        try:
            from advanced_analysis import generate_comprehensive_analysis_report

            analysis_results = generate_comprehensive_analysis_report(
                results,
                f"{config.output.base_output_dir}/comprehensive_analysis"
            )

            print("✅ Advanced analysis completed!")
            print("Generated insights:")
            if 'optimal_thresholds' in analysis_results:
                opt_f1 = analysis_results['optimal_thresholds'].get('f1', 0.5)
                print(f"  - Optimal F1 threshold: {opt_f1:.3f}")
            print("  - Dataset difficulty ranking")
            print("  - Error pattern analysis")
            print("  - Performance consistency assessment")

        except ImportError:
            print("⚠️  Advanced analysis not available")
        except Exception as e:
            print(f"⚠️  Advanced analysis failed: {e}")

        # 4. Run logits analysis for file-based evaluation
        if strategy_type == "file":
            print("\n4️⃣  Running logits analysis...")
            try:
                from TRANSFORMER.utils.aggregation_utils import analyze_logits_statistics
                analyze_logits_statistics(results,
                                        f"{config.output.base_output_dir}/logits_analysis.png")
                print("✅ Logits analysis completed!")
            except Exception as e:
                print(f"⚠️  Logits analysis failed: {e}")

        # 5. Generate summary report
        print("\n5️⃣  Generating summary report...")
        try:
            summary_path = f"{config.output.base_output_dir}/pipeline_summary.md"
            generate_pipeline_summary(results, strategy_type, config, summary_path)
            print(f"✅ Pipeline summary saved to: {summary_path}")
        except Exception as e:
            print(f"⚠️  Summary generation failed: {e}")

        print("\n✅ UNIFIED ANALYSIS PIPELINE COMPLETE")
        print(f"📁 All results saved to: {config.output.base_output_dir}")
        print(f"📊 Processed {len(results)} datasets with {strategy_type}-based evaluation")

        return results

    return run_complete_analysis_pipeline


def generate_pipeline_summary(results, strategy_type, config, summary_path):
    """Generate a comprehensive summary of the pipeline execution"""
    import os
    from datetime import datetime

    os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    with open(summary_path, 'w') as f:
        f.write("# Unified Analysis Pipeline Summary\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Strategy:** {strategy_type.title()}-based evaluation\n")
        f.write(f"**Model:** {config.model.model_path}\n\n")

        # Dataset results summary
        f.write("## Dataset Results\n\n")
        f.write("| Dataset | Files | Accuracy | F1 Score | Precision | Recall |\n")
        f.write("|---------|-------|----------|----------|-----------|--------|\n")

        for dataset_name, metrics in results.items():
            f.write(f"| {dataset_name} | {metrics.get('num_files', 0)} | "
                   f"{metrics.get('file_level_accuracy', 0):.3f} | "
                   f"{metrics.get('file_level_f1', 0):.3f} | "
                   f"{metrics.get('file_level_precision', 0):.3f} | "
                   f"{metrics.get('file_level_recall', 0):.3f} |\n")

        # Overall statistics
        f.write("\n## Overall Statistics\n\n")
        total_files = sum(metrics.get('num_files', 0) for metrics in results.values())
        avg_accuracy = sum(metrics.get('file_level_accuracy', 0) for metrics in results.values()) / len(results)
        avg_f1 = sum(metrics.get('file_level_f1', 0) for metrics in results.values()) / len(results)

        f.write(f"- **Total Datasets:** {len(results)}\n")
        f.write(f"- **Total Files Processed:** {total_files:,}\n")
        f.write(f"- **Average Accuracy:** {avg_accuracy:.3f}\n")
        f.write(f"- **Average F1 Score:** {avg_f1:.3f}\n")

        # Configuration used
        f.write("\n## Configuration\n\n")
        f.write("```json\n")
        import json
        f.write(json.dumps(config.to_dict(), indent=2))
        f.write("\n```\n")

        # Generated files
        f.write("\n## Generated Files\n\n")
        f.write("The pipeline generated the following analysis files:\n")
        f.write("- Probability distribution plots\n")
        f.write("- Threshold analysis charts\n")
        f.write("- Detailed CSV results\n")
        f.write("- Performance metrics\n")

        if strategy_type == "file":
            f.write("- Logits analysis\n")

        f.write("- Comprehensive analysis report (if available)\n")
        f.write("- Compatible format data for external analysis\n")
