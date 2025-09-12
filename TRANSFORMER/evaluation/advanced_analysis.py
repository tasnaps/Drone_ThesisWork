#!/usr/bin/env python3
"""
Result analysis and visualization utilities for transformer model evaluation.
This module provides comprehensive analysis tools for evaluation results.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from scipy import stats
from TRANSFORMER.config.config import ClipEvaluationConfig, FileEvaluationConfig


def generate_comprehensive_analysis_report(results, output_dir="analysis_report", evaluation_type="clip"):
    """
    Generate a comprehensive analysis report combining multiple visualization and analysis techniques.

    Args:
        results: Dictionary of evaluation results by dataset
        output_dir: Directory to save analysis outputs
        evaluation_type: Type of evaluation ("clip" or "file") to determine correct threshold
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n=== Generating Comprehensive Analysis Report ===")

    # Get the correct threshold from configuration
    if evaluation_type == "clip":
        config_threshold = ClipEvaluationConfig().threshold
    else:
        config_threshold = FileEvaluationConfig().threshold

    # 1. Advanced threshold optimization
    optimal_thresholds = analyze_optimal_thresholds(results, output_dir, config_threshold)

    # 2. Class imbalance analysis
    imbalance_analysis = analyze_class_imbalance(results, output_dir)

    # 3. Performance consistency analysis
    consistency_analysis = analyze_performance_consistency(results, output_dir)

    # 4. Dataset difficulty analysis
    difficulty_analysis = analyze_dataset_difficulty(results, output_dir, config_threshold)

    # 5. Error pattern analysis
    error_patterns = analyze_error_patterns(results, output_dir, config_threshold)

    # 6. Logit analysis by dataset
    logit_analysis = analyze_logits_by_dataset(results, output_dir, config_threshold)

    # 7. Generate executive summary
    generate_executive_summary(
        results, optimal_thresholds, imbalance_analysis,
        consistency_analysis, difficulty_analysis, error_patterns,
        output_dir, config_threshold
    )

    print(f"Comprehensive analysis complete. Results saved to: {output_dir}")
    return {
        'optimal_thresholds': optimal_thresholds,
        'imbalance_analysis': imbalance_analysis,
        'consistency_analysis': consistency_analysis,
        'difficulty_analysis': difficulty_analysis,
        'error_patterns': error_patterns,
        'logit_analysis': logit_analysis  # Add missing logit_analysis to return
    }


def analyze_optimal_thresholds(results, output_dir, config_threshold):
    """Analyze optimal thresholds using multiple criteria."""
    print("Analyzing optimal thresholds...")

    # Combine all data
    all_probs = []
    all_labels = []
    for dataset_results in results.values():
        if 'probabilities' in dataset_results and 'true_labels' in dataset_results:
            all_probs.extend(dataset_results['probabilities'])
            all_labels.extend(dataset_results['true_labels'])

    if len(set(all_labels)) <= 1:
        return {'error': 'Insufficient class diversity for threshold analysis'}

    thresholds = np.linspace(0.01, 0.99, 100)
    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'specificity': [],
        'balanced_accuracy': [],
        'mcc': []  # Matthews Correlation Coefficient
    }

    for threshold in thresholds:
        preds = (np.array(all_probs) >= threshold).astype(int)

        tp = np.sum((preds == 1) & (np.array(all_labels) == 1))
        fp = np.sum((preds == 1) & (np.array(all_labels) == 0))
        tn = np.sum((preds == 0) & (np.array(all_labels) == 0))
        fn = np.sum((preds == 0) & (np.array(all_labels) == 1))

        # Calculate metrics
        accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        balanced_acc = (recall + specificity) / 2

        # Matthews Correlation Coefficient
        mcc_denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = ((tp * tn) - (fp * fn)) / mcc_denom if mcc_denom > 0 else 0

        metrics['accuracy'].append(accuracy)
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['f1'].append(f1)
        metrics['specificity'].append(specificity)
        metrics['balanced_accuracy'].append(balanced_acc)
        metrics['mcc'].append(mcc)

    # Find optimal thresholds
    optimal_results = {
        'accuracy': thresholds[np.argmax(metrics['accuracy'])],
        'f1': thresholds[np.argmax(metrics['f1'])],
        'balanced_accuracy': thresholds[np.argmax(metrics['balanced_accuracy'])],
        'mcc': thresholds[np.argmax(metrics['mcc'])],
        'youden_j': thresholds[np.argmax(np.array(metrics['recall']) + np.array(metrics['specificity']) - 1)]
    }

    # Create threshold optimization plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Comprehensive Threshold Optimization Analysis', fontsize=16)

    # Plot metrics vs threshold
    ax1 = axes[0, 0]
    ax1.plot(thresholds, metrics['f1'], 'b-', linewidth=2, label='F1 Score')
    ax1.plot(thresholds, metrics['balanced_accuracy'], 'g-', linewidth=2, label='Balanced Accuracy')
    ax1.plot(thresholds, metrics['mcc'], 'r-', linewidth=2, label='MCC')
    ax1.axvline(x=optimal_results['f1'], color='blue', linestyle='--', alpha=0.7, label=f'F1 Optimal ({optimal_results["f1"]:.3f})')
    ax1.axvline(x=config_threshold, color='black', linestyle='-', alpha=0.8, linewidth=2, label=f'Current ({config_threshold:.3f})')  # Add current threshold line
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Score')
    ax1.set_title('Primary Metrics vs Threshold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot precision/recall vs threshold
    ax2 = axes[0, 1]
    ax2.plot(thresholds, metrics['precision'], 'r-', linewidth=2, label='Precision')
    ax2.plot(thresholds, metrics['recall'], 'b-', linewidth=2, label='Recall')
    ax2.plot(thresholds, metrics['specificity'], 'g-', linewidth=2, label='Specificity')
    ax2.axvline(x=optimal_results['youden_j'], color='purple', linestyle='--', alpha=0.7, label="Youden's J")
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('Score')
    ax2.set_title('Precision/Recall/Specificity vs Threshold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # ROC curve
    ax3 = axes[0, 2]
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    ax3.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
    ax3.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax3.set_xlabel('False Positive Rate')
    ax3.set_ylabel('True Positive Rate')
    ax3.set_title('ROC Curve')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Precision-Recall curve
    ax4 = axes[1, 0]
    precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_probs)
    avg_precision = average_precision_score(all_labels, all_probs)
    ax4.plot(recall_curve, precision_curve, 'g-', linewidth=2, label=f'PR (AP = {avg_precision:.3f})')
    ax4.set_xlabel('Recall')
    ax4.set_ylabel('Precision')
    ax4.set_title('Precision-Recall Curve')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Optimal thresholds comparison
    ax5 = axes[1, 1]
    opt_names = list(optimal_results.keys())
    opt_values = list(optimal_results.values())
    bars = ax5.bar(opt_names, opt_values, alpha=0.7, color=['blue', 'green', 'red', 'orange', 'purple'])
    ax5.set_ylabel('Optimal Threshold')
    ax5.set_title('Optimal Thresholds by Criterion')
    ax5.tick_params(axis='x', rotation=45)
    ax5.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, value in zip(bars, opt_values):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom')

    # Cost analysis
    ax6 = axes[1, 2]
    # Assume FP cost = 1, FN cost = 3 (missing drone is worse)
    fp_costs = np.array(metrics['precision']) * len(all_labels) * (1 - np.array(metrics['precision']))
    fn_costs = np.array(metrics['recall']) * len(all_labels) * (1 - np.array(metrics['recall'])) * 3
    total_costs = fp_costs + fn_costs

    optimal_cost_idx = np.argmin(total_costs)
    optimal_cost_threshold = thresholds[optimal_cost_idx]

    ax6.plot(thresholds, total_costs, 'purple', linewidth=2, label='Total Cost')
    ax6.axvline(x=optimal_cost_threshold, color='red', linestyle='--', alpha=0.7,
               label=f'Cost Optimal ({optimal_cost_threshold:.3f})')
    ax6.set_xlabel('Threshold')
    ax6.set_ylabel('Total Cost')
    ax6.set_title('Cost Analysis (FP=1, FN=3)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'threshold_optimization_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

    return optimal_results


def analyze_class_imbalance(results, output_dir):
    """Analyze the impact of class imbalance on model performance."""
    print("Analyzing class imbalance effects...")

    # Suppress specific warnings that we'll handle properly
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered in subtract")
        warnings.filterwarnings("ignore", message="invalid value encountered in divide")

        imbalance_data = []

        for dataset_name, dataset_results in results.items():
            if 'true_labels' not in dataset_results:
                continue

            labels = np.array(dataset_results['true_labels'])
            n_positive = np.sum(labels == 1)
            n_negative = np.sum(labels == 0)
            total = len(labels)

            if total == 0:
                continue

            # Safe division with zero check
            imbalance_ratio = n_positive / n_negative if n_negative > 0 else float('inf') if n_positive > 0 else 1.0
            positive_ratio = n_positive / total

            # Calculate performance metrics
            if 'file_level_accuracy' in dataset_results:
                accuracy = dataset_results['file_level_accuracy']
                f1 = dataset_results.get('file_level_f1', 0)
                precision = dataset_results.get('file_level_precision', 0)
                recall = dataset_results.get('file_level_recall', 0)
            else:
                accuracy = f1 = precision = recall = 0

            imbalance_data.append({
                'dataset': dataset_name,
                'total_samples': total,
                'positive_samples': n_positive,
                'negative_samples': n_negative,
                'imbalance_ratio': imbalance_ratio,
                'positive_ratio': positive_ratio,
                'accuracy': accuracy,
                'f1': f1,
                'precision': precision,
                'recall': recall
            })

        if not imbalance_data:
            return {'error': 'No valid data for imbalance analysis'}

        # Create imbalance analysis plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Class Imbalance Impact Analysis', fontsize=16)

        df = pd.DataFrame(imbalance_data)

        # Plot 1: Imbalance ratio vs F1 score
        ax1 = axes[0, 0]

        # Filter out infinite values for plotting
        finite_mask = np.isfinite(df['imbalance_ratio']) & np.isfinite(df['f1'])
        df_finite = df[finite_mask]

        if len(df_finite) > 0:
            scatter = ax1.scatter(df_finite['imbalance_ratio'], df_finite['f1'],
                                s=df_finite['total_samples']/10, alpha=0.6,
                                c=df_finite['accuracy'], cmap='viridis')
            ax1.set_xlabel('Imbalance Ratio (Positive/Negative)')
            ax1.set_ylabel('F1 Score')
            ax1.set_title('F1 Score vs Class Imbalance')
            ax1.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax1, label='Accuracy')

            # Add dataset labels for finite values only
            for i, row in df_finite.iterrows():
                if len(row['dataset']) < 15:  # Only label shorter names to avoid clutter
                    ax1.annotate(row['dataset'], (row['imbalance_ratio'], row['f1']),
                                xytext=(5, 5), textcoords='offset points', fontsize=8)
        else:
            ax1.text(0.5, 0.5, 'No valid data\nfor imbalance plot',
                    ha='center', va='center', transform=ax1.transAxes, fontsize=12)
            ax1.set_title('F1 Score vs Class Imbalance (No Data)')

        # Plot 2: Sample size distribution
        ax2 = axes[0, 1]
        ax2.bar(range(len(df)), df['positive_samples'], alpha=0.7, label='Positive', color='red')
        ax2.bar(range(len(df)), df['negative_samples'], bottom=df['positive_samples'],
                alpha=0.7, label='Negative', color='blue')
        ax2.set_xlabel('Dataset Index')
        ax2.set_ylabel('Number of Samples')
        ax2.set_title('Sample Distribution by Dataset')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Performance correlation matrix with robust error handling
        ax3 = axes[1, 0]
        try:
            # Calculate correlation matrix with proper NaN handling
            corr_columns = ['imbalance_ratio', 'positive_ratio', 'accuracy', 'f1', 'precision', 'recall']
            corr_data_subset = df[corr_columns].copy()

            # Replace infinite values with NaN before processing
            corr_data_subset = corr_data_subset.replace([np.inf, -np.inf], np.nan)

            # Remove columns with zero variance or all NaN values
            valid_columns = []
            for col in corr_columns:
                col_data = corr_data_subset[col].dropna()
                if len(col_data) > 1 and col_data.std() > 1e-10:  # Only include columns with meaningful variance
                    valid_columns.append(col)

            if len(valid_columns) >= 2:
                # Calculate correlation only for valid columns
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    corr_data = corr_data_subset[valid_columns].corr()

                # Replace any remaining NaN or inf values
                corr_data = corr_data.fillna(0)
                corr_data = corr_data.replace([np.inf, -np.inf], 0)

                # Check if we have a valid correlation matrix
                if corr_data.shape[0] > 1 and not corr_data.isna().all().all():
                    im = ax3.imshow(corr_data, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
                    ax3.set_xticks(range(len(corr_data.columns)))
                    ax3.set_yticks(range(len(corr_data.columns)))
                    ax3.set_xticklabels(corr_data.columns, rotation=45)
                    ax3.set_yticklabels(corr_data.columns)
                    ax3.set_title('Performance Correlation Matrix')

                    # Add correlation values
                    for i in range(len(corr_data.columns)):
                        for j in range(len(corr_data.columns)):
                            value = corr_data.iloc[i, j]
                            if not np.isnan(value) and not np.isinf(value):
                                ax3.text(j, i, f'{value:.2f}',
                                        ha='center', va='center',
                                        color='white' if abs(value) > 0.5 else 'black')

                    plt.colorbar(im, ax=ax3)
                else:
                    ax3.text(0.5, 0.5, 'No valid correlation data\navailable',
                            ha='center', va='center', transform=ax3.transAxes, fontsize=12)
                    ax3.set_title('Performance Correlation Matrix')
                    corr_data = pd.DataFrame()
            else:
                # Not enough valid columns for correlation analysis
                ax3.text(0.5, 0.5, 'Insufficient variance\nfor correlation analysis',
                        ha='center', va='center', transform=ax3.transAxes, fontsize=12)
                ax3.set_title('Performance Correlation Matrix')
                corr_data = pd.DataFrame()  # Empty dataframe

        except Exception as e:
            # Handle any other correlation calculation errors
            ax3.text(0.5, 0.5, f'Correlation analysis failed:\n{str(e)[:50]}...',
                    ha='center', va='center', transform=ax3.transAxes, fontsize=10)
            ax3.set_title('Performance Correlation Matrix (Error)')
            corr_data = pd.DataFrame()  # Empty dataframe

        # Plot 4: Recommendations based on imbalance
        ax4 = axes[1, 1]
        # Categorize datasets by imbalance severity (handle infinite values)
        finite_ratio_mask = np.isfinite(df['imbalance_ratio'])
        df_finite_ratio = df[finite_ratio_mask]

        if len(df_finite_ratio) > 0:
            balanced = df_finite_ratio[df_finite_ratio['imbalance_ratio'].between(0.5, 2.0)]
            moderate_imbalance = df_finite_ratio[(df_finite_ratio['imbalance_ratio'] < 0.5) |
                                               (df_finite_ratio['imbalance_ratio'] > 2.0)]
            severe_imbalance = df_finite_ratio[(df_finite_ratio['imbalance_ratio'] < 0.1) |
                                             (df_finite_ratio['imbalance_ratio'] > 10.0)]
        else:
            balanced = moderate_imbalance = severe_imbalance = pd.DataFrame()

        categories = ['Balanced\n(0.5-2.0)', 'Moderate\nImbalance', 'Severe\nImbalance']
        avg_f1s = [
            balanced['f1'].mean() if len(balanced) > 0 and not balanced['f1'].isna().all() else 0,
            moderate_imbalance['f1'].mean() if len(moderate_imbalance) > 0 and not moderate_imbalance['f1'].isna().all() else 0,
            severe_imbalance['f1'].mean() if len(severe_imbalance) > 0 and not severe_imbalance['f1'].isna().all() else 0
        ]
        # Replace NaN values with 0
        avg_f1s = [0 if np.isnan(x) else x for x in avg_f1s]

        counts = [len(balanced), len(moderate_imbalance), len(severe_imbalance)]

        bars = ax4.bar(categories, avg_f1s, alpha=0.7, color=['green', 'orange', 'red'])
        ax4.set_ylabel('Average F1 Score')
        ax4.set_title('Performance by Imbalance Category')
        ax4.grid(True, alpha=0.3)

        # Add count labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'n={count}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'class_imbalance_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()

        return {
            'imbalance_data': imbalance_data,
            'balanced_datasets': len(balanced),
            'moderate_imbalance_datasets': len(moderate_imbalance),
            'severe_imbalance_datasets': len(severe_imbalance),
            'correlation_matrix': corr_data.to_dict() if not corr_data.empty else {}
        }


def analyze_performance_consistency(results, output_dir):
    """Analyze consistency of model performance across different datasets."""
    print("Analyzing performance consistency...")

    # Extract performance metrics
    metrics_data = []
    for dataset_name, dataset_results in results.items():
        if 'file_level_accuracy' in dataset_results:
            metrics_data.append({
                'dataset': dataset_name,
                'accuracy': dataset_results.get('file_level_accuracy', 0),
                'f1': dataset_results.get('file_level_f1', 0),
                'precision': dataset_results.get('file_level_precision', 0),
                'recall': dataset_results.get('file_level_recall', 0),
                'num_files': dataset_results.get('num_files', 0)
            })

    if len(metrics_data) < 2:
        return {'error': 'Insufficient datasets for consistency analysis'}

    df = pd.DataFrame(metrics_data)

    # Calculate consistency metrics
    consistency_results = {
        'accuracy_std': df['accuracy'].std(),
        'f1_std': df['f1'].std(),
        'precision_std': df['precision'].std(),
        'recall_std': df['recall'].std(),
        'accuracy_cv': df['accuracy'].std() / df['accuracy'].mean() if df['accuracy'].mean() > 0 else float('inf'),
        'f1_cv': df['f1'].std() / df['f1'].mean() if df['f1'].mean() > 0 else float('inf'),
        'worst_performing_dataset': df.loc[df['f1'].idxmin(), 'dataset'],
        'best_performing_dataset': df.loc[df['f1'].idxmax(), 'dataset'],
        'performance_range': df['f1'].max() - df['f1'].min()
    }

    # Create consistency analysis plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Performance Consistency Analysis', fontsize=16)

    # Plot 1: Performance metrics distribution
    ax1 = axes[0, 0]
    metrics_to_plot = ['accuracy', 'f1', 'precision', 'recall']
    box_data = [df[metric] for metric in metrics_to_plot]
    box_plot = ax1.boxplot(box_data, labels=metrics_to_plot, patch_artist=True)

    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)

    ax1.set_ylabel('Score')
    ax1.set_title('Performance Metrics Distribution')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Performance by dataset size
    ax2 = axes[0, 1]
    scatter = ax2.scatter(df['num_files'], df['f1'], alpha=0.6, s=100)
    ax2.set_xlabel('Number of Files')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('Performance vs Dataset Size')
    ax2.grid(True, alpha=0.3)

    # Add trend line
    if len(df) > 2:
        z = np.polyfit(df['num_files'], df['f1'], 1)
        p = np.poly1d(z)
        ax2.plot(df['num_files'], p(df['num_files']), "r--", alpha=0.8)

    # Plot 3: Radar chart of average performance
    ax3 = axes[1, 0]
    categories = ['Accuracy', 'F1', 'Precision', 'Recall']
    values = [df['accuracy'].mean(), df['f1'].mean(), df['precision'].mean(), df['recall'].mean()]

    # Convert to polar coordinates
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    values += values[:1]  # Complete the circle
    angles += angles[:1]

    ax3 = plt.subplot(2, 2, 3, projection='polar')
    ax3.plot(angles, values, 'o-', linewidth=2)
    ax3.fill(angles, values, alpha=0.25)
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(categories)
    ax3.set_ylim(0, 1)
    ax3.set_title('Average Performance Profile', y=1.08)

    # Plot 4: Performance consistency heatmap
    ax4 = axes[1, 1]

    # Create a performance matrix (datasets vs metrics)
    performance_matrix = df.set_index('dataset')[['accuracy', 'f1', 'precision', 'recall']].T

    im = ax4.imshow(performance_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax4.set_xticks(range(len(performance_matrix.columns)))
    ax4.set_yticks(range(len(performance_matrix.index)))
    ax4.set_xticklabels(performance_matrix.columns, rotation=45)
    ax4.set_yticklabels(performance_matrix.index)
    ax4.set_title('Performance Heatmap by Dataset')

    plt.colorbar(im, ax=ax4)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_consistency_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

    return consistency_results


def analyze_dataset_difficulty(results, output_dir, config_threshold):
    """Analyze which datasets are more difficult for the model."""
    print("Analyzing dataset difficulty...")

    difficulty_data = []

    for dataset_name, dataset_results in results.items():
        if 'probabilities' not in dataset_results or 'true_labels' not in dataset_results:
            continue

        probs = np.array(dataset_results['probabilities'])
        labels = np.array(dataset_results['true_labels'])

        # Calculate difficulty metrics
        confidence = np.abs(probs - 0.5)  # Distance from decision boundary
        avg_confidence = np.mean(confidence)
        min_confidence = np.min(confidence)

        # Prediction entropy
        epsilon = 1e-8
        probs_safe = np.clip(probs, epsilon, 1 - epsilon)
        entropy = -np.mean(probs_safe * np.log2(probs_safe) + (1 - probs_safe) * np.log2(1 - probs_safe))

        # Class separation
        if len(np.unique(labels)) > 1:
            drone_probs = probs[labels == 1]
            non_drone_probs = probs[labels == 0]

            if len(drone_probs) > 0 and len(non_drone_probs) > 0:
                separation = np.abs(np.mean(drone_probs) - np.mean(non_drone_probs))
                overlap = calculate_distribution_overlap(drone_probs, non_drone_probs)
            else:
                separation = 0
                overlap = 1
        else:
            separation = 0
            overlap = 1

        # Performance metrics
        f1 = dataset_results.get('file_level_f1', 0)
        accuracy = dataset_results.get('file_level_accuracy', 0)

        difficulty_score = (1 - avg_confidence) + entropy + overlap - separation

        difficulty_data.append({
            'dataset': dataset_name,
            'avg_confidence': avg_confidence,
            'min_confidence': min_confidence,
            'entropy': entropy,
            'separation': separation,
            'overlap': overlap,
            'difficulty_score': difficulty_score,
            'f1': f1,
            'accuracy': accuracy,
            'num_samples': len(probs)
        })

    if not difficulty_data:
        return {'error': 'No valid data for difficulty analysis'}

    df = pd.DataFrame(difficulty_data)
    df = df.sort_values('difficulty_score', ascending=False)

    # Create difficulty analysis plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Dataset Difficulty Analysis', fontsize=16)

    # Plot 1: Difficulty ranking
    ax1 = axes[0, 0]
    bars = ax1.barh(range(len(df)), df['difficulty_score'],
                    color=plt.cm.RdYlGn_r(df['difficulty_score'] / df['difficulty_score'].max()))
    ax1.set_yticks(range(len(df)))
    ax1.set_yticklabels(df['dataset'])
    ax1.set_xlabel('Difficulty Score')
    ax1.set_title('Dataset Difficulty Ranking')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Difficulty vs Performance
    ax2 = axes[0, 1]
    scatter = ax2.scatter(df['difficulty_score'], df['f1'],
                         s=df['num_samples']/10, alpha=0.6, c=df['accuracy'], cmap='viridis')
    ax2.set_xlabel('Difficulty Score')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('Performance vs Difficulty')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax2, label='Accuracy')

    # Add trend line
    if len(df) > 2:
        z = np.polyfit(df['difficulty_score'], df['f1'], 1)
        p = np.poly1d(z)
        ax2.plot(df['difficulty_score'], p(df['difficulty_score']), "r--", alpha=0.8)

    # Plot 3: Difficulty components
    ax3 = axes[1, 0]
    components = ['avg_confidence', 'entropy', 'separation', 'overlap']
    component_data = df[components].values.T

    x = np.arange(len(df))
    width = 0.2

    for i, component in enumerate(components):
        ax3.bar(x + i*width, component_data[i], width, label=component, alpha=0.7)

    ax3.set_xlabel('Dataset Index')
    ax3.set_ylabel('Score')
    ax3.set_title('Difficulty Components by Dataset')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Easy vs Hard datasets comparison
    ax4 = axes[1, 1]

    # Split into easy and hard datasets
    median_difficulty = df['difficulty_score'].median()
    easy_datasets = df[df['difficulty_score'] <= median_difficulty]
    hard_datasets = df[df['difficulty_score'] > median_difficulty]

    categories = ['Easy Datasets', 'Hard Datasets']
    avg_f1s = [easy_datasets['f1'].mean(), hard_datasets['f1'].mean()]
    avg_accs = [easy_datasets['accuracy'].mean(), hard_datasets['accuracy'].mean()]

    x = np.arange(len(categories))
    width = 0.35

    ax4.bar(x - width/2, avg_f1s, width, label='F1 Score', alpha=0.7)
    ax4.bar(x + width/2, avg_accs, width, label='Accuracy', alpha=0.7)

    ax4.set_ylabel('Score')
    ax4.set_title('Easy vs Hard Datasets Performance')
    ax4.set_xticks(x)
    ax4.set_xticklabels(categories)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dataset_difficulty_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

    return {
        'difficulty_ranking': df[['dataset', 'difficulty_score', 'f1']].to_dict('records'),
        'easiest_dataset': df.iloc[-1]['dataset'],
        'hardest_dataset': df.iloc[0]['dataset'],
        'difficulty_performance_correlation': df['difficulty_score'].corr(df['f1'])
    }


def calculate_distribution_overlap(dist1, dist2):
    """Calculate overlap between two distributions."""
    if len(dist1) == 0 or len(dist2) == 0:
        return 1.0

    # Use kernel density estimation
    try:
        kde1 = stats.gaussian_kde(dist1)
        kde2 = stats.gaussian_kde(dist2)

        # Calculate overlap over a range
        min_val = min(np.min(dist1), np.min(dist2))
        max_val = max(np.max(dist1), np.max(dist2))
        x = np.linspace(min_val, max_val, 100)

        overlap = np.trapz(np.minimum(kde1(x), kde2(x)), x)
        return overlap
    except:
        # Fallback to simple range overlap
        range1 = (np.min(dist1), np.max(dist1))
        range2 = (np.min(dist2), np.max(dist2))

        overlap_start = max(range1[0], range2[0])
        overlap_end = min(range1[1], range2[1])

        if overlap_start < overlap_end:
            overlap_length = overlap_end - overlap_start
            total_range = max(range1[1], range2[1]) - min(range1[0], range2[0])
            return overlap_length / total_range if total_range > 0 else 0
        else:
            return 0


def analyze_error_patterns(results, output_dir, config_threshold):
    """Analyze patterns in model errors."""
    print("Analyzing error patterns...")

    error_data = []

    for dataset_name, dataset_results in results.items():
        if 'probabilities' not in dataset_results or 'true_labels' not in dataset_results:
            continue

        probs = np.array(dataset_results['probabilities'])
        labels = np.array(dataset_results['true_labels'])
        preds = (probs > config_threshold).astype(int)  # Use config_threshold instead of 0.5

        # Identify error types
        false_positives = (preds == 1) & (labels == 0)
        false_negatives = (preds == 0) & (labels == 1)

        fp_probs = probs[false_positives] if np.any(false_positives) else np.array([])
        fn_probs = probs[false_negatives] if np.any(false_negatives) else np.array([])

        error_data.append({
            'dataset': dataset_name,
            'total_samples': len(labels),
            'false_positives': np.sum(false_positives),
            'false_negatives': np.sum(false_negatives),
            'fp_rate': np.sum(false_positives) / len(labels),
            'fn_rate': np.sum(false_negatives) / len(labels),
            'fp_avg_confidence': np.mean(fp_probs) if len(fp_probs) > 0 else 0,
            'fn_avg_confidence': np.mean(fn_probs) if len(fn_probs) > 0 else 0,
            'fp_probs': fp_probs,
            'fn_probs': fn_probs
        })

    if not error_data:
        return {'error': 'No valid data for error analysis'}

    # Create error pattern analysis plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Error Pattern Analysis', fontsize=16)

    df = pd.DataFrame([{k: v for k, v in item.items() if k not in ['fp_probs', 'fn_probs']}
                      for item in error_data])

    # Plot 1: Error rates by dataset
    ax1 = axes[0, 0]
    x = np.arange(len(df))
    width = 0.35

    ax1.bar(x - width/2, df['fp_rate'], width, label='False Positive Rate', alpha=0.7, color='red')
    ax1.bar(x + width/2, df['fn_rate'], width, label='False Negative Rate', alpha=0.7, color='blue')

    ax1.set_xlabel('Dataset')
    ax1.set_ylabel('Error Rate')
    ax1.set_title('Error Rates by Dataset')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['dataset'], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Error confidence distribution
    ax2 = axes[0, 1]

    all_fp_probs = np.concatenate([item['fp_probs'] for item in error_data if len(item['fp_probs']) > 0])
    all_fn_probs = np.concatenate([item['fn_probs'] for item in error_data if len(item['fn_probs']) > 0])

    if len(all_fp_probs) > 0:
        ax2.hist(all_fp_probs, bins=30, alpha=0.7, label='False Positives', color='red', density=True)
    if len(all_fn_probs) > 0:
        ax2.hist(all_fn_probs, bins=30, alpha=0.7, label='False Negatives', color='blue', density=True)

    ax2.axvline(x=config_threshold, color='black', linestyle='--', alpha=0.7, label=f'Decision Threshold ({config_threshold:.3f})')  # Use config_threshold
    ax2.set_xlabel('Model Confidence')
    ax2.set_ylabel('Density')
    ax2.set_title('Error Confidence Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Error count vs dataset size
    ax3 = axes[1, 0]
    scatter1 = ax3.scatter(df['total_samples'], df['false_positives'],
                          alpha=0.6, color='red', s=100, label='False Positives')
    scatter2 = ax3.scatter(df['total_samples'], df['false_negatives'],
                          alpha=0.6, color='blue', s=100, label='False Negatives')

    ax3.set_xlabel('Total Samples')
    ax3.set_ylabel('Error Count')
    ax3.set_title('Error Count vs Dataset Size')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Error type preference by dataset
    ax4 = axes[1, 1]

    # Calculate error bias (FP vs FN preference)
    df['error_bias'] = (df['fp_rate'] - df['fn_rate']) / (df['fp_rate'] + df['fn_rate'] + 1e-8)

    colors = ['red' if bias > 0 else 'blue' for bias in df['error_bias']]
    bars = ax4.bar(range(len(df)), np.abs(df['error_bias']), color=colors, alpha=0.7)

    ax4.set_xlabel('Dataset')
    ax4.set_ylabel('Error Bias (|FP - FN| / Total Errors)')
    ax4.set_title('Error Type Bias by Dataset')
    ax4.set_xticks(range(len(df)))
    ax4.set_xticklabels(df['dataset'], rotation=45)
    ax4.grid(True, alpha=0.3)

    # Add legend for colors
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', alpha=0.7, label='FP Bias'),
                      Patch(facecolor='blue', alpha=0.7, label='FN Bias')]
    ax4.legend(handles=legend_elements)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_pattern_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

    return {
        'total_false_positives': df['false_positives'].sum(),
        'total_false_negatives': df['false_negatives'].sum(),
        'avg_fp_rate': df['fp_rate'].mean(),
        'avg_fn_rate': df['fn_rate'].mean(),
        'fp_confidence_avg': np.mean(all_fp_probs) if len(all_fp_probs) > 0 else 0,
        'fn_confidence_avg': np.mean(all_fn_probs) if len(all_fn_probs) > 0 else 0,
        'error_bias_analysis': df[['dataset', 'error_bias']].to_dict('records')
    }


def analyze_logits_by_dataset(results, output_dir, config_threshold):
    """Analyze model logits and predictions by dataset to identify patterns."""
    print("Analyzing logits by dataset...")

    logit_data = []

    for dataset_name, dataset_results in results.items():
        if 'probabilities' not in dataset_results or 'true_labels' not in dataset_results:
            continue

        probs = np.array(dataset_results['probabilities'])
        labels = np.array(dataset_results['true_labels'])

        # Enhanced logit conversion with better error handling
        epsilon = 1e-7  # Small value to avoid log(0)
        probs_clipped = np.clip(probs, epsilon, 1 - epsilon)

        # Handle edge cases where probabilities are exactly 0 or 1
        try:
            logits = np.log(probs_clipped / (1 - probs_clipped))
            # Check for any remaining invalid values
            if np.any(~np.isfinite(logits)):
                print(f"Warning: Found invalid logit values in dataset {dataset_name}, replacing with safe values")
                logits = np.where(~np.isfinite(logits), 0, logits)
        except Exception as e:
            print(f"Warning: Logit conversion failed for dataset {dataset_name}: {e}")
            logits = np.zeros_like(probs)  # Fallback to zeros

        # Calculate logit statistics with error handling
        logit_mean = np.mean(logits) if len(logits) > 0 else 0
        logit_std = np.std(logits) if len(logits) > 1 else 0

        # Separate by true class
        drone_logits = logits[labels == 1]
        non_drone_logits = logits[labels == 0]

        drone_logit_mean = np.mean(drone_logits) if len(drone_logits) > 0 else 0
        non_drone_logit_mean = np.mean(non_drone_logits) if len(non_drone_logits) > 0 else 0

        # Calculate threshold in logit space with error handling
        try:
            if config_threshold <= 0 or config_threshold >= 1:
                print(f"Warning: Invalid threshold {config_threshold}, using default 0.006")
                config_threshold = 0.006
            threshold_logit = np.log(config_threshold / (1 - config_threshold))
        except Exception as e:
            print(f"Warning: Threshold logit conversion failed: {e}")
            threshold_logit = np.log(0.006 / (1 - 0.006))  # Fallback to default

        # Predictions using config threshold
        preds = (probs >= config_threshold).astype(int)
        correct_preds = (preds == labels).sum()
        accuracy = correct_preds / len(labels) if len(labels) > 0 else 0

        logit_data.append({
            'dataset': dataset_name,
            'logit_mean': logit_mean,
            'logit_std': logit_std,
            'drone_logit_mean': drone_logit_mean,
            'non_drone_logit_mean': non_drone_logit_mean,
            'logit_separation': abs(drone_logit_mean - non_drone_logit_mean),
            'threshold_logit': threshold_logit,
            'accuracy': accuracy,
            'num_samples': len(logits),
            'num_drone_samples': len(drone_logits),
            'num_non_drone_samples': len(non_drone_logits),
            'logits': logits,
            'drone_logits': drone_logits,
            'non_drone_logits': non_drone_logits,
            'labels': labels
        })

    if not logit_data:
        return {'error': 'No valid data for logit analysis'}

    # Create logit analysis plots with better error handling
    try:
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Logit Analysis by Dataset', fontsize=16)

        df = pd.DataFrame([{k: v for k, v in item.items()
                           if k not in ['logits', 'drone_logits', 'non_drone_logits', 'labels']}
                          for item in logit_data])

        # Plot 1: Logit distribution by dataset (box plot) with better dataset naming
        ax1 = axes[0, 0]
        all_logits_by_dataset = [item['logits'] for item in logit_data]
        dataset_names = [item['dataset'] for item in logit_data]

        # Improved dataset name handling
        short_names = []
        for i, name in enumerate(dataset_names):
            if len(name) > 12:
                short_name = name[:12] + "..."
            else:
                short_name = name
            short_names.append(f"{i}\n{short_name}")

        if len(all_logits_by_dataset) > 0:
            box_plot = ax1.boxplot(all_logits_by_dataset, labels=range(len(dataset_names)), patch_artist=True)
            ax1.set_xlabel('Dataset Index')
            ax1.set_ylabel('Logit Values')
            ax1.set_title('Logit Distribution by Dataset')
            ax1.grid(True, alpha=0.3)
            ax1.set_xticklabels(short_names, rotation=45, fontsize=8)

            # Color boxes by accuracy with safety check
            if len(df['accuracy']) > 0 and not df['accuracy'].isna().all():
                colors = plt.cm.RdYlGn(df['accuracy'])
                for patch, color in zip(box_plot['boxes'], colors):
                    patch.set_facecolor(color)

        # Plot 2: Logit separation by class
        ax2 = axes[0, 1]
        x = np.arange(len(df))
        width = 0.35

        bars1 = ax2.bar(x - width/2, df['drone_logit_mean'], width,
                       label='Drone Samples', alpha=0.7, color='red')
        bars2 = ax2.bar(x + width/2, df['non_drone_logit_mean'], width,
                       label='Non-Drone Samples', alpha=0.7, color='blue')

        # Add threshold line
        ax2.axhline(y=df['threshold_logit'].iloc[0], color='black', linestyle='--',
                    alpha=0.7, label=f'Threshold (logit={df["threshold_logit"].iloc[0]:.2f})')

        ax2.set_xlabel('Dataset Index')
        ax2.set_ylabel('Mean Logit Value')
        ax2.set_title('Mean Logit by Class and Dataset')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Logit separation vs accuracy
        ax3 = axes[0, 2]
        scatter = ax3.scatter(df['logit_separation'], df['accuracy'],
                             s=df['num_samples']/10, alpha=0.6, c=df['logit_std'], cmap='viridis')
        ax3.set_xlabel('Logit Separation (|Drone - Non-Drone|)')
        ax3.set_ylabel('Accuracy')
        ax3.set_title('Accuracy vs Logit Separation')
        ax3.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax3, label='Logit Std Dev')

        # Add trend line
        if len(df) > 2:
            z = np.polyfit(df['logit_separation'], df['accuracy'], 1)
            p = np.poly1d(z)
            ax3.plot(df['logit_separation'], p(df['logit_separation']), "r--", alpha=0.8)

        # Plot 4: Logit histograms for selected datasets (top 3 by separation)
        ax4 = axes[1, 0]
        top_datasets = df.nlargest(3, 'logit_separation')

        colors = ['red', 'blue', 'green']
        for i, (_, row) in enumerate(top_datasets.iterrows()):
            dataset_idx = df[df['dataset'] == row['dataset']].index[0]
            logits = logit_data[dataset_idx]['logits']
            ax4.hist(logits, bins=30, alpha=0.6, label=f"{row['dataset'][:15]}...",
                    color=colors[i], density=True)

        ax4.axvline(x=df['threshold_logit'].iloc[0], color='black', linestyle='--',
                    alpha=0.7, label='Threshold')
        ax4.set_xlabel('Logit Value')
        ax4.set_ylabel('Density')
        ax4.set_title('Logit Distributions (Top 3 Separation)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # Plot 5: Class-specific logit distributions
        ax5 = axes[1, 1]
        all_drone_logits = np.concatenate([item['drone_logits'] for item in logit_data
                                          if len(item['drone_logits']) > 0])
        all_non_drone_logits = np.concatenate([item['non_drone_logits'] for item in logit_data
                                              if len(item['non_drone_logits']) > 0])

        if len(all_drone_logits) > 0:
            ax5.hist(all_drone_logits, bins=50, alpha=0.6, label='Drone', color='red', density=True)
        if len(all_non_drone_logits) > 0:
            ax5.hist(all_non_drone_logits, bins=50, alpha=0.6, label='Non-Drone', color='blue', density=True)

        ax5.axvline(x=df['threshold_logit'].iloc[0], color='black', linestyle='--',
                    alpha=0.7, label='Threshold')
        ax5.set_xlabel('Logit Value')
        ax5.set_ylabel('Density')
        ax5.set_title('Overall Logit Distribution by Class')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # Plot 6: Dataset performance metrics
        ax6 = axes[1, 2]
        metrics = ['accuracy', 'logit_separation', 'logit_std']
        metric_data = df[metrics].values.T

        x = np.arange(len(df))
        width = 0.25

        for i, metric in enumerate(metrics):
            # Normalize metrics to 0-1 scale for comparison
            normalized_data = (metric_data[i] - metric_data[i].min()) / (metric_data[i].max() - metric_data[i].min() + 1e-8)
            ax6.bar(x + i*width, normalized_data, width, label=metric, alpha=0.7)

        ax6.set_xlabel('Dataset Index')
        ax6.set_ylabel('Normalized Score')
        ax6.set_title('Dataset Performance Metrics (Normalized)')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'logit_analysis_by_dataset.png'), dpi=300, bbox_inches='tight')
        plt.close()

        return {
            'dataset_logit_stats': df[['dataset', 'logit_mean', 'logit_std', 'logit_separation', 'accuracy']].to_dict('records'),
            'overall_drone_logit_mean': np.mean(all_drone_logits) if len(all_drone_logits) > 0 else 0,
            'overall_non_drone_logit_mean': np.mean(all_non_drone_logits) if len(all_non_drone_logits) > 0 else 0,
            'threshold_logit': df['threshold_logit'].iloc[0],
            'separation_accuracy_correlation': df['logit_separation'].corr(df['accuracy'])
        }
    except Exception as e:
        print(f"Error during logit analysis plotting: {e}")
        return {'error': 'Logit analysis plotting failed due to an error'}


def generate_executive_summary(results, optimal_thresholds, imbalance_analysis,
                             consistency_analysis, difficulty_analysis, error_patterns, output_dir, config_threshold):
    """Generate an executive summary of the comprehensive analysis."""

    with open(os.path.join(output_dir, 'executive_summary.md'), 'w', encoding='utf-8') as f:
        f.write("# Model Evaluation Executive Summary\n\n")

        # Overall performance
        f.write("## Overall Performance\n\n")
        total_datasets = len(results)
        f.write(f"- **Total Datasets Evaluated**: {total_datasets}\n")

        if total_datasets > 0:
            avg_f1 = np.mean([r.get('file_level_f1', 0) for r in results.values()])
            avg_accuracy = np.mean([r.get('file_level_accuracy', 0) for r in results.values()])
            f.write(f"- **Average F1 Score**: {avg_f1:.3f}\n")
            f.write(f"- **Average Accuracy**: {avg_accuracy:.3f}\n")

        # Threshold recommendations
        f.write("\n## Threshold Optimization\n\n")
        if 'error' not in optimal_thresholds:
            f.write(f"- **Recommended F1-optimal threshold**: {optimal_thresholds['f1']:.3f}\n")
            f.write(f"- **Recommended balanced accuracy threshold**: {optimal_thresholds['balanced_accuracy']:.3f}\n")
            f.write(f"- **Current default threshold**: {config_threshold:.3f}\n")

            if abs(optimal_thresholds['f1'] - config_threshold) > 0.05:
                f.write(f"- **⚠️ RECOMMENDATION**: Consider changing threshold from {config_threshold:.3f} to {optimal_thresholds['f1']:.3f}\n")

        # Class imbalance insights
        f.write("\n## Class Imbalance Analysis\n\n")
        if 'error' not in imbalance_analysis:
            f.write(f"- **Balanced datasets**: {imbalance_analysis['balanced_datasets']}\n")
            f.write(f"- **Moderately imbalanced datasets**: {imbalance_analysis['moderate_imbalance_datasets']}\n")
            f.write(f"- **Severely imbalanced datasets**: {imbalance_analysis['severe_imbalance_datasets']}\n")

        # Performance consistency
        f.write("\n## Performance Consistency\n\n")
        if 'error' not in consistency_analysis:
            f.write(f"- **F1 Score Standard Deviation**: {consistency_analysis['f1_std']:.3f}\n")
            f.write(f"- **F1 Score Coefficient of Variation**: {consistency_analysis['f1_cv']:.3f}\n")
            f.write(f"- **Best Performing Dataset**: {consistency_analysis['best_performing_dataset']}\n")
            f.write(f"- **Worst Performing Dataset**: {consistency_analysis['worst_performing_dataset']}\n")
            f.write(f"- **Performance Range**: {consistency_analysis['performance_range']:.3f}\n")

        # Dataset difficulty
        f.write("\n## Dataset Difficulty Assessment\n\n")
        if 'error' not in difficulty_analysis:
            f.write(f"- **Hardest Dataset**: {difficulty_analysis['hardest_dataset']}\n")
            f.write(f"- **Easiest Dataset**: {difficulty_analysis['easiest_dataset']}\n")
            f.write(f"- **Difficulty-Performance Correlation**: {difficulty_analysis['difficulty_performance_correlation']:.3f}\n")

        # Error patterns
        f.write("\n## Error Pattern Analysis\n\n")
        if 'error' not in error_patterns:
            f.write(f"- **Total False Positives**: {error_patterns['total_false_positives']}\n")
            f.write(f"- **Total False Negatives**: {error_patterns['total_false_negatives']}\n")
            f.write(f"- **Average False Positive Rate**: {error_patterns['avg_fp_rate']:.3f}\n")
            f.write(f"- **Average False Negative Rate**: {error_patterns['avg_fn_rate']:.3f}\n")

        # Recommendations
        f.write("\n## Key Recommendations\n\n")
        f.write("1. **Threshold Optimization**: ")
        if 'error' not in optimal_thresholds and abs(optimal_thresholds['f1'] - config_threshold) > 0.05:
            f.write(f"Adjust threshold to {optimal_thresholds['f1']:.3f} for better F1 performance\n")
        else:
            f.write("Current threshold appears optimal\n")

        f.write("2. **Class Imbalance**: ")
        if 'error' not in imbalance_analysis and imbalance_analysis['severe_imbalance_datasets'] > 0:
            f.write("Consider data augmentation or resampling for severely imbalanced datasets\n")
        else:
            f.write("Class balance appears manageable\n")

        f.write("3. **Performance Consistency**: ")
        if 'error' not in consistency_analysis and consistency_analysis['f1_cv'] > 0.3:
            f.write("High performance variance detected - investigate dataset-specific issues\n")
        else:
            f.write("Performance is reasonably consistent across datasets\n")

        f.write("4. **Error Patterns**: ")
        if 'error' not in error_patterns:
            if error_patterns['avg_fp_rate'] > error_patterns['avg_fn_rate']:
                f.write("Model tends toward false positives - consider raising threshold\n")
            elif error_patterns['avg_fn_rate'] > error_patterns['avg_fp_rate']:
                f.write("Model tends toward false negatives - consider lowering threshold\n")
            else:
                f.write("Error patterns appear balanced\n")

    print(f"Executive summary saved to: {os.path.join(output_dir, 'executive_summary.md')}")

