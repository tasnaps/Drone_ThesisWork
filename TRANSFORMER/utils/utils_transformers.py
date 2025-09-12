import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import json
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def plot_cv_results(fold_results):
    """
    Plot cross-validation results including metrics over folds and training history.

    Args:
        fold_results: List of dictionaries containing fold results
    """
    if not fold_results or len(fold_results) == 0:
        print("No fold results to plot")
        return

    # Create plots directory if it doesn't exist
    Path("./plots").mkdir(exist_ok=True)

    # Plot metrics across folds
    metrics = ['f1', 'accuracy', 'precision', 'recall']
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        values = []
        for fold in fold_results:
            # Try different possible key formats
            value = fold.get(metric, fold.get(f'eval_{metric}', fold.get(f'test_{metric}', None)))
            if value is not None:
                values.append(value)

        if values:
            axes[i].bar(range(1, len(values) + 1), values)
            axes[i].set_title(f'{metric.capitalize()} across folds')
            axes[i].set_xlabel('Fold')
            axes[i].set_ylabel(metric.capitalize())
            axes[i].set_ylim(0, 1)

            # Add mean line
            mean_val = np.mean(values)
            axes[i].axhline(y=mean_val, color='r', linestyle='--',
                           label=f'Mean: {mean_val:.3f}')
            axes[i].legend()

    plt.tight_layout()
    plt.savefig("./plots/cv_metrics_summary.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Plot training history if available
    if 'history' in fold_results[0]:
        plot_cv_training_history(fold_results)


def plot_cv_training_history(fold_results):
    """Plot training history across all CV folds."""
    metrics_to_plot = ['eval_loss', 'eval_f1', 'eval_accuracy', 'eval_precision']

    for metric in metrics_to_plot:
        plt.figure(figsize=(10, 6))

        for fold_idx, fold in enumerate(fold_results):
            if 'history' not in fold:
                continue

            history = fold['history']
            epochs = []
            values = []

            for log_entry in history:
                if metric in log_entry and 'epoch' in log_entry:
                    epochs.append(log_entry['epoch'])
                    values.append(log_entry[metric])

            if epochs and values:
                plt.plot(epochs, values, label=f'Fold {fold_idx + 1}', alpha=0.7)

        plt.xlabel('Epoch')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(f'{metric.replace("_", " ").title()} Across CV Folds')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"./plots/cv_{metric}_history.png", dpi=300, bbox_inches='tight')
        plt.show()


def generate_detailed_report(trainer, test_dataset):
    """
    Generate detailed classification report and confusion matrix.

    Args:
        trainer: The trained model trainer
        test_dataset: Test dataset for evaluation

    Returns:
        tuple: (detailed_report_dict, confusion_matrix)
    """
    # Get predictions
    predictions = trainer.predict(test_dataset)
    y_true = predictions.label_ids

    # Convert logits to probabilities and then to predictions
    probs = torch.softmax(torch.tensor(predictions.predictions), dim=-1).cpu().numpy()
    y_pred = np.argmax(probs, axis=1)

    # Calculate metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    accuracy = accuracy_score(y_true, y_pred)

    # Create detailed report
    class_names = ['Other', 'Drone']  # Assuming binary classification
    detailed_report = {
        'accuracy': float(accuracy),
        'macro_avg': {
            'precision': float(np.mean(precision)),
            'recall': float(np.mean(recall)),
            'f1-score': float(np.mean(f1)),
            'support': int(np.sum(support))
        },
        'weighted_avg': {
            'precision': float(np.average(precision, weights=support)),
            'recall': float(np.average(recall, weights=support)),
            'f1-score': float(np.average(f1, weights=support)),
            'support': int(np.sum(support))
        }
    }

    # Add per-class metrics
    for i, class_name in enumerate(class_names):
        detailed_report[class_name] = {
            'precision': float(precision[i]) if i < len(precision) else 0.0,
            'recall': float(recall[i]) if i < len(recall) else 0.0,
            'f1-score': float(f1[i]) if i < len(f1) else 0.0,
            'support': int(support[i]) if i < len(support) else 0
        }

    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot and save confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig("./plots/confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Save classification report
    sklearn_report = classification_report(y_true, y_pred, target_names=class_names)
    with open("./reports/classification_report.txt", 'w') as f:
        f.write(sklearn_report)

    print("Classification Report:")
    print(sklearn_report)

    return detailed_report, cm


def plot_training_history(trainer):
    """
    Plot training history from trainer logs.

    Args:
        trainer: The trained model trainer
    """
    if not hasattr(trainer, 'state') or not trainer.state.log_history:
        print("No training history available")
        return

    # Extract training and validation metrics
    train_logs = []
    eval_logs = []

    for log in trainer.state.log_history:
        if 'loss' in log and 'epoch' in log:
            train_logs.append(log)
        elif any(key.startswith('eval_') for key in log.keys()) and 'epoch' in log:
            eval_logs.append(log)

    if not train_logs and not eval_logs:
        print("No training history found in trainer logs")
        return

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot training loss
    if train_logs:
        epochs = [log['epoch'] for log in train_logs if 'loss' in log]
        losses = [log['loss'] for log in train_logs if 'loss' in log]

        if epochs and losses:
            axes[0, 0].plot(epochs, losses, 'b-', label='Training Loss')
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

    # Plot validation metrics
    metrics_to_plot = [
        ('eval_loss', 'Validation Loss', (0, 1)),
        ('eval_f1', 'Validation F1-Score', (1, 0)),
        ('eval_accuracy', 'Validation Accuracy', (1, 1))
    ]

    for metric, title, pos in metrics_to_plot:
        if eval_logs:
            epochs = [log['epoch'] for log in eval_logs if metric in log]
            values = [log[metric] for log in eval_logs if metric in log]

            if epochs and values:
                axes[pos].plot(epochs, values, 'r-', label=title)
                axes[pos].set_title(title)
                axes[pos].set_xlabel('Epoch')
                axes[pos].set_ylabel(metric.replace('eval_', '').replace('_', ' ').title())
                axes[pos].legend()
                axes[pos].grid(True, alpha=0.3)

                if metric != 'eval_loss':  # Set y-axis limit for metrics (not loss)
                    axes[pos].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig("./plots/training_history.png", dpi=300, bbox_inches='tight')
    plt.show()


def save_model_and_results(trainer, results):
    """
    Save the trained model and results.

    Args:
        trainer: The trained model trainer
        results: Dictionary containing all results
    """
    # Create directories
    Path("./models").mkdir(exist_ok=True)
    Path("./reports").mkdir(exist_ok=True)

    # Save the model
    model_path = "./models/final_transformer_model"
    trainer.save_model(model_path)
    print(f"Model saved to {model_path}")

    # Save results as JSON
    results_path = "./reports/final_results.json"

    # Convert numpy types to Python native types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj

    serializable_results = convert_to_serializable(results)

    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print(f"Results saved to {results_path}")

    # Save a summary report
    summary_path = "./reports/summary_report.txt"
    with open(summary_path, 'w') as f:
        f.write("=== Model Performance Summary ===\n\n")

        if 'validation' in results:
            f.write("Validation Results:\n")
            for key, value in results['validation'].items():
                if isinstance(value, (int, float)):
                    f.write(f"  {key}: {value:.4f}\n")
            f.write("\n")

        if 'test' in results:
            f.write("Test Results:\n")
            for key, value in results['test'].items():
                if isinstance(value, (int, float)):
                    f.write(f"  {key}: {value:.4f}\n")
            f.write("\n")

        if 'detailed_report' in results:
            f.write("Detailed Classification Metrics:\n")
            dr = results['detailed_report']
            f.write(f"  Overall Accuracy: {dr.get('accuracy', 'N/A'):.4f}\n")

            if 'macro_avg' in dr:
                f.write("  Macro Average:\n")
                for metric, value in dr['macro_avg'].items():
                    if isinstance(value, (int, float)):
                        f.write(f"    {metric}: {value:.4f}\n")

    print(f"Summary report saved to {summary_path}")
