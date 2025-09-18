import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

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