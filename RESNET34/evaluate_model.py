"""
Evaluation script for trained ResNet-34 audio classification model.
Uses 1-second sliding windows to match training scale, then aggregates with chosen metric. In thesis we used noisy-OR.
"""

import os
import json
import pandas as pd
from datetime import datetime
from collections import Counter
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
from config import DATASETS, SAMPLE_RATE
import librosa
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)

from resnet_34 import (
    ResNetForAudioClassification,
    get_label_mappings,
    preprocess_audio_to_resnet_input,
    device
)

LABEL2ID = {"unknown": 0, "yes_drone": 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

@dataclass
class EvaluationConfig:
    """Configuration for ResNet-34 evaluation."""
    model_path: str
    data_dir: str
    output_dir: str
    batch_size: int = 64
    window_duration: float = 1.0  # Fixed 1-second windows to match training
    hop_duration: float = 0.5    # 50% overlap
    threshold: float = 0.5
    sample_rate: int = SAMPLE_RATE
    seed: int = 42

class WindowProcessor:
    """Processes audio files using 1-second sliding windows to match training scale."""

    def __init__(self, config: EvaluationConfig):
        self.config = config

    def load_audio_file(self, file_path: str) -> Tuple[Optional[np.ndarray], float]:
        """Load audio file and return waveform and duration."""
        try:
            waveform, sr = librosa.load(file_path, sr=self.config.sample_rate)
            duration = len(waveform) / sr
            return waveform, duration
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None, 0.0

    def extract_windows(self, waveform: np.ndarray, duration: float) -> List[np.ndarray]:
        """Extract 1-second windows with specified hop size."""
        window_samples = int(self.config.window_duration * self.config.sample_rate)
        hop_samples = int(self.config.hop_duration * self.config.sample_rate)

        windows = []
        start = 0

        while start + window_samples <= len(waveform):
            window = waveform[start:start + window_samples]
            windows.append(window)
            start += hop_samples

        # Handle remaining audio if significant (>50% of window)
        if len(waveform) - start > window_samples * 0.5:
            remaining = waveform[start:]
            # Pad to full window length
            if len(remaining) < window_samples:
                pad_length = window_samples - len(remaining)
                remaining = np.concatenate([remaining, np.zeros(pad_length)])
            windows.append(remaining[:window_samples])

        return windows

    def aggregate_predictions(self, predictions: List[float], method: str = "noisy_or") -> float:
        """Aggregate predictions from multiple windows."""
        if not predictions:
            return 0.0

        if method == "mean":
            return float(np.mean(predictions))
        elif method == "max":
            return float(np.max(predictions))
        elif method == "median":
            return float(np.median(predictions))
        elif method == "noisy_or":
            # Noisy-OR: P(file) = 1 - ∏(1 - p_window)
            # Clip probabilities to avoid numerical issues
            predictions_clipped = [max(1e-6, min(1-1e-6, p)) for p in predictions]
            log_prob_not_drone = np.sum([np.log1p(-p) for p in predictions_clipped])
            return float(1.0 - np.exp(log_prob_not_drone))
        elif method == "top_k_noisy_or":
            # Top-k Noisy-OR (k=3) to reduce single hot window domination
            k = min(3, len(predictions))
            top_k_preds = sorted(predictions, reverse=True)[:k]
            return self.aggregate_predictions(top_k_preds, "noisy_or")
        else:
            return float(np.mean(predictions))

def get_label_for_file(file_path: str, label_override: Optional[str]) -> str:
    """Get the appropriate label for a file."""
    if label_override:
        return label_override

    # Extract from folder structure
    parent = Path(file_path).parent.name.lower()

    # Direct mapping
    if parent in LABEL2ID:
        return parent

    # Pattern matching
    drone_patterns = ['drone', 'yes_drone', 'positive', 'uav', 'quadcopter', 'target', '1']
    non_drone_patterns = ['unknown', 'non_drone', 'negative', 'background', 'noise', '0']

    for pattern in drone_patterns:
        if pattern in parent:
            return 'yes_drone'

    for pattern in non_drone_patterns:
        if pattern in parent:
            return 'unknown'

    print(f"Warning: Could not determine label for '{parent}', defaulting to 'unknown'")
    return 'unknown'

def load_model(model_path: str) -> ResNetForAudioClassification:
    """Load trained model from checkpoint."""
    print(f"Loading model from {model_path}")

    # Initialize model architecture
    label2id, _ = get_label_mappings()
    model = ResNetForAudioClassification(num_labels=len(label2id), freeze_blocks=True)

    # Try different checkpoint formats
    checkpoint_files = [
        "pytorch_model.bin",
        "model.safetensors",
        "best_model.pth",
        "final_model.pth"
    ]

    model_loaded = False
    for checkpoint_file in checkpoint_files:
        checkpoint_path = os.path.join(model_path, checkpoint_file)
        if os.path.exists(checkpoint_path):
            try:
                if checkpoint_file.endswith('.bin') or checkpoint_file.endswith('.pth'):
                    state_dict = torch.load(checkpoint_path, map_location=device)
                    if 'state_dict' in state_dict:
                        state_dict = state_dict['state_dict']
                    model.load_state_dict(state_dict)
                    print(f"✓ Loaded model from {checkpoint_file}")
                    model_loaded = True
                    break
                elif checkpoint_file.endswith('.safetensors'):
                    try:
                        from safetensors.torch import load_file
                        state_dict = load_file(checkpoint_path, device=str(device))
                        model.load_state_dict(state_dict)
                        print(f"✓ Loaded model from {checkpoint_file}")
                        model_loaded = True
                        break
                    except ImportError:
                        print(f"safetensors not available, skipping {checkpoint_file}")
                        continue
            except Exception as e:
                print(f"Failed to load {checkpoint_file}: {e}")
                continue

    if not model_loaded:
        raise FileNotFoundError(f"No valid model checkpoint found in {model_path}")

    model.to(device)
    model.eval()
    return model

def calibrate_threshold(model: ResNetForAudioClassification, config: EvaluationConfig, dataset_config: Dict) -> float:
    """
    Calibration wasn't in use in the reported resutls, as the csv's were analyzed in separate program.
    """
    print("\n=== Threshold Calibration ===")

    processor = WindowProcessor(config)
    all_probs = []
    all_labels = []

    # Use a subset for calibration to be faster
    file_count = 0
    max_calibration_files = 200

    for root, dirs, files in os.walk(config.data_dir):
        if file_count >= max_calibration_files:
            break

        for file in files:
            if file_count >= max_calibration_files:
                break

            if file.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
                file_path = os.path.join(root, file)

                # Get label
                label_string = get_label_for_file(file_path, dataset_config.get('label_override'))
                true_label = LABEL2ID.get(label_string, 0)

                # Process file
                waveform, duration = processor.load_audio_file(file_path)
                if waveform is None:
                    continue

                windows = processor.extract_windows(waveform, duration)
                if not windows:
                    continue

                # Get predictions for all windows
                window_predictions = []
                for i in range(0, len(windows), config.batch_size):
                    batch_windows = windows[i:i + config.batch_size]

                    # Preprocess batch
                    batch_tensors = []
                    for window in batch_windows:
                        img = preprocess_audio_to_resnet_input(window)
                        batch_tensors.append(img)

                    if batch_tensors:
                        batch_tensor = torch.stack(batch_tensors).to(device)

                        with torch.no_grad():
                            outputs = model(batch_tensor)
                            logits = outputs.logits
                            probabilities = F.softmax(logits, dim=-1)
                            drone_probs = probabilities[:, 1].cpu().numpy()
                            window_predictions.extend(drone_probs.tolist())

                # Aggregate using Noisy-OR
                if window_predictions:
                    file_prob = processor.aggregate_predictions(window_predictions, "noisy_or")
                    all_probs.append(file_prob)
                    all_labels.append(true_label)
                    file_count += 1

    if len(all_probs) < 10:
        print("Warning: Too few files for proper calibration, using default threshold 0.5")
        return 0.5

    # Check if we have both classes for calibration
    unique_labels = set(all_labels)
    if len(unique_labels) < 2:
        print(f"Warning: Only one class found in calibration data ({unique_labels}), using default threshold 0.5")
        return 0.5

    # Find best threshold by F1 score
    try:
        precisions, recalls, thresholds = precision_recall_curve(all_labels, all_probs)
        f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
        best_idx = np.nanargmax(f1_scores)
        best_threshold = thresholds[best_idx] if len(thresholds) > 0 else 0.5

        # Sanity check the threshold
        if best_threshold <= 0.0 or best_threshold >= 1.0 or np.isnan(best_threshold):
            print(f"Warning: Invalid threshold {best_threshold}, using default 0.5")
            return 0.5

        print(f"Calibration used {len(all_probs)} files")
        print(f"Best threshold: {best_threshold:.4f} (F1: {f1_scores[best_idx]:.4f})")

        # Save threshold for future use
        threshold_file = os.path.join(config.model_path, "calibrated_threshold.json")
        threshold_data = {
            "threshold": float(best_threshold),
            "f1_score": float(f1_scores[best_idx]),
            "calibration_files": len(all_probs),
            "calibration_date": datetime.now().isoformat()
        }
        with open(threshold_file, 'w') as f:
            json.dump(threshold_data, f, indent=2)
        print(f"Threshold saved to: {threshold_file}")

        return float(best_threshold)

    except Exception as e:
        print(f"Warning: Threshold calibration failed ({e}), using default threshold 0.5")
        return 0.5

def evaluate_model_with_windows(model: ResNetForAudioClassification, config: EvaluationConfig, dataset_name: str, dataset_config: Dict) -> Dict:
    """Evaluate model using 1-second sliding windows."""
    model.eval()
    processor = WindowProcessor(config)

    all_results = []
    file_count = 0

    print(f"\nEvaluating dataset: {dataset_name}")
    print(f"Description: {dataset_config.get('description', 'No description')}")
    if dataset_config.get('label_override'):
        print(f"Label override: {dataset_config['label_override']}")

    print(f"Window duration: {config.window_duration}s")
    print(f"Hop duration: {config.hop_duration}s")
    print(f"Threshold: {config.threshold:.4f}")
    print(f"Expected seconds per cell: ~0.143s (1s window / 224px * 32 stride)")

    # Walk through dataset directory
    for root, dirs, files in os.walk(config.data_dir):
        for file in files:
            if file.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
                file_path = os.path.join(root, file)

                # Get label
                label_string = get_label_for_file(file_path, dataset_config.get('label_override'))
                true_label = LABEL2ID.get(label_string, 0)

                # Load and process file
                waveform, duration = processor.load_audio_file(file_path)
                if waveform is None:
                    continue

                # Extract 1-second windows
                windows = processor.extract_windows(waveform, duration)
                if not windows:
                    continue

                # Get predictions for all windows
                window_predictions = []
                for i in range(0, len(windows), config.batch_size):
                    batch_windows = windows[i:i + config.batch_size]

                    # Preprocess batch using shared function
                    batch_tensors = []
                    for window in batch_windows:
                        img = preprocess_audio_to_resnet_input(window)
                        batch_tensors.append(img)

                    if batch_tensors:
                        batch_tensor = torch.stack(batch_tensors).to(device)

                        # Model inference
                        with torch.no_grad():
                            outputs = model(batch_tensor)
                            logits = outputs.logits
                            probabilities = F.softmax(logits, dim=-1)
                            drone_probs = probabilities[:, 1].cpu().numpy()
                            window_predictions.extend(drone_probs.tolist())

                # Aggregate using Noisy-OR
                if window_predictions:
                    aggregated_prob = processor.aggregate_predictions(window_predictions, "noisy_or")
                    predicted_label = 1 if aggregated_prob > config.threshold else 0

                    result = {
                        "file_id": file_count,
                        "filename": os.path.basename(file_path),
                        "file_path": file_path,
                        "true_label": true_label,
                        "true_label_name": ID2LABEL[true_label],
                        "predicted_label": predicted_label,
                        "predicted_label_name": ID2LABEL[predicted_label],
                        "drone_probability": aggregated_prob,
                        "file_duration": duration,
                        "num_windows": len(windows),
                        "window_predictions": window_predictions,
                        "aggregation_method": "noisy_or"
                    }

                    all_results.append(result)
                    file_count += 1

                    if file_count % 50 == 0:
                        print(f"  Processed {file_count} files...")

    print(f"\nProcessed {file_count} files total")

    # Calculate metrics
    if not all_results:
        return {"error": "No files processed"}

    true_labels = [r["true_label"] for r in all_results]
    predicted_labels = [r["predicted_label"] for r in all_results]
    probabilities = [r["drone_probability"] for r in all_results]

    # Basic metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, average="binary", zero_division=0
    )

    # ROC curve and AUC
    try:
        auc = roc_auc_score(true_labels, probabilities)
        fpr, tpr, roc_thresholds = roc_curve(true_labels, probabilities)
    except ValueError:
        auc = 0.0
        fpr = tpr = roc_thresholds = None

    # Precision-Recall curve
    try:
        prec_curve, rec_curve, pr_thresholds = precision_recall_curve(true_labels, probabilities)
    except ValueError:
        prec_curve = rec_curve = pr_thresholds = None

    # Confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)

    # Class distribution
    class_distribution = Counter(true_labels)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "confusion_matrix": cm.tolist(),
        "class_distribution": dict(class_distribution),
        "total_files": file_count,
        "dataset_name": dataset_name,
        "threshold_used": config.threshold
    }

    return {
        "metrics": metrics,
        "results": all_results,
        "roc_data": {
            "fpr": fpr.tolist() if fpr is not None else None,
            "tpr": tpr.tolist() if tpr is not None else None,
            "thresholds": roc_thresholds.tolist() if roc_thresholds is not None else None
        },
        "pr_data": {
            "precision": prec_curve.tolist() if prec_curve is not None else None,
            "recall": rec_curve.tolist() if rec_curve is not None else None,
            "thresholds": pr_thresholds.tolist() if pr_thresholds is not None else None
        }
    }

def save_results_to_csv(results: Dict, output_path: str, dataset_name: str = None):
    """Save evaluation results to CSV file in the specified format."""
    # Create CSV data in the exact format requested
    csv_data = []
    for r in results["results"]:
        # Determine split type based on number of windows
        split = "whole_file" if r["num_windows"] == 1 else f"{r['num_windows']}_windows"

        csv_row = {
            "file_id": r["file_id"],
            "true_label": r["true_label"],
            "predicted_label": r["predicted_label"],
            "drone_probability": f"{r['drone_probability']:.8f}",  # Force decimal notation with 6 decimal places
            "aggregation_method": r["aggregation_method"],
            "aggregation_threshold": f"{r.get('aggregation_threshold', results['metrics']['threshold_used']):.6f}",  # Force decimal notation
            "split": split
        }
        csv_data.append(csv_row)

    # Generate dataset-specific filename if dataset name is provided
    if dataset_name:
        # Extract directory and base filename
        output_dir = os.path.dirname(output_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_output_path = os.path.join(output_dir, f"{dataset_name}_results_{timestamp}.csv")
    else:
        dataset_output_path = output_path

    df = pd.DataFrame(csv_data)
    df.to_csv(dataset_output_path, index=False)
    print(f"Results saved to: {dataset_output_path}")

    # Save summary metrics with dataset-specific filename
    metrics_path = dataset_output_path.replace('.csv', '_metrics.txt')
    with open(metrics_path, 'w') as f:
        metrics = results["metrics"]
        f.write(f"Dataset: {metrics['dataset_name']}\n")
        f.write(f"Total files: {metrics['total_files']}\n")
        f.write(f"Threshold used: {metrics['threshold_used']:.8f}\n")  # Force decimal notation
        f.write(f"Accuracy: {metrics['accuracy']:.8f}\n")  # Force decimal notation
        f.write(f"Precision: {metrics['precision']:.8f}\n")  # Force decimal notation
        f.write(f"Recall: {metrics['recall']:.8f}\n")  # Force decimal notation
        f.write(f"F1-Score: {metrics['f1']:.8f}\n")  # Force decimal notation
        f.write(f"AUC: {metrics['auc']:.8f}\n")  # Force decimal notation
        f.write(f"Class distribution: {metrics['class_distribution']}\n")
        f.write(f"Confusion Matrix:\n{np.array(metrics['confusion_matrix'])}\n")

    print(f"Metrics saved to: {metrics_path}")
    return dataset_output_path

def main():
    """Main evaluation script."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate ResNet-34 audio classification model")
    parser.add_argument("--model_path", type=str, required=True,
                      help="Path to trained model directory")
    parser.add_argument("--dataset", type=str, default=None,
                      choices=list(DATASETS.keys()),
                      help="Dataset name to evaluate (if not provided, evaluates all datasets)")
    parser.add_argument("--output_dir", type=str,
                      default=None,
                      help="Output directory for results (defaults to timestamped folder on desktop)")
    parser.add_argument("--batch_size", type=int, default=64,
                      help="Batch size for window processing")
    parser.add_argument("--hop_duration", type=float, default=0.5,
                      help="Hop duration between windows in seconds")
    parser.add_argument("--calibrate", action="store_true",
                      help="Calibrate threshold on this dataset")
    parser.add_argument("--threshold", type=float, default=None,
                      help="Manual threshold override")

    args = parser.parse_args()

    # Create timestamped output directory on desktop if not specified
    if args.output_dir is None:
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.dataset:
            args.output_dir = os.path.join(desktop_path, f"resnet34_evaluation_{args.dataset}_{timestamp}")
        else:
            args.output_dir = os.path.join(desktop_path, f"resnet34_evaluation_all_datasets_{timestamp}")

    # Determine which datasets to evaluate
    if args.dataset:
        datasets_to_evaluate = [args.dataset]
        print(f"Evaluating single dataset: {args.dataset}")
    else:
        datasets_to_evaluate = list(DATASETS.keys())
        print(f"Evaluating all {len(datasets_to_evaluate)} datasets")

    try:
        model = load_model(args.model_path)
        print(f"✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return

    # Create main output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Results will be saved to: {args.output_dir}")

    # Track overall results
    all_dataset_results = {}

    # Iterate through datasets to evaluate
    for dataset_name in datasets_to_evaluate:
        dataset_config = DATASETS[dataset_name]

        # Check if dataset path exists
        if not os.path.exists(dataset_config["path"]):
            print(f"Skipping {dataset_name}: Dataset path does not exist: {dataset_config['path']}")
            continue

        print(f"\n{'='*80}")
        print(f"EVALUATING DATASET: {dataset_name} ({datasets_to_evaluate.index(dataset_name) + 1}/{len(datasets_to_evaluate)})")
        print(f"{'='*80}")

        # Determine threshold for this dataset
        threshold = args.threshold
        if threshold is None:
            # Try to load saved calibrated threshold
            threshold_file = os.path.join(args.model_path, "calibrated_threshold.json")
            if os.path.exists(threshold_file):
                try:
                    with open(threshold_file, 'r') as f:
                        threshold_data = json.load(f)
                    threshold = threshold_data["threshold"]
                    print(f"✓ Using saved calibrated threshold: {threshold:.4f}")
                except:
                    threshold = 0.5
                    print("Warning: Could not load saved threshold, using 0.5")
            else:
                threshold = 0.5
                print("No calibrated threshold found, using default 0.5")

        # Create configuration for this dataset
        config = EvaluationConfig(
            model_path=args.model_path,
            data_dir=dataset_config["path"],
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            hop_duration=args.hop_duration,
            threshold=threshold
        )

        # Calibrate threshold if requested (only for first dataset or single dataset)
        if args.calibrate and (args.dataset or dataset_name == datasets_to_evaluate[0]):
            calibrated_threshold = calibrate_threshold(model, config, dataset_config)
            config.threshold = calibrated_threshold

        # Run evaluation for this dataset
        results = evaluate_model_with_windows(model, config, dataset_name, dataset_config)

        if "error" in results:
            print(f"Evaluation failed for {dataset_name}: {results['error']}")
            continue

        # Store results
        all_dataset_results[dataset_name] = results

        # Print results for this dataset
        metrics = results["metrics"]
        print(f"\n{'='*60}")
        print(f"RESULTS - {dataset_name}")
        print(f"{'='*60}")
        print(f"Total files processed: {metrics['total_files']}")
        print(f"Threshold used: {metrics['threshold_used']:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1']:.4f}")
        print(f"AUC: {metrics['auc']:.4f}")
        print(f"Class distribution: {metrics['class_distribution']}")
        print(f"Confusion Matrix:")
        print(np.array(metrics['confusion_matrix']))

        # Save results for this dataset
        dataset_output_dir = os.path.join(args.output_dir, dataset_name)
        os.makedirs(dataset_output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(dataset_output_dir, f"{dataset_name}_results_{timestamp}.csv")
        save_results_to_csv(results, output_file, dataset_name)

    # Create summary report if multiple datasets were evaluated
    if len(all_dataset_results) > 1:
        print(f"\n{'='*80}")
        print(f"SUMMARY ACROSS ALL DATASETS")
        print(f"{'='*80}")

        summary_data = []
        for dataset_name, results in all_dataset_results.items():
            metrics = results["metrics"]
            summary_data.append({
                "Dataset": dataset_name,
                "Total_Files": metrics['total_files'],
                "Accuracy": f"{metrics['accuracy']:.4f}",
                "Precision": f"{metrics['precision']:.4f}",
                "Recall": f"{metrics['recall']:.4f}",
                "F1_Score": f"{metrics['f1']:.4f}",
                "AUC": f"{metrics['auc']:.4f}",
                "Threshold": f"{metrics['threshold_used']:.4f}",
                "Description": DATASETS[dataset_name]["description"]
            })

        # Save summary to CSV
        summary_df = pd.DataFrame(summary_data)
        summary_file = os.path.join(args.output_dir, f"evaluation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        summary_df.to_csv(summary_file, index=False)
        print(f"Summary saved to: {summary_file}")

        # Print summary table
        print("\nSummary Table:")
        print(summary_df.to_string(index=False))

    print(f"\n✓ Evaluation completed successfully!")
    print(f"All results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
