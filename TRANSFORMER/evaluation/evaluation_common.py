#!/usr/bin/env python3
"""
Common evaluation functionality for transformer models.
This module contains shared functions for audio processing, model evaluation, and result analysis.
"""

import time
import gc
import numpy as np
import torch
from datasets import load_dataset, Audio, Dataset
from scipy.special import softmax
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import warnings

# Suppress sklearn warnings about undefined metrics
warnings.filterwarnings("ignore", message=".*(Precision|F-score|Recall) is ill-defined.*")

def create_data_collator(feat_ext, sample_rate):
    """Create a data collator function for the trainer"""
    def data_collator_dynamic(features):
        audio_arrays = [f["input_arrays"] for f in features]
        inputs = feat_ext(
            audio_arrays,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True
        )

        # Return only the tensors the model expects
        return {
            "input_values": inputs.input_values,
            "attention_mask": inputs.attention_mask,
            "labels": torch.tensor(
                [f["labels"] for f in features],
                dtype=torch.long
            )
        }
    return data_collator_dynamic

def create_compute_metrics_function(evaluation_type="file"):
    """Create compute_metrics function for different evaluation types"""
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predicted_classes = np.argmax(predictions, axis=1)

        accuracy = accuracy_score(labels, predicted_classes)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predicted_classes, average='binary', zero_division=0.0
        )

        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    return compute_metrics

def get_file_size_mb(file_path):
    """Get file size in MB"""
    try:
        import os
        size_bytes = os.path.getsize(file_path)
        return size_bytes / (1024 * 1024)
    except:
        return 0


def process_predictions(predictions, threshold=None):
    """Process raw model predictions into probabilities and classifications"""
    # Get threshold from config if not provided
    if threshold is None:
        from TRANSFORMER.config.config import EvaluationConfig
        config = EvaluationConfig()
        threshold = config.file.threshold  # Default to file threshold

    # Use scipy.special.softmax for performance
    probabilities = softmax(predictions, axis=1)
    drone_probabilities = probabilities[:, 1]  # P(yes_drone)

    # Handle negative thresholds (inverted logic for poor-performing models)
    if threshold < 0:
        # Negative threshold means use inverted logic: predict drone when prob < |threshold|
        abs_threshold = abs(threshold)
        predicted_classes = (drone_probabilities < abs_threshold).astype(int)
    else:
        # Normal logic: predict drone when prob > threshold
        predicted_classes = (drone_probabilities > threshold).astype(int)

    return predicted_classes, drone_probabilities

def calculate_performance_metrics(true_labels, predictions, probabilities):
    """Calculate comprehensive performance metrics for evaluation results"""
    unique_labels = np.unique(true_labels)
    unique_preds = np.unique(predictions)

    accuracy = accuracy_score(true_labels, predictions)

    # Enhanced handling for single-class datasets
    if len(unique_labels) == 1:
        true_class = unique_labels[0]
        true_class_name = "yes_drone" if true_class == 1 else "unknown"

        if len(unique_preds) == 1:
            pred_class = unique_preds[0]
            pred_class_name = "yes_drone" if pred_class == 1 else "unknown"

            if true_class == pred_class:
                precision = 1.0
                recall = 1.0
                f1 = 1.0
                classification_quality = "Perfect"
            else:
                precision = 0.0
                recall = 0.0
                f1 = 0.0
                classification_quality = "Complete Miss"

            print(f"  Single-class dataset ({true_class_name}): {classification_quality}")
            print(f"  Model predicted: {pred_class_name} for all samples")
        else:
            # Mixed predictions for single true class
            correct_preds = np.sum(predictions == true_class)
            total_preds = len(predictions)

            if true_class == 1:  # All samples are "yes_drone"
                recall = correct_preds / total_preds
                drone_detections = np.sum(predictions == 1)
                precision = correct_preds / drone_detections if drone_detections > 0 else 0.0
            else:  # All samples are "unknown"
                recall = correct_preds / total_preds
                precision = 0.0

            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            print(f"  Single-class dataset ({true_class_name}): Mixed predictions")
            print(f"  Correct predictions: {correct_preds}/{total_preds} ({correct_preds/total_preds:.3f})")
    else:
        # Normal case with multiple classes
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='binary', zero_division=0.0
        )

    # Calculate additional statistics for single-class datasets
    additional_stats = {}
    if len(unique_labels) == 1:
        true_class = unique_labels[0]
        if true_class == 1:  # All "yes_drone"
            false_negative_rate = np.sum(predictions == 0) / len(predictions)
            detection_rate = 1 - false_negative_rate
            additional_stats = {
                'detection_rate': detection_rate,
                'false_negative_rate': false_negative_rate,
                'dataset_type': 'all_drone'
            }
        else:  # All "unknown"
            false_positive_rate = np.sum(predictions == 1) / len(predictions)
            specificity = 1 - false_positive_rate
            additional_stats = {
                'specificity': specificity,
                'false_positive_rate': false_positive_rate,
                'dataset_type': 'all_unknown'
            }
    else:
        additional_stats = {'dataset_type': 'mixed'}

    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'unique_true_labels': len(unique_labels),
        'unique_pred_labels': len(unique_preds),
        **additional_stats
    }

def print_performance_statistics(dataset_name, metrics, additional_stats):
    """Print performance statistics in a consistent format"""
    print(f"{dataset_name} → Accuracy={metrics['accuracy']:.3f}, F1={metrics['f1']:.3f}, "
          f"Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}")

    # Print additional statistics for single-class datasets
    if additional_stats['dataset_type'] == 'all_drone':
        print(f"  Detection Rate: {additional_stats['detection_rate']:.3f} (how many drones were found)")
        print(f"  Miss Rate: {additional_stats['false_negative_rate']:.3f} (how many drones were missed)")
    elif additional_stats['dataset_type'] == 'all_unknown':
        print(f"  Specificity: {additional_stats['specificity']:.3f} (how well it rejects non-drones)")
        print(f"  False Positive Rate: {additional_stats['false_positive_rate']:.3f} (how often it falsely detects drones)")

def safe_cuda_cleanup():
    """Safely clean up CUDA memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()

def monitor_cuda_memory(context=""):
    """Monitor and report CUDA memory usage"""
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
        memory_reserved = torch.cuda.memory_reserved() / (1024**3)   # GB
        print(f"    GPU Memory {context}: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
        return memory_allocated, memory_reserved
    return 0, 0

def estimate_memory_requirements(audio_length, sample_rate):
    """Estimate memory requirements for processing audio"""
    # Rough estimate for float32 audio processing
    estimated_memory_gb = (audio_length * sample_rate * 4) / (1024**3)
    return estimated_memory_gb

def aggregate_predictions_by_confidence(predictions, labels, file_ids, probabilities, threshold=0.2, verbose=True):
    """
    Aggregate predictions using confidence-weighted averaging.
    This is the common aggregation method used across evaluation scripts.
    """
    file_probabilities = {}
    file_true_labels = {}

    # Group probabilities by file_id
    for pred, label, file_id, prob in zip(predictions, labels, file_ids, probabilities):
        if file_id not in file_probabilities:
            file_probabilities[file_id] = []
            file_true_labels[file_id] = label

        file_probabilities[file_id].append(prob)

    # Aggregate using confidence-weighted averaging
    aggregated_preds = []
    aggregated_labels = []
    aggregated_avg_probs = []

    for file_id in sorted(file_probabilities.keys()):
        clip_probs = file_probabilities[file_id]
        true_label = file_true_labels[file_id]

        # Calculate average probability across all clips
        avg_prob = np.mean(clip_probs)

        # Apply threshold to average probability
        file_pred = int(avg_prob > threshold)

        aggregated_preds.append(file_pred)
        aggregated_labels.append(true_label)
        aggregated_avg_probs.append(avg_prob)

        if verbose:
            print(f"File {file_id}: {len(clip_probs)} clips, avg_prob: {avg_prob:.4f}, "
                  f"threshold: {threshold}, prediction: {file_pred}, true: {true_label}")

    return np.array(aggregated_preds), np.array(aggregated_labels), np.array(aggregated_avg_probs)

def create_trainer_config(model, feat_ext, sample_rate, batch_size, output_dir="./eval_results"):
    """Create a standardized trainer configuration"""
    from transformers import Trainer, TrainingArguments

    eval_args = TrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=batch_size,
        do_train=False,
        do_eval=True,
        logging_strategy="no",
        save_strategy="no",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=eval_args,
        data_collator=create_data_collator(feat_ext, sample_rate),
        compute_metrics=create_compute_metrics_function(),
    )

    return trainer

def load_model_and_feature_extractor(model_path):
    """Load model and feature extractor with proper device handling"""
    from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model to {device}…")

    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path).to(device)
    feat_ext = Wav2Vec2FeatureExtractor.from_pretrained(model_path)

    return model, feat_ext, device
