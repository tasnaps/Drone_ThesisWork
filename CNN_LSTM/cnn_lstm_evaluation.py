"""
CNN-LSTM Evaluation script.
"""
import os
import traceback
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
import datetime
import librosa
from typing import List, Dict, Tuple, Optional
import warnings
from dataclasses import dataclass
import argparse
import json
from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support,
        confusion_matrix, roc_auc_score
    )
from collections import Counter
from cnn_lstm_model import CNNLSTMModel
from common import SAMPLE_RATE, N_MELS, HOP_LENGTH, DATASETS, CALIBRATION_SET, LABEL2ID, ID2LABEL
# Suppress warnings
warnings.filterwarnings("ignore")

@dataclass
class EvaluationConfig:
    """Configuration for CNN-LSTM evaluation."""
    model_path: str
    data_dir: str
    output_dir: str
    batch_size: int = 8
    max_file_length: float = 180.0  #files longer than this get split
    overlap_seconds: float = 1.5  # 1.5-second overlap
    chunk_duration: float = 180.0  # large chunks for file splitting
    model_chunk_duration: float = 1.0  # sliding window size that matches training
    sample_rate: int = SAMPLE_RATE
    threshold: float = 0.5
    n_mels: int = N_MELS
    model_hop_seconds: float = 1.0



class AudioPreprocessor:
    """Audio preprocessing similar to data.py but for evaluation."""

    def __init__(self, sample_rate: int = SAMPLE_RATE, n_mels: int = N_MELS):
        self.sample_rate = sample_rate
        self.n_mels = n_mels

        # Initialize transforms (similar to data.py)
        import torchaudio.transforms as T
        self.spec_extractor = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=1024,
            hop_length=HOP_LENGTH,
            power=1.0  # For magnitude spectrogram for PCEN
        )

    def preprocess_audio_chunk(self, waveform: torch.Tensor) -> torch.Tensor:
        """Preprocess a single audio chunk similar to prepare_batch in data.py.
        Expects window-sized inputs (e.g., 1s) and does not crop the beginning only.
        """
        # Ensure waveform is 1D
        if waveform.ndim > 1:
            waveform = waveform.mean(dim=0)

        # Convert to tensor if numpy
        if isinstance(waveform, np.ndarray):
            waveform = torch.tensor(waveform, dtype=torch.float32)

        # Do not forcibly crop to 3s; assume upstream provided fixed window duration.
        # Only ensure a minimal length for STFT safety.
        min_len = 1024
        if waveform.size(-1) < min_len:
            pad = torch.zeros(min_len - waveform.size(-1), dtype=waveform.dtype)
            waveform = torch.cat([waveform, pad], dim=-1)

        # Extract mel spectrogram
        mel = self.spec_extractor(waveform)

        # Apply PCEN (same as data.py)
        mel_np = mel.numpy()
        mel_scaled_np = mel_np * (2**31)
        pcen_mel_np = librosa.pcen(mel_scaled_np, sr=self.sample_rate, hop_length=HOP_LENGTH)
        mel = torch.tensor(pcen_mel_np, dtype=torch.float32)

        # Add channel dimension
        return mel.unsqueeze(0)  # Shape: (1, n_mels, time)


class LargeFileHandler:
    """Handles splitting and aggregation of large audio files."""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.preprocessor = AudioPreprocessor(config.sample_rate, config.n_mels)

    def load_audio_file(self, file_path: str) -> Tuple[Optional[np.ndarray], float]:
        """Load audio file and return waveform and duration."""
        try:
            waveform, sr = librosa.load(file_path, sr=self.config.sample_rate)
            duration = len(waveform) / sr
            return waveform, duration
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None, 0.0

    def split_long_audio(self, waveform: np.ndarray, duration: float) -> List[np.ndarray]:
        """Split long audio into overlapping large chunks (~180s) as a safety valve."""
        if duration <= self.config.max_file_length:
            return [waveform]

        chunk_samples = int(self.config.chunk_duration * self.config.sample_rate)
        overlap_samples = int(self.config.overlap_seconds * self.config.sample_rate)
        step_samples = max(1, chunk_samples - overlap_samples)

        chunks = []
        start = 0
        while start < len(waveform):
            end = min(start + chunk_samples, len(waveform))
            chunk = waveform[start:end]

            # Pad if chunk is too short
            if len(chunk) < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), mode='constant')

            chunks.append(chunk)
            start += step_samples

            if end >= len(waveform):
                break

        return chunks

    def split_into_model_windows(self, waveform: np.ndarray) -> List[np.ndarray]:
        """Split a (possibly large) waveform into fixed-size model windows (e.g., 1s) with hop.
        Ensures full coverage including the tail without excessive overlap.
        """
        sr = self.config.sample_rate
        win = int(self.config.model_chunk_duration * sr)
        hop = max(1, int(self.config.model_hop_seconds * sr))

        # If shorter than a single window, pad to full window
        if len(waveform) <= win:
            pad_len = max(0, win - len(waveform))
            return [np.pad(waveform, (0, pad_len), mode='constant')]

        windows: List[np.ndarray] = []

        # Create regular windows with hop
        for start in range(0, len(waveform) - win + 1, hop):
            windows.append(waveform[start:start + win])

        # Handle remaining tail if there's uncovered audio
        last_window_end = (len(windows) - 1) * hop + win if windows else 0
        if last_window_end < len(waveform):
            # Extract the remaining audio and pad it
            tail_start = last_window_end
            tail_audio = waveform[tail_start:]

            # Pad the tail to window size
            pad_len = win - len(tail_audio)
            padded_tail = np.pad(tail_audio, (0, pad_len), mode='constant')
            windows.append(padded_tail)

        return windows

    def aggregate_predictions(self, predictions: List[float], method: str = "mean") -> float:
        """Aggregate predictions from multiple windows or chunks."""
        if not predictions:
            return 0.0

        if method == "mean":
            return float(np.mean(predictions))
        elif method == "max":
            return float(np.max(predictions))
        elif method == "median":
            return float(np.median(predictions))
        else:
            return float(np.mean(predictions))


class CNNLSTMEvaluator:
    """Main evaluator for CNN-LSTM model with large file handling."""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.file_handler = LargeFileHandler(config)
        self.model = None
        # Default to index 1 (common convention), will auto-infer during calibration
        self.drone_class_index: int = 1

        print(f"Using device: {self.device}")

    def load_model(self):
        """Load the trained CNN-LSTM model."""
        print(f"Loading model from: {self.config.model_path}")

        # Handle both file paths and directory paths
        model_file_path = self.config.model_path

        # If it's a directory, look for model files inside (including nested subdirectories)
        if os.path.isdir(self.config.model_path):
            # Try different common model file names
            potential_files = [
                "pytorch_model.bin",
                "model.pth",
                "best_model.pth",
                "final_model.pth",
                "cnn_lstm_model.pth",
                "model.safetensors"
            ]

            model_loaded = False

            # First, try the direct directory
            for filename in potential_files:
                potential_path = os.path.join(self.config.model_path, filename)
                if os.path.exists(potential_path):
                    model_file_path = potential_path
                    print(f"Found model file: {filename}")
                    model_loaded = True
                    break

            # If not found, search in subdirectories (one level deep)
            if not model_loaded:
                for subdir in os.listdir(self.config.model_path):
                    subdir_path = os.path.join(self.config.model_path, subdir)
                    if os.path.isdir(subdir_path):
                        for filename in potential_files:
                            potential_path = os.path.join(subdir_path, filename)
                            if os.path.exists(potential_path):
                                model_file_path = potential_path
                                print(f"Found model file in subdirectory: {subdir}/{filename}")
                                model_loaded = True
                                break
                        if model_loaded:
                            break

            if not model_loaded:
                # List available files to help debug
                available_files = []
                for root, dirs, files in os.walk(self.config.model_path):
                    for file in files:
                        if file.endswith(('.pth', '.bin', '.safetensors')):
                            rel_path = os.path.relpath(os.path.join(root, file), self.config.model_path)
                            available_files.append(rel_path)

                raise FileNotFoundError(
                    f"No recognized model file found in {self.config.model_path}. "
                    f"Available model files: {available_files}. "
                    f"Expected one of: {potential_files}"
                )

        # Rest of the loading logic remains the same...
        try:
            if model_file_path.endswith('.safetensors'):
                # Handle safetensors files
                try:
                    from safetensors.torch import load_file
                    checkpoint = load_file(model_file_path, device=str(self.device))
                    print(f"✓ Successfully loaded safetensors from: {os.path.basename(model_file_path)}")

                    # For safetensors, we use the exact training script parameters
                    model_config = {
                        'num_labels': 2,  # Binary classification
                        'hidden_size': 128,  # Default from training script
                        'lstm_layers': 1,
                    }
                    print(f"Using training script model config: {model_config}")

                    # Initialize model with exact training parameters
                    self.model = CNNLSTMModel(
                        num_labels=model_config['num_labels'],
                        hidden_size=model_config['hidden_size'],
                        lstm_layers=model_config['lstm_layers']
                    )

                    # Load state dict directly (safetensors already provides the state dict)
                    self.model.load_state_dict(checkpoint)

                except ImportError:
                    raise RuntimeError("safetensors library not available. Install with: pip install safetensors")

            else:
                # Handle regular PyTorch files (.pth, .bin)
                checkpoint = torch.load(model_file_path, map_location=self.device, weights_only=False)
                print(f"✓ Successfully loaded checkpoint from: {os.path.basename(model_file_path)}")

                # Extract model configuration or use defaults
                if 'model_config' not in checkpoint:
                    print("Warning: No model_config found in checkpoint, using defaults")
                    model_config = {
                        'n_mels': 128,
                        'hidden_dim': 128,
                        'num_layers': 2,
                        'num_classes': 2
                    }
                else:
                    model_config = checkpoint['model_config']

                # Initialize model
                self.model = CNNLSTMModel(
                    num_labels=model_config.get('num_classes', 2),
                    hidden_size=model_config.get('hidden_dim', 128),
                    lstm_layers=model_config.get('num_layers', 1)
                )

                # Load state dict
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])
                else:
                    # Try loading the checkpoint directly as state dict
                    self.model.load_state_dict(checkpoint)

        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_file_path}: {e}")

        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded successfully!")

    def predict_chunks(self, chunks: List[np.ndarray]) -> List[float]:
        """Predict probabilities for a list of audio windows (1s by default)."""
        if not chunks:
            return []

        predictions = []

        # Process chunks in batches
        for i in range(0, len(chunks), self.config.batch_size):
            batch_chunks = chunks[i:i + self.config.batch_size]

            # Preprocess chunks
            batch_spectrograms = []
            for chunk in batch_chunks:
                chunk_tensor = torch.tensor(chunk, dtype=torch.float32)
                spectrogram = self.file_handler.preprocessor.preprocess_audio_chunk(chunk_tensor)
                batch_spectrograms.append(spectrogram)

            # Stack and pad spectrograms
            max_time = max(spec.size(-1) for spec in batch_spectrograms)
            padded_specs = []
            for spec in batch_spectrograms:
                padded = F.pad(spec, (0, max_time - spec.size(-1)))
                padded_specs.append(padded)

            batch_tensor = torch.stack(padded_specs).to(self.device)

            # Model inference
            with torch.no_grad():
                outputs = self.model(batch_tensor)
                probabilities = torch.softmax(outputs.logits, dim=1)
                # Use the inferred class index for the 'drone' class
                drone_probs = probabilities[:, self.drone_class_index].cpu().numpy()
                predictions.extend(drone_probs.tolist())

        return predictions

    def infer_drone_class_index(self, calibration_dir: str, max_files: int = 100) -> int:
        """Infer whether class index 0 or 1 corresponds to 'yes_drone' using a small sample.
        Chooses the index that yields higher mean probability for drone-labeled files
        than for unknown-labeled files, maximizing separation.
        """
        if not os.path.isdir(calibration_dir):
            print("Calibration directory not found for class index inference; keeping default index 1")
            return self.drone_class_index

        # Gather a small balanced sample
        sample_files: List[Tuple[str, int]] = []
        drone_count = unknown_count = 0
        for root, _, files in os.walk(calibration_dir):
            for file in files:
                if not file.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
                    continue
                file_path = os.path.join(root, file)
                label_str = get_label_for_file(file_path, None)
                label = LABEL2ID.get(label_str, 0)
                if label == 1 and drone_count < max_files // 2:
                    sample_files.append((file_path, label))
                    drone_count += 1
                elif label == 0 and unknown_count < max_files // 2:
                    sample_files.append((file_path, label))
                    unknown_count += 1
                if len(sample_files) >= max_files:
                    break
            if len(sample_files) >= max_files:
                break

        if len(sample_files) < 10:
            print("Not enough files to infer class index; keeping default index 1")
            return self.drone_class_index

        def eval_with_index(idx: int) -> Tuple[float, float]:
            saved_idx = self.drone_class_index
            self.drone_class_index = idx
            drone_probs_list, unknown_probs_list = [], []
            for fp, lab in sample_files:
                res = self.evaluate_file(fp, lab)
                if res is None:
                    continue
                p = float(res['drone_probability'])
                if lab == 1:
                    drone_probs_list.append(p)
                else:
                    unknown_probs_list.append(p)
            self.drone_class_index = saved_idx
            d_mean = float(np.mean(drone_probs_list)) if drone_probs_list else 0.0
            u_mean = float(np.mean(unknown_probs_list)) if unknown_probs_list else 0.0
            return d_mean, u_mean

        d0, u0 = eval_with_index(0)
        d1, u1 = eval_with_index(1)

        # Choose the index that maximizes (mean_drone - mean_unknown)
        sep0 = d0 - u0
        sep1 = d1 - u1
        chosen = 1 if sep1 >= sep0 else 0
        self.drone_class_index = chosen
        print(f"Inferred drone class index: {chosen} (sep0={sep0:.4f}, sep1={sep1:.4f}, d0={d0:.4f}, u0={u0:.4f}, d1={d1:.4f}, u1={u1:.4f})")
        return chosen

    def evaluate_file(self, file_path: str, true_label: int) -> Optional[Dict]:
        """Evaluate a single audio file using 1-second sliding windows over the entire file."""
        # Load audio
        waveform, duration = self.file_handler.load_audio_file(file_path)
        if waveform is None:
            return None

        # Outer split into large chunks if necessary (for memory safety)
        outer_chunks = self.file_handler.split_long_audio(waveform, duration)

        # Create 1s windows inside each outer chunk
        windows: List[np.ndarray] = []
        for chunk in outer_chunks:
            windows.extend(self.file_handler.split_into_model_windows(chunk))

        # Predict over all windows and aggregate
        window_predictions = self.predict_chunks(windows)
        aggregated_prob = self.file_handler.aggregate_predictions(window_predictions, method="mean")
        predicted_label = 1 if aggregated_prob > self.config.threshold else 0

        # Determine split type
        split_type = "whole_file" if len(outer_chunks) == 1 else f"{len(outer_chunks)}_chunks"
        aggregation_method = "mean_over_windows"

        return {
            "file_path": file_path,
            "true_label": true_label,
            "predicted_label": predicted_label,
            "drone_probability": aggregated_prob,
            "aggregation_method": aggregation_method,
            "aggregation_threshold": self.config.threshold,
            "split": split_type,
            "duration": duration,
            "num_chunks": len(outer_chunks)
        }

    def evaluate_dataset(self, dataset_name: str, data_dir: str) -> List[Dict]:
        """Evaluate all files in a dataset."""
        print(f"\nEvaluating dataset: {dataset_name}")
        print(f"Data directory: {data_dir}")

        results = []
        file_count = 0

        # Retrieve dataset config for potential label override
        dataset_config = DATASETS.get(dataset_name, {})
        label_override = dataset_config.get('label_override')

        # Walk through dataset directory
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
                    file_path = os.path.join(root, file)

                    # Determine label using helper with optional override
                    label_string = get_label_for_file(file_path, label_override)
                    true_label = LABEL2ID.get(label_string, 0)

                    result = self.evaluate_file(file_path, true_label)
                    if result:
                        result['file_id'] = file_count
                        results.append(result)
                        file_count += 1

                        if file_count % 50 == 0:
                            print(f"  Processed {file_count} files...")

        print(f"  Completed: {file_count} files processed")
        return results

    def save_results(self, dataset_results: Dict[str, List[Dict]]):
        """Save results in the same CSV format as ResNet evaluation."""
        # Use the provided output directory (created in main)
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"\nSaving results to: {output_dir}")

        # Save individual dataset results with ResNet-style CSV format
        for dataset_name, results in dataset_results.items():
            if not results:
                continue

            # Create dataset subdirectory
            dataset_dir = output_dir / dataset_name
            dataset_dir.mkdir(exist_ok=True)

            # Convert to DataFrame with exact ResNet format
            df_data = []
            for result in results:
                df_data.append({
                    "file_id": result["file_id"],
                    "true_label": result["true_label"],
                    "predicted_label": result["predicted_label"],
                    "drone_probability": f"{result['drone_probability']:.8f}",  # Force 8 decimal places
                    "aggregation_method": result["aggregation_method"],
                    "aggregation_threshold": f"{self.config.threshold:.6f}",  # Force 6 decimal places
                    "split": result["split"]
                })

            df = pd.DataFrame(df_data)

            # Save detailed CSV (matches ResNet format)
            csv_path = dataset_dir / f"{dataset_name}_detailed_files.csv"
            df.to_csv(csv_path, index=False)
            print(f"  Saved: {csv_path}")

        # Create summary statistics (matches ResNet format)
        summary_data = []
        for dataset_name, results in dataset_results.items():
            if not results:
                continue

            # Calculate metrics
            true_labels = [r["true_label"] for r in results]
            predicted_labels = [r["predicted_label"] for r in results]
            probabilities = [r["drone_probability"] for r in results]

            # Basic metrics
            accuracy = np.mean([t == p for t, p in zip(true_labels, predicted_labels)])

            # Count metrics
            tp = sum([t == 1 and p == 1 for t, p in zip(true_labels, predicted_labels)])
            tn = sum([t == 0 and p == 0 for t, p in zip(true_labels, predicted_labels)])
            fp = sum([t == 0 and p == 1 for t, p in zip(true_labels, predicted_labels)])
            fn = sum([t == 1 and p == 0 for t, p in zip(true_labels, predicted_labels)])

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            # File splitting statistics
            whole_files = sum([1 for r in results if r["split"] == "whole_file"])
            split_files = len(results) - whole_files
            avg_chunks = np.mean([r["num_chunks"] for r in results])

            summary_data.append({
                "Dataset": dataset_name,
                "Total_Files": len(results),
                "Whole_Files": whole_files,
                "Split_Files": split_files,
                "Avg_Chunks_Per_File": f"{avg_chunks:.2f}",
                "Accuracy": f"{accuracy:.4f}",
                "Precision": f"{precision:.4f}",
                "Recall": f"{recall:.4f}",
                "F1_Score": f"{f1:.4f}",
                "Mean_Drone_Prob": f"{np.mean(probabilities):.6f}",
                "Std_Drone_Prob": f"{np.std(probabilities):.6f}",
                "Threshold": f"{self.config.threshold:.6f}",
                "Description": DATASETS.get(dataset_name, {}).get("description", "Unknown dataset")
            })

        # Save summary (matches ResNet format)
        summary_df = pd.DataFrame(summary_data)
        summary_path = output_dir / f"evaluation_summary_{timestamp}.csv"
        summary_df.to_csv(summary_path, index=False)

        # Save evaluation config
        config_data = {
            "model_path": self.config.model_path,
            "max_file_length": self.config.max_file_length,
            "overlap_seconds": self.config.overlap_seconds,
            "chunk_duration": self.config.chunk_duration,
            "threshold": f"{self.config.threshold:.6f}",  # Force decimal notation
            "batch_size": self.config.batch_size,
            "timestamp": timestamp,
            "evaluation_type": "CNN-LSTM",
            "model_chunk_duration": self.config.model_chunk_duration,
            "model_hop_seconds": self.config.model_hop_seconds,
        }

        config_df = pd.DataFrame([config_data])
        config_path = output_dir / "evaluation_config.csv"
        config_df.to_csv(config_path, index=False)

        print(f"\n Evaluation completed!")
        print(f"Results saved to: {output_dir}")
        print(f"Summary statistics saved to: {summary_path}")

        return output_dir

LABEL2ID = {"unknown": 0, "yes_drone": 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
drone_pattern = ["yes_drone"]
no_drone_pattern = ["unknown"]

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

def calibrate_threshold(model: nn.Module, config: EvaluationConfig, calibration_data_dir: str = None,
                        strategy: str = "f1", force_drone_index: Optional[int] = None) -> Tuple[float, 'CNNLSTMEvaluator']:
    """Robust threshold calibration with improved label inference and tie-breaking."""
    print("\n=== Threshold Calibration ===")
    print(f"Using optimization strategy: {strategy}")

    data_dir = calibration_data_dir if calibration_data_dir else config.data_dir
    print(f"Using calibration dataset: {data_dir}")

    evaluator = CNNLSTMEvaluator(config)
    evaluator.model = model  # model already on device and in eval mode

    # NEW: infer which class index corresponds to 'drone' using the calibration set
    if force_drone_index is not None:
        evaluator.drone_class_index = force_drone_index
        print(f"Using forced drone class index: {force_drone_index}")
    else:
        try:
            evaluator.infer_drone_class_index(data_dir, max_files=100)
        except Exception as e:
            print(f"Warning: Failed to infer class index ({e}); continuing with index {evaluator.drone_class_index}")

    all_probs: List[float] = []
    all_labels: List[int] = []

    # Collect aggregated per-file probabilities
    for root, _, files in os.walk(data_dir):
        for file in files:
            if not file.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
                continue
            file_path = os.path.join(root, file)
            # Determine label from folder structure (no override for calibration set)
            label_string = get_label_for_file(file_path, None)
            true_label = LABEL2ID.get(label_string, 0)

            result = evaluator.evaluate_file(file_path, true_label)
            if result is not None:
                all_probs.append(float(result["drone_probability"]))
                all_labels.append(true_label)

    if len(all_probs) < 10:
        print("Warning: Too few files for proper calibration, using default 0.5")
        return 0.5, evaluator

    labels = np.asarray(all_labels, dtype=np.int32)
    probs = np.asarray(all_probs, dtype=np.float64)

    # Per-class diagnostics
    pos_mask = labels == 1
    neg_mask = labels == 0
    n_pos, n_neg = int(pos_mask.sum()), int(neg_mask.sum())

    if n_pos == 0 or n_neg == 0:
        print(
            f"Warning: Only one class present in calibration data (unknown: {n_neg}, drone: {n_pos}), using default 0.5")
        return 0.5, evaluator

    prevalence = float(labels.mean())

    def safe_stats(x: np.ndarray):
        if x.size == 0: return float("nan"), float("nan"), float("nan"), float("nan")
        return float(np.min(x)), float(np.max(x)), float(np.mean(x)), float(np.std(x))

    p_min, p_max, p_mean, p_std = safe_stats(probs)
    n_min, n_max, n_mean, n_std = safe_stats(probs[neg_mask])
    d_min, d_max, d_mean, d_std = safe_stats(probs[pos_mask])
    n_unique = int(np.unique(np.round(probs, 8)).size)

    print(f"Calibration used {len(probs)} files")
    print(f"Class counts -> unknown: {n_neg}, drone: {n_pos}, prevalence: {prevalence:.4f}")
    print(
        f"All probs -> min: {p_min:.8f}, max: {p_max:.8f}, mean: {p_mean:.8f}, std: {p_std:.8f}, unique(~1e-8): {n_unique}")
    print(f"Unknown probs -> min: {n_min:.8f}, max: {n_max:.8f}, mean: {n_mean:.8f}, std: {n_std:.8f}")
    print(f"Drone probs   -> min: {d_min:.8f}, max: {d_max:.8f}, mean: {d_mean:.8f}, std: {d_std:.8f}")

    # If drone mean is still lower and we haven't forced an index yet, try the other index
    if d_mean < n_mean and force_drone_index is None:
        print(f"Switching drone class index to 0 due to inverted probabilities (d_mean={d_mean:.4f} < n_mean={n_mean:.4f})")
        # Use the opposite index (0 if we tried 1, 1 if we tried 0)
        corrected_index = 0 if evaluator.drone_class_index == 1 else 1
        return calibrate_threshold(model, config, calibration_data_dir, strategy, force_drone_index=corrected_index)

    if not np.isfinite(p_std) or p_std < 1e-8 or n_unique <= 2:
        print("Warning: Degenerate probability distribution; returning 0.5")
        return 0.5, evaluator

    # Candidate thresholds
    grid = np.linspace(1e-6, 1 - 1e-6, 1001)
    quantiles = np.quantile(probs, np.linspace(0.01, 0.99, 99))
    candidates = np.unique(np.clip(np.concatenate([grid, quantiles]), 1e-6, 1 - 1e-6))

    # Metrics per threshold
    f1_scores, ba_scores, youden_scores, pos_rates = [], [], [], []

    for thr in candidates:
        preds = (probs >= thr).astype(np.int32)
        tp = np.sum((labels == 1) & (preds == 1))
        tn = np.sum((labels == 0) & (preds == 0))
        fp = np.sum((labels == 0) & (preds == 1))
        fn = np.sum((labels == 1) & (preds == 0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        f1_scores.append((2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0)
        ba_scores.append(0.5 * (recall + specificity))
        youden_scores.append(recall + specificity - 1.0)
        pos_rates.append(np.mean(preds))
    #Removed balanced acc since same as youden. And we dont need the interpretability.
    metrics_map = {"f1": np.array(f1_scores, dtype=np.float64),
                   "youden": np.array(youden_scores, dtype=np.float64)}
    metric = metrics_map.get(strategy, metrics_map["f1"])
    if strategy not in metrics_map:
        print(f"Warning: Unknown strategy '{strategy}', falling back to f1")

    # Replace non-finite with -inf so they won't win
    metric = np.where(np.isfinite(metric), metric, -np.inf)
    if not np.isfinite(metric).any() or np.all(metric == -np.inf):
        print("Warning: Metric evaluation failed; returning default 0.5")
        return 0.5, evaluator

    best_score = float(np.max(metric))
    best_indices = np.where(np.isclose(metric, best_score, atol=1e-9))[0]

    # Tie-breaking toward prevalence, then median threshold
    best_thresholds = candidates[best_indices]
    best_pos_rates = np.array(pos_rates)[best_indices]

    min_rate_diff = float(np.min(np.abs(best_pos_rates - prevalence)))
    tied_indices = np.where(np.isclose(np.abs(best_pos_rates - prevalence), min_rate_diff, atol=1e-9))[0]

    final_candidates = best_thresholds[tied_indices]
    best_threshold = float(np.median(final_candidates))

    if not (0.0 < best_threshold < 1.0) or not np.isfinite(best_threshold):
        print(f"Warning: Invalid threshold {best_threshold}, using default 0.5")
        return 0.5, evaluator

    print(f"Best threshold ({strategy}): {best_threshold:.6f} (score: {best_score:.4f})")

    # Save threshold with diagnostics
    threshold_file = os.path.join(config.model_path, "CNN_LSTM_calibrated_threshold_chosen.json")
    threshold_data = {
        "threshold": best_threshold, "strategy": strategy, "score": best_score,
        "calibration_files": len(probs),
        "class_counts": {"unknown": n_neg, "drone": n_pos, "prevalence": prevalence},
        "unknown_stats": {"min": n_min, "max": n_max, "mean": n_mean, "std": n_std},
        "drone_stats": {"min": d_min, "max": d_max, "mean": d_mean, "std": d_std},
        "all_stats": {"min": p_min, "max": p_max, "mean": p_mean, "std": p_std, "unique": n_unique},
        "calibration_date": datetime.datetime.now().isoformat(),
        "drone_class_index": evaluator.drone_class_index
    }

    try:
        with open(threshold_file, 'w') as f:
            json.dump(threshold_data, f, indent=2)
        print(f"Threshold saved to: {threshold_file}")
    except Exception as e:
        print(f"Warning: Failed to save threshold file: {e}")



    return best_threshold, evaluator


def print_dataset_results(dataset_name: str, results: List[Dict]):
    """Print evaluation results for a dataset (matching ResNet format)."""
    if not results:
        print(f"No results for dataset {dataset_name}")
        return

    # Calculate metrics
    true_labels = [r["true_label"] for r in results]
    predicted_labels = [r["predicted_label"] for r in results]
    probabilities = [r["drone_probability"] for r in results]

    # Basic metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, average="binary", zero_division=0
    )

    # ROC AUC
    try:
        auc = roc_auc_score(true_labels, probabilities)
    except ValueError:
        auc = 0.0

    # Confusion matrix and class distribution
    cm = confusion_matrix(true_labels, predicted_labels)
    class_distribution = Counter(true_labels)

    # Get threshold from first result
    threshold_used = results[0].get("aggregation_threshold", 0.5)

    # Print results
    print(f"\n{'=' * 60}")
    print(f"RESULTS - {dataset_name}")
    print(f"{'=' * 60}")
    print(f"Total files processed: {len(results)}")
    print(f"Threshold used: {threshold_used:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Class distribution: {dict(class_distribution)}")
    print(f"Confusion Matrix:")
    print(np.array(cm))

def get_model_paths(model_input: str) -> Dict[str, str]:
    """
    Parses a model input path. Can be a single file or a directory containing multiple models.
    If a directory, it looks for subdirectories (each containing a model) or model files directly.
    """
    models = {}
    if os.path.isfile(model_input):
        # Handle case where a single model file is provided
        model_name = Path(model_input).stem
        models[model_name] = model_input
        return models
    elif os.path.isdir(model_input):
        # Handle case where a directory of models is provided
        for item in os.listdir(model_input):
            item_path = os.path.join(model_input, item)
            if os.path.isdir(item_path):
                # Use the subdirectory name as the model name (e.g., "run-1")
                models[item] = item_path
            elif item.endswith(('.pth', '.bin', '.safetensors')):
                # Use the file stem as the model name
                model_name = Path(item).stem
                models[model_name] = item_path
        if not models:
            raise ValueError(f"No models or model subdirectories found in: {model_input}")
        return models
    else:
        raise FileNotFoundError(f"Model path does not exist: {model_input}")


def main():
    """Main evaluation function with argparse and multi-model support."""
    parser = argparse.ArgumentParser(description="Evaluate one or more CNN-LSTM audio classification models.")
    parser.add_argument("--model_path", type=str, required=True,
                      help="Path to a trained model file, or a directory containing multiple model folders (e.g., run-1, run-5).")
    parser.add_argument("--dataset", type=str, default=None,
                      choices=list(DATASETS.keys()),
                      help="Specific dataset to evaluate (if not provided, evaluates all datasets).")
    parser.add_argument("--output_dir", type=str, default=None,
                      help="Base output directory for the evaluation series (defaults to a folder on the desktop).")
    parser.add_argument("--batch_size", type=int, default=8,
                      help="Batch size for chunk processing.")
    parser.add_argument("--hop_duration", type=float, default=1.5,
                      help="Overlap duration between large chunks in seconds (outer 180s).")
    parser.add_argument("--window_duration", type=float, default=1.0,
                      help="Sliding window duration in seconds (default 1.0).")
    parser.add_argument("--window_hop", type=float, default=1.0,
                      help="Sliding window hop in seconds (default 1.0).")
    parser.add_argument("--threshold_strategy", type=str, default="f1",
                        choices=["f1", "youden", "balanced_acc"],
                        help="Strategy for threshold optimization.")
    args = parser.parse_args()

    # --- 1. Set up Base Output Directory ---
    if args.output_dir:
        base_output_dir = Path(args.output_dir)
    else:
        desktop_path = Path.home() / "Desktop"
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_output_dir = desktop_path / f"cnn_lstm_evaluation_series_{timestamp}"
    base_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Base output directory for all models: {base_output_dir}")

    # --- 2. Find Models and Datasets ---
    try:
        model_paths = get_model_paths(args.model_path)
        print(f"Found {len(model_paths)} model(s) to evaluate: {list(model_paths.keys())}")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        exit(1)

    datasets_to_evaluate = [args.dataset] if args.dataset else list(DATASETS.keys())
    print(f"Will evaluate against {len(datasets_to_evaluate)} dataset(s): {datasets_to_evaluate}")

    # --- 3. Main Loop for Each Model ---
    all_models_results = {}
    for model_name, model_path in model_paths.items():
        print(f"\n{'='*100}")
        print(f"EVALUATING MODEL: {model_name} ({list(model_paths.keys()).index(model_name) + 1}/{len(model_paths)})")
        print(f"Path: {model_path}")
        print(f"{'='*100}")

        model_output_dir = base_output_dir / model_name
        model_output_dir.mkdir(exist_ok=True)

        # --- Create and Calibrate a Single Evaluator Per Model ---
        try:
            # Create a single evaluator instance for this model
            eval_config = EvaluationConfig(
                model_path=model_path,
                data_dir="",  # Will be updated per dataset
                output_dir=str(model_output_dir),
                batch_size=args.batch_size,
                model_hop_seconds=args.window_hop,
                model_chunk_duration=args.window_duration,
                overlap_seconds=args.hop_duration
            )
            evaluator = CNNLSTMEvaluator(eval_config)
            evaluator.load_model()

            if isinstance(CALIBRATION_SET, dict):
                calibration_path = CALIBRATION_SET['path']
            elif isinstance(CALIBRATION_SET, str):
                calibration_path = CALIBRATION_SET
            else:
                raise ValueError(f"CALIBRATION_SET must be dict or str, got {type(CALIBRATION_SET)}")

            calibrated_threshold, calibrated_evaluator = calibrate_threshold(
                evaluator.model,
                eval_config,
                calibration_path,
                args.threshold_strategy,
                force_drone_index=0
            )

            evaluator.config.threshold = calibrated_threshold
            evaluator.drone_class_index = calibrated_evaluator.drone_class_index

            if evaluator.drone_class_index == 0:
                # Model uses 0 for drones, 1 for unknown - update global mapping
                global LABEL2ID, ID2LABEL
                LABEL2ID = {"yes_drone": 0, "unknown": 1}
                ID2LABEL = {0: "yes_drone", 1: "unknown"}
                print("Updated label mapping to match model convention: drone=0, unknown=1")
            print(f"\nUsing calibrated threshold: {evaluator.config.threshold:.4f} and drone_class_index: {evaluator.drone_class_index} for all subsequent datasets.")

        except Exception as e:
            print(f"FATAL: Could not load or calibrate model '{model_name}'. Skipping. Error: {e}")
            traceback.print_exc()
            continue  # Skip to the next model

        # --- Inner Loop for Datasets (using the single, calibrated evaluator) ---
        model_dataset_results = {}
        for dataset_name in datasets_to_evaluate:
            dataset_config = DATASETS.get(dataset_name, {})
            dataset_path = dataset_config.get("path")

            if not dataset_path or not os.path.exists(dataset_path):
                print(f"\nSkipping dataset '{dataset_name}': Path not found or not configured.")
                continue

            # The evaluator instance now correctly holds the calibrated threshold and index
            dataset_results = evaluator.evaluate_dataset(dataset_name, dataset_path)
            print_dataset_results(dataset_name, dataset_results)
            model_dataset_results[dataset_name] = dataset_results

        # --- Save all results for the current model ---
        if model_dataset_results:
            evaluator.save_results(model_dataset_results)
            all_models_results[model_name] = model_dataset_results
        else:
            print(f"No datasets were successfully evaluated for model '{model_name}'.")

    print("\nAll model evaluations complete.")

    # --- 4. Final Summary Across All Models ---
    if len(all_models_results) > 1:
        print(f"\n{'='*100}")
        print("CROSS-MODEL SUMMARY")
        print(f"{'='*100}")
        for model_name, datasets in all_models_results.items():
            print(f"\nModel: {model_name}")
            for dataset_name, results in datasets.items():
                if results:
                    accuracy = np.mean([r["true_label"] == r["predicted_label"] for r in results])
                    print(f"  - {dataset_name}: {len(results)} files, Accuracy: {accuracy:.4f}")
            if not datasets:
                print("  - No results recorded.")

    print("\nEvaluation series finished.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        traceback.print_exc()
        exit(1)
