"""
CNN-LSTM Evaluation script.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
import datetime
import librosa
from typing import List, Dict, Tuple
import warnings
from dataclasses import dataclass
import argparse
import json
from sklearn.metrics import precision_recall_curve
from cnn_lstm_model import CNNLSTMModel
from common import SAMPLE_RATE, N_MELS, HOP_LENGTH, DATASETS

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

    def load_audio_file(self, file_path: str) -> Tuple[np.ndarray, float]:
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
            return np.mean(predictions)
        elif method == "max":
            return np.max(predictions)
        elif method == "median":
            return np.median(predictions)
        else:
            return np.mean(predictions)


class CNNLSTMEvaluator:
    """Main evaluator for CNN-LSTM model with large file handling."""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.file_handler = LargeFileHandler(config)
        self.model = None

        print(f"Using device: {self.device}")

    def load_model(self):
        """Load the trained CNN-LSTM model."""
        print(f"Loading model from: {self.config.model_path}")

        # Handle both file paths and directory paths
        model_file_path = self.config.model_path

        # If it's a directory, look for model files inside
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
            for filename in potential_files:
                potential_path = os.path.join(self.config.model_path, filename)
                if os.path.exists(potential_path):
                    model_file_path = potential_path
                    print(f"Found model file: {filename}")
                    model_loaded = True
                    break

            if not model_loaded:
                # List available files to help debug
                available_files = [f for f in os.listdir(self.config.model_path)
                                 if f.endswith(('.pth', '.bin', '.safetensors'))]
                raise FileNotFoundError(
                    f"No recognized model file found in {self.config.model_path}. "
                    f"Available model files: {available_files}. "
                    f"Expected one of: {potential_files}"
                )

        # Load the model file
        try:
            if model_file_path.endswith('.safetensors'):
                # Handle safetensors files
                try:
                    from safetensors.torch import load_file
                    checkpoint = load_file(model_file_path, device=str(self.device))
                    print(f"✓ Successfully loaded safetensors from: {os.path.basename(model_file_path)}")

                    # For safetensors, we use the exact training script parameters
                    model_config = {
                        'num_labels': 2,  # Binary classification - matches training script
                        'hidden_size': 128,  # Default from training script
                        'lstm_layers': 1,  # Default from training script (not 2!)
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
                    # The evaluation model signature differs from training checkpoint metadata; use defaults
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

        # Print training info if available (only for PyTorch checkpoints)
        if not model_file_path.endswith('.safetensors'):
            if 'training_info' in checkpoint:
                training_info = checkpoint['training_info']
                if 'best_val_acc' in training_info:
                    print(f"Best training accuracy: {training_info['best_val_acc']:.2f}%")
            elif 'best_val_acc' in checkpoint:
                print(f"Best training accuracy: {checkpoint['best_val_acc']:.2f}%")

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
                drone_probs = probabilities[:, 1].cpu().numpy()  # Probability of drone class
                predictions.extend(drone_probs.tolist())

        return predictions

    def evaluate_file(self, file_path: str, true_label: int) -> Dict:
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

        # Walk through dataset directory
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
                    file_path = os.path.join(root, file)

                    # Determine label from directory structure
                    # Assuming structure: dataset/class_name/files
                    class_name = os.path.basename(root).lower()
                    if 'drone' in class_name or 'yes' in class_name or '1' in class_name:
                        true_label = 1
                    else:
                        true_label = 0

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

def calibrate_threshold(model: nn.Module, config: EvaluationConfig) -> float:
    """Calibrate threshold using validation data with the same 1s windowing logic."""
    print("\n=== Threshold Calibration ===")

    evaluator = CNNLSTMEvaluator(config)
    evaluator.model = model  # Use existing model

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

                # Determine label from directory structure
                class_name = os.path.basename(root).lower()
                if 'drone' in class_name or 'yes' in class_name or '1' in class_name:
                    true_label = 1
                else:
                    true_label = 0

                # Use the same evaluation path to compute file-level probability
                result = evaluator.evaluate_file(file_path, true_label)
                if result is not None:
                    all_probs.append(float(result["drone_probability"]))
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
        print(f"Best threshold: {best_threshold:.6f} (F1: {f1_scores[best_idx]:.4f})")

        # Save threshold for future use
        threshold_file = os.path.join(config.model_path, "calibrated_threshold.json")
        threshold_data = {
            "threshold": float(best_threshold),
            "f1_score": float(f1_scores[best_idx]),
            "calibration_files": len(all_probs),
            "calibration_date": datetime.datetime.now().isoformat()
        }
        with open(threshold_file, 'w') as f:
            json.dump(threshold_data, f, indent=2)
        print(f"Threshold saved to: {threshold_file}")

        return float(best_threshold)

    except Exception as e:
        print(f"Warning: Threshold calibration failed ({e}), using default threshold 0.5")
        return 0.5

def main():
    """Main evaluation function with argparse support."""
    parser = argparse.ArgumentParser(description="Evaluate CNN-LSTM audio classification model")
    parser.add_argument("--model_path", type=str, required=True,
                      help="Path to trained model directory or .pth file")
    parser.add_argument("--dataset", type=str, default=None,
                      choices=list(DATASETS.keys()),
                      help="Dataset name to evaluate (if not provided, evaluates all datasets)")
    parser.add_argument("--output_dir", type=str, default=None,
                      help="Output directory for results (defaults to timestamped folder on desktop)")
    parser.add_argument("--batch_size", type=int, default=8,
                      help="Batch size for chunk processing")
    parser.add_argument("--hop_duration", type=float, default=1.5,
                      help="Overlap duration between large chunks in seconds (outer 180s)")
    # Optional: sliding window parameters
    parser.add_argument("--window_duration", type=float, default=1.0,
                      help="Sliding window duration in seconds (default 1.0)")
    parser.add_argument("--window_hop", type=float, default=1.0,
                      help="Sliding window hop in seconds (default 1.0; set <duration for overlap)")
    parser.add_argument("--calibrate", action="store_true",
                      help="Calibrate threshold on this dataset")
    parser.add_argument("--threshold", type=float, default=None,
                      help="Manual threshold override")

    args = parser.parse_args()

    # Create timestamped output directory on desktop if not specified
    if args.output_dir is None:
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.dataset:
            args.output_dir = os.path.join(desktop_path, f"cnn_lstm_evaluation_{args.dataset}_{timestamp}")
        else:
            args.output_dir = os.path.join(desktop_path, f"cnn_lstm_evaluation_all_datasets_{timestamp}")

    # Determine which datasets to evaluate
    if args.dataset:
        datasets_to_evaluate = [args.dataset]
        print(f"Evaluating single dataset: {args.dataset}")
    else:
        datasets_to_evaluate = list(DATASETS.keys())
        print(f"Evaluating all {len(datasets_to_evaluate)} datasets")

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
            print(f" Skipping {dataset_name}: Dataset path does not exist: {dataset_config['path']}")
            continue

        print(f"\n{'='*80}")
        print(f"EVALUATING DATASET: {dataset_name} ({datasets_to_evaluate.index(dataset_name) + 1}/{len(datasets_to_evaluate)})")
        print(f"{'='*80}")
        print(f"Description: {dataset_config['description']}")

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
                    print(f"✓ Using saved calibrated threshold: {threshold:.6f}")
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
            overlap_seconds=args.hop_duration,
            threshold=threshold,
            model_chunk_duration=args.window_duration,
            model_hop_seconds=args.window_hop,
        )

        print(f"Max file length: {config.max_file_length}s")
        print(f"Chunk duration: {config.chunk_duration}s")
        print(f"Overlap: {config.overlap_seconds}s")
        print(f"Batch size: {config.batch_size}")
        print(f"Window: {config.model_chunk_duration}s, Hop: {config.model_hop_seconds}s")
        print(f"Threshold: {config.threshold:.6f}")

        try:
            # Initialize evaluator and load model
            evaluator = CNNLSTMEvaluator(config)
            evaluator.load_model()

            # Calibrate threshold if requested (only for first dataset or single dataset)
            if args.calibrate and (args.dataset or dataset_name == datasets_to_evaluate[0]):
                calibrated_threshold = calibrate_threshold(evaluator.model, config)
                config.threshold = calibrated_threshold
                evaluator.config.threshold = calibrated_threshold

            # Evaluate dataset
            results = evaluator.evaluate_dataset(dataset_name, dataset_config["path"])
            all_dataset_results[dataset_name] = results

            # Print results for this dataset
            if results:
                whole_files = sum([1 for r in results if r["split"] == "whole_file"])
                split_files = len(results) - whole_files
                accuracy = np.mean([r["true_label"] == r["predicted_label"] for r in results])

                print(f"\n{'='*60}")
                print(f"RESULTS - {dataset_name}")
                print(f"{'='*60}")
                print(f"Total files processed: {len(results)}")
                print(f"Whole files: {whole_files}")
                print(f"Split files: {split_files}")
                print(f"Threshold used: {config.threshold:.6f}")
                print(f"Accuracy: {accuracy:.4f}")
                if split_files > 0:
                    avg_chunks = np.mean([r["num_chunks"] for r in results if r["num_chunks"] > 1])
                    print(f"Avg chunks per split file: {avg_chunks:.1f}")

        except Exception as e:
            print(f"Evaluation failed for {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save all results
    if all_dataset_results:
        # Create a temporary evaluator just for saving results
        temp_config = EvaluationConfig(
            model_path=args.model_path,
            data_dir="",
            output_dir=args.output_dir,
            threshold=threshold
        )
        temp_evaluator = CNNLSTMEvaluator(temp_config)
        output_dir = temp_evaluator.save_results(all_dataset_results)

        # Create summary report if multiple datasets were evaluated
        if len(all_dataset_results) > 1:
            print(f"\n{'='*80}")
            print(f"SUMMARY ACROSS ALL DATASETS")
            print(f"{'='*80}")

            for dataset_name, results in all_dataset_results.items():
                if results:
                    whole_files = sum([1 for r in results if r["split"] == "whole_file"])
                    split_files = len(results) - whole_files
                    accuracy = np.mean([r["true_label"] == r["predicted_label"] for r in results])
                    print(f"{dataset_name}:")
                    print(f"  Total files: {len(results)}")
                    print(f"  Accuracy: {accuracy:.4f}")
                    print(f"  Whole files: {whole_files}, Split files: {split_files}")

        print(f"\n✓ Evaluation completed successfully!")
        print(f" All results saved to: {output_dir}")
    else:
        print("No datasets were successfully evaluated!")

if __name__ == "__main__":
    main()
