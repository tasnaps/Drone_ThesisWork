#!/usr/bin/env python3
"""
Configuration management for transformer model evaluation.
This module centralizes all configuration parameters and provides environment-based overrides.
"""

import os
from dataclasses import dataclass
from typing import Optional
from pathlib import Path



@dataclass
class ModelConfigForEval:
    """Configuration for model paths and basic settings. This is for eval only"""

    model_path: str = "C:/Users/tapio/PycharmProjects/gradu/TransformerModel/transformer_final/checkpoint-2930"
    sample_rate: int = 16000

    @classmethod
    def from_env(cls):
        """Create config from environment variables."""
        return cls(
            model_path=os.getenv('MODEL_PATH', cls.model_path),
            sample_rate=int(os.getenv('SAMPLE_RATE', cls.sample_rate))
        )

    def validate(self):
        """Validate model configuration."""
        errors = []
        warnings = []

        # Check if model path exists
        model_path = Path(self.model_path)
        if not model_path.exists():
            errors.append(f"Model path does not exist: {self.model_path}")
        elif not model_path.is_dir():
            errors.append(f"Model path is not a directory: {self.model_path}")
        else:
            # Check for required model files
            required_files = ['config.json', 'pytorch_model.bin']
            safetensors_file = model_path / 'model.safetensors'
            pytorch_file = model_path / 'pytorch_model.bin'

            if not (safetensors_file.exists() or pytorch_file.exists()):
                errors.append(f"No model weights found in {self.model_path} (looking for model.safetensors or pytorch_model.bin)")

            config_file = model_path / 'config.json'
            if not config_file.exists():
                errors.append(f"Model configuration file not found: {config_file}")

        # Validate sample rate
        if self.sample_rate <= 0:
            errors.append(f"Sample rate must be positive, got: {self.sample_rate}")
        elif self.sample_rate not in [8000, 16000, 22050, 44100, 48000]:
            warnings.append(f"Unusual sample rate: {self.sample_rate}. Common rates are 16000, 22050, 44100, 48000")

        return errors, warnings


#note that we shouldn't use the clip based evaluation until its properly tested
@dataclass
class ClipEvaluationConfig:
    """Configuration specific to clip-based evaluation."""
    clip_duration: float = 1.0
    batch_size: int = 4
    max_clips_per_dataset: int = 10000
    threshold: float = 0.006  # Updated from 0.0028 to optimized threshold
    aggregation_threshold: float = 0.006  # Updated from 0.2 to optimized threshold

    @property
    def clip_samples(self) -> int:
        """Calculate clip samples based on duration and sample rate."""
        return int(ModelConfigForEval().sample_rate * self.clip_duration)

    @classmethod
    def from_env(cls):
        """Create config from environment variables."""
        return cls(
            clip_duration=float(os.getenv('CLIP_DURATION', cls.clip_duration)),
            batch_size=int(os.getenv('CLIP_BATCH_SIZE', cls.batch_size)),
            max_clips_per_dataset=int(os.getenv('MAX_CLIPS_PER_DATASET', cls.max_clips_per_dataset)),
            threshold=float(os.getenv('CLIP_THRESHOLD', cls.threshold)),
            aggregation_threshold=float(os.getenv('AGGREGATION_THRESHOLD', cls.aggregation_threshold))
        )

    def validate(self):
        """Validate clip evaluation configuration."""
        errors = []
        warnings = []

        # Validate clip duration
        if self.clip_duration <= 0:
            errors.append(f"Clip duration must be positive, got: {self.clip_duration}")
        elif self.clip_duration < 0.5:
            warnings.append(f"Very short clip duration: {self.clip_duration}s. May not capture enough audio context")
        elif self.clip_duration > 10:
            warnings.append(f"Very long clip duration: {self.clip_duration}s. May increase memory usage significantly")

        # Validate batch size
        if self.batch_size <= 0:
            errors.append(f"Batch size must be positive, got: {self.batch_size}")
        elif self.batch_size > 64:
            warnings.append(f"Large batch size: {self.batch_size}. May cause memory issues")

        # Validate max clips
        if self.max_clips_per_dataset <= 0:
            errors.append(f"Max clips per dataset must be positive, got: {self.max_clips_per_dataset}")
        elif self.max_clips_per_dataset < 100:
            warnings.append(f"Low max clips per dataset: {self.max_clips_per_dataset}. May not provide reliable statistics")

        # Validate thresholds
        if not (0 <= self.threshold <= 1):
            errors.append(f"Threshold must be between 0 and 1, got: {self.threshold}")
        if not (0 <= self.aggregation_threshold <= 1):
            errors.append(f"Aggregation threshold must be between 0 and 1, got: {self.aggregation_threshold}")

        return errors, warnings


@dataclass
class FileEvaluationConfig:
    """Settings for the whole file evaluation."""
    batch_size: int = 6
    threshold: float = 0.006
    large_file_threshold: float = 45.0  # processed alone
    very_large_threshold: float = 180.0  # Files longer than this are processed in chunks
    max_file_length: float = 180.0  # Match very_large_threshold
    max_file_size_mb: int = 100  # Max size before chunking
    overlap_seconds: float = 1.5  # Overlap between chunks

    @classmethod
    def from_env(cls):
        """Create config from environment variables."""
        return cls(
            batch_size=int(os.getenv('FILE_BATCH_SIZE', cls.batch_size)),
            threshold=float(os.getenv('FILE_THRESHOLD', cls.threshold)),
            large_file_threshold=float(os.getenv('LARGE_FILE_THRESHOLD', cls.large_file_threshold)),
            very_large_threshold=float(os.getenv('VERY_LARGE_THRESHOLD', cls.very_large_threshold)),
            max_file_length=float(os.getenv('MAX_FILE_LENGTH', cls.max_file_length)),
            max_file_size_mb=int(os.getenv('MAX_FILE_SIZE_MB', cls.max_file_size_mb)),
            overlap_seconds=float(os.getenv('OVERLAP_SECONDS', cls.overlap_seconds))
        )

    def validate(self):
        """Validate file evaluation configuration."""
        errors = []
        warnings = []

        # Validate batch size
        if self.batch_size <= 0:
            errors.append(f"Batch size must be positive, got: {self.batch_size}")
        elif self.batch_size > 8:
            warnings.append(f"Large batch size for file evaluation: {self.batch_size}. May cause memory issues with large files")

        # Validate threshold
        if not (0 <= self.threshold <= 1):
            errors.append(f"Threshold must be between 0 and 1, got: {self.threshold}")

        # Validate file thresholds
        if self.large_file_threshold <= 0:
            errors.append(f"Large file threshold must be positive, got: {self.large_file_threshold}")
        if self.very_large_threshold <= self.large_file_threshold:
            errors.append(f"Very large threshold ({self.very_large_threshold}) must be greater than large threshold ({self.large_file_threshold})")
        if self.max_file_length <= self.very_large_threshold:
            warnings.append(f"Max file length ({self.max_file_length}) is close to very large threshold ({self.very_large_threshold})")

        # Validate file size
        if self.max_file_size_mb <= 0:
            errors.append(f"Max file size must be positive, got: {self.max_file_size_mb}")
        elif self.max_file_size_mb > 1000:
            warnings.append(f"Very large max file size: {self.max_file_size_mb}MB. May cause memory issues")

        # Validate overlap
        if self.overlap_seconds < 0:
            errors.append(f"Overlap seconds must be non-negative, got: {self.overlap_seconds}")
        elif self.overlap_seconds > 10:
            warnings.append(f"Large overlap: {self.overlap_seconds}s. May cause excessive redundancy")

        return errors, warnings

@dataclass
class OutputConfig:
    """Configuration for output directories and file naming."""
    base_output_dir: str = "./eval_results"
    plot_dpi: int = 300
    save_individual_csvs: bool = True
    save_combined_csvs: bool = True
    save_plots: bool = True

    @classmethod
    def from_env(cls):
        """Create config from environment variables."""
        return cls(
            base_output_dir=os.getenv('OUTPUT_DIR', cls.base_output_dir),
            plot_dpi=int(os.getenv('PLOT_DPI', cls.plot_dpi)),
            save_individual_csvs=os.getenv('SAVE_INDIVIDUAL_CSVS', 'true').lower() == 'true',
            save_combined_csvs=os.getenv('SAVE_COMBINED_CSVS', 'true').lower() == 'true',
            save_plots=os.getenv('SAVE_PLOTS', 'true').lower() == 'true'
        )

    def validate(self):
        """Validate output configuration."""
        errors = []
        warnings = []

        # Check if output directory can be created
        output_path = Path(self.base_output_dir)
        try:
            output_path.mkdir(parents=True, exist_ok=True)
            # Test write permissions
            test_file = output_path / 'test_write.tmp'
            test_file.touch()
            test_file.unlink()
        except PermissionError:
            errors.append(f"No write permission for output directory: {self.base_output_dir}")
        except Exception as e:
            errors.append(f"Cannot create output directory {self.base_output_dir}: {e}")

        # Validate plot DPI
        if self.plot_dpi <= 0:
            errors.append(f"Plot DPI must be positive, got: {self.plot_dpi}")
        elif self.plot_dpi < 72:
            warnings.append(f"Low plot DPI: {self.plot_dpi}. Images may appear pixelated")
        elif self.plot_dpi > 600:
            warnings.append(f"Very high plot DPI: {self.plot_dpi}. May create large image files")

        return errors, warnings

class EvaluationConfig:
    """Main configuration class that combines all config sections."""

    def __init__(self, model_config: Optional[ModelConfigForEval] = None,
                 clip_config: Optional[ClipEvaluationConfig] = None,
                 file_config: Optional[FileEvaluationConfig] = None,
                 output_config: Optional[OutputConfig] = None):
        self.model = model_config or ModelConfigForEval.from_env()
        self.clip = clip_config or ClipEvaluationConfig.from_env()
        self.file = file_config or FileEvaluationConfig.from_env()
        self.output = output_config or OutputConfig.from_env()

    @classmethod
    def from_env(cls):
        """Create complete config from environment variables."""
        return cls(
            ModelConfigForEval.from_env(),
            ClipEvaluationConfig.from_env(),
            FileEvaluationConfig.from_env(),
            OutputConfig.from_env()
        )

    @classmethod
    def default(cls):
        """Create config with all default values."""
        return cls()

    def validate(self):
        """Validate entire configuration."""
        all_errors = []
        all_warnings = []

        # Validate each section
        for config_name, config_obj in [
            ('Model', self.model),
            ('Clip', self.clip),
            ('File', self.file),
            ('Output', self.output)
        ]:
            errors, warnings = config_obj.validate()
            all_errors.extend([f"{config_name}: {error}" for error in errors])
            all_warnings.extend([f"{config_name}: {warning}" for warning in warnings])

        # Validate dataset paths
        dataset_errors, dataset_warnings, accessible_datasets = validate_dataset_paths()
        all_errors.extend([f"Dataset: {error}" for error in dataset_errors])
        all_warnings.extend([f"Dataset: {warning}" for warning in dataset_warnings])

        return {
            'errors': all_errors,
            'warnings': all_warnings,
            'accessible_datasets': accessible_datasets,
            'is_valid': len(all_errors) == 0
        }

    def to_dict(self):
        """Convert config to dictionary for logging/debugging."""
        return {
            'model': {
                'model_path': self.model.model_path,
                'sample_rate': self.model.sample_rate
            },
            'clip': {
                'clip_duration': self.clip.clip_duration,
                'batch_size': self.clip.batch_size,
                'max_clips_per_dataset': self.clip.max_clips_per_dataset,
                'threshold': self.clip.threshold,
                'aggregation_threshold': self.clip.aggregation_threshold
            },
            'file': {
                'batch_size': self.file.batch_size,
                'threshold': self.file.threshold,
                'large_file_threshold': self.file.large_file_threshold,
                'very_large_threshold': self.file.very_large_threshold,
                'max_file_length': self.file.max_file_length,
                'max_file_size_mb': self.file.max_file_size_mb,
                'overlap_seconds': self.file.overlap_seconds
            },
            'output': {
                'base_output_dir': self.output.base_output_dir,
                'plot_dpi': self.output.plot_dpi,
                'save_individual_csvs': self.output.save_individual_csvs,
                'save_combined_csvs': self.output.save_combined_csvs,
                'save_plots': self.output.save_plots
            }
        }

def validate_dataset_paths():
    """Validate dataset paths from dataset configuration."""
    from dataset_config import ENHANCED_DATASETS

    errors = []
    warnings = []
    accessible_datasets = []

    for dataset_name, config in ENHANCED_DATASETS.items():
        dataset_path = Path(config['path'])

        if not dataset_path.exists():
            errors.append(f"Dataset '{dataset_name}' path does not exist: {config['path']}")
        elif not dataset_path.is_dir():
            errors.append(f"Dataset '{dataset_name}' path is not a directory: {config['path']}")
        else:
            # Check for audio files
            audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
            audio_files = []

            try:
                for ext in audio_extensions:
                    audio_files.extend(list(dataset_path.rglob(f'*{ext}')))

                if not audio_files:
                    warnings.append(f"Dataset '{dataset_name}' contains no audio files")
                elif len(audio_files) < 10:
                    warnings.append(f"Dataset '{dataset_name}' has very few audio files ({len(audio_files)})")
                else:
                    accessible_datasets.append(dataset_name)

            except PermissionError:
                errors.append(f"No read permission for dataset '{dataset_name}': {config['path']}")
            except Exception as e:
                warnings.append(f"Error scanning dataset '{dataset_name}': {e}")

    return errors, warnings, accessible_datasets


def get_config() -> EvaluationConfig:
    """Get the global configuration instance."""
    return EvaluationConfig.from_env()


def print_config(config: EvaluationConfig):
    """Print configuration for debugging."""
    print("=== Evaluation Configuration ===")
    import json
    print(json.dumps(config.to_dict(), indent=2))
    print("=" * 35)
