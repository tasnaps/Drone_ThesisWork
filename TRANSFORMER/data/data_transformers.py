import os
import torch
from datasets import load_dataset, Audio, DatasetDict
from transformers import Wav2Vec2FeatureExtractor, DataCollatorWithPadding
from typing import Dict, List, Any
import numpy as np
from audiomentations import AddGaussianNoise, BandPassFilter, Compose


# 1) load the pretrained feature extractor
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    "ALM/wav2vec2-large-audioset"
)

# Audio Augmentation Configuration
class AugmentationConfig:
    """Configuration class for audio augmentations"""

    # Individual augmentation settings
    GAUSSIAN_NOISE = {
        "enabled": True,
        "min_amplitude": 0.001,
        "max_amplitude": 0.05,
        "p": 0.5
    }

    BANDPASS_FILTER = {
        "enabled": True,
        "min_center_freq": 200,
        "max_center_freq": 4000,
        "min_bandwidth_fraction": 0.1,
        "max_bandwidth_fraction": 0.8,
        "p": 0.3
    }

    # Augmentation modes
    AUGMENTATION_MODE = "none"  # Options: "none", "gaussian_only", "bandpass_only", "all"

# Audio Length Configuration
class AudioLengthConfig:
    """Configuration class for audio length modification during training"""

    # Audio shortening settings
    SHORTEN_AUDIO = True  # Enable/disable audio shortening
    MIN_LENGTH_RATIO = 0.5  # Minimum length as fraction of original (50%)
    MAX_LENGTH_RATIO = 0.75  # Maximum length as fraction of original (75%)
    RANDOM_CROP = True  # If True, randomly crop from audio. If False, take from beginning

    # For debugging and analysis
    PRESERVE_ORIGINAL_LENGTH_INFO = True  # Keep track of original lengths for analysis

def create_augmentation_pipeline(mode="all"):
    """
    Create augmentation pipeline based on specified mode

    Args:
        mode: str - "none", "gaussian_only", "bandpass_only", "all"

    Returns:
        Augmentation transform or None
    """
    if mode == "none":
        return None

    transforms = []

    if mode in ["gaussian_only", "all"] and AugmentationConfig.GAUSSIAN_NOISE["enabled"]:
        transforms.append(
            AddGaussianNoise(
                min_amplitude=AugmentationConfig.GAUSSIAN_NOISE["min_amplitude"],
                max_amplitude=AugmentationConfig.GAUSSIAN_NOISE["max_amplitude"],
                p=AugmentationConfig.GAUSSIAN_NOISE["p"]
            )
        )

    if mode in ["bandpass_only", "all"] and AugmentationConfig.BANDPASS_FILTER["enabled"]:
        transforms.append(
            BandPassFilter(
                min_center_freq=AugmentationConfig.BANDPASS_FILTER["min_center_freq"],
                max_center_freq=AugmentationConfig.BANDPASS_FILTER["max_center_freq"],
                min_bandwidth_fraction=AugmentationConfig.BANDPASS_FILTER["min_bandwidth_fraction"],
                max_bandwidth_fraction=AugmentationConfig.BANDPASS_FILTER["max_bandwidth_fraction"],
                p=AugmentationConfig.BANDPASS_FILTER["p"]
            )
        )

    if not transforms:
        return None

    # If only one transform, return it directly
    if len(transforms) == 1:
        return transforms[0]

    # If multiple transforms, compose them
    return Compose(transforms)

# Initialize audio augmentation with current config
augment_transform = create_augmentation_pipeline(AugmentationConfig.AUGMENTATION_MODE)

def update_augmentation_mode(new_mode="all"):
    """
    Update the global augmentation pipeline with a new mode

    Args:
        new_mode: str - "none", "gaussian_only", "bandpass_only", "all"
    """
    global augment_transform
    AugmentationConfig.AUGMENTATION_MODE = new_mode
    augment_transform = create_augmentation_pipeline(new_mode)

    if augment_transform is None:
        print(f"🔇 Augmentation disabled (mode: {new_mode})")
    else:
        active_augmentations = []
        if new_mode in ["gaussian_only", "all"] and AugmentationConfig.GAUSSIAN_NOISE["enabled"]:
            active_augmentations.append("GaussianNoise")
        if new_mode in ["bandpass_only", "all"] and AugmentationConfig.BANDPASS_FILTER["enabled"]:
            active_augmentations.append("BandPassFilter")
        print(f"🎵 Augmentation updated (mode: {new_mode}) - Active: {', '.join(active_augmentations)}")

def print_augmentation_info():
    """Print current augmentation configuration"""
    print("\n=== Audio Augmentation Configuration ===")
    print(f"Mode: {AugmentationConfig.AUGMENTATION_MODE}")
    print("\nGaussian Noise:")
    print(f"  Enabled: {AugmentationConfig.GAUSSIAN_NOISE['enabled']}")
    print(f"  Amplitude: {AugmentationConfig.GAUSSIAN_NOISE['min_amplitude']}-{AugmentationConfig.GAUSSIAN_NOISE['max_amplitude']}")
    print(f"  Probability: {AugmentationConfig.GAUSSIAN_NOISE['p']}")
    print("\nBandPass Filter:")
    print(f"  Enabled: {AugmentationConfig.BANDPASS_FILTER['enabled']}")
    print(f"  Center Frequency: {AugmentationConfig.BANDPASS_FILTER['min_center_freq']}-{AugmentationConfig.BANDPASS_FILTER['max_center_freq']} Hz")
    print(f"  Bandwidth Fraction: {AugmentationConfig.BANDPASS_FILTER['min_bandwidth_fraction']}-{AugmentationConfig.BANDPASS_FILTER['max_bandwidth_fraction']}")
    print(f"  Probability: {AugmentationConfig.BANDPASS_FILTER['p']}")

    active_augs = []
    if AugmentationConfig.AUGMENTATION_MODE != "none":
        if AugmentationConfig.AUGMENTATION_MODE in ["gaussian_only", "all"] and AugmentationConfig.GAUSSIAN_NOISE["enabled"]:
            active_augs.append("GaussianNoise")
        if AugmentationConfig.AUGMENTATION_MODE in ["bandpass_only", "all"] and AugmentationConfig.BANDPASS_FILTER["enabled"]:
            active_augs.append("BandPassFilter")

    print(f"\nCurrently Active: {', '.join(active_augs) if active_augs else 'None'}")
    print("="*45)

# Print initial configuration
# print_augmentation_info()

# 2) split just like before
def load_and_split(
    data_dir: str,
    val_size: float = 0.2,
    seed: int = 42
) -> DatasetDict:
    raw = load_dataset("audiofolder", data_dir=data_dir)
    tmp = raw["train"].train_test_split(
        test_size=val_size,
        seed=seed,
        stratify_by_column="label"
    )
    return DatasetDict({
        "train": tmp["train"],
        "validation": tmp["test"],
    })

# 3) Simplified preprocessing function for individual examples (like model.py)
def preprocess_single_example(example, augment=False, shorten_for_training=False):
    """
    Process a single audio example into input_values.

    Args:
        example: Dataset example with audio and label
        augment: Whether to apply audio augmentation
        shorten_for_training: Whether to apply audio shortening
    """
    audio = example["audio"]
    array = audio["array"]
    sr = audio["sampling_rate"]

    if array is None or len(array) == 0:
        raise ValueError("Empty audio detected in example")

    # Apply audio shortening if requested
    if shorten_for_training and AudioLengthConfig.SHORTEN_AUDIO:
        array, _ = shorten_audio_array(array, sr)

    # Apply augmentation if requested
    if augment and augment_transform is not None:
        array = augment_transform(samples=array, sample_rate=sr)

    # Extract features
    inputs = feature_extractor(
        array,
        sampling_rate=sr,
        return_tensors="np"
    )

    # Return only essential data - no metadata
    return {
        "input_values": inputs.input_values[0],
        "labels": example["label"]
    }

# 4) Fast preprocessing function that processes all data upfront
def prepare_dataset_fast(ds, augment=False, shorten_for_training=False):
    """
    Fast preprocessing that mimics model.py approach with optional audio shortening:
    1. Cast audio column to 16kHz
    2. Process all audio files in parallel upfront (with optional shortening)
    3. Remove raw audio columns
    4. Set format to torch tensors

    Args:
        ds: Dataset to process
        augment: Whether to apply audio augmentation (typically True for training, False for val/test)
        shorten_for_training: Whether to apply audio shortening (typically True for training, False for val/test)
    """
    # Cast to 16kHz (same as model.py)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    # Determine description based on enabled features
    desc_parts = []
    if shorten_for_training and AudioLengthConfig.SHORTEN_AUDIO:
        desc_parts.append("with audio shortening")
    if augment:
        desc_parts.append("with augmentation")
    if not desc_parts:
        desc_parts.append("without modifications")

    description = f"Processing audio files {' and '.join(desc_parts)}"

    # Process all audio files in parallel (like model.py)
    ds = ds.map(
        lambda example: preprocess_single_example(
            example,
            augment=augment,
            shorten_for_training=shorten_for_training
        ),
        remove_columns=["audio", "label"],  # Remove raw data
        num_proc=os.cpu_count(),  # Parallel processing
        load_from_cache_file=True,  # Cache results
        desc=description
    )

    # Determine which columns to set as torch tensors
    torch_columns = ["input_values", "labels"]

    # Add length metadata columns if they exist (only if preserve_original_length_info is enabled)
    #if AudioLengthConfig.PRESERVE_ORIGINAL_LENGTH_INFO and len(ds) > 0:
    #    # Check if length metadata columns exist in the first example
    #    first_example = ds[0]
    #    length_cols = [col for col in first_example.keys() if col.startswith("length_")]
    #    torch_columns.extend(length_cols)

    # Convert to torch tensors for fast training
    ds.set_format(type="torch", columns=torch_columns)

    return ds



# Create the data collator instance (same as the fast model.py)
data_collator = DataCollatorWithPadding(feature_extractor, padding=True)

def update_audio_length_config(shorten_audio=None, min_ratio=None, max_ratio=None, random_crop=None):
    """
    Update the global audio length configuration

    Args:
        shorten_audio: bool - Enable/disable audio shortening
        min_ratio: float - Minimum length ratio (0.0 to 1.0)
        max_ratio: float - Maximum length ratio (0.0 to 1.0)
        random_crop: bool - Whether to use random cropping vs start cropping
    """
    if shorten_audio is not None:
        AudioLengthConfig.SHORTEN_AUDIO = shorten_audio
    if min_ratio is not None:
        AudioLengthConfig.MIN_LENGTH_RATIO = min_ratio
    if max_ratio is not None:
        AudioLengthConfig.MAX_LENGTH_RATIO = max_ratio
    if random_crop is not None:
        AudioLengthConfig.RANDOM_CROP = random_crop

    print_audio_length_info()

def print_audio_length_info():
    """Print current audio length configuration"""
    print("\n=== Audio Length Configuration ===")
    print(f"Shorten Audio: {AudioLengthConfig.SHORTEN_AUDIO}")
    if AudioLengthConfig.SHORTEN_AUDIO:
        print(f"Length Range: {AudioLengthConfig.MIN_LENGTH_RATIO*100:.0f}% - {AudioLengthConfig.MAX_LENGTH_RATIO*100:.0f}% of original")
        print(f"Cropping Mode: {'Random' if AudioLengthConfig.RANDOM_CROP else 'From start'}")
        print(f"Preserve Original Info: {AudioLengthConfig.PRESERVE_ORIGINAL_LENGTH_INFO}")
    else:
        print("Audio length modification disabled - using full audio files")
    print("="*40)


def shorten_audio_array(audio_array, sample_rate=16000):
    """
    Shorten audio array based on current configuration.
    Always returns consistent metadata structure.

    Args:
        audio_array: numpy array of audio samples
        sample_rate: sampling rate (default 16kHz)

    Returns:
        tuple: (audio_array, metadata_dict)
    """
    original_length = len(audio_array)

    # Initialize consistent metadata structure
    metadata = {
        'original_length': original_length,
        'original_duration_s': original_length / sample_rate,
        'shortened': False,
        'target_ratio': 1.0,
        'length_ratio': 1.0,
        'crop_start': 0,
        'crop_end': original_length
    }

    # Return early if shortening is disabled
    if not AudioLengthConfig.SHORTEN_AUDIO:
        metadata['final_length'] = original_length
        metadata['final_duration_s'] = original_length / sample_rate
        return audio_array, metadata

    # Calculate target length
    length_ratio = np.random.uniform(
        AudioLengthConfig.MIN_LENGTH_RATIO,
        AudioLengthConfig.MAX_LENGTH_RATIO
    )
    target_length = int(original_length * length_ratio)

    # Ensure minimum length (1 second)
    min_samples = sample_rate
    target_length = max(target_length, min_samples)

    # If target >= original, return original
    if target_length >= original_length:
        metadata['final_length'] = original_length
        metadata['final_duration_s'] = original_length / sample_rate
        return audio_array, metadata

    # Determine crop position
    if AudioLengthConfig.RANDOM_CROP:
        max_start = original_length - target_length
        crop_start = np.random.randint(0, max_start + 1)
    else:
        crop_start = 0

    crop_end = crop_start + target_length
    shortened_array = audio_array[crop_start:crop_end]

    # Update metadata with actual values
    # Create metadata
    metadata = {
        'original_length': original_length,
        'final_length': len(shortened_array),
        'length_ratio': len(shortened_array) / original_length,
        'crop_start': crop_start,
        'crop_end': crop_end,
        'shortened': True,
        'target_ratio': length_ratio,
        'original_duration_s': original_length / sample_rate,
        'final_duration_s': len(shortened_array) / sample_rate
    }

    return shortened_array, metadata