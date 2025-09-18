#!/usr/bin/env python3
"""
Audio processing utilities for transformer model evaluation.
This module contains functions for audio loading, preprocessing, and chunking operations.
"""

import numpy as np
from datasets import load_dataset, Audio, Dataset
from TRANSFORMER.config.dataset_config import SAMPLE_RATE, LABEL2ID, CHUNK_SIZE, extract_label_from_path



def split_audio_to_clips(audio_array, clip_samples):
    """Split audio into 1-second clips, handling files shorter than 1 second"""
    clips = []

    # If the entire audio is shorter than 1 second, pad it to 1 second
    if len(audio_array) <= clip_samples:
        padded_clip = np.pad(audio_array, (0, clip_samples - len(audio_array)), 'constant', constant_values=0)
        clips.append(padded_clip)
        return clips

    # For longer audio, split into clips
    for start in range(0, len(audio_array), clip_samples):
        end = min(start + clip_samples, len(audio_array))
        clip = audio_array[start:end]

        # Pad the last clip if it's shorter than clip_samples
        if len(clip) < clip_samples:
            clip = np.pad(clip, (0, clip_samples - len(clip)), 'constant', constant_values=0)

        clips.append(clip)

    return clips


def prepare_audio_for_model(audio_array):
    """
    Prepare audio for model input by handling variable lengths.
    - Files are processed at their original length (no truncation)
    - Empty or extremely short files are handled by padding to minimum length
    """
    # Wav2Vec2 requires minimum length due to multiple convolutional layers
    # The model has multiple conv layers with different kernel sizes and strides
    # We need to ensure the input survives all downsampling operations
    MIN_SAMPLES = 1600  # Minimum 100ms at 16kHz (safely handles all conv layers)

    if len(audio_array) == 0:
        # Handle empty audio files - return minimum length of silence
        return np.zeros(MIN_SAMPLES, dtype=np.float32)

    if len(audio_array) < MIN_SAMPLES:
        # Pad extremely short files to minimum length
        padded_audio = np.pad(audio_array, (0, MIN_SAMPLES - len(audio_array)), 'constant', constant_values=0)
        return padded_audio.astype(np.float32)

    # Return the audio as-is (no truncation, padding will be handled by the collator)
    return audio_array.astype(np.float32)


def split_large_audio_file(audio_array, sample_rate, max_length_seconds, overlap_seconds):
    """
    Split a large audio file into smaller chunks with overlap.

    Args:
        audio_array: numpy array of audio samples
        sample_rate: sample rate in Hz
        max_length_seconds: maximum length per chunk in seconds
        overlap_seconds: overlap between chunks in seconds

    Returns:
        List of audio chunks (numpy arrays)
    """
    max_samples = int(max_length_seconds * sample_rate)
    overlap_samples = int(overlap_seconds * sample_rate)

    if len(audio_array) <= max_samples:
        return [audio_array]

    chunks = []
    start = 0

    while start < len(audio_array):
        end = min(start + max_samples, len(audio_array))
        chunk = audio_array[start:end]
        chunks.append(chunk)

        # Move start position (with overlap)
        start = end - overlap_samples

        # Avoid tiny chunks at the end
        if start >= len(audio_array) - overlap_samples:
            break

    return chunks


def split_large_audio_file_with_aggregation(audio_array, sample_rate, max_length_seconds, overlap_seconds):
    """
    Split a large audio file into smaller chunks with overlap.
    Returns chunks and metadata needed for proper aggregation.

    Args:
        audio_array: numpy array of audio samples
        sample_rate: sample rate in Hz
        max_length_seconds: maximum length per chunk in seconds
        overlap_seconds: overlap between chunks in seconds

    Returns:
        List of tuples: (chunk_audio, start_time, end_time, chunk_idx)
    """
    max_samples = int(max_length_seconds * sample_rate)
    overlap_samples = int(overlap_seconds * sample_rate)

    if len(audio_array) <= max_samples:
        return [(audio_array, 0.0, len(audio_array) / sample_rate, 0)]

    chunks_with_metadata = []
    start = 0
    chunk_idx = 0

    while start < len(audio_array):
        end = min(start + max_samples, len(audio_array))
        chunk = audio_array[start:end]

        start_time = start / sample_rate
        end_time = end / sample_rate

        chunks_with_metadata.append((chunk, start_time, end_time, chunk_idx))

        # Move start position (with overlap)
        start = end - overlap_samples
        chunk_idx += 1

        # Avoid tiny chunks at the end
        if start >= len(audio_array) - overlap_samples:
            break

    return chunks_with_metadata


def load_audio_dataset(path: str, label_override: str = None):
    """
    Load audio dataset with proper configuration.

    Args:
        path: Directory path containing audio files
        label_override: Override label for all files (if provided)

    Returns:
        Loaded HuggingFace dataset with audio column properly configured
    """
    raw = load_dataset("audiofolder", data_dir=path, split="train")
    raw = raw.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
    return raw


def create_chunked_dataset(input_arrays, labels, file_ids, lengths, max_clips_per_split):
    """
    Create a chunked dataset from processed audio data.

    Args:
        input_arrays: List of processed audio arrays
        labels: List of labels
        file_ids: List of file IDs
        lengths: List of audio lengths
        max_clips_per_split: Maximum clips per dataset split

    Returns:
        List of dataset splits
    """
    if not input_arrays:  # Only if we have data
        return []

    chunk_data = {
        "input_arrays": input_arrays,
        "labels": labels,
        "input_length": lengths,
        "file_ids": file_ids
    }

    chunk_ds = Dataset.from_dict(chunk_data)
    splits = []

    # Split this chunk if it's too large
    if len(chunk_ds) > max_clips_per_split:
        for i in range(0, len(chunk_ds), max_clips_per_split):
            end_idx = min(i + max_clips_per_split, len(chunk_ds))
            split_ds = chunk_ds.select(range(i, end_idx))
            splits.append(split_ds)
    else:
        splits.append(chunk_ds)

    return splits


def categorize_files_by_size(raw_dataset, large_threshold, very_large_threshold, max_file_length, max_file_size_mb, get_file_size_mb_func):
    """
    Categorize files by size for different processing strategies.

    Args:
        raw_dataset: HuggingFace dataset
        large_threshold: Threshold for large files (seconds)
        very_large_threshold: Threshold for very large files (seconds)
        max_file_length: Maximum file length (seconds)
        max_file_size_mb: Maximum file size (MB)
        get_file_size_mb_func: Function to get file size in MB

    Returns:
        Tuple of (regular_files, large_files, very_large_files)
    """
    regular_files = []
    large_files = []
    very_large_files = []

    for file_idx in range(len(raw_dataset)):
        item = raw_dataset[file_idx]
        audio = item["audio"]
        arr = audio["array"]
        duration_seconds = len(arr) / SAMPLE_RATE
        file_size_mb = get_file_size_mb_func(audio["path"])

        if duration_seconds > max_file_length or file_size_mb > max_file_size_mb:
            very_large_files.append((file_idx, item, len(arr)))
            print(f"  Very large file detected: {audio['path']} ({duration_seconds:.1f}s, {file_size_mb:.1f}MB) - will be split")
        elif duration_seconds > large_threshold:
            large_files.append((file_idx, item, len(arr)))
            print(f"  Large file detected: {audio['path']} ({duration_seconds:.1f}s) - will process individually")
        else:
            regular_files.append((file_idx, item, len(arr)))

    print(f"  Regular files: {len(regular_files)} (will be batched)")
    print(f"  Large files: {len(large_files)} (will be processed individually)")
    print(f"  Very large files: {len(very_large_files)} (will be split into chunks)")

    return regular_files, large_files, very_large_files


def process_audio_chunk(files_chunk, label_override, audio_processor_func, clip_samples=None):
    """
    Process a chunk of audio files into input arrays and labels.

    Args:
        files_chunk: List of (file_idx, item, original_length) tuples
        label_override: Label override if provided
        audio_processor_func: Function to process audio (split_audio_to_clips or prepare_audio_for_model)
        clip_samples: Number of samples per clip (for clip-based processing)

    Returns:
        Tuple of (input_arrays, labels, file_ids, lengths)
    """
    chunk_input_arrays = []
    chunk_labels = []
    chunk_file_ids = []
    chunk_lengths = []

    for file_idx, item, original_length in files_chunk:
        audio = item["audio"]
        arr = audio["array"]

        # Use label override if provided, otherwise extract from path
        original_label = label_override or extract_label_from_path(audio["path"])

        if clip_samples is not None:
            # Clip-based processing
            clips = audio_processor_func(arr, clip_samples)
            # Add each clip with the same file ID and original label
            for clip in clips:
                chunk_input_arrays.append(clip)
                chunk_lengths.append(len(clip))
                chunk_labels.append(LABEL2ID[original_label])
                chunk_file_ids.append(file_idx)  # Use file index as file ID
        else:
            # Whole-file processing
            processed_audio = audio_processor_func(arr)
            chunk_input_arrays.append(processed_audio)
            chunk_lengths.append(len(processed_audio))
            chunk_labels.append(LABEL2ID[original_label])
            chunk_file_ids.append(file_idx)

            # Print processing info for very short files
            duration_seconds = original_length / SAMPLE_RATE
            if duration_seconds < 0.5:
                print(f"    Short file: {audio['path']} ({duration_seconds:.2f}s) - padded")

    return chunk_input_arrays, chunk_labels, chunk_file_ids, chunk_lengths
