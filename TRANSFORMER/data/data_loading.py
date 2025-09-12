import gc
import math
import torch
from datasets import Dataset
from TRANSFORMER.utils.audio_processing import (
    load_audio_dataset, categorize_files_by_size, process_audio_chunk,
    create_chunked_dataset, split_audio_to_clips, prepare_audio_for_model
)
from TRANSFORMER.evaluation.evaluation_common import get_file_size_mb
from TRANSFORMER.utils.batch_optimization import create_optimized_file_groups, get_batch_stats
from TRANSFORMER.config.dataset_config import SAMPLE_RATE, LABEL2ID, CHUNK_SIZE

def _cleanup_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def load_and_prepare_clip_based(path: str, label_override: str = None, max_clips_per_split=10000, clip_samples=16000):
    """
    Load and prepare audio files for clip-based evaluation.

    Args:
        path: Directory path containing audio files
        label_override: Override label for all files (if provided)
        max_clips_per_split: Maximum clips per dataset split
        clip_samples: Number of samples per clip

    Returns:
        Tuple of (dataset_splits, num_files)
    """
    raw = load_audio_dataset(path, label_override)
    print(f"Processing {len(raw)} audio files in chunks...")

    all_splits = []

    for chunk_start in range(0, len(raw), CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, len(raw))
        print(f"  Processing files {chunk_start+1}-{chunk_end} of {len(raw)}")

        # Create chunk of files to process
        files_chunk = []
        for file_idx in range(chunk_start, chunk_end):
            item = raw[file_idx]
            files_chunk.append((file_idx, item, len(item["audio"]["array"])))

        # Process this chunk
        chunk_input_arrays, chunk_labels, chunk_file_ids, chunk_lengths = process_audio_chunk(
            files_chunk, label_override, split_audio_to_clips, clip_samples
        )

        # Create dataset splits from this chunk
        chunk_splits = create_chunked_dataset(
            chunk_input_arrays, chunk_labels, chunk_file_ids, chunk_lengths, max_clips_per_split
        )
        all_splits.extend(chunk_splits)

        # Clean up chunk data to free memory
        del chunk_input_arrays, chunk_labels, chunk_file_ids, chunk_lengths
        gc.collect()

    print(f"Created {len(all_splits)} splits from {len(raw)} files")

    # Sort each split by input length for efficient batching
    for i, split in enumerate(all_splits):
        all_splits[i] = split.sort("input_length")

    return all_splits, len(raw)


def load_and_prepare_whole_file(path: str, label_override: str = None, max_files_per_split=1000,
                               large_file_threshold=15.0, very_large_threshold=45.0,
                               max_file_length=90.0, max_file_size_mb=50, trainer=None):
    """
    Load and prepare audio files for whole-file evaluation with special handling for large files.

    Args:
        path: Directory path containing audio files
        label_override: Override label for all files (if provided)
        max_files_per_split: Maximum files per dataset split
        large_file_threshold: Files >X seconds processed individually
        very_large_threshold: Files >X seconds get special handling
        max_file_length: Absolute maximum - split files longer than this
        max_file_size_mb: Maximum file size in MB before splitting
        trainer: Trainer instance for processing chunks

    Returns:
        Tuple of (dataset_splits, num_files)
    """
    raw = load_audio_dataset(path, label_override)
    print(f"Processing {len(raw)} audio files as whole files...")

    # Categorize files by size for different processing strategies
    regular_files, large_files, very_large_files = categorize_files_by_size(
        raw, large_file_threshold, very_large_threshold, max_file_length, max_file_size_mb, get_file_size_mb
    )

    all_splits = []

    # Process regular files in chunks
    if regular_files:
        regular_splits = _process_regular_files(regular_files, label_override, max_files_per_split)
        all_splits.extend(regular_splits)

    # Process large files individually (but not split)
    if large_files:
        large_splits = _process_large_files(large_files, label_override)
        all_splits.extend(large_splits)

    # Process very large files by splitting them with proper aggregation
    if very_large_files:
        very_large_splits = _process_very_large_files(very_large_files, label_override, trainer)
        all_splits.extend(very_large_splits)

    print(f"Created {len(all_splits)} splits from {len(raw)} files")

    # Sort regular file splits by original length for more efficient batching
    _sort_regular_file_splits(all_splits, len(large_files), len(very_large_files))

    return all_splits, len(raw)


def load_and_prepare_whole_file_optimized(path: str, label_override: str = None, max_files_per_split=1000,
                               large_file_threshold=45.0, very_large_threshold=180.0,
                               max_file_length=180.0, max_file_size_mb=100, trainer=None,
                               target_batch_size=6):
    """
    Load and prepare audio files for whole-file evaluation with intelligent batch optimization.
    This version uses smart file grouping to minimize batch count and maximize GPU efficiency.

    Args:
        path: Directory path containing audio files
        label_override: Override label for all files (if provided)
        max_files_per_split: Maximum files per dataset split
        large_file_threshold: Files >X seconds processed individually
        very_large_threshold: Files >X seconds get special handling
        max_file_length: Absolute maximum - split files longer than this
        max_file_size_mb: Maximum file size in MB before splitting
        trainer: Trainer instance for processing chunks
        target_batch_size: Target batch size for regular files

    Returns:
        Tuple of (dataset_splits, num_files)
    """
    raw = load_audio_dataset(path, label_override)
    print(f"🚀 Processing {len(raw)} audio files with intelligent batch optimization...")

    # Use intelligent batch optimization instead of simple categorization
    optimized_groups = create_optimized_file_groups(
        raw, large_file_threshold, very_large_threshold, max_file_length,
        max_file_size_mb, get_file_size_mb, target_batch_size
    )

    # Print optimization statistics
    stats = get_batch_stats(optimized_groups)
    print("📈 Optimization Results:")
    for category, stat in stats.items():
        print(f"  {category.title()}: {stat['num_batches']} batches, {stat['total_files']} files (avg {stat['avg_batch_size']:.1f}/batch)")

    all_splits = []

    # Process optimized regular file batches
    if "regular" in optimized_groups:
        print("⚡ Processing optimized regular file batches...")
        regular_splits = []
        for batch_idx, file_batch in enumerate(optimized_groups["regular"]):
            print(f"  Batch {batch_idx + 1}/{len(optimized_groups['regular'])}: {len(file_batch)} files")
            splits = _process_optimized_regular_batch(file_batch, label_override, max_files_per_split)
            regular_splits.extend(splits)
        
        all_splits.extend(regular_splits)

    # Process large files (individually)
    if "large" in optimized_groups:
        print("🔄 Processing large files individually...")
        large_splits = []
        for file_batch in optimized_groups["large"]:
            # Each batch contains exactly one large file
            splits = _process_large_files(file_batch, label_override)
            large_splits.extend(splits)
        
        all_splits.extend(large_splits)

    # Process very large files (with chunking)
    if "very_large" in optimized_groups:
        print("🔀 Processing very large files with chunking...")
        very_large_splits = []
        for file_batch in optimized_groups["very_large"]:
            # Each batch contains exactly one very large file
            splits = _process_very_large_files(file_batch, label_override, trainer)
            very_large_splits.extend(splits)
        
        all_splits.extend(very_large_splits)

    print(f"✅ Created {len(all_splits)} optimized splits from {len(raw)} files")

    # Sort splits by type and complexity for optimal processing order
    _sort_optimized_splits(all_splits, optimized_groups)

    return all_splits, len(raw)


def _process_regular_files(regular_files, label_override, max_files_per_split):
    """Process regular-sized files in batches."""
    print("Processing regular files in chunks...")
    splits = []

    for chunk_start in range(0, len(regular_files), CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, len(regular_files))
        print(f"  Processing regular files {chunk_start+1}-{chunk_end} of {len(regular_files)}")

        files_chunk = regular_files[chunk_start:chunk_end]

        # Process this chunk of regular files
        chunk_input_arrays, chunk_labels, chunk_file_ids, chunk_lengths = process_audio_chunk(
            files_chunk, label_override, prepare_audio_for_model
        )

        # Add original lengths for whole-file processing
        chunk_original_lengths = [original_length for _, _, original_length in files_chunk]

        # Create dataset splits from this chunk
        if chunk_input_arrays:
            chunk_data = {
                "input_arrays": chunk_input_arrays,
                "labels": chunk_labels,
                "input_length": chunk_lengths,
                "original_length": chunk_original_lengths,
                "file_ids": chunk_file_ids
            }

            chunk_ds = Dataset.from_dict(chunk_data)

            # Split this chunk if it's too large
            if len(chunk_ds) > max_files_per_split:
                for i in range(0, len(chunk_ds), max_files_per_split):
                    end_idx = min(i + max_files_per_split, len(chunk_ds))
                    split_ds = chunk_ds.select(range(i, end_idx))
                    splits.append(split_ds)
            else:
                splits.append(chunk_ds)

            # Clean up chunk data
            del chunk_input_arrays, chunk_labels, chunk_file_ids, chunk_lengths, chunk_original_lengths, chunk_data, chunk_ds
            gc.collect()

    return splits


def _process_large_files(large_files, label_override):
    """Process large files individually with better memory management."""
    print("Processing large files individually...")
    splits = []

    for file_idx, item, original_length in large_files:
        audio = item["audio"]
        arr = audio["array"]
        duration_seconds = original_length / SAMPLE_RATE

        print(f"  Processing large file: {audio['path']} ({duration_seconds:.1f}s)")

        # Use memory cleanup for large files too
        _cleanup_memory()

        files_chunk = [(file_idx, item, original_length)]
        chunk_input_arrays, chunk_labels, chunk_file_ids, chunk_lengths = process_audio_chunk(
            files_chunk, label_override, prepare_audio_for_model
        )

        # Create individual dataset for this large file
        if chunk_input_arrays:
            large_file_data = {
                "input_arrays": chunk_input_arrays,
                "labels": chunk_labels,
                "input_length": chunk_lengths,
                "original_length": [original_length],
                "file_ids": chunk_file_ids
            }

            large_file_ds = Dataset.from_dict(large_file_data)
            splits.append(large_file_ds)

            # Clean up with proper memory management
            del large_file_data, large_file_ds, chunk_input_arrays, chunk_labels, chunk_file_ids, chunk_lengths
            _cleanup_memory()

    return splits


def _process_very_large_files(very_large_files, label_override, trainer=None):
    """
    Process very large files by splitting them for data preparation.
    IMPORTANT: This function prepares data only - predictions are handled elsewhere.
    Enhanced with better memory management for extremely large files.
    """
    print("Processing very large files with splitting for data preparation...")
    splits = []

    # Process each very large file individually to keep its chunks together
    for file_idx, item, original_length in very_large_files:
        audio = item["audio"]
        arr = audio["array"]
        duration_seconds = original_length / SAMPLE_RATE
        file_size_mb = get_file_size_mb(audio["path"])

        print(f"  🔀 Splitting very large file: {audio['path']} ({duration_seconds:.1f}s, {file_size_mb:.1f}MB)")

        # Decide the number of parts for data preparation
        max_part_sec = 180
        num_parts = math.ceil(duration_seconds / max_part_sec)
        print(f"Splitting into {num_parts} parts of ~{max_part_sec:.1f}s each")

        # Create parts for data preparation
        parts = []
        for idx in range(num_parts):
            start_sec = idx * max_part_sec
            end_sec = min((idx + 1) * max_part_sec, duration_seconds)
            start_sample = int(start_sec * SAMPLE_RATE)
            end_sample = int(end_sec * SAMPLE_RATE)
            part_audio = arr[start_sample:end_sample]
            parts.append((part_audio, start_sec, end_sec, idx))

        # Process each part for data preparation (no predictions)
        for part_audio, start_sec, end_sec, idx in parts:
            print(f"  Preparing part {idx + 1} of {num_parts}")
            _cleanup_memory()

            # Prepare audio for model (data preparation only)
            processed = prepare_audio_for_model(part_audio)

            # Create dataset entry for this part
            part_data = {
                "input_arrays": [processed],
                "labels": [LABEL2ID[label_override or "unknown"]],
                "input_length": [len(processed)],
                "original_length": [len(part_audio)],
                "file_ids": [file_idx * 10000 + idx],  # Unique ID
                # Add metadata for tracking the part
                "part_metadata": [{
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "part_index": idx,
                    "total_parts": num_parts,
                    "original_file_id": file_idx
                }]
            }

            part_ds = Dataset.from_dict(part_data)
            splits.append(part_ds)

            print(f"  Part {idx + 1} prepared for evaluation")

    return splits


def _sort_regular_file_splits(all_splits, num_large_files, num_very_large_files):
    """Sort regular file splits by original length for more efficient batching."""
    # Regular file splits count (excluding large and very large files)
    regular_splits_count = len(all_splits) - num_large_files - num_very_large_files

    # Sort regular file splits by original length for more efficient batching
    for i in range(max(0, regular_splits_count)):
        if i < len(all_splits):
            all_splits[i] = all_splits[i].sort("original_length")


def _process_optimized_regular_batch(file_batch, label_override, max_files_per_split):
    """Process an optimized batch of regular files."""
    # Process this optimized batch of regular files
    chunk_input_arrays, chunk_labels, chunk_file_ids, chunk_lengths = process_audio_chunk(
        file_batch, label_override, prepare_audio_for_model
    )

    # Add original lengths for whole-file processing
    chunk_original_lengths = [original_length for _, _, original_length in file_batch]

    splits = []
    if chunk_input_arrays:
        chunk_data = {
            "input_arrays": chunk_input_arrays,
            "labels": chunk_labels,
            "input_length": chunk_lengths,
            "original_length": chunk_original_lengths,
            "file_ids": chunk_file_ids
        }

        chunk_ds = Dataset.from_dict(chunk_data)

        # Split this chunk if it's too large
        if len(chunk_ds) > max_files_per_split:
            for i in range(0, len(chunk_ds), max_files_per_split):
                end_idx = min(i + max_files_per_split, len(chunk_ds))
                split_ds = chunk_ds.select(range(i, end_idx))
                splits.append(split_ds)
        else:
            splits.append(chunk_ds)

        # Clean up chunk data
        del chunk_input_arrays, chunk_labels, chunk_file_ids, chunk_lengths, chunk_original_lengths, chunk_data, chunk_ds
        gc.collect()

    return splits


def _sort_optimized_splits(all_splits, optimized_groups):
    """Sort optimized splits for optimal processing order."""
    # Regular file splits are already optimized by duration similarity
    # Just ensure they're sorted by average duration within each split
    regular_count = len(optimized_groups.get("regular", []))

    for i in range(min(regular_count, len(all_splits))):
        if i < len(all_splits) and "original_length" in all_splits[i].column_names:
            all_splits[i] = all_splits[i].sort("original_length")
