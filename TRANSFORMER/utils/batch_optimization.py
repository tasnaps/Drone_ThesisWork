#!/usr/bin/env python3
"""
Batch optimization utilities for efficient file processing.
This module implements intelligent file grouping to minimize the number of batches
and maximize GPU utilization efficiency.
"""

import os
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class FileInfo:
    """Information about a file for batch optimization."""
    file_idx: int
    item: Any
    original_length: int
    duration_seconds: float
    size_mb: float
    file_path: str

    @property
    def size_category(self) -> str:
        """Get the size category for this file."""
        return categorize_file_size(self.duration_seconds, self.size_mb)


def categorize_file_size(duration_seconds: float, size_mb: float,
                        large_file_threshold: float = 45.0,
                        very_large_threshold: float = 180.0,
                        max_file_size_mb: int = 100) -> str:
    """Categorize a file by its size."""
    if duration_seconds > very_large_threshold or size_mb > max_file_size_mb:
        return "very_large"
    elif duration_seconds > large_file_threshold:
        return "large"
    else:
        return "regular"


class BatchOptimizer:
    """Optimizes file batching for efficient processing."""

    def __init__(self, target_batch_size: int = 6, max_files_per_split: int = 1000):
        self.target_batch_size = target_batch_size
        self.max_files_per_split = max_files_per_split

    def optimize_file_grouping(self, files: List[Tuple[int, Any, int]],
                             large_file_threshold: float,
                             very_large_threshold: float,
                             max_file_size_mb: int,
                             get_file_size_mb_func) -> Dict[str, List[FileInfo]]:
        """
        Group files intelligently for optimal batch processing.

        Args:
            files: List of (file_idx, item, original_length) tuples
            large_file_threshold: Threshold for large files
            very_large_threshold: Threshold for very large files
            max_file_size_mb: Maximum file size in MB
            get_file_size_mb_func: Function to get file size

        Returns:
            Dict with categorized and optimized file groups
        """
        print("🔍 Analyzing files for optimal batching...")

        # Convert to FileInfo objects with metadata
        file_infos = []
        for file_idx, item, original_length in files:
            duration_seconds = original_length / 16000  # SAMPLE_RATE
            size_mb = get_file_size_mb_func(item["audio"]["path"])
            file_path = item["audio"]["path"]

            file_info = FileInfo(
                file_idx=file_idx,
                item=item,
                original_length=original_length,
                duration_seconds=duration_seconds,
                size_mb=size_mb,
                file_path=file_path
            )
            file_infos.append(file_info)

        # Group by category
        categorized_files = defaultdict(list)
        for file_info in file_infos:
            category = categorize_file_size(
                file_info.duration_seconds,
                file_info.size_mb,
                large_file_threshold,
                very_large_threshold,
                max_file_size_mb
            )
            categorized_files[category].append(file_info)

        # Optimize each category
        optimized_groups = {}

        # Regular files - optimize for similar durations
        if categorized_files["regular"]:
            optimized_groups["regular"] = self._optimize_regular_files(categorized_files["regular"])

        # Large files - keep individual but sort by size
        if categorized_files["large"]:
            optimized_groups["large"] = self._optimize_large_files(categorized_files["large"])

        # Very large files - sort by complexity
        if categorized_files["very_large"]:
            optimized_groups["very_large"] = self._optimize_very_large_files(categorized_files["very_large"])

        self._print_optimization_summary(categorized_files, optimized_groups)

        return optimized_groups

    def _optimize_regular_files(self, regular_files: List[FileInfo]) -> List[List[FileInfo]]:
        """
        Optimize regular files for batching using Best-Fit Decreasing algorithm.
        This algorithm sorts files by duration (descending) and places each file
        into the batch with the least leftover space after placement.
        """
        print(f"    🎯 Applying Best-Fit Decreasing algorithm to {len(regular_files)} regular files...")

        # Sort files in descending order of duration (Best-Fit Decreasing)
        regular_files.sort(key=lambda f: f.duration_seconds, reverse=True)

        # Initialize batches with capacity tracking
        batches = []
        batch_capacities = []  # Track remaining capacity for each batch

        # Calculate target capacity based on longest files that should fit together
        # For regular files (≤45s), we want to be able to fit multiple files efficiently
        max_regular_duration = max(f.duration_seconds for f in regular_files) if regular_files else 45.0
        avg_duration = sum(f.duration_seconds for f in regular_files) / len(regular_files)

        # Target capacity should allow for efficient packing of longer files
        # Use the larger of: (max_duration + buffer) or (avg_duration * batch_size)
        capacity_option_1 = max_regular_duration * 2.5  # Allow for 2-3 longer files
        capacity_option_2 = avg_duration * self.target_batch_size * 1.5  # Traditional approach with buffer
        target_batch_capacity = max(capacity_option_1, capacity_option_2, 45.0)  # Ensure minimum 45s

        print(f"    📊 Target batch capacity: {target_batch_capacity:.1f}s (max file: {max_regular_duration:.1f}s, avg: {avg_duration:.1f}s)")

        # Process each file using Best-Fit Decreasing
        for file_info in regular_files:
            file_duration = file_info.duration_seconds
            best_batch_idx = -1
            best_remaining_space = float('inf')

            # Find the batch with the least remaining space that can still fit this file
            for batch_idx, remaining_capacity in enumerate(batch_capacities):
                if remaining_capacity >= file_duration:
                    remaining_space_after = remaining_capacity - file_duration
                    if remaining_space_after < best_remaining_space:
                        best_remaining_space = remaining_space_after
                        best_batch_idx = batch_idx

            # If we found a suitable batch, add the file there
            if best_batch_idx != -1:
                batches[best_batch_idx].append(file_info)
                batch_capacities[best_batch_idx] -= file_duration
                print(f"      📦 File {file_info.file_path.split('/')[-1]} ({file_duration:.1f}s) → Batch {best_batch_idx + 1} (remaining: {batch_capacities[best_batch_idx]:.1f}s)")
            else:
                # Create a new batch for this file
                batches.append([file_info])
                batch_capacities.append(target_batch_capacity - file_duration)
                print(f"      📦 File {file_info.file_path.split('/')[-1]} ({file_duration:.1f}s) → New Batch {len(batches)} (remaining: {batch_capacities[-1]:.1f}s)")

        # Final optimization: merge small batches if beneficial
        optimized_batches = self._merge_small_batches(batches, target_batch_capacity)

        # Print final statistics
        batch_sizes = [len(batch) for batch in optimized_batches]
        batch_durations = [sum(f.duration_seconds for f in batch) for batch in optimized_batches]

        print(f"    ✅ Best-Fit Decreasing result: {len(optimized_batches)} batches")
        print(f"    📈 Batch sizes: min={min(batch_sizes)}, max={max(batch_sizes)}, avg={sum(batch_sizes)/len(batch_sizes):.1f}")
        print(f"    ⏱️ Batch durations: min={min(batch_durations):.1f}s, max={max(batch_durations):.1f}s, avg={sum(batch_durations)/len(batch_durations):.1f}s")

        return optimized_batches

    def _merge_small_batches(self, batches: List[List[FileInfo]], target_capacity: float) -> List[List[FileInfo]]:
        """
        Merge small batches together to improve efficiency.
        This is a post-processing step after Best-Fit Decreasing.
        """
        if len(batches) <= 1:
            return batches

        print(f"    🔗 Merging small batches (target capacity: {target_capacity:.1f}s)...")

        optimized_batches = []
        i = 0

        while i < len(batches):
            current_batch = batches[i].copy()
            current_duration = sum(f.duration_seconds for f in current_batch)
            current_size = len(current_batch)

            # Try to merge with subsequent small batches
            j = i + 1
            while j < len(batches) and current_size < self.target_batch_size:
                next_batch = batches[j]
                next_duration = sum(f.duration_seconds for f in next_batch)
                next_size = len(next_batch)

                # Check if merging is beneficial
                if (current_size + next_size <= self.target_batch_size and
                    current_duration + next_duration <= target_capacity * 1.5):

                    print(f"      🔗 Merging batch {i+1} ({current_size} files, {current_duration:.1f}s) with batch {j+1} ({next_size} files, {next_duration:.1f}s)")
                    current_batch.extend(next_batch)
                    current_duration += next_duration
                    current_size += next_size
                    j += 1
                else:
                    break

            optimized_batches.append(current_batch)
            i = j if j > i + 1 else i + 1

        return optimized_batches

    def _optimize_large_files(self, large_files: List[FileInfo]) -> List[List[FileInfo]]:
        """
        Optimize large files - process individually but in optimal order.
        Sort by duration (shortest first) for faster initial processing.
        """
        # Sort by duration (process shorter files first for quicker feedback)
        large_files.sort(key=lambda f: f.duration_seconds)

        # Each large file gets its own batch
        return [[file_info] for file_info in large_files]

    def _optimize_very_large_files(self, very_large_files: List[FileInfo]) -> List[List[FileInfo]]:
        """
        Optimize very large files - sort by complexity (size + duration).
        Process smaller/simpler files first.
        IMPORTANT: Each very large file gets its own processing unit to ensure
        all chunks from the same file are processed together.
        """
        # Sort by complexity score (combination of duration and file size)
        very_large_files.sort(key=lambda f: f.duration_seconds * (1 + f.size_mb / 100))

        # Each very large file gets its own processing unit
        # This ensures all chunks from the same file stay together
        return [[file_info] for file_info in very_large_files]

    def _print_optimization_summary(self, original_categorized: Dict, optimized_groups: Dict):
        """Print summary of the optimization results."""
        print("📊 Batch Optimization Summary:")

        for category in ["regular", "large", "very_large"]:
            if category in original_categorized:
                original_count = len(original_categorized[category])
                if category in optimized_groups:
                    batch_count = len(optimized_groups[category])
                    if category == "regular":
                        avg_batch_size = sum(len(batch) for batch in optimized_groups[category]) / batch_count
                        print(f"  📁 {category.title()} files: {original_count} files → {batch_count} batches (avg {avg_batch_size:.1f} files/batch)")
                    else:
                        print(f"  📁 {category.title()} files: {original_count} files → {batch_count} individual processes")

        # Calculate efficiency improvement
        total_files = sum(len(files) for files in original_categorized.values())
        regular_batches = len(optimized_groups.get("regular", []))
        large_batches = len(optimized_groups.get("large", []))
        very_large_batches = len(optimized_groups.get("very_large", []))
        total_processing_units = regular_batches + large_batches + very_large_batches

        if total_processing_units > 0:
            efficiency_ratio = total_files / total_processing_units
            print(f"  ⚡ Overall efficiency: {efficiency_ratio:.1f} files per processing unit")


def create_optimized_file_groups(raw_dataset, large_file_threshold: float,
                                very_large_threshold: float, max_file_length: float,
                                max_file_size_mb: int, get_file_size_mb_func,
                                target_batch_size: int = 6) -> Dict[str, List[Any]]:
    """
    Create optimized file groups for efficient batch processing.
    This is the main entry point for the optimization system.
    """
    # Create file tuples in the expected format
    all_files = []
    for file_idx in range(len(raw_dataset)):
        item = raw_dataset[file_idx]
        original_length = len(item["audio"]["array"])
        all_files.append((file_idx, item, original_length))

    # Initialize optimizer
    optimizer = BatchOptimizer(target_batch_size=target_batch_size)

    # Get optimized groups
    optimized_groups = optimizer.optimize_file_grouping(
        all_files, large_file_threshold, very_large_threshold,
        max_file_size_mb, get_file_size_mb_func
    )

    # Convert back to the expected format for existing processing functions
    result = {
        "regular": [],
        "large": [],
        "very_large": []
    }

    for category, batches in optimized_groups.items():
        for batch in batches:
            # Convert FileInfo objects back to tuples
            file_tuples = [(f.file_idx, f.item, f.original_length) for f in batch]
            result[category].append(file_tuples)

    return result


# Utility functions for integration with existing code
def get_batch_stats(optimized_groups: Dict[str, List[Any]]) -> Dict[str, Any]:
    """Get statistics about the optimized batches."""
    stats = {}

    for category, batches in optimized_groups.items():
        if batches:
            batch_sizes = [len(batch) for batch in batches]
            stats[category] = {
                "num_batches": len(batches),
                "total_files": sum(batch_sizes),
                "avg_batch_size": sum(batch_sizes) / len(batches),
                "min_batch_size": min(batch_sizes),
                "max_batch_size": max(batch_sizes)
            }

    return stats
