from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple, Dict, Any, Optional, Callable
import numpy as np
import time
import warnings
from functools import lru_cache
from collections import namedtuple

try:
    import torch
    from torch.utils.data import Subset
except Exception:
    torch = None
    Subset = None


# Named tuple for file information
FileInfo = namedtuple('FileInfo', ['file_idx', 'item', 'original_length', 'duration_sec', 'size_mb'])


# LRU cache for dataset length extraction
@lru_cache(maxsize=8)
def _get_dataset_fingerprint(ds) -> str:
    """Generate a fingerprint for dataset caching"""
    try:
        if hasattr(ds, "_fingerprint"):
            return str(ds._fingerprint)
        return f"{id(ds)}_{len(ds)}"
    except:
        return f"{id(ds)}_unknown"


def _extract_lengths(ds, sample_rate: int = 16000) -> List[int]:
    """
    Robustly extract per-item lengths (in samples).
    Tries multiple strategies with fallbacks for different dataset formats.
    """
    # Cache key for this dataset
    cache_key = _get_dataset_fingerprint(ds)

    # Check if we have a cached result in module memory
    if hasattr(_extract_lengths, "_cache"):
        if cache_key in _extract_lengths._cache:
            return _extract_lengths._cache[cache_key]
    else:
        _extract_lengths._cache = {}

    # Try vectorized access methods in order of efficiency
    methods = [
        lambda: ds["original_length"],
        lambda: [len(x[0]) if isinstance(x, (list, np.ndarray)) and len(x) > 0
                 else len(x) for x in ds["input_values"]],
        lambda: ds["length"],
        lambda: ds["duration"] * sample_rate
    ]

    for method in methods:
        try:
            lengths = method()
            if isinstance(lengths, (list, np.ndarray)) and len(lengths) == len(ds):
                result = [int(x) for x in lengths]
                _extract_lengths._cache[cache_key] = result
                return result
        except Exception:
            continue

    # Fallback: item-by-item iteration with progress tracking for large datasets
    start_time = time.time()
    show_progress = len(ds) > 1000
    out = []

    for i in range(len(ds)):
        if show_progress and i % 1000 == 0:
            elapsed = time.time() - start_time
            print(f"Extracting lengths: {i}/{len(ds)} items processed ({elapsed:.1f}s elapsed)")

        try:
            item = ds[i]
            if "original_length" in item:
                out.append(int(item["original_length"]))
            elif "input_values" in item and isinstance(item["input_values"], (list, np.ndarray)):
                # Handle multi-channel audio (take longest channel)
                x = item["input_values"]
                if hasattr(x, "shape") and len(x.shape) > 1:
                    out.append(int(x.shape[-1]))  # Assume last dim is time
                elif hasattr(x, "__len__"):
                    if len(x) > 0 and hasattr(x[0], "__len__"):
                        # Get max length across channels
                        out.append(int(max(len(ch) for ch in x if hasattr(ch, "__len__"))))
                    else:
                        out.append(int(len(x)))
                else:
                    out.append(int(sample_rate))  # 1s fallback
            elif "length" in item:
                out.append(int(item["length"]))
            elif "duration" in item:
                out.append(int(float(item["duration"]) * sample_rate))
            else:
                out.append(int(sample_rate))  # 1s fallback
        except Exception:
            out.append(int(sample_rate))  # 1s fallback on any error

    _extract_lengths._cache[cache_key] = out
    return out


def _compute_budget_seconds(lengths: Sequence[int], batch_size: int,
                            large_sec: float | None = None,
                            very_large_sec: float | None = None,
                            sample_rate: int = 16000,
                            memory_safety_factor: float = 0.85) -> float:
    """
    Compute adaptive per-batch time budget (in seconds) based on data distribution.

    Args:
        lengths: Sequence of audio lengths in samples
        batch_size: Target batch size
        large_sec: Threshold for large files (seconds)
        very_large_sec: Threshold for very large files (seconds)
        sample_rate: Audio sample rate
        memory_safety_factor: Safety factor for memory usage (0.0-1.0)

    Returns:
        Budget in seconds for each batch
    """
    if not lengths:
        return max(1.0, float(batch_size))  # safe fallback

    # Use p80 and p95 for more robust estimation
    p80 = float(np.percentile(lengths, 80)) / sample_rate
    p95 = float(np.percentile(lengths, 95)) / sample_rate
    ref = p80 if p80 > 0 else 1.0

    # Check for severe outliers that might cause OOM
    outlier_factor = p95 / max(0.1, p80)
    if outlier_factor > 3.0:
        warnings.warn(f"Dataset has severe outliers (p95/p80={outlier_factor:.1f}). "
                      f"Consider separate handling for files > {p95:.1f}s.")

    if very_large_sec:
        ref = min(ref, float(very_large_sec))

    # Calculate baseline budget
    budget = ref * max(1, int(batch_size)) * 1.2

    # Adjust based on distribution properties
    mean_len = float(np.mean(lengths)) / sample_rate
    if mean_len < ref * 0.3:  # Highly skewed distribution
        # Allow packing more small files
        budget = max(budget, mean_len * batch_size * 2.5)

    # Apply constraint from large_sec if provided
    if large_sec:
        budget = min(budget, float(large_sec) * max(1, int(batch_size)) * 2.0)

    # Apply memory safety factor
    budget *= memory_safety_factor

    # At least 2 seconds per batch
    return max(budget, 2.0)


def plan_length_packed_batches(
        lengths: Sequence[int],
        max_seconds_per_batch: float,
        sample_rate: int = 16000,
        max_items_per_batch: int | None = None,
        algorithm: str = "ffd",  # "ffd", "bfd", or "balanced"
        balance_factor: float = 0.0  # 0.0-1.0: higher = more balanced batches
) -> List[List[int]]:
    """
    Pack items by length into batches constrained by total seconds.

    Args:
        lengths: Sequence of audio lengths in samples
        max_seconds_per_batch: Maximum seconds of audio per batch
        sample_rate: Audio sample rate
        max_items_per_batch: Maximum items per batch
        algorithm: Packing algorithm (ffd=First-Fit-Decreasing, bfd=Best-Fit-Decreasing, balanced=balanced distribution)
        balance_factor: How much to prioritize balanced batches vs. minimum number of batches

    Returns:
        List of batches (each batch is a list of indices)
    """
    if not lengths:
        return []

    max_samples = int(max_seconds_per_batch * sample_rate)

    # Indices sorted by descending length
    order = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)

    batches: List[List[int]] = []
    used = [False] * len(lengths)

    # Separate very large items that exceed budget
    solo_items = []
    for idx in order:
        if lengths[idx] >= max_samples:
            solo_items.append(idx)
            used[idx] = True
            batches.append([idx])

    # Filter remaining items
    remaining = [i for i in order if not used[i]]

    if algorithm == "balanced" and balance_factor > 0 and remaining:
        # Balanced distribution algorithm
        target_batches = max(1, len(remaining) // max_items_per_batch) if max_items_per_batch else max(1, int(len(remaining) ** 0.5))

        # Sort remaining by length for better distribution
        remaining.sort(key=lambda i: lengths[i], reverse=True)

        # Create empty batches
        new_batches = [[] for _ in range(target_batches)]
        batch_sizes = [0] * target_batches

        # Distribute items using a greedy approach
        for idx in remaining:
            # Find batch with minimum current size
            min_batch = min(range(target_batches), key=lambda i: batch_sizes[i])
            new_batches[min_batch].append(idx)
            batch_sizes[min_batch] += lengths[idx]
            used[idx] = True

        # Only add non-empty batches
        batches.extend([b for b in new_batches if b])

    elif algorithm == "bfd" and remaining:
        # Best-Fit-Decreasing
        bfd_batches = []  # New batches for BFD
        batch_sizes = []  # Track current size of each batch

        for idx in remaining:
            item_size = lengths[idx]

            # Find the best batch (with smallest remaining space after adding item)
            best_batch = -1
            min_remaining_space = float('inf')

            for i, size in enumerate(batch_sizes):
                if size + item_size <= max_samples:
                    remaining_space = max_samples - (size + item_size)
                    if remaining_space < min_remaining_space:
                        min_remaining_space = remaining_space
                        best_batch = i

            if best_batch >= 0:
                # Add to existing batch
                bfd_batches[best_batch].append(idx)
                batch_sizes[best_batch] += item_size
            else:
                # Create new batch
                bfd_batches.append([idx])
                batch_sizes.append(item_size)

            used[idx] = True

        batches.extend(bfd_batches)

    elif remaining:
        # First-Fit-Decreasing (default)
        for idx in remaining:
            if used[idx]:
                continue
            used[idx] = True

            # Start new batch with this item
            current = [idx]
            current_sum = lengths[idx]

            # Try to pack additional items (smallest first after FFD seed)
            for j in sorted((i for i in remaining if not used[i]), key=lambda i: lengths[i]):
                # Max items constraint
                if max_items_per_batch is not None and len(current) >= max_items_per_batch:
                    break

                cand_sum = current_sum + lengths[j]
                if cand_sum <= max_samples:
                    current.append(j)
                    current_sum = cand_sum
                    used[j] = True

            batches.append(current)

    # Calculate diagnostic statistics
    if batches:
        batch_lens = [sum(lengths[i] for i in b) / sample_rate for b in batches]
        # Keep minimal logging for important information only

    return batches


def subset_dataset(ds, indices: Sequence[int]):
    """
    Create a dataset view with the given indices, optimized for different dataset types.
    """
    if not indices:
        raise ValueError("Empty indices list for subset_dataset")

    # Fast path for HuggingFace Datasets
    if hasattr(ds, "select"):
        return ds.select(list(indices))

    # PyTorch Dataset with GPU-aware caching
    if Subset is not None and hasattr(ds, "__getitem__") and hasattr(ds, "__len__"):
        class CachedSubset(Subset):
            """Memory-efficient subset with lazy loading and caching"""

            def __init__(self, dataset, indices, cache_size=4):
                super().__init__(dataset, indices)
                self._cache = {}
                self._max_cache = cache_size

            def __getitem__(self, idx):
                if idx in self._cache:
                    return self._cache[idx]

                item = super().__getitem__(idx)

                # Cache management
                if len(self._cache) >= self._max_cache:
                    # Simple LRU: remove random item
                    self._cache.pop(next(iter(self._cache)))

                self._cache[idx] = item
                return item

            # Pass through column access
            def __getattr__(self, name):
                # Try to delegate to the dataset
                try:
                    attr = getattr(self.dataset, name)
                    if callable(attr):
                        return attr
                    # If attribute is a sequence, subset it
                    if hasattr(attr, "__getitem__") and len(attr) == len(self.dataset):
                        return [attr[i] for i in self.indices]
                except AttributeError:
                    pass

                # Default
                return super().__getattr__(name)

        return CachedSubset(ds, list(indices))

    # Plain Python sequence with additional functionality
    class EnhancedSeqWrapper:
        def __init__(self, base, idxs):
            self._base = base
            self._idxs = list(idxs)
            self._cache = {}

        def __len__(self):
            return len(self._idxs)

        def __getitem__(self, i):
            if i not in self._cache:
                self._cache[i] = self._base[self._idxs[i]]
            return self._cache[i]

        # Column access with dynamic projection
        def __getattr__(self, name):
            # Try base attribute
            try:
                attr = getattr(self._base, name)
                if callable(attr):
                    return attr
                # If attribute is a sequence, subset it
                if hasattr(attr, "__getitem__") and len(attr) == len(self._base):
                    return [attr[i] for i in self._idxs]
            except AttributeError:
                pass

            # Implement common column access patterns
            if hasattr(self._base, "__getitem__"):
                # Try to extract a column from dictionaries
                try:
                    return [self._base[idx].get(name) for idx in self._idxs]
                except (TypeError, AttributeError):
                    pass

            raise AttributeError(f"{type(self._base).__name__} has no attribute {name}")

    return EnhancedSeqWrapper(ds, indices)


def plan_batches_for_dataset(
        ds,
        batch_size: int,
        sample_rate: int = 16000,
        large_file_threshold: float | None = None,
        very_large_threshold: float | None = None,
        max_items_per_batch: int | None = None,
        algorithm: str = "ffd",
        memory_safety_factor: float = 0.85,
        diagnostics: bool = True
) -> Tuple[List[List[int]], List[int], Dict[str, Any]]:
    """
    Plan optimized length-aware batches for a dataset with enhanced diagnostics.

    Returns:
        Tuple of (batches, lengths, diagnostics) where:
          - batches is a list of index lists
          - lengths is the list of per-item lengths
          - diagnostics is a dict with planning statistics
    """
    start_time = time.time()

    # Extract lengths with enhanced error handling
    try:
        lengths = _extract_lengths(ds, sample_rate=sample_rate)
    except Exception as e:
        print(f"Warning: Failed to extract lengths ({e}). Using default 1s length.")
        lengths = [sample_rate] * len(ds)

    # Compute budget with safety factor
    budget_sec = _compute_budget_seconds(
        lengths,
        batch_size=batch_size,
        large_sec=large_file_threshold,
        very_large_sec=very_large_threshold,
        sample_rate=sample_rate,
        memory_safety_factor=memory_safety_factor
    )

    # Allow packing more tiny files than nominal batch size
    max_items = max_items_per_batch if max_items_per_batch is not None else max(batch_size * 4, batch_size + 16)

    # Plan batches with selected algorithm
    batches = plan_length_packed_batches(
        lengths,
        max_seconds_per_batch=budget_sec,
        sample_rate=sample_rate,
        max_items_per_batch=max_items,
        algorithm=algorithm
    )

    # Collect diagnostics
    diagnostic_info = {
        "dataset_size": len(ds),
        "num_batches": len(batches),
        "budget_seconds": budget_sec,
        "planning_time": time.time() - start_time,
        "total_audio_seconds": sum(lengths) / sample_rate,
        "mean_batch_size": np.mean([len(b) for b in batches]) if batches else 0,
        "algorithm": algorithm,
    }

    if diagnostics:
        print(f"  Dataset: {len(ds)} items, ~{sum(lengths) / sample_rate:.1f}s audio")
        print(f"  Created {len(batches)} batches (avg {diagnostic_info['mean_batch_size']:.1f} items/batch)")

    return batches, lengths, diagnostic_info


def create_optimized_file_groups(
        raw_dataset,
        large_file_threshold: float,
        very_large_threshold: float,
        max_file_length: float,
        max_file_size_mb: float,
        get_file_size_mb_func,
        target_batch_size: int = 6,
        sample_rate: int = 16000
) -> Dict[str, List[List]]:
    """
    Create optimized file groups categorized by size and duration.

    Args:
        raw_dataset: Raw audio dataset
        large_file_threshold: Threshold in seconds for large files
        very_large_threshold: Threshold in seconds for very large files
        max_file_length: Maximum file length in seconds before splitting
        max_file_size_mb: Maximum file size in MB
        get_file_size_mb_func: Function to get file size in MB
        target_batch_size: Target number of files per batch for regular files
        sample_rate: Audio sample rate

    Returns:
        Dictionary with keys 'regular', 'large', 'very_large', each containing lists of file batches
    """
    # Categorize files
    regular_files = []
    large_files = []
    very_large_files = []

    for file_idx in range(len(raw_dataset)):
        try:
            item = raw_dataset[file_idx]
            audio_array = item["audio"]["array"]
            audio_path = item["audio"]["path"]

            original_length = len(audio_array)
            duration_sec = original_length / sample_rate

            # Get file size
            try:
                file_size_mb = get_file_size_mb_func(audio_path)
            except Exception:
                file_size_mb = 0.0

            # Create file info
            file_info = FileInfo(
                file_idx=file_idx,
                item=item,
                original_length=original_length,
                duration_sec=duration_sec,
                size_mb=file_size_mb
            )

            # Categorize based on thresholds
            if duration_sec >= very_large_threshold or file_size_mb >= max_file_size_mb:
                very_large_files.append(file_info)
            elif duration_sec >= large_file_threshold:
                large_files.append(file_info)
            else:
                regular_files.append(file_info)

        except Exception as e:
            print(f"Warning: Error processing file {file_idx}: {e}")
            continue

    # Create optimized batches for each category
    result = {}

    # Regular files: pack into batches using intelligent grouping
    if regular_files:
        # Sort by duration for better batching
        regular_files.sort(key=lambda f: f.duration_sec)

        # Extract durations for batch planning
        durations = [f.duration_sec for f in regular_files]

        # Compute optimal batch budget (target total duration per batch)
        if durations:
            median_duration = np.median(durations)
            batch_budget_sec = median_duration * target_batch_size * 1.2
        else:
            batch_budget_sec = 60.0  # 60 second default

        # Use FFD packing to create batches
        regular_batches = []
        current_batch = []
        current_duration = 0.0

        for f_info in regular_files:
            # Check if adding this file would exceed budget or max batch size
            if (current_duration + f_info.duration_sec > batch_budget_sec and current_batch) or \
               len(current_batch) >= target_batch_size * 2:
                # Flush current batch
                regular_batches.append(current_batch)
                current_batch = []
                current_duration = 0.0

            current_batch.append(f_info)
            current_duration += f_info.duration_sec

        # Add remaining batch
        if current_batch:
            regular_batches.append(current_batch)

        result["regular"] = regular_batches

    # Large files: one file per batch
    if large_files:
        result["large"] = [[f_info] for f_info in large_files]

    # Very large files: one file per batch (will be split during processing)
    if very_large_files:
        result["very_large"] = [[f_info] for f_info in very_large_files]

    return result
