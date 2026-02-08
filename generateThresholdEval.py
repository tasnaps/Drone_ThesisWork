import os
import shutil
import random
from pathlib import Path
import hashlib
from typing import Dict, List, Optional, Tuple
import statistics
import contextlib
import wave
import aifc
import math

# Optional: duration extraction via mutagen (if installed)
try:
    from mutagen import File as MutagenFile  # type: ignore
    HAS_MUTAGEN = True
except Exception:
    HAS_MUTAGEN = False

# --- CONFIGURATION: PLEASE EDIT THESE PATHS ---

# A list of all the directories containing your CLEANED evaluation datasets.
# The script will look for 'drone' and 'unknown' subfolders inside each of these.
SOURCE_DATASETS = [
    "C:/Users/XXX/Desktop/Datasests/DronePrint/DronePrint/Dataset/DS1/ExperimentallyCollected",
    "C:/Users/XXX/Desktop/Datasests/EmoSoundscapes/Parsed",
    "C:/Users/XXX/Desktop/Datasests/ESC-50-master/audio",
    "C:/Users/XXX/Desktop/Datasests/H-2/converted",
    "C:/Users/XXX/Desktop/Datasests/Svanström & Englund/Drone-detection-dataset/Data/Audio",
    "C:/Users/XXX/Desktop/Datasests/TapioCollection",
    "C:/Users/XXX/Desktop/Datasests/UrbanSound8K/mergedFolder",
    "C:/Users/XXX/Desktop/Datasests/Wonjun",
]

# The destination directory where the calibration files will be moved.
# This directory will be created if it doesn't exist.
CALIBRATION_DIR = "C:/Users/XXX/Desktop/Aineistot/eval_threshold"

# Deprecated: no longer used (we allocate by fairness, not percentage)
# CALIBRATION_SPLIT_RATIO = 0.2

# The maximum number of files to move from any single dataset.
# This helps prevent large datasets from dominating the calibration set.
MAX_FILES_PER_DATASET = 250

# The maximum total number of files to move per class across all datasets.
# This ensures the final calibration set size per class stays bounded.
MAX_TOTAL_PER_CLASS = 1000

# New: global total duration target across both classes (seconds)
TOTAL_DURATION_TARGET = 10000.0
# New: per-dataset percentage cap per class (e.g. 0.2 means up to 20% of that dataset/class)
MAX_PERCENT_PER_DATASET = 0.20

# Optional per-class percentage and total caps (override the global values above if desired)
MAX_PERCENT_PER_DATASET_BY_CLASS: Dict[str, float] = {"drone": MAX_PERCENT_PER_DATASET, "unknown": MAX_PERCENT_PER_DATASET}
MAX_TOTAL_PER_CLASS_BY_CLASS: Dict[str, int] = {"drone": MAX_TOTAL_PER_CLASS, "unknown": MAX_TOTAL_PER_CLASS}

# Soft relaxation controls: if under global duration target, gradually relax caps and re-plan
RELAX_CAPS_IF_UNDER_TARGET = True
# Consider target met if within this relative tolerance (e.g. 0.02 = 2% shortfall allowed)
TOTAL_DURATION_TOL = 0.02
# Stepwise relaxation values for per-dataset percent cap and per-dataset file cap
RELAX_PERCENT_STEPS: List[float] = [MAX_PERCENT_PER_DATASET, 0.3, 0.4, 0.6, 0.8, 1.0]
RELAX_FILE_CAP_STEPS: List[int] = [MAX_FILES_PER_DATASET, max(2 * MAX_FILES_PER_DATASET, 500), 10**9]

# Balancing / execution modes
# count | duration | hybrid
BALANCE_MODE = "duration"
# Class-level balancing target: equal_count | equal_duration
CLASS_BALANCE = "equal_duration"
# Allowed relative mismatch between class durations when CLASS_BALANCE == 'equal_duration'
CLASS_DURATION_TOL = 0.05  # 5%
# If True, only print the plan and do not move/copy files
DRY_RUN = False
# move | copy (used when DRY_RUN is False)
MOVE_MODE = "move"

# --- SCRIPT LOGIC ---

def _dataset_slug_from_path(p: Path) -> str:
    """Create a short, stable slug from dataset folder name + short hash of full path."""
    full = str(p.resolve())
    h = hashlib.sha1(full.encode("utf-8")).hexdigest()[:6]
    return f"{p.name}-{h}"

# Helper: normalize and de-duplicate source dataset paths
def _dedup_source_paths(paths: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for p in paths:
        try:
            rp = str(Path(p).resolve())
        except Exception:
            rp = str(Path(p))
        key = rp.replace("\\", "/").rstrip("/").lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(rp)
    return out

# Helper: get audio duration in seconds (best-effort)
def _get_audio_duration_seconds(path: Path) -> Optional[float]:
    suffix = path.suffix.lower()
    # Try mutagen first (most formats)
    if HAS_MUTAGEN:
        try:
            mf = MutagenFile(str(path))
            if mf is not None and getattr(mf, "info", None) is not None:
                dur = getattr(mf.info, "length", None)
                if isinstance(dur, (int, float)) and dur > 0:
                    return float(dur)
        except Exception:
            pass
    # WAV
    if suffix == ".wav":
        try:
            with contextlib.closing(wave.open(str(path), "rb")) as w:
                frames = w.getnframes()
                rate = w.getframerate()
                if rate > 0:
                    return frames / float(rate)
        except Exception:
            return None
    # AIFF/AIFF-C
    if suffix in (".aiff", ".aif"):
        try:
            with contextlib.closing(aifc.open(str(path), "rb")) as af:
                frames = af.getnframes()
                rate = af.getframerate()
                if rate > 0:
                    return frames / float(rate)
        except Exception:
            return None
    # Unknown format or failed
    return None

# Helper: water-filling fair allocation of integer quotas up to target_total
def _water_fill_allocate(capacity_by_ds: Dict[str, int], target_total: int) -> Dict[str, int]:
    # Filter only positive capacities
    caps = {k: max(0, int(v)) for k, v in capacity_by_ds.items() if int(v) > 0}
    if not caps or target_total <= 0:
        return {k: 0 for k in capacity_by_ds.keys()}
    # Initial equal share
    K = len(caps)
    base = target_total // K
    alloc = {k: min(base, caps[k]) for k in caps}
    used = sum(alloc.values())
    remaining = max(0, target_total - used)
    # Round-robin distribute remainder honoring capacities
    if remaining > 0:
        # Stable order for determinism
        keys = sorted(caps.keys())
        idx = 0
        while remaining > 0:
            k = keys[idx % len(keys)]
            if alloc[k] < caps[k]:
                alloc[k] += 1
                remaining -= 1
            idx += 1
            # If no one can take more, break
            if all(alloc[kk] >= caps[kk] for kk in keys):
                break
    # Ensure all original keys are present
    full = {k: 0 for k in capacity_by_ds.keys()}
    full.update(alloc)
    return full

# Helper: median of durations ignoring None
def _median_duration(durations: List[Optional[float]]) -> Optional[float]:
    vals = [d for d in durations if isinstance(d, (int, float)) and d is not None]
    if not vals:
        return None
    try:
        return float(statistics.median(vals))
    except Exception:
        return None

# Helper: select files for a class using count/duration/hybrid balancing across datasets
def _select_files_for_class(
    class_label: str,
    per_ds_files: Dict[str, List[Path]],
    alloc_by_ds: Dict[str, int],
    duration_cache: Dict[Path, Optional[float]],
    mode: str,
) -> List[Tuple[Path, str]]:
    selected: List[Tuple[Path, str]] = []

    # Precompute per-dataset candidate lists with durations
    ds_candidates: Dict[str, List[Tuple[Path, float]]] = {}
    ds_median: Dict[str, float] = {}

    for ds_slug, files in per_ds_files.items():
        if not files or alloc_by_ds.get(ds_slug, 0) <= 0:
            continue
        # Deterministic order then random shuffle for tie-breaking
        files_sorted = sorted(files, key=lambda p: str(p).lower())
        # Cache durations and compute median for None fallbacks
        durs = [duration_cache.get(p) for p in files_sorted]
        med = _median_duration(durs)
        if med is None:
            med = 1.0  # neutral fallback
        ds_median[ds_slug] = med
        # Build candidate tuples with numeric duration (fallback to median when None)
        cand = []
        for p in files_sorted:
            d = duration_cache.get(p)
            if not isinstance(d, (int, float)) or d is None or d <= 0:
                d = med
            cand.append((p, float(d)))
        # For duration/hybrid, prefer longer first within each dataset
        cand.sort(key=lambda t: (-t[1], str(t[0]).lower()))
        ds_candidates[ds_slug] = cand

    if not ds_candidates:
        return selected

    if mode == "count":
        # Simple per-dataset random pick respecting allocations
        for ds_slug, cand in ds_candidates.items():
            need = alloc_by_ds.get(ds_slug, 0)
            if need <= 0:
                continue
            # reproducible shuffle
            rnd = list(cand)
            random.shuffle(rnd)
            for p, _ in rnd[:need]:
                selected.append((p, ds_slug))
        return selected

    # duration or hybrid: round-robin by smallest accumulated duration
    acc_dur_by_ds: Dict[str, float] = {ds: 0.0 for ds in ds_candidates.keys()}
    idx_ptr_by_ds: Dict[str, int] = {ds: 0 for ds in ds_candidates.keys()}
    remaining_by_ds: Dict[str, int] = {ds: alloc_by_ds.get(ds, 0) for ds in ds_candidates.keys()}

    total_to_pick = sum(max(0, v) for v in remaining_by_ds.values())
    if total_to_pick <= 0:
        return selected

    # Stable iteration order
    ds_order = sorted(ds_candidates.keys())

    picked = 0
    while picked < total_to_pick:
        # Choose dataset with minimal accumulated duration that still has quota and candidates
        eligible = [
            ds for ds in ds_order
            if remaining_by_ds.get(ds, 0) > 0 and idx_ptr_by_ds.get(ds, 0) < len(ds_candidates[ds])
        ]
        if not eligible:
            break
        # tie-break by slug order
        ds_min = min(eligible, key=lambda ds: (acc_dur_by_ds[ds], ds))
        # pick next longest file from that dataset
        i = idx_ptr_by_ds[ds_min]
        p, d = ds_candidates[ds_min][i]
        idx_ptr_by_ds[ds_min] += 1
        remaining_by_ds[ds_min] -= 1
        acc_dur_by_ds[ds_min] += d
        selected.append((p, ds_min))
        picked += 1

    return selected

# Helper: select up to a target total duration for a class, fairly across datasets
# Respects per-dataset count capacities (cap_by_ds) and a global max count (max_total_count)
# New: respects optional per-dataset duration caps and can prioritize smaller datasets first
def _select_files_for_class_to_duration_target(
    class_label: str,
    per_ds_files: Dict[str, List[Path]],
    cap_by_ds: Dict[str, int],
    duration_cache: Dict[Path, Optional[float]],
    mode: str,
    duration_target: float,
    max_total_count: int,
    per_ds_duration_cap: Optional[Dict[str, float]] = None,
    preferred_ds_order: Optional[List[str]] = None,
) -> List[Tuple[Path, str]]:
    if duration_target <= 0 or max_total_count <= 0:
        return []

    # Build candidates per dataset with durations (fallbacks applied), sorted by duration desc
    ds_candidates: Dict[str, List[Tuple[Path, float]]] = {}
    for ds_slug, files in per_ds_files.items():
        cap = max(0, int(cap_by_ds.get(ds_slug, 0)))
        if cap <= 0:
            continue
        files_sorted = sorted(files, key=lambda p: str(p).lower())
        # Compute median for fallbacks
        durs = [duration_cache.get(p) for p in files_sorted]
        med = _median_duration(durs) or 1.0
        cand: List[Tuple[Path, float]] = []
        for p in files_sorted:
            d = duration_cache.get(p)
            if not isinstance(d, (int, float)) or d is None or d <= 0:
                d = med
            cand.append((p, float(d)))
        cand.sort(key=lambda t: (-t[1], str(t[0]).lower()))
        # Only need up to cap candidates per dataset
        ds_candidates[ds_slug] = cand[:cap]

    if not ds_candidates:
        return []

    # duration/hybrid behave the same here (count mode cannot target duration meaningfully)
    acc_dur_by_ds: Dict[str, float] = {ds: 0.0 for ds in ds_candidates.keys()}
    idx_ptr_by_ds: Dict[str, int] = {ds: 0 for ds in ds_candidates.keys()}
    remaining_by_ds: Dict[str, int] = {ds: len(cand) for ds, cand in ds_candidates.items()}

    total_picked = 0
    total_duration = 0.0
    selected: List[Tuple[Path, str]] = []

    # Dataset caps for duration
    ds_dur_caps = per_ds_duration_cap or {}

    # Preferred dataset order: start with smaller datasets (lower available duration)
    if preferred_ds_order:
        ds_order = [ds for ds in preferred_ds_order if ds in ds_candidates]
        # Append any missing in sorted order for determinism
        ds_order += [ds for ds in sorted(ds_candidates.keys()) if ds not in ds_order]
    else:
        ds_order = sorted(ds_candidates.keys())

    # Helper to find next candidate index that fits duration cap for a dataset
    def _next_fitting_index(ds: str) -> Optional[int]:
        start = idx_ptr_by_ds[ds]
        cap = ds_dur_caps.get(ds, float('inf'))
        for j in range(start, len(ds_candidates[ds])):
            p, d = ds_candidates[ds][j]
            if acc_dur_by_ds[ds] + d <= cap + 1e-6:  # allow small epsilon
                return j
        return None

    while total_picked < max_total_count and total_duration < duration_target:
        eligible = []
        for ds in ds_order:
            if remaining_by_ds.get(ds, 0) <= 0:
                continue
            j = _next_fitting_index(ds)
            if j is not None:
                eligible.append(ds)
            else:
                # no candidate fits remaining dur cap; mark exhausted
                remaining_by_ds[ds] = 0
        if not eligible:
            break
        # Choose dataset with minimal accumulated duration (tie-break by ds_order position)
        ds_min = min(eligible, key=lambda ds: (acc_dur_by_ds[ds], ds_order.index(ds)))
        j = _next_fitting_index(ds_min)
        if j is None:
            remaining_by_ds[ds_min] = 0
            continue
        # If we had to skip some oversized items, advance pointer to the fitting index
        idx_ptr_by_ds[ds_min] = j
        p, d = ds_candidates[ds_min][j]
        idx_ptr_by_ds[ds_min] += 1
        remaining_by_ds[ds_min] -= 1
        acc_dur_by_ds[ds_min] += d
        total_duration += d
        selected.append((p, ds_min))
        total_picked += 1

    return selected


def main():
    """
    Creates a calibration dataset by moving a fraction of files
    from source directories to a new calibration directory.
    """
    print("--- Starting Calibration Set Preparation ---")

    # 1. Create the main calibration directory and its subdirectories
    drone_dest_dir = Path(CALIBRATION_DIR) / "yes_drone"
    no_drone_dest_dir = Path(CALIBRATION_DIR) / "unknown"

    print(f"Creating destination directory: {CALIBRATION_DIR}")
    drone_dest_dir.mkdir(parents=True, exist_ok=True)
    no_drone_dest_dir.mkdir(parents=True, exist_ok=True)

    total_files_moved = 0
    # Track totals per class to enforce MAX_TOTAL_PER_CLASS
    moved_per_class = {"drone": 0, "unknown": 0}
    
    # A dictionary to hold all files, categorized by dataset (slug) and class label
    all_files_by_dataset: Dict[str, Dict[str, List[Path]]] = {}
    # Map slug -> human label (last folder name)
    dataset_labels: Dict[str, str] = {}

    # 2. Collect all audio files from all source datasets
    print("\nCollecting all audio files from source datasets...")

    # Resolve and de-duplicate sources
    unique_sources = _dedup_source_paths(SOURCE_DATASETS)
    if len(unique_sources) != len(SOURCE_DATASETS):
        print(f"De-duplicated sources: {len(SOURCE_DATASETS)} -> {len(unique_sources)}")
    
    # Shuffle dataset order for fairness when hitting global caps (reproducible via seed)
    datasets_order = list(unique_sources)
    random.shuffle(datasets_order)

    for source_path_str in datasets_order:
        source_path = Path(source_path_str)
        if not source_path.is_dir():
            print(f"Warning: Source directory not found, skipping: {source_path}")
            continue
        
        dataset_name = source_path.name
        dataset_slug = _dataset_slug_from_path(source_path)
        all_files_by_dataset[dataset_slug] = {"drone": [], "unknown": []}
        dataset_labels[dataset_slug] = dataset_name
        
        print(f"Scanning: {dataset_name} [{dataset_slug}]")
        
        # Check for both possible drone folder names
        drone_folders = ["yes_drone", "drone"]
        drone_dir = None
        for folder_name in drone_folders:
            potential_dir = source_path / folder_name
            if potential_dir.is_dir():
                drone_dir = potential_dir
                print(f"  Found drone folder: {folder_name}")
                break
        
        # Check for unknown folder
        unknown_dir = source_path / "unknown"
        
        # Process drone files
        if drone_dir and drone_dir.is_dir():
            # Find all common audio file types
            audio_files = list(drone_dir.glob('*.wav')) + \
                          list(drone_dir.glob('*.mp3')) + \
                          list(drone_dir.glob('*.flac'))+ \
                          list(drone_dir.glob('*.ogg')) + \
                          list(drone_dir.glob('*.m4a')) + \
                          list(drone_dir.glob('*.aiff')) + \
                          list(drone_dir.glob('*.aac'))
            all_files_by_dataset[dataset_slug]["drone"].extend(audio_files)
            print(f"  Found {len(audio_files)} drone files")
        else:
            print(f"  No drone folder found (checked: {', '.join(drone_folders)})")
        
        # Process unknown files
        if unknown_dir.is_dir():
            audio_files = list(unknown_dir.glob('*.wav')) + \
                          list(unknown_dir.glob('*.mp3')) + \
                          list(unknown_dir.glob('*.flac'))+ \
                          list(unknown_dir.glob('*.ogg')) + \
                          list(unknown_dir.glob('*.m4a')) + \
                          list(unknown_dir.glob('*.aiff')) + \
                          list(unknown_dir.glob('*.aac'))
            all_files_by_dataset[dataset_slug]["unknown"].extend(audio_files)
            print(f"  Found {len(audio_files)} unknown files")
        else:
            print(f"  No 'unknown' folder found")

    # Print dataset summary
    print("\nDataset Summary:")
    total_drone_files = 0
    total_unknown_files = 0
    for ds_slug, files_dict in all_files_by_dataset.items():
        drone_count = len(files_dict["drone"])
        unknown_count = len(files_dict["unknown"])
        total_drone_files += drone_count
        total_unknown_files += unknown_count
        print(f"  {dataset_labels[ds_slug]} [{ds_slug}]: {drone_count} drone, {unknown_count} unknown")
    
    print(f"\nOverall totals: {total_drone_files} drone files, {total_unknown_files} unknown files")

    # 3. Compute durations (best-effort) for all files to support duration/hybrid balancing
    print("\nComputing durations (best-effort; mutagen installed: {}):".format(HAS_MUTAGEN))
    duration_cache: Dict[Path, Optional[float]] = {}
    all_paths: List[Path] = []
    for ds_slug, files_dict in all_files_by_dataset.items():
        for class_label, flist in files_dict.items():
            all_paths.extend(flist)
    # Deterministic order then iterate
    for p in sorted(all_paths, key=lambda q: str(q).lower()):
        try:
            duration_cache[p] = _get_audio_duration_seconds(p)
        except Exception:
            duration_cache[p] = None

    # Helper: compute available duration per dataset/class with median fallback
    def _available_duration_ds_class(ds_slug: str, class_label: str) -> float:
        files = all_files_by_dataset[ds_slug][class_label]
        if not files:
            return 0.0
        durs = [duration_cache.get(p) for p in files]
        med = _median_duration(durs) or 1.0
        total = 0.0
        for p in files:
            d = duration_cache.get(p)
            if not isinstance(d, (int, float)) or d is None or d <= 0:
                d = med
            total += float(d)
        return total

    # 4. Balanced allocation across datasets per class (water-filling) and selection
    print("\nAllocating balanced quotas and selecting files...")

    # Process order: for equal-duration balance, select unknown first to define the duration target
    classes = ["unknown", "drone"] if CLASS_BALANCE == "equal_duration" and BALANCE_MODE in ("duration", "hybrid") else ["drone", "unknown"]

    # Nested plan builder so we can re-run with relaxed caps if needed
    def _build_plan(percent_caps: Dict[str, float], max_total_by_class: Dict[str, int], per_dataset_file_cap: int) -> Tuple[Dict[str, List[Tuple[Path, str]]], Dict[str, float]]:
        rem_cap_by_ds: Dict[str, int] = {ds: per_dataset_file_cap for ds in all_files_by_dataset.keys()}
        selections_by_class_local: Dict[str, List[Tuple[Path, str]]] = {"drone": [], "unknown": []}
        class_total_duration_local: Dict[str, float] = {"drone": 0.0, "unknown": 0.0}

        desired_per_class_duration = TOTAL_DURATION_TARGET / 2.0 if (BALANCE_MODE in ("duration", "hybrid") and CLASS_BALANCE == "equal_duration") else None

        for class_label in classes:
            # Build per-dataset capacities for this class (respect remaining per-dataset caps and percent caps)
            cap_by_ds: Dict[str, int] = {}
            per_ds_dur_cap: Dict[str, float] = {}
            per_ds_available_dur: Dict[str, float] = {}
            pct_for_class = float(min(1.0, max(0.0, percent_caps.get(class_label, MAX_PERCENT_PER_DATASET))))
            for ds_slug, files_dict in all_files_by_dataset.items():
                avail_count = len(files_dict.get(class_label, []))
                if avail_count <= 0 or rem_cap_by_ds.get(ds_slug, 0) <= 0:
                    cap_by_ds[ds_slug] = 0
                    per_ds_dur_cap[ds_slug] = 0.0
                    per_ds_available_dur[ds_slug] = 0.0
                    continue
                # Count cap respects per-dataset limit and percentage cap per class
                pct_cap = int(math.floor(avail_count * pct_for_class))
                if avail_count > 0:
                    # ensure at least 1 file is allowed when available
                    pct_cap = max(1, pct_cap)
                count_cap = min(avail_count, rem_cap_by_ds[ds_slug], pct_cap)
                cap_by_ds[ds_slug] = count_cap
                # Duration cap per dataset/class based on percentage of available duration
                avail_dur = _available_duration_ds_class(ds_slug, class_label)
                per_ds_available_dur[ds_slug] = avail_dur
                per_ds_dur_cap[ds_slug] = avail_dur * pct_for_class
            total_capacity = sum(cap_by_ds.values())
            if total_capacity <= 0:
                print(f"  Class '{class_label}': no capacity available under current caps, skipping")
                continue

            # Determine preferred order: start with smallest datasets by available duration for this class
            preferred_order = [ds for ds, _ in sorted(per_ds_available_dur.items(), key=lambda kv: (kv[1], kv[0])) if cap_by_ds.get(ds, 0) > 0]

            # Compute available duration across all datasets under percent caps (upper bound)
            available_under_caps = sum(per_ds_dur_cap.values())

            if BALANCE_MODE in ("duration", "hybrid") and CLASS_BALANCE == "equal_duration" and any(selections_by_class_local.values()) and class_label == "drone":
                # Second class: target to match the other class duration within tolerance and global desired target
                ref_duration = class_total_duration_local.get("unknown", 0.0)
                requested = desired_per_class_duration if desired_per_class_duration is not None else ref_duration
                # Allow slight overage to reach ref within tolerance, but cap by available and count
                target_duration = min(available_under_caps, max(0.0, ref_duration * (1 + CLASS_DURATION_TOL)))
                if requested is not None:
                    target_duration = min(target_duration, requested)
                max_count_cap = min(max_total_by_class.get(class_label, MAX_TOTAL_PER_CLASS), total_capacity)
                per_ds_files = {ds: all_files_by_dataset[ds][class_label] for ds in all_files_by_dataset.keys()}
                selected = _select_files_for_class_to_duration_target(
                    class_label,
                    per_ds_files,
                    cap_by_ds,
                    duration_cache,
                    BALANCE_MODE,
                    duration_target=target_duration,
                    max_total_count=max_count_cap,
                    per_ds_duration_cap=per_ds_dur_cap,
                    preferred_ds_order=preferred_order,
                )
                print(f"\nClass '{class_label}': duration-targeted to ~{target_duration:.1f}s (tol {CLASS_DURATION_TOL*100:.0f}%), max {max_count_cap} files, per-dataset cap {int(pct_for_class*100)}%")
            elif BALANCE_MODE in ("duration", "hybrid") and CLASS_BALANCE == "equal_duration" and class_label == "unknown":
                # First class: target half of TOTAL_DURATION_TARGET (or available)
                requested = desired_per_class_duration or available_under_caps
                target_duration = min(available_under_caps, requested)
                max_count_cap = min(max_total_by_class.get(class_label, MAX_TOTAL_PER_CLASS), total_capacity)
                per_ds_files = {ds: all_files_by_dataset[ds][class_label] for ds in all_files_by_dataset.keys()}
                selected = _select_files_for_class_to_duration_target(
                    class_label,
                    per_ds_files,
                    cap_by_ds,
                    duration_cache,
                    BALANCE_MODE,
                    duration_target=target_duration,
                    max_total_count=max_count_cap,
                    per_ds_duration_cap=per_ds_dur_cap,
                    preferred_ds_order=preferred_order,
                )
                print(f"\nClass '{class_label}': duration-targeted to ~{target_duration:.1f}s (global half target {desired_per_class_duration:.1f}s), max {max_count_cap} files, per-dataset cap {int(pct_for_class*100)}%")
            else:
                # Default fair count allocation up to per-class max total
                target_total = min(max_total_by_class.get(class_label, MAX_TOTAL_PER_CLASS), total_capacity)
                alloc_by_ds = _water_fill_allocate(cap_by_ds, target_total)
                print(f"\nClass '{class_label}': target {target_total} (sum capacity {total_capacity}), per-dataset cap {int(pct_for_class*100)}%")
                for ds_slug in sorted(all_files_by_dataset.keys()):
                    if cap_by_ds.get(ds_slug, 0) > 0 or alloc_by_ds.get(ds_slug, 0) > 0:
                        print(f"  {dataset_labels[ds_slug]} [{ds_slug}]: avail={cap_by_ds.get(ds_slug,0)} alloc={alloc_by_ds.get(ds_slug,0)} rem_ds_cap={rem_cap_by_ds.get(ds_slug,0)}")
                per_ds_files = {ds: all_files_by_dataset[ds][class_label] for ds in all_files_by_dataset.keys()}
                selected = _select_files_for_class(class_label, per_ds_files, alloc_by_ds, duration_cache, BALANCE_MODE)

            selections_by_class_local[class_label] = selected

            # Update totals and remaining per-dataset caps
            picked_counts: Dict[str, int] = {}
            tot_dur = 0.0
            for p, ds in selected:
                picked_counts[ds] = picked_counts.get(ds, 0) + 1
                d = duration_cache.get(p)
                if isinstance(d, (int, float)) and d is not None and d > 0:
                    tot_dur += float(d)
            class_total_duration_local[class_label] = tot_dur
            for ds, cnt in picked_counts.items():
                rem_cap_by_ds[ds] = max(0, rem_cap_by_ds.get(ds, 0) - cnt)

        return selections_by_class_local, class_total_duration_local

    # Initial plan with configured caps
    percent_caps_init = dict(MAX_PERCENT_PER_DATASET_BY_CLASS)
    max_total_by_class = dict(MAX_TOTAL_PER_CLASS_BY_CLASS)

    selections_by_class, class_total_duration = _build_plan(percent_caps_init, max_total_by_class, MAX_FILES_PER_DATASET)

    total_planned_duration = sum(class_total_duration.values())
    target_threshold = TOTAL_DURATION_TARGET * (1.0 - TOTAL_DURATION_TOL)

    # If under target, try relaxing caps stepwise
    used_percent_caps = dict(percent_caps_init)
    used_file_cap = MAX_FILES_PER_DATASET
    if RELAX_CAPS_IF_UNDER_TARGET and BALANCE_MODE in ("duration", "hybrid") and total_planned_duration < target_threshold:
        print(f"\nPlanned duration ~{total_planned_duration:.1f}s is below target {TOTAL_DURATION_TARGET:.1f}s (tol {TOTAL_DURATION_TOL*100:.0f}%). Trying relaxed caps...")
        best_plan = (selections_by_class, class_total_duration)
        best_total = total_planned_duration
        for pct in RELAX_PERCENT_STEPS:
            for fcap in RELAX_FILE_CAP_STEPS:
                pct = float(min(1.0, max(pct, 0.0)))
                trial_caps = {"drone": max(percent_caps_init.get("drone", pct), pct), "unknown": max(percent_caps_init.get("unknown", pct), pct)}
                selections_trial, durations_trial = _build_plan(trial_caps, max_total_by_class, int(fcap))
                tot = sum(durations_trial.values())
                print(f"  Relax attempt: per-dataset cap ~{int(pct*100)}%, file cap {int(fcap)} -> total ~{tot:.1f}s")
                if tot > best_total:
                    best_plan = (selections_trial, durations_trial)
                    best_total = tot
                    used_percent_caps = dict(trial_caps)
                    used_file_cap = int(fcap)
                if tot >= target_threshold:
                    break
            if best_total >= target_threshold:
                break
        selections_by_class, class_total_duration = best_plan
        total_planned_duration = best_total
        print(f"Using caps: per-dataset % unknown={int(used_percent_caps['unknown']*100)}%, drone={int(used_percent_caps['drone']*100)}%; per-dataset file cap={used_file_cap}")

    # 5. Execute plan (DRY_RUN/COPY/MOVE)
    # Compute summary
    for class_label in ["drone", "unknown"]:
        cnt = len(selections_by_class[class_label])
        total_dur = class_total_duration.get(class_label, 0.0)
        if total_dur <= 0:
            # compute if not set (e.g., first class in equal_count mode)
            total_dur = 0.0
            for p, _ in selections_by_class[class_label]:
                d = duration_cache.get(p)
                if isinstance(d, (int, float)) and d is not None and d > 0:
                    total_dur += float(d)
        print(f"\nPlanned selection for '{class_label}': {cnt} files, ~{total_dur:.1f}s total")

    total_planned_duration = sum(class_total_duration.values())
    print(f"\nGlobal planned total duration: ~{total_planned_duration:.1f}s (target {TOTAL_DURATION_TARGET:.1f}s)")

    # New: Print per-dataset contribution to the planned split (counts and durations per class)
    # Use the same median-fallback strategy as selection, so datasets with unreadable durations
    # don’t show up as 0.0s.
    med_by_ds_class: Dict[Tuple[str, str], float] = {}
    for ds_slug in all_files_by_dataset.keys():
        for cls in ["drone", "unknown"]:
            files = all_files_by_dataset[ds_slug][cls]
            durs = [duration_cache.get(p) for p in files]
            med_by_ds_class[(ds_slug, cls)] = _median_duration(durs) or 1.0

    ds_contrib: Dict[str, Dict[str, float]] = {}
    for class_label in ["drone", "unknown"]:
        for p, ds in selections_by_class[class_label]:
            d = duration_cache.get(p)
            if not isinstance(d, (int, float)) or d is None or d <= 0:
                d = med_by_ds_class.get((ds, class_label), 1.0)
            dur = float(d)
            if ds not in ds_contrib:
                ds_contrib[ds] = {
                    "drone_count": 0, "unknown_count": 0,
                    "drone_dur": 0.0, "unknown_dur": 0.0,
                }
            ds_contrib[ds][f"{class_label}_count"] += 1
            ds_contrib[ds][f"{class_label}_dur"] += dur

    if ds_contrib:
        print("\nDataset contribution summary (planned):")
        # Sort by total duration desc, then name
        def _totals(k: str):
            v = ds_contrib[k]
            return (-(v["drone_dur"] + v["unknown_dur"]), dataset_labels.get(k, k))
        for ds in sorted(ds_contrib.keys(), key=_totals):
            v = ds_contrib[ds]
            d_cnt = int(v["drone_count"])  # type: ignore[arg-type]
            u_cnt = int(v["unknown_count"])  # type: ignore[arg-type]
            d_dur = v["drone_dur"]
            u_dur = v["unknown_dur"]
            t_cnt = d_cnt + u_cnt
            t_dur = d_dur + u_dur
            if t_cnt == 0:
                continue
            label = dataset_labels.get(ds, ds)
            print(
                f"  {label} [{ds}]: drone={d_cnt} (~{d_dur:.1f}s), "
                f"unknown={u_cnt} (~{u_dur:.1f}s), total={t_cnt} (~{t_dur:.1f}s)"
            )

    if DRY_RUN:
        print("\nDRY_RUN enabled: no files will be moved or copied. Review the plan above.")
        print("--- Dry-run Complete ---")
        return

    # Perform copy/move
    for class_label in classes:
        destination_folder = drone_dest_dir if class_label == "drone" else no_drone_dest_dir
        for f_path, ds_slug in selections_by_class[class_label]:
            try:
                original_name = f_path.name
                new_name = f"{ds_slug}_{original_name}"
                destination_path = destination_folder / new_name

                # Collision-safe: append counter if file already exists
                if destination_path.exists():
                    stem = destination_path.stem
                    suffix = destination_path.suffix
                    i = 1
                    while True:
                        candidate = destination_folder / f"{stem}_{i}{suffix}"
                        if not candidate.exists():
                            destination_path = candidate
                            break
                        i += 1
                if MOVE_MODE == "move":
                    shutil.move(str(f_path), str(destination_path))
                else:
                    shutil.copy2(str(f_path), str(destination_path))
                total_files_moved += 1
                moved_per_class[class_label] += 1
            except Exception as e:
                print(f"    Error processing file {f_path}: {e}")

    print("\n--- Preparation Complete ---")
    print(f"Total files written to calibration set: {total_files_moved}")
    print(f"Per-class totals: drone={moved_per_class['drone']}, unknown={moved_per_class['unknown']}")
    print(f"The directory '{CALIBRATION_DIR}' is now ready for use.")
    print("The remaining files in the source directories now form your final, clean test set.")


if __name__ == "__main__":
    # Set a seed for reproducibility of the random split
    random.seed(42)
    main()