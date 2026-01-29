#!/usr/bin/env python3
"""
Shared dataset configuration and labeling utilities for transformer model evaluation.
This module centralizes dataset definitions and labeling logic to avoid duplication.
"""

import pathlib
import os

# Common configuration constants
SAMPLE_RATE = 16000  # Audio sample rate in Hz
LABEL2ID = {"unknown": 0, "yes_drone": 1}
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '100'))  # Allow env override

# Enhanced dataset configuration with metadata
ENHANCED_DATASETS = {
    "H-2": {
        "path": "C:/Users/tapio/Desktop/Aineistot/H-2/converted",
        "label_override": None,
        "expected_structure": "folder_based",  # unknown/yes_drone folders
        "description": "Mixed dataset with folder structure"
    },
    #"Fusion": {
    #    "path": "C:/Users/tapio/Desktop/Aineistot/FusionDataset",
    #    "label_override": None,  # Fixed: Has both yes_drone and unknown folders
    #    "expected_structure": "folder_based",
    #    "description": "Mixed dataset with folder structure"
    #},
    "DronePrint": {
        "path": "C:/Users/tapio/Desktop/Aineistot/DronePrint/DronePrint/Dataset/DS1/ExperimentallyCollected",
        "label_override": "yes_drone",
        "expected_structure": "all_same",
        "description": "All drone samples"
    },
    "MerilainenCompiledSounds": {
        "path": "C:/Users/tapio/Desktop/Aineistot/MerilainenCompiledSounds",
        "label_override": None,
        "expected_structure": "folder_based",
        "description": "Mixed dataset with folder structure"
    },
    "S&E": {
        "path": "C:/Users/tapio/Desktop/Aineistot/Svanström & Englund/Drone-detection-dataset/Data/Audio",
        "label_override": None,
        "expected_structure": "folder_based",
        "description": "Mixed dataset with folder structure"
    },
    "Emo": {
        "path": "C:/Users/tapio/Desktop/Aineistot/EmoSoundscapes/Parsed",
        "label_override": "unknown",
        "expected_structure": "all_same",
        "description": "All environmental sounds"
    },
    "ESC-50": {
        "path": "C:/Users/tapio/Desktop/Aineistot/ESC-50-master/audio",
        "label_override": "unknown",
        "expected_structure": "all_same",
        "description": "All environmental sounds"
    },
    "UrbanSound": {
        "path": "C:/Users/tapio/Desktop/Aineistot/UrbanSound8K/mergedFolder",
        "label_override": "unknown",
        "expected_structure": "all_same",
        "description": "All urban sounds"
    },
    "Wonjun_Yi": {
        "path": "C:/Users/tapio/Desktop/Aineistot/Wonjun",
        "label_override": "yes_drone",
        "expected_structure": "all_same",
        "description": "All drone samples"
    },
    "CalibrationDataset": {
        "path": "C:/Users/tapio/Desktop/Aineistot/eval_threshold",
        "label_override": None,
        "expected_structure": "folder_based",
        "description": "Mixed dataset with folder structure"
    },
}

def smart_label_detection(dataset_path, dataset_name, label_override=None):
    """
    Intelligently detect labels based on multiple strategies:
    1. Label override (manual specification)
    2. Folder structure analysis
    3. Filename pattern analysis
    4. Dataset-specific rules
    """

    # Strategy 1: Manual override takes precedence
    if label_override:
        return label_override

    # Strategy 2: Analyze folder structure
    path_obj = pathlib.Path(dataset_path)

    # Check for common drone/non-drone folder patterns
    drone_indicators = ['drone', 'yes_drone', 'positive', 'target', 'uav', 'quadcopter']
    non_drone_indicators = ['non_drone', 'unknown', 'negative', 'background', 'noise']

    # Get all subdirectories (only if path exists)
    try:
        if path_obj.exists():
            subdirs = [d.name.lower() for d in path_obj.iterdir() if d.is_dir()]

            # If we find clear folder structure, use it
            if any(indicator in ' '.join(subdirs) for indicator in drone_indicators):
                return None  # Use folder-based labeling
    except Exception:
        # If we can't access the path, continue to other strategies
        pass

    # Strategy 3: Dataset-specific intelligent defaults
    if dataset_name:  # Check if dataset_name is not None and not empty
        dataset_rules = {
            'al-emadi': None,          # Has folder structure
            'h-2': None,               # Has folder structure
            'fusion': None,            # Has folder structure (corrected)
            'droneprint': 'yes_drone', # All samples are drones
            'tapio': None,
            # Has folder structure
            's&e': None,               # Has folder structure
            'emo': 'unknown',          # All environmental sounds
            'esc-50': 'unknown',       # All environmental sounds
            'urbansound': 'unknown',   # All urban sounds (no drones)
            'wonjun': 'yes_drone',     # All samples are drones
        }

        # Match dataset name to rules (case-insensitive partial matching)
        dataset_key = dataset_name.lower().replace(' ', '').replace('-', '').replace('_', '')
        for rule_key, default_label in dataset_rules.items():
            rule_key_clean = rule_key.replace('-', '').replace('_', '')
            if rule_key_clean in dataset_key or dataset_key in rule_key_clean:
                return default_label

    # Strategy 4: Analyze actual file structure as fallback
    try:
        if path_obj.exists():
            # Sample a few files to detect patterns
            audio_files = list(path_obj.rglob("*.wav"))[:50]  # Sample first 50 files

            # Check parent directory names of files
            parent_dirs = [f.parent.name.lower() for f in audio_files]

            drone_count = sum(1 for d in parent_dirs if any(ind in d for ind in drone_indicators))
            non_drone_count = sum(1 for d in parent_dirs if any(ind in d for ind in non_drone_indicators))

            if drone_count > non_drone_count * 2:
                return 'yes_drone'
            elif non_drone_count > drone_count * 2:
                return 'unknown'
            else:
                return None  # Mixed dataset, use folder structure

    except Exception as e:
        print(f"Warning: Could not analyze {dataset_name or 'unknown'} structure: {e}")

    # Default fallback
    return 'unknown'  # Safe default

def get_label_for_file(file_path, dataset_name, label_override):
    """
    Get the appropriate label for a specific file
    """
    # Use smart detection to get the strategy
    detected_strategy = smart_label_detection(
        pathlib.Path(file_path).parent,
        dataset_name,
        label_override
    )

    if detected_strategy:
        # Use the detected/overridden label for all files
        return detected_strategy
    else:
        # Use folder-based detection (your current extract_label_from_path)
        return extract_label_from_path(file_path)

def extract_label_from_path(path: str) -> str:
    """Extract label from file path based on parent directory name"""
    parent = pathlib.Path(path).parent.name
    return parent if parent in LABEL2ID else "unknown"

def validate_dataset_labels(dataset_path, dataset_name, label_override):
    """
    Validate and report on the labeling strategy for a dataset
    """
    print(f"\n=== Label Analysis for {dataset_name} ===")

    path_obj = pathlib.Path(dataset_path)
    if not path_obj.exists():
        print(f"❌ Path does not exist: {dataset_path}")
        return False

    # Get sample of files to analyze
    audio_files = list(path_obj.rglob("*.wav"))[:100]

    if not audio_files:
        print(f"❌ No audio files found")
        return False

    # Analyze folder structure
    parent_dirs = set(f.parent.name for f in audio_files)
    print(f"Found {len(parent_dirs)} unique parent directories:")
    for dir_name in sorted(parent_dirs):
        count = sum(1 for f in audio_files if f.parent.name == dir_name)
        print(f"   {dir_name}: {count} files")

    # Analyze labeling strategy
    if label_override:
        print(f"Using override label: '{label_override}' for all files")
    else:
        print(f"Using folder-based labeling:")
        label_counts = {}
        for f in audio_files[:20]:  # Sample first 20
            label = extract_label_from_path(str(f))
            label_counts[label] = label_counts.get(label, 0) + 1

        for label, count in label_counts.items():
            print(f"   {label}: {count} files (sample)")

    return True
