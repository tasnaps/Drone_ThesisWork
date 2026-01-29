"""Common constants and label mappings shared across CNN-LSTM pipeline.

Keep a single source of truth for label maps and audio/spectrogram parameters.
"""
from typing import Dict, Tuple

# Label mappings
LABEL2ID: Dict[str, int] = {"unknown": 0, "yes_drone": 1}
ID2LABEL: Dict[int, str] = {v: k for k, v in LABEL2ID.items()}

def get_label_mappings() -> Tuple[Dict[str, int], Dict[int, str]]:
    """Return label-to-id and id-to-label mappings."""
    return LABEL2ID, ID2LABEL

# Audio and spectrogram parameters
SAMPLE_RATE: int = 16000
N_MELS: int = 128
N_FFT: int = 1024
HOP_LENGTH: int = 256



#Datasets for eval and training Swap these for your paths

TRAIN_SET = "C:/Users/tapio/Desktop/Aineistot/TrainingDataset/Al-Emadi/Binary_Drone_Audio"
CALIBRATION_SET = "C:/Users/tapio/Desktop/Aineistot/eval_threshold"
DATASETS = {
    "H-2": {
        "path": "C:/Users/tapio/Desktop/Aineistot/H-2/converted",
        "description": "All drone samples",
        "label_override": "yes_drone"
    },
    "Fusion": {
        "path": "C:/Users/tapio/Desktop/Aineistot/FusionDataset",
        "description": "Mixed dataset with folder structure"
    },
    "DronePrint": {
        "path": "C:/Users/tapio/Desktop/Aineistot/DronePrint/DronePrint/Dataset/DS1/ExperimentallyCollected",
        "description": "All drone samples",
        "label_override": "yes_drone"
    },
    "Tapio": {
        "path": "C:/Users/tapio/Desktop/Aineistot/MerilainenCompiledSounds",
        "description": "Mixed dataset with folder structure"
    },
    "S&E": {
        "path": "C:/Users/tapio/Desktop/Aineistot/Svanström & Englund/Drone-detection-dataset/Data/Audio",
        "description": "Mixed dataset with folder structure"
    },
    "Emo": {
        "path": "C:/Users/tapio/Desktop/Aineistot/EmoSoundscapes/Parsed",
        "description": "All environmental sounds",
        "label_override": "unknown"
    },
    "ESC-50": {
        "path": "C:/Users/tapio/Desktop/Aineistot/ESC-50-master/audio",
        "description": "All environmental sounds",
        "label_override": "unknown"
    },
    "UrbanSound": {
        "path": "C:/Users/tapio/Desktop/Aineistot/UrbanSound8K/mergedFolder",
        "description": "All urban sounds",
        "label_override": "unknown"
    },
    "WonjunYi": {
        "path": "C:/Users/tapio/Desktop/Aineistot/Wonjun",
        "description": "All drone samples",
        "label_override": "yes_drone"
    },
    "CalibrationDataset": {
        "path": "C:/Users/tapio/Desktop/Aineistot/eval_threshold",
        "description": "Mixed dataset with folder structure"
    },
}

## Decimal notation for the .csv outputs for the evaluation script
def format_decimal(value: float, decimals: int = 6) -> str:
    """Format a float without scientific notation, fixed decimals.

    Useful for writing CSVs where exponential notation is undesirable.
    """
    fmt = f"{{0:.{decimals}f}}"
    return fmt.format(float(value))
