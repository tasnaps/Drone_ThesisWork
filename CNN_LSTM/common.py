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

TRAIN_SET = "C:/Gradu Juttui/Datasets/DroneAudioDataset_Saraalemadi/Binary_Drone_Audio"

DATASETS = {
    "H-2": {
        "path": "C:/Gradu Juttui/Datasets/H-2/converted/",
        "description": "Mixed dataset with folder structure"
    },
    "Fusion": {
        "path": "C:/Gradu Juttui/Datasets/FusionDataset/",
        "description": "Mixed dataset with folder structure"
    },
    "DronePrint": {
        "path": "C:/Gradu Juttui/Datasets/DronePrint/DronePrint/Dataset/DS1/ExperimentallyCollected",
        "description": "All drone samples"
    },
    "Tapio": {
        "path": "C:/Gradu Juttui/Datasets/TapioCollection",
        "description": "Mixed dataset with folder structure"
    },
    "S&E": {
        "path": "C:/Gradu Juttui/Datasets/Svanström & Englund/Drone-detection-dataset/Data/Audio",
        "description": "Mixed dataset with folder structure"
    },
    "Emo": {
        "path": "C:/Gradu Juttui/Datasets/Emo-Soundscapes/Emo-Soundscapes-Audio/Parsed",
        "description": "All environmental sounds"
    },
    "ESC-50": {
        "path": "C:/Gradu Juttui/Datasets/ESC-50-master/audio",
        "description": "All environmental sounds"
    },
    "UrbanSound": {
        "path": "C:/Gradu Juttui/Datasets/UrbanSound8K/mergedFolder",
        "description": "All urban sounds"
    },
    "DroneAudio": {
        "path": "C:/Gradu Juttui/Datasets/DroneAudioDataset_Saraalemadi/Binary_Drone_Audio",
        "description": "Training dataset used to train the model"
    }
}

## Decimal notation for the .csv outputs for the evaluation script
def format_decimal(value: float, decimals: int = 6) -> str:
    """Format a float without scientific notation, fixed decimals.

    Useful for writing CSVs where exponential notation is undesirable.
    """
    fmt = f"{{0:.{decimals}f}}"
    return fmt.format(float(value))
