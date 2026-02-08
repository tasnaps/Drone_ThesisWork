#TODO: Update your own paths. See CNN-LSTM: common.py for details

DATASETS = {
    "H-2": {
        "path": "C:/Users/XXX/Desktop/Datasets/H-2/converted",
        "description": "All drone samples",
        "label_override": "yes_drone"
    },
    "Fusion": {
        "path": "C:/Users/XXX/Desktop/Datasets/FusionDataset",
        "description": "Mixed dataset with folder structure"
    },
    "DronePrint": {
        "path": "C:/Users/XXX/Desktop/Datasets/DronePrint/DronePrint/Dataset/DS1/ExperimentallyCollected",
        "description": "All drone samples",
        "label_override": "yes_drone"
    },
    "Tapio": {
        "path": "C:/Users/XXX/Desktop/Datasets/MerilainenCompiledSounds",
        "description": "Mixed dataset with folder structure"
    },
    "S&E": {
        "path": "C:/Users/XXX/Desktop/Datasets/Svanström & Englund/Drone-detection-dataset/Data/Audio",
        "description": "Mixed dataset with folder structure"
    },
    "Emo": {
        "path": "C:/Users/XXX/Desktop/Datasets/EmoSoundscapes/Parsed",
        "description": "All environmental sounds",
        "label_override": "unknown"
    },
    "ESC-50": {
        "path": "C:/Users/XXX/Desktop/Datasets/ESC-50-master/audio",
        "description": "All environmental sounds",
        "label_override": "unknown"
    },
    "UrbanSound": {
        "path": "C:/Users/XXX/Desktop/Datasets/UrbanSound8K/mergedFolder",
        "description": "All urban sounds",
        "label_override": "unknown"
    },
    "WonjunYi": {
        "path": "C:/Users/XXX/Desktop/Datasets/Wonjun",
        "description": "All drone samples",
        "label_override": "yes_drone"
    },
    "CalibrationDataset": {
        "path": "C:/Users/XXX/Desktop/Datasets/eval_threshold",
        "description": "Mixed dataset with folder structure"
    },
}
TRAIN_SET = "C:/Users/XXX/Desktop/Datasets/TrainingDataset/Al-Emadi/Binary_Drone_Audio"
CALIBRATION_SET = "C:/Users/XXX/Desktop/Datasets/eval_threshold"

#For specific epoch runs.
RUNS = [1, 5, 10, 20, 50, 100]
SAMPLE_RATE = 16000