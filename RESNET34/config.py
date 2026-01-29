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
TRAIN_SET = "C:/Users/tapio/Desktop/Aineistot/TrainingDataset/Al-Emadi/Binary_Drone_Audio"
CALIBRATION_SET = "C:/Users/tapio/Desktop/Aineistot/eval_threshold"

RUNS = [1, 5, 10, 20, 50, 100]
SAMPLE_RATE = 16000