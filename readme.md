# Drone Audio Classification - Thesis Work

A comprehensive audio classification pipeline for drone detection using multiple deep learning architectures: **CNN-LSTM**, **ResNet-34**, and **Transformer-based models**.

---

## Table of Contents

- [Requirements](#requirements)
- [CNN-LSTM](#cnn-lstm)
- [ResNet-34](#resnet-34)
- [Transformer Pipeline](#transformer-pipeline)
- [Evaluation](#evaluation)
- [Troubleshooting](#troubleshooting)

---

## Requirements

- **Python Version:** 3.11 is supported
- Install dependencies: `pip install -r requirements.txt`

> **IMPORTANT:** Due to incompatibility issues with ffmpeg (if you get it working, use newer versions), we use older numpy and datasets versions. This works if you edit:
> `\datasets\formatting\formatting.py` so that return arrays in lines 196-197 have arguments like this:
> ```python
> return np.array(array, dtype=object)
> return np.array(array)
> ```

---

## CNN-LSTM

### Configuration Files

| File | Configuration |
|------|---------------|
| `main.py` | **Line 34:** Set training data directory |
| `common.py` | **Lines 24-74:** Paths for eval datasets. Can also set training dataset path here if you change `main.py` line 35 |

### Training

Run `main.py` after configuring the paths.

---

## ResNet-34

### Configuration Files

| File | Configuration |
|------|---------------|
| `config.py` | Dataset paths and similar settings as CNN-LSTM code |
| `resnet_34.py` | **Line 294:** Training directory<br>**Line 313:** Change path for model output files |

### Training

Configure paths in `config.py` and `resnet_34.py`, then run the training script.

---

## Transformer Pipeline

### Configuration Overview

| File | Purpose |
|------|---------|
| `config/config.py` | Basic config. Dataset directories should be set here. Use label override if dataset only has one class. Not the most elegant solution for dataset config - but it works |
| `main_transformers.py` | **Lines 12-16:** Adjust if ffmpeg and dependencies work together<br>**Line 128:** Training directory + parameters |
| `models/model_transformer.py` | Setting model names and output locations |
| `evaluation/evaluation_strategy_factory.py` | Output directory (can be bypassed via CLI arguments) |
| `evaluation/batch_evaluation.py` | Needs work |

### Data Processing

| File | Purpose                                                                                                                                                                                                                                                                                                                                                                                                                 |
|------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `utils/augment_audio_files.py` | Use this script if you want to augment audio files. Copy your training data and use the wanted augmentation                                                                                                                                                                                                                                                                                                             |
| `data/data_transformers.py` | Contains parameters for augmenting audio files inside the pipeline (no need to duplicate files). **Note:** There's a bug in choosing the augment method so keep the augmentation parameter disabled (`self.augment_data = False` in `main_transformers.py`). The clipping method might differ from `augment_audio_files.py` (not random places - either head, tails, or both). Also contains selector for feature extractor |

### Training

1. Check all paths in configuration files
2. Run `main_transformers.py`

---

## Evaluation

### Using the CLI (Recommended)

Always use the CLI to evaluate transformers models from the **root directory**:

```bash
# Single model evaluation
python -m TRANSFORMER.cli file --model-path <path to saved model>

# Multi-model evaluation
python -m TRANSFORMER.cli multi-model --base-strategy file
```

### CLI Flags

| Flag | Description |
|------|-------------|
| `--calibrate` | Enable calibration |
| `--calibration-key <key>` | Specify calibration dataset (e.g., `"CalibrationDataset"`) |
| `--force-recalibrate` | Force recalibration |

### CNN-LSTM & Resnet-34

For evaluation the non-SSL models: you can simply run the evaluation scripts after checkinig the paths.

### Output

The evaluation outputs `.csv` files for each dataset, which were used for thesis analysis.

> **Note:** The code contains logic for different evaluation styles (clip-based) which cuts audio files and won't do aggregation for decisions. **This shouldn't be used** as it has poor performance and likely broke at some point during development.

---

### Generating Calibration Datasets.

To use fraction of existing datasets for calibrration. We have a script called `generateThresholdEval.py` which generates a subset of the datasets for calibration by moving the files from directories defined in the script.


### Existing analysis

Results and analysis directory contains excel where I've collected the results from thesis work.

Analysis directory contains plots and graphs.

Results directory contains both model specific csv files, and csv's that were generated by our analysis scripts (in the `Analysis Scripts` subdirectory)

## Troubleshooting

### FFmpeg Issues
If you encounter ffmpeg compatibility issues, the codebase uses older numpy and datasets versions as a workaround. See the [Requirements](#-requirements) section for the fix.

### Dataset Configuration
If your dataset only has one class, make sure to use the label override option in `config.py`.

---

## Project Structure

```
├── CNN_LSTM/          # CNN-LSTM architecture
├── RESNET34/          # ResNet-34 architecture  
├── TRANSFORMER/       # Transformer-based models
└── Results&Analysis/  # Results and analysis scripts
```

---

## License

This project is licensed under a non-commercial license. See [LICENSE](LICENSE) for details.
Commercial use is prohibited.
