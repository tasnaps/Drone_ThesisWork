""" File for data handling, preprocessing, etc."""
import os
import torch
import torch.nn.functional as F
from datasets import load_dataset, Audio, DatasetDict, Dataset
import torchaudio.transforms as T
import librosa
from common import SAMPLE_RATE, N_MELS, N_FFT, HOP_LENGTH, LABEL2ID, ID2LABEL
from pathlib import Path
import shutil
import tempfile
import hashlib
# Spectrogram + augmentation transforms
spec_extractor = T.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_mels=N_MELS,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    power=1.0  # Changed to 1.0 for magnitude spectrogram for PCEN
)
spec_augment = T.FrequencyMasking(freq_mask_param=15)
time_augment = T.TimeMasking(time_mask_param=35)

# Label mappings (single source of truth via common)
id2label = ID2LABEL
label2id = LABEL2ID


def load_and_split(
    data_dir: str,
    val_size: float = 0.2,
    test_size: float = 0.0,
    seed: int = 42
) -> DatasetDict:
    """
    Load an audio dataset from folders and split into train/validation/test with stratification.

    Args:
        data_dir: Path to root directory containing class subfolders of audio files.
        val_size: Fraction of data to reserve for validation (0.0 = no validation split).
        test_size: Fraction of data to reserve for test (0.0 = no test split).
        seed: Random seed for reproducibility.

    Returns:
        A DatasetDict with keys 'train' and optionally 'validation' and/or 'test'.
    """
    root = Path(data_dir).resolve()

    # Load with drop_labels=False to prevent automatic split detection
    raw = load_dataset(
        "audiofolder",
        data_dir=str(root),
        drop_labels=False
    )

    # Get all data from whatever split was created
    split_name = next(iter(raw.keys()))
    all_data = raw[split_name]
    print(f"Loaded {len(all_data)} examples from {data_dir} (split: {split_name})")

    # No splitting needed
    if val_size == 0 and test_size == 0:
        return DatasetDict({"train": all_data})

    # Helper function for stratified split
    def do_split(data, size, seed_val):
        if "label" in data.column_names:
            try:
                return data.train_test_split(
                    test_size=size,
                    seed=seed_val,
                    stratify_by_column="label"
                )
            except (ValueError, TypeError):
                return data.train_test_split(test_size=size, seed=seed_val)
        return data.train_test_split(test_size=size, seed=seed_val)

    # Split into train and (validation + test)
    if val_size > 0 and test_size > 0:
        # First split: train vs (val+test)
        tmp1 = do_split(all_data, val_size + test_size, seed)
        # Second split: validation vs test
        test_ratio = test_size / (val_size + test_size)
        tmp2 = do_split(tmp1["test"], test_ratio, seed)
        return DatasetDict({
            "train": tmp1["train"],
            "validation": tmp2["train"],
            "test": tmp2["test"],
        })
    elif val_size > 0:
        # Only validation split
        tmp = do_split(all_data, val_size, seed)
        return DatasetDict({
            "train": tmp["train"],
            "validation": tmp["test"],
        })
    else:
        # Only test split
        tmp = do_split(all_data, test_size, seed)
        return DatasetDict({
            "train": tmp["train"],
            "test": tmp["test"],
        })

def prepare_batch(
    batch: dict,
    augment: bool = False
) -> dict:
    """
    Convert a batch of raw examples into model-ready features.

    Processes each example's waveform into a Mel-spectrogram,
    applies optional augmentations, and collects pixel_values,
    labels (if present), and file paths.

    Args:
        batch: A batch dict with:
            - 'audio': list of {'array': np.ndarray, 'path': str} entries
            - optionally 'label': list of ints
        augment: Whether to apply random augmentations (noise, time/freq masking).

    Returns:
        A dict with keys:
            'pixel_values': list of Tensors, each shape (1, n_mels, T_i)
            'path'        : list of file paths (str)
            'labels'      : list of ints (if original batch had 'label')
    """
    pixel_values = []
    labels = []
    paths = []

    for idx, audio in enumerate(batch["audio"]):
        # 1) waveform -> tensor
        waveform = torch.tensor(audio["array"], dtype=torch.float32)
        if waveform.ndim > 1:
            waveform = waveform.mean(dim=0)

        # 2) pad or crop to fixed length
        max_len = SAMPLE_RATE * 3
        if waveform.size(-1) > max_len:
            waveform = waveform[..., :max_len]
        min_len = 1024
        if waveform.size(-1) < min_len:
            pad = torch.zeros(min_len - waveform.size(-1), dtype=waveform.dtype)
            waveform = torch.cat([waveform, pad], dim=-1)

        # 3) optional augmentations on waveform
        if augment:
            waveform += torch.randn_like(waveform) * torch.empty(1).uniform_(0.001, 0.01)
            waveform *= torch.empty(1).uniform_(0.8, 1.2)

        # 4) to Mel-spectrogram
        mel = spec_extractor(waveform) # mel is a Tensor: (n_mels, time)

        # Apply PCEN
        mel_np = mel.numpy()
        mel_scaled_np = mel_np * (2**31)
        # Apply PCEN. sr and hop_length match spec_extractor
        pcen_mel_np = librosa.pcen(mel_scaled_np, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
        mel = torch.tensor(pcen_mel_np, dtype=torch.float32)

        # Apply spectral augmentations after PCEN
        if augment:
            mel = spec_augment(mel)
            mel = time_augment(mel)

        pixel_values.append(mel.unsqueeze(0)) # Add channel dim: (1, n_mels, time)
        paths.append(audio["path"])
        if "label" in batch:
            labels.append(batch["label"][idx])

    out = {"pixel_values": pixel_values, "path": paths}
    if labels:
        out["labels"] = labels
    return out

def collate_fn(features: list) -> dict:
    """
    Collate a list of examples into a batched tensor dict.

    Pads variable-length spectrograms along the time axis and stacks them.

    Args:
        features: List of dicts from `prepare_batch`, each with:
            - 'pixel_values': Tensor or nested list (1, n_mels, T_i)
            - optionally 'labels': int

    Returns:
        A dict with:
            'pixel_values': Tensor of shape (batch_size, 1, n_mels, max_T)
            'labels'      : LongTensor (batch_size,) if present
    """
    pixel_vals = []
    for f in features:
        pv = f["pixel_values"]
        # convert any nested list into a Tensor
        if not torch.is_tensor(pv):
            pv = torch.tensor(pv, dtype=torch.float32)
        pixel_vals.append(pv)

    # find max time axis
    max_t = max(p.size(-1) for p in pixel_vals)
    # pad and stack
    padded = [F.pad(p, (0, max_t - p.size(-1))) for p in pixel_vals]
    batch = {"pixel_values": torch.stack(padded)}

    # labels, if present
    if "labels" in features[0]:
        batch["labels"] = torch.tensor([f["labels"] for f in features], dtype=torch.long)
    return batch

def preprocess_split(
    ds: Dataset,
    augment: bool = False
) -> Dataset:
    """
    Apply feature extraction and optional augmentation to an entire dataset in batched mode.

    Args:
        ds: A HuggingFace Dataset with columns 'audio' and optionally 'label'.
        augment: Whether to include random augmentations.

    Returns:
        A new Dataset with columns 'pixel_values', 'path', and optionally 'labels'.
    """
    ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
    remove_cols = ["audio"]
    if "label" in ds.column_names:
        remove_cols.append("label")
    return ds.map(
        lambda batch: prepare_batch(batch, augment),
        remove_columns=remove_cols,
        batched=True,
        batch_size=1000,
        num_proc=os.cpu_count()
    )
