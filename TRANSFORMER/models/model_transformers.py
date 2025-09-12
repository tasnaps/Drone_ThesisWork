from transformers import Wav2Vec2ForSequenceClassification

# === CENTRALIZED MODEL CONFIGURATION ===
class ModelConfig:
    """Configuration for Wav2Vec2 model training."""

    # Model settings
    MODEL_NAME = "ALM/wav2vec2-large-audioset"

    # Output directories
    OUTPUT_DIR = "./wav2vec2-drone-Wonjun-0.7Epoch-NoAug"
    LR_FINDER_OUTPUT_DIR = "./lr_finder_results-ALM-EarlyStopping"

    # Training settings that might be shared
    IGNORE_MISMATCHED_SIZES = True

def model_init(num_labels: int):

    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        ModelConfig.MODEL_NAME,
        num_labels=num_labels,
        ignore_mismatched_sizes=ModelConfig.IGNORE_MISMATCHED_SIZES,
    )
    return model
