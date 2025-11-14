# src/config.py
from dataclasses import dataclass
import torch

@dataclass
class CFG:
    seed: int = 42

    # Model names
    model_name_or_path: str = "roberta-base"
    embedding_model: str = "thenlper/gte-large"

    # Training settings
    train_batch_size: int = 16
    eval_batch_size: int = 32
    lr: float = 3e-5
    max_length: int = 128

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Training tricks
    fp16: bool = True   # enable FP16 for speed
    use_checkpointing: bool = False  # Do not use gradient checkpointing

    # Add these new fields in CFG:
    use_precomputed_gte: bool = True
    accumulation_steps: int = 2      # Effective batch size = train_batch_size × 2
    epochs: int = 11
    warmup_ratio: float = 0.1
    save_every_epoch: bool = True

    # Emotions (final 11)
    label_list = [
        "joy",
        "sadness",
        "neutral",
        "anger",
        "love",
        "fear",
        "disgust",
        "confusion",
        "surprise",
        "shame",
        "guilt"
    ]

    # Workers
    num_workers: int = 0

cfg = CFG()
