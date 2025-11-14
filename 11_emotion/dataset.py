# src/dataset.py
import torch
import numpy as np
import pandas as pd
import ast
import os
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from src.config import cfg

tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path)

class EmotionDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

        # Load GTE embeddings if available
        self.use_gte = cfg.use_precomputed_gte
        if self.use_gte:
            npy_name = os.path.basename(csv_path).replace(".csv", "_gte.npy")
            npy_path = f"data/processed/{npy_name}"
            self.gte_vectors = np.load(npy_path)
        else:
            self.gte_vectors = None

        self.texts = self.df["text"].astype(str).tolist()
        self.labels = [
            torch.tensor(ast.literal_eval(v), dtype=torch.float32)
            for v in self.df["label_vector"]
        ]
            
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "text": self.texts[idx],
            "gte": torch.tensor(self.gte_vectors[idx], dtype=torch.float32) if self.use_gte else None,
            "labels": self.labels[idx]
        }

def collate_fn(batch):
    texts = [b["text"] for b in batch]
    gte = torch.stack([b["gte"] for b in batch]) if batch[0]["gte"] is not None else None
    labels = torch.stack([b["labels"] for b in batch])

    tokenized = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=cfg.max_length,
        return_tensors="pt"
    )

    tokenized["texts"] = texts
    tokenized["gte"] = gte
    tokenized["labels"] = labels

    return tokenized


def get_loader(path, batch_size, shuffle=True):
    ds = EmotionDataset(path)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True
    )
