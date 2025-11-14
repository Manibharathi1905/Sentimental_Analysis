# scripts/tune_thresholds.py
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import argparse
import ast
from src.hybrid_model import HybridEmotionModel
from src.dataset import get_loader
from src.config import cfg
from src.utils import multi_label_metrics


def evaluate_thresholds(probs, labels, threshold):
    preds = (probs >= threshold).astype(int)
    return multi_label_metrics(labels, preds)["f1_macro"]


def main(ckpt, val_file):
    print("Loading validation loader...")
    val_loader = get_loader(val_file, batch_size=cfg.eval_batch_size, shuffle=False)

    print("Loading model checkpoint...")
    model = HybridEmotionModel(cfg).to(cfg.device)
    checkpoint = torch.load(ckpt, map_location=cfg.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_probs = []
    all_labels = []

    print("Running model on validation set...")
    for batch in tqdm(val_loader):
        with torch.no_grad():
            logits = model(
                batch["input_ids"].to(cfg.device),
                batch["attention_mask"].to(cfg.device),
                batch["texts"],
                batch["gte"].to(cfg.device),
            )
            probs = torch.sigmoid(logits).cpu().numpy()
            labels = batch["labels"].cpu().numpy()

            all_probs.append(probs)
            all_labels.append(labels)

    all_probs = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)

    num_classes = all_labels.shape[1]
    best_thresholds = np.zeros(num_classes)

    print("\n🔍 Tuning thresholds...")
    for c in range(num_classes):
        best_f1 = -1
        best_t = 0.5

        for t in np.arange(0.1, 0.91, 0.01):
            f1 = evaluate_thresholds(all_probs[:, c], all_labels[:, c], t)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t

        best_thresholds[c] = best_t
        print(f"Class {c}: best threshold = {best_t:.2f}")

    np.save("outputs/thresholds.npy", best_thresholds)
    print("\n✔ Saved thresholds → outputs/thresholds.npy")

    # also save JSON for readability
    import json
    json.dump(best_thresholds.tolist(), open("outputs/thresholds.json", "w"), indent=2)

    print("✔ Saved readable thresholds → outputs/thresholds.json")
    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--val_file", required=True)
    args = parser.parse_args()

    main(args.ckpt, args.val_file)
