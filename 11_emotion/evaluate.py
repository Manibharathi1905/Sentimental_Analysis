# src/evaluate.py
import torch
import numpy as np
from src.config import cfg
from src.dataset import get_loader
from src.hybrid_model import HybridEmotionModel
from src.utils import multi_label_metrics

@torch.no_grad()
def evaluate_model(ckpt, test_file, threshold=0.5):
    device = cfg.device
    model = HybridEmotionModel(cfg).to(device)

    print("Loading checkpoint:", ckpt)
    ck = torch.load(ckpt, map_location=device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    loader = get_loader(test_file, cfg.eval_batch_size, shuffle=False)

    preds, trues = [], []

    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        texts = batch["texts"]
        labels = batch["labels"].cpu().numpy()

        logits = model(ids, mask, texts, batch["gte"].to(cfg.device))
        probs = torch.sigmoid(logits).cpu().numpy()

        preds.append((probs >= threshold).astype(int))
        trues.append(labels)

    y_pred = np.vstack(preds)
    y_true = np.vstack(trues)

    metrics = multi_label_metrics(y_true, y_pred)
    print("Test Metrics:", metrics)

    return metrics


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="outputs/checkpoints/best_model.pt")
    p.add_argument("--test_file", default="data/processed/test_final_11_balanced.csv")
    args = p.parse_args()

    evaluate_model(args.ckpt, args.test_file)
