# src/train.py
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np
import os
import pandas as pd
from transformers import get_linear_schedule_with_warmup

from src.config import cfg
from src.dataset import get_loader
from src.hybrid_model import HybridEmotionModel
from src.utils import compute_pos_weight, multi_label_metrics


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                new_val = param.data.clone()
                old_val = self.shadow[name]
                self.shadow[name] = old_val * self.decay + new_val * (1 - self.decay)

    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.shadow[name].clone()


def train_one_epoch(model, loader, optimizer, criterion, scaler, scheduler, ema):
    model.train()
    total_loss = 0

    optimizer.zero_grad()
    for step, batch in enumerate(tqdm(loader, desc="Training")):

        with torch.amp.autocast("cuda", enabled=cfg.fp16):
            logits = model(
                input_ids=batch["input_ids"].to(cfg.device),
                attention_mask=batch["attention_mask"].to(cfg.device),
                texts=batch["texts"],
                gte=batch["gte"].to(cfg.device),
            )
            loss = criterion(logits, batch["labels"].to(cfg.device))

        # accumulate true loss for logging
        total_loss += loss.item()

        # scale for gradient accumulation
        scaled_loss = loss / cfg.accumulation_steps
        scaler.scale(scaled_loss).backward()
        if (step + 1) % cfg.accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()

            ema.update(model)  # Add this

            optimizer.zero_grad()
            scheduler.step()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, threshold=0.5):
    model.eval()
    preds, trues = [], []

    for batch in tqdm(loader, desc="Validating"):
        input_ids = batch["input_ids"].to(cfg.device)
        mask = batch["attention_mask"].to(cfg.device)
        labels = batch["labels"].cpu().numpy()
        texts = batch["texts"]

        logits = model(input_ids, mask, texts, batch["gte"].to(cfg.device))
        probs = torch.sigmoid(logits).cpu().numpy()

        preds.append((probs >= threshold).astype(int))
        trues.append(labels)

    y_pred = np.vstack(preds)
    y_true = np.vstack(trues)

    metrics = multi_label_metrics(y_true, y_pred)
    return metrics


def main(train_file, val_file, resume=None):
    # --------------------------
    # 1) Load data
    # --------------------------
    train_loader = get_loader(train_file, cfg.train_batch_size)
    val_loader = get_loader(val_file, cfg.eval_batch_size, shuffle=False)

    # --------------------------
    # 2) Optimizer + Scheduler
    # --------------------------
    # Load label vectors
    df = pd.read_csv(train_file)
    import ast
    vectors = [ast.literal_eval(v) for v in df["label_vector"]]
    pos_weight = compute_pos_weight(vectors).to(cfg.device)

    model = HybridEmotionModel(cfg).to(cfg.device)

    start_epoch = 0
    best_micro = 0

    # --------------------------
    # Resume if --resume is given
    # --------------------------
    if resume is not None:
        try:
            ckpt = torch.load(resume, map_location=cfg.device)
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"✔ Loaded checkpoint: {resume}")
            start_epoch = ckpt.get("epoch", 8)
            if "best_micro" in ckpt:
                best_micro = ckpt["best_micro"]
            print(f"Resuming from epoch {start_epoch}")
        except Exception as e:
            print("⚠ Resume failed, training from scratch:", e)

    # Freeze only first 2 layers
    for name, param in model.roberta.named_parameters():
        if "layer." in name:
            if int(name.split("layer.")[1].split(".")[0]) < 2:
                param.requires_grad = False
            else:
                param.requires_grad = True

    optimizer = AdamW(model.parameters(), lr=cfg.lr)

    # EMA
    ema = EMA(model)

    num_training_steps = len(train_loader) * cfg.epochs
    num_warmup = int(num_training_steps * cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup,
        num_training_steps=num_training_steps
    )

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # --------------------------
    # 3) AMP Scaler (Mixed Precision)
    # --------------------------
    scaler = torch.amp.GradScaler(enabled=cfg.fp16)

    save_path = "outputs/checkpoints/best_model.pt"
    os.makedirs("outputs/checkpoints", exist_ok=True)

    for epoch in range(start_epoch, cfg.epochs):

        print(f"EPOCH {epoch+1}/{cfg.epochs}")
        print("============================")
        loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, scheduler, ema)
        print(f"Train Loss = {loss:.4f}")

        # Use EMA for evaluation
        original_params = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                original_params[name] = param.data.clone()
                param.data = ema.shadow[name]

        metrics = evaluate(model, val_loader)
        print("Val metrics:", metrics)

        # Restore original weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = original_params[name]

        if cfg.save_every_epoch:
            epoch_path = f"outputs/checkpoints/epoch_{epoch+1}.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "best_micro": best_micro
            }, epoch_path)
            print(f"✔ Saved checkpoint for epoch {epoch+1}.")

        if metrics["f1_micro"] > best_micro:
            best_micro = metrics["f1_micro"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "best_micro": best_micro
            }, save_path)
            print("✔ Saved new best checkpoint.")

    print("\nTraining complete!")
    print(f"Best MICRO-F1 = {best_micro:.4f}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--train_file", default="data/processed/train_final_11_balanced.csv")
    p.add_argument("--val_file", default="data/processed/val_final_11_balanced.csv")
    p.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    args = p.parse_args()

    main(args.train_file, args.val_file, args.resume)
