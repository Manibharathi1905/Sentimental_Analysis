#!/usr/bin/env python3
"""
train.py

Usage:
  python src/train.py --config configs/config.yaml

This script:
 - loads a DistilBERT model for multi-label classification (20 emotions)
 - trains on GoEmotions + EmpatheticDialogues with weighted BCEWithLogitsLoss
 - evaluates on validation set and saves best model based on macro F1
"""

import argparse
import json
import os
import yaml
import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd

class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_length, text_col, label_col, num_labels=20):
        if label_col not in df.columns:
            raise ValueError(f"Column '{label_col}' not found in DataFrame. Available columns: {list(df.columns)}")
        if text_col not in df.columns:
            raise ValueError(f"Column '{text_col}' not found in DataFrame. Available columns: {list(df.columns)}")
        self.texts = df[text_col].tolist()
        self.labels = []
        for x in df[label_col]:
            try:
                label = eval(x) if isinstance(x, str) else x
                if not isinstance(label, (list, tuple)) or len(label) != num_labels:
                    raise ValueError(f"Invalid label format: {x}")
                self.labels.append(label)
            except Exception as e:
                raise ValueError(f"Failed to process label: {x}. Error: {e}")
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float)
        }

def compute_metrics_multi(y_true, y_pred_probs, threshold=0.3):
    y_pred = (y_pred_probs > threshold).astype(int)
    micro_f1 = f1_score(y_true, y_pred, average='micro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    return {'micro_f1': micro_f1, 'macro_f1': macro_f1}

def train(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    model = AutoModelForSequenceClassification.from_pretrained(
        config['model_name'],
        num_labels=20,
        problem_type="multi_label_classification"
    ).to(device)
    
    train_df = pd.read_csv(config['train_csv'])
    val_df = pd.read_csv(config['val_csv'])
    
    print(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}")
    if len(train_df) < 1000:
        raise ValueError("Training dataset is too small. Expected ~70k samples. Check download_and_prepare.py")
    
    train_ds = EmotionDataset(train_df, tokenizer, config['max_length'], config['text_col'], config['label_col'])
    val_ds = EmotionDataset(val_df, tokenizer, config['max_length'], config['text_col'], config['label_col'])
    
    # Validate label distribution
    label_counts = [sum(x[i] for x in train_ds.labels) for i in range(20)]
    print(f"Training label counts: {label_counts}")
    if all(count < 10 for count in label_counts):
        raise ValueError("Label counts are too low. Check label mapping in download_and_prepare.py")
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=config['batch_size'])
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
    label_counts = [sum(x[i] for x in train_ds.labels) for i in range(20)]
    total_samples = len(train_ds)
    pos_weight = torch.tensor([(total_samples - c) / (c + 1e-5) for c in label_counts]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    best_macro_f1 = 0
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")
        
        model.eval()
        y_true, y_pred_probs = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask).logits
                y_true.extend(labels.cpu().numpy())
                y_pred_probs.extend(torch.sigmoid(outputs).cpu().numpy())
        y_true = np.array(y_true)
        y_pred_probs = np.array(y_pred_probs)
        metrics = compute_metrics_multi(y_true, y_pred_probs, config['threshold'])
        print(f"Epoch {epoch+1} validation metrics: {metrics}")
        
        if metrics['macro_f1'] > best_macro_f1:
            best_macro_f1 = metrics['macro_f1']
            os.makedirs(config['output_dir'], exist_ok=True)
            torch.save(model.state_dict(), f"{config['output_dir']}/best_model.pt")
            print(f"Saved best model with macro_f1={best_macro_f1:.4f}")
    
    print("Training complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    args = parser.parse_args()
    train(args.config)