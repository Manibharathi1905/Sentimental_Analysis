# src/test_accuracy.py
import torch
import yaml
import pandas as pd
import numpy as np
from torch import nn
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score, classification_report

class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_length, text_col, label_col):
        self.texts = df[text_col].tolist()
        self.labels = [eval(x) if isinstance(x, str) else x for x in df[label_col].tolist()]
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

def test_accuracy(config_path, test_csv):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "No GPU")
    
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    model = AutoModelForSequenceClassification.from_pretrained(
        config['model_name'],
        num_labels=20,
        problem_type="multi_label_classification"
    ).to(device)
    model.load_state_dict(torch.load(f"{config['output_dir']}/best_model.pt"))
    model.eval()
    
    test_df = pd.read_csv(test_csv)
    test_ds = EmotionDataset(test_df, tokenizer, config['max_length'], config['text_col'], config['label_col'])
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=config['batch_size'])
    
    y_true, y_pred_probs = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask).logits
            y_true.extend(labels.cpu().numpy())
            y_pred_probs.extend(torch.sigmoid(outputs).cpu().numpy())
    
    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    
    thresholds = [0.2, 0.25, 0.3, 0.35]
    for thresh in thresholds:
        metrics = compute_metrics_multi(y_true, y_pred_probs, thresh)
        print(f"Test Metrics (threshold={thresh}): {metrics}")
    
    # Detailed report for best threshold
    best_threshold = 0.3
    y_pred = (y_pred_probs > best_threshold).astype(int)
    with open(config['label_map'], 'r') as f:
        label_map = json.load(f)
    print("Per-label F1 Scores:")
    for i in range(20):
        f1 = f1_score(y_true[:, i], y_pred[:, i])
        print(f"{label_map[str(i)]}: {f1:.4f}")
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=[label_map[str(i)] for i in range(20)]))
    
    results = {'metrics': metrics, 'per_label_f1': {label_map[str(i)]: f1_score(y_true[:, i], y_pred[:, i]) for i in range(20)}}
    with open('test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Results saved to test_results.json")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--test_csv', type=str, default='data/processed/goemotions_test.csv')
    args = parser.parse_args()
    test_accuracy(args.config, args.test_csv)