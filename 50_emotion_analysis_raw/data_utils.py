from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
import json
import os

def load_label_map(path="data/label_map.json"):
    with open(path, "r") as f:
        return json.load(f)

class EmotionDataset:
    def __init__(self, tokenizer_name="distilbert-base-uncased", max_length=256, label_map_path="data/label_map.json"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.label_map = load_label_map(label_map_path)
        self.num_labels = len(self.label_map)

    def encode_examples(self, texts, labels_multi_hot):
        """texts: list[str], labels_multi_hot: list[list[int]] (same length)"""
        enc = self.tokenizer(texts, truncation=True, padding="max_length", max_length=self.max_length)
        enc["labels"] = labels_multi_hot
        return Dataset.from_dict(enc)

    def prepare_from_csv(self, csv_path, text_col="text", label_col="labels"):
        """
        Expects label_col to contain comma-separated emotion strings or JSON list.
        """
        import pandas as pd
        df = pd.read_csv(csv_path)
        texts = df[text_col].astype(str).tolist()
        all_labels = []
        for cell in df[label_col].astype(str):
            # attempt JSON parse
            try:
                labs = json.loads(cell)
            except:
                labs = [x.strip() for x in cell.split(",") if x.strip()]
            vec = [0]*self.num_labels
            for lab in labs:
                key = lab.lower()
                if key in self.label_map:
                    vec[self.label_map[key]] = 1
            all_labels.append(vec)
        return self.encode_examples(texts, all_labels)
