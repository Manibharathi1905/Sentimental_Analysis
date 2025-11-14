#!/usr/bin/env python3
"""
check_labels.py

Script to check label counts in the processed CSV files.
"""

import pandas as pd
import os

def check_labels(csv_path):
    if not os.path.exists(csv_path):
        print(f"File {csv_path} not found.")
        return
    df = pd.read_csv(csv_path)
    if 'labels' not in df.columns:
        print(f"No 'labels' column in {csv_path}")
        return
    # Convert string labels to lists if needed
    df['labels'] = df['labels'].apply(lambda x: eval(x) if isinstance(x, str) else x)
    num_labels = len(df['labels'].iloc[0]) if len(df) > 0 else 0
    label_counts = [sum(df['labels'].apply(lambda x: x[i])) for i in range(num_labels)]
    print(f"Label counts in {csv_path}: {label_counts}")
    print(f"Total samples: {len(df)}")

if __name__ == "__main__":
    files = [
        "data/processed/goemotions_train.csv",
        "data/processed/goemotions_validation.csv",
        "data/processed/goemotions_test.csv",
        "data/processed/empathetic_train.csv",
        "data/processed/empathetic_validation.csv",
        "data/processed/empathetic_test.csv",
        "data/processed/goemotions_empathetic_train.csv",
        "data/processed/goemotions_empathetic_val.csv"
    ]
    for f in files:
        check_labels(f)
        print()
