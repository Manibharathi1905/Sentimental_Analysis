
#!/usr/bin/env python3
"""
download_and_prepare.py

Usage:
  python src/download_and_prepare.py --out_dir data/processed --label_map data/label_map.json

This script:
 - downloads GoEmotions (simplified) and EmpatheticDialogues via HuggingFace
 - maps their labels to 20 mental health emotions
 - oversamples rare emotions for class balancing
 - saves CSVs with text and multi-hot labels
"""

import argparse
import json
import os
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm

def load_label_map(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_csv(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")

def map_goemotions_labels(original_labels, label_map):
    # GoEmotions simplified labels are indices (0-27); map to 20 emotions
    goemotions_emotions = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity',
        'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
        'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
        'relief', 'remorse', 'sadness', 'surprise', 'neutral'
    ]
    goemotions_to_new = {
        'admiration': 0, 'amusement': 0, 'joy': 0, 'love': 0, 'optimism': 0, 'excitement': 0,  # Happiness
        'grief': 1, 'sadness': 1, 'disappointment': 1, 'remorse': 1,  # Sadness
        'anger': 2, 'annoyance': 2, 'disapproval': 2,  # Anger
        'fear': 3, 'nervousness': 3,  # Fear
        'nervousness': 4, 'desire': 4,  # Anxiety
        'remorse': 5, 'disapproval': 5,  # Guilt
        'optimism': 6, 'hope': 6,  # Hope
        'caring': 7, 'love': 7,  # Loneliness (proxied)
        'embarrassment': 8,  # Shame
        'disappointment': 9, 'annoyance': 9,  # Frustration
        'relief': 10,  # Relief
        'grief': 11, 'sadness': 11,  # Despair
        'pride': 12,  # Pride
        'confusion': 13, 'realization': 13,  # Confusion
        'gratitude': 14,  # Gratitude
        'disapproval': 15, 'desire': 15,  # Envy (proxied)
        'disgust': 16,  # Disgust
        'approval': 17, 'caring': 17,  # Trust
        'curiosity': 18,  # Boredom (proxied)
        'excitement': 19, 'surprise': 19  # Excitement
    }
    new_labels = [0] * len(label_map)
    for idx in original_labels:
        if idx < len(goemotions_emotions):
            emotion = goemotions_emotions[idx]
            if emotion in goemotions_to_new and emotion != 'neutral':  # Exclude neutral
                new_labels[goemotions_to_new[emotion]] = 1
    return new_labels
                
def map_empathetic_labels(original_emotion, label_map):
    # EmpatheticDialogues emotions to 20 emotions
    empathetic_to_new = {
        'content': 0, 'excited': 0, 'happy': 0, 'joyful': 0, 'nostalgic': 0, 'proud': 0, 'surprised': 0, 'impressed': 0,  # Happiness
        'sad': 1, 'devastated': 1, 'sentimental': 1, 'disappointed': 1, 'lonely': 1,  # Sadness
        'angry': 2, 'annoyed': 2, 'furious': 2,  # Anger
        'afraid': 3, 'terrified': 3, 'scared': 3,  # Fear
        'anxious': 4, 'apprehensive': 4, 'nervous': 4,  # Anxiety
        'ashamed': 5, 'guilty': 5,  # Guilt
        'hopeful': 6, 'optimistic': 6, 'prepared': 6,  # Hope
        'lonely': 7, 'isolated': 7, 'sentimental': 7,  # Loneliness
        'embarrassed': 8,  # Shame
        'frustrated': 9, 'annoyed': 9,  # Frustration
        'relieved': 10, 'grateful': 10,  # Relief
        'hopeless': 11, 'sad': 11, 'devastated': 11,  # Despair
        'proud': 12, 'impressed': 12,  # Pride
        'confused': 13, 'disoriented': 13,  # Confusion
        'grateful': 14, 'appreciative': 14,  # Gratitude
        'jealous': 15, 'envious': 15,  # Envy
        'disgusted': 16,  # Disgust
        'trusting': 17, 'confident': 17,  # Trust
        'bored': 18,  # Boredom
        'anticipating': 19, 'excited': 19, 'surprised': 19  # Excitement
    }
    new_labels = [0] * len(label_map)
    emotion = original_emotion.lower().strip()
    if emotion in empathetic_to_new:
        new_labels[empathetic_to_new[emotion]] = 1
    return new_labels
        
def convert_goemotions(out_dir, label_map):
    print("Loading GoEmotions (simplified)...")
    ds = load_dataset("go_emotions", "simplified")
    goemotions_emotions = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity',
        'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
        'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
        'relief', 'remorse', 'sadness', 'surprise', 'neutral'
    ]
    def ds_to_df(split):
        rows = []
        label_counts = [0] * len(label_map)
        for x in tqdm(ds[split], desc=f"Processing GoEmotions {split}"):
            text = x.get("text", "")
            labels = x.get("labels", [])  # List of emotion indices
            new_labels = map_goemotions_labels(labels, label_map)
            for i, val in enumerate(new_labels):
                label_counts[i] += val
            rows.append({"text": text, "labels": new_labels})
        print(f"GoEmotions {split} label counts: {label_counts}")
        return pd.DataFrame(rows)

    train_df = ds_to_df("train")
    val_df = ds_to_df("validation")
    test_df = ds_to_df("test")

    save_csv(train_df, os.path.join(out_dir, "goemotions_train.csv"))
    save_csv(val_df, os.path.join(out_dir, "goemotions_validation.csv"))
    save_csv(test_df, os.path.join(out_dir, "goemotions_test.csv"))
    print("Saved GoEmotions CSVs to", out_dir)
    return train_df, val_df, test_df
                
def convert_empathetic(out_dir, label_map):
    print("Loading EmpatheticDialogues...")
    try:
        ds = load_dataset("facebook/empathetic_dialogues")
    except Exception as e:
        print(f"Error loading EmpatheticDialogues: {e}")
        print("Falling back to alternative path...")
        ds = load_dataset("empathetic_dialogues", download_mode="force_redownload")

    def to_df(split):
        rows = []
        emotion_counts = {}
        label_counts = [0] * len(label_map)
        for x in tqdm(ds[split], desc=f"Processing EmpatheticDialogues {split}"):
            text = x.get("utterance", x.get("context", ""))
            # Try 'context' as emotion source, as 'emotion' was empty
            emotion = x.get("context", "").lower().strip()
            if not emotion:
                emotion_counts['empty'] = emotion_counts.get('empty', 0) + 1
                print(f"Sample keys: {x.keys()}, context: {x.get('context', '')[:50]}, prompt: {x.get('prompt', '')[:50]}")
            new_labels = map_empathetic_labels(emotion, label_map)
            for i, val in enumerate(new_labels):
                label_counts[i] += val
            rows.append({"text": text, "labels": new_labels})
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        print(f"EmpatheticDialogues {split} emotion counts: {emotion_counts}")
        print(f"EmpatheticDialogues {split} label counts: {label_counts}")
        return pd.DataFrame(rows)

    train_df = to_df("train")
    val_df = to_df("validation")
    test_df = to_df("test")

    save_csv(train_df, os.path.join(out_dir, "empathetic_train.csv"))
    save_csv(val_df, os.path.join(out_dir, "empathetic_validation.csv"))
    save_csv(test_df, os.path.join(out_dir, "empathetic_test.csv"))
    print("Saved EmpatheticDialogues CSVs to", out_dir)
    return train_df, val_df, test_df
                
def combine_and_split(out_dir, label_map):
    ge_train = pd.read_csv(os.path.join(out_dir, "goemotions_train.csv"))
    ge_val = pd.read_csv(os.path.join(out_dir, "goemotions_validation.csv"))
    emp_train = pd.read_csv(os.path.join(out_dir, "empathetic_train.csv"))
    emp_val = pd.read_csv(os.path.join(out_dir, "empathetic_validation.csv"))

    # Convert string labels to lists
    for df in [ge_train, ge_val, emp_train, emp_val]:
        df['labels'] = df['labels'].apply(lambda x: eval(x) if isinstance(x, str) else x)

    train_comb = pd.concat([ge_train, emp_train], ignore_index=True)
    val_comb = pd.concat([ge_val, emp_val], ignore_index=True)

    # Oversample rare labels
    num_labels = len(label_map)
    label_counts = [sum(train_comb['labels'].apply(lambda x: x[i])) for i in range(num_labels)]
    print(f"Label counts before oversampling: {label_counts}")
    rare_threshold = 1000
    rare_labels = [i for i, count in enumerate(label_counts) if count < rare_threshold and count > 0]
    if rare_labels:
        oversampled = train_comb[train_comb['labels'].apply(lambda x: any(x[i] for i in rare_labels))]
        train_comb = pd.concat([train_comb, oversampled]).sample(frac=1, random_state=42).reset_index(drop=True)
    val_comb = val_comb.sample(frac=1, random_state=42).reset_index(drop=True)

    save_csv(train_comb, os.path.join(out_dir, "goemotions_empathetic_train.csv"))
    save_csv(val_comb, os.path.join(out_dir, "goemotions_empathetic_val.csv"))
    print("Saved combined train/val CSVs.")
    print(f"Training samples: {len(train_comb)}, Validation samples: {len(val_comb)}")
        
def main(out_dir, label_map_path):
    os.makedirs(out_dir, exist_ok=True)
    label_map = load_label_map(label_map_path)
    ge_train, ge_val, ge_test = convert_goemotions(out_dir, label_map)
    emp_train, emp_val, emp_test = convert_empathetic(out_dir, label_map)
    combine_and_split(out_dir, label_map)
    save_csv(ge_test, os.path.join(out_dir, "goemotions_test.csv"))
    print("All done. Look for CSV files in:", out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="data/processed")
    parser.add_argument("--label_map", type=str, default="data/label_map.json")
    args = parser.parse_args()
    main(args.out_dir, args.label_map)