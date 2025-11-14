# src/preprocess.py
import os
import pandas as pd
from src.config import cfg

def load_csvs(input_dir):
    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".csv")]
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print("Failed read", f, e)
    return dfs

def detect_cols(df):
    text_cols = [c for c in df.columns if c.lower() in ("text","sentence","tweet","content","message")]
    emo_cols = [c for c in df.columns if c.lower() in ("emotion","label","labels","sentiment","mapped","mapped_label","mapped_emotion")]
    if not text_cols:
        # try any string-like column as text fallback
        strs = [c for c in df.columns if df[c].dtype == object]
        if strs:
            text_cols = [strs[0]]
    if not emo_cols:
        # allow absence (we'll leave emotion blank)
        emo_cols = []
    # prepare base
    new = {}
    if text_cols:
        new["text"] = df[text_cols[0]].astype(str)
    else:
        raise ValueError("No text-like column found.")
    if emo_cols:
        new["emotion"] = df[emo_cols[0]].astype(str)
    else:
        new["emotion"] = ""
    # additional fields preserved if exist
    if "source" in df.columns:
        new["source"] = df["source"]
    else:
        new["source"] = "unknown"
    return pd.DataFrame(new)

def basic_merge(input_dir, out_path):
    dfs = load_csvs(input_dir)
    normalized = []
    for df in dfs:
        try:
            normalized.append(detect_cols(df))
        except Exception as e:
            print("skip file", e)
    if not normalized:
        raise ValueError("No input CSVs normalized.")
    merged = pd.concat(normalized, ignore_index=True)
    # add placeholders for mapped and label_vector if missing
    if "mapped" not in merged.columns:
        merged["mapped"] = merged["emotion"].fillna("").astype(str)
    if "label_vector" not in merged.columns:
        # create empty vector (all zeros) placeholder for later mapping
        merged["label_vector"] = merged.apply(lambda _: str([0]*len(cfg.label_list)), axis=1)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    merged.to_csv(out_path, index=False)
    print("Saved merged to", out_path)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", default="data/raw")
    p.add_argument("--out_path", default="data/processed/all_raw.csv")
    args = p.parse_args()
    basic_merge(args.input_dir, args.out_path)
