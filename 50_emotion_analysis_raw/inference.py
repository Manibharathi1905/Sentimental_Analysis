# --- robust loading for inference.py (replace current load logic) ---
import torch
import numpy as np
from transformers import AutoTokenizer
from model import EmotionClassifier
from utils import load_json
import os, sys, traceback

class EmotionPredictor:
    def __init__(self, model_path, model_name="distilbert-base-uncased", label_map_path="data/label_map.json", device=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() and device != "cpu" else "cpu")
        print(f"\nUsing device: {self.device}")
        if self.device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        self.label_map = load_json(label_map_path)           # expects {"sadness": 0, "joy": 1, ...}
        self.labels = {int(v): k for k, v in self.label_map.items()}  # reverse: index -> name
        print("Label map size:", len(self.label_map))

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # build model
        n_labels = len(self.label_map)
        self.model = EmotionClassifier(model_name=model_name, n_labels=n_labels)
        
        # robust loading:
        try:
            ckpt = torch.load(model_path, map_location=self.device)
            # If wrapped checkpoint e.g., {"state_dict": ...}
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                state = ckpt["state_dict"]
            elif isinstance(ckpt, dict) and any(k.startswith("backbone") or k.startswith("classifier") for k in ckpt.keys()):
                state = ckpt
            else:
                # probably a full model object
                try:
                    # if it's an nn.Module saved directly
                    if isinstance(ckpt, torch.nn.Module):
                        print("Loaded a full model object from file.")
                        self.model = ckpt.to(self.device)
                        self.model.eval()
                        print("Model loaded as full object. Skipping state_dict load.")
                        return
                    else:
                        # attempt to extract state_dict attribute if present
                        if hasattr(ckpt, "state_dict"):
                            state = ckpt.state_dict()
                        else:
                            state = ckpt
                except Exception:
                    state = ckpt

            # If state is set, load into model
            if isinstance(state, dict):
                # convert keys if they contain 'module.' prefix from DataParallel training
                new_state = {}
                for k,v in state.items():
                    new_k = k.replace("module.", "") if k.startswith("module.") else k
                    new_state[new_k] = v
                self.model.load_state_dict(new_state, strict=False)
                print("Loaded state_dict into EmotionClassifier (strict=False).")
            else:
                print("Unexpected checkpoint format; loaded object type:", type(state))
        except Exception as e:
            print("ERROR loading model checkpoint:", e)
            traceback.print_exc()
            raise e

        # finalize
        self.model.to(self.device)
        self.model.eval()
        # debug prints
        try:
            out_dim = getattr(self.model.classifier, "out_features", None) or getattr(self.model, "classifier_out_dim", None)
            print("Model classifier out dimension (out_features):", out_dim)
            print("Expected #labels (label_map):", len(self.label_map))
        except Exception as e:
            print("Couldn't introspect classifier dims:", e)

    def predict(self, text, threshold=0.4):
        enc = self.tokenizer(text, truncation=True, padding="max_length", max_length=256, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        # ensure label keys are ints
        labels = [self.labels[i] for i,p in enumerate(probs) if p>=threshold and i in self.labels]
        scores = {self.labels[i]: float(probs[i]) for i in range(len(probs)) if i in self.labels}
        return labels, scores
