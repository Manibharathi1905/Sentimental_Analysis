# utils/emotion_detector.py
# Deep Emotion Detection Utility

from typing import Dict, Optional
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
# Use built-in generics (PEP 585) for annotations on Python 3.9+
from config import config

class EmotionDetector:
    def __init__(self):
        # Defer heavy model loading until first use
        self.device = None  # will be set lazily
        self._tokenizer = None
        self._model = None
        self.emotion_pipeline = None
        self._has_pipeline = False
        self.emotion_labels = [
            'anger', 'fear', 'joy', 'love', 'sadness', 'surprise', 'neutral', 'optimism'
        ]

    def _ensure_pipeline(self):
        if self._has_pipeline:
            return
        try:
            # Lazy import torch to avoid top-level side-effects during app import
            import torch
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # Try to load tokenizer and model and create pipeline
            self._tokenizer = AutoTokenizer.from_pretrained(config.emotion_model_name)
            model_dtype = torch.float16 if (self.device and self.device.type == "cuda") else None
            # Some HF versions accept dtype, others do not; guard it
            kwargs = {}
            if model_dtype is not None:
                kwargs['torch_dtype'] = model_dtype
            self._model = AutoModelForSequenceClassification.from_pretrained(
                config.emotion_model_name,
                **kwargs,
            )
            try:
                self._model.to(self.device)
            except Exception:
                pass

            self.emotion_pipeline = pipeline(
                "text-classification",
                model=self._model,
                tokenizer=self._tokenizer,
                device=0 if self.device.type == "cuda" else -1,
                top_k=None,
            )
            self._has_pipeline = True
        except Exception as e:
            # If pipeline cannot be created (missing deps, torch version), fall back to lightweight keyword rules
            print("Warning: emotion pipeline unavailable, using lightweight fallback. ", e)
            self._has_pipeline = False
    
    def detect_emotions(self, text: Optional[str]) -> Dict[str, float]:
        """
        Detect emotions from user input text.
        Returns dict of emotion: probability.
        """
        # Defensive: handle None or non-string inputs
        if text is None or not isinstance(text, str) or not text.strip():
            return {"neutral": 1.0}

        # Truncate text to the model's maximum sequence length
        max_length = 512  # Maximum token length supported by the model
        if len(text) > max_length:
            text = text[:max_length]
            # Log truncation instead of printing to avoid clutter
            import logging
            logging.warning("Input text truncated to fit model's maximum sequence length.")

        # Prefer the full pipeline if available; otherwise use a simple keyword-based fallback
        self._ensure_pipeline()
        emotions = {}
        if self._has_pipeline and self.emotion_pipeline is not None:
            try:
                # Use pipeline in batch mode for efficiency
                results = self.emotion_pipeline([text])[0]
                # results is a list of dicts like [{'label': 'JOY', 'score': 0.9}, ...]
                for item in results:
                    label = item.get('label', '').lower()
                    score = item.get('score', 0.0)
                    emotions[label] = score
            except Exception as e:
                logging.error(f"Error during emotion detection: {e}")
                emotions = {"neutral": 1.0}
        else:
            # Lightweight heuristic: look for keywords indicating sadness or loss
            lower = text.lower()
            if any(w in lower for w in ["lost my", "lost", "loss", "missing", "goodbye", "farewell"]) and any(pet in lower for pet in ["cat", "dog", "pet"]):
                emotions = {"sadness": 0.99, "neutral": 0.01}
            elif any(w in lower for w in ["happy", "joy", "glad", "excited"]):
                emotions = {"joy": 0.9, "neutral": 0.1}
            else:
                emotions = {"neutral": 1.0}

        # Normalize to ensure sum=1
        total = sum(emotions.values())
        if total > 0:
            emotions = {k: v / total for k, v in emotions.items()}

        return emotions
    
    def is_low_emotion(self, emotions: Dict[str, float]) -> bool:
        """
        Check if detected emotions indicate low mood (e.g., high sadness/fear/anger).
        """
        low_score = sum(emotions.get(emo, 0) for emo in config.emotions_to_monitor)
        return low_score > config.low_emotion_threshold

# Global instance
emotion_detector = EmotionDetector()
