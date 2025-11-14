# utils/emotion_classifier.py
import logging
try:
    import torch
except Exception:
    # Defer detailed torch usage to runtime; keep module importable when torch is missing
    torch = None
import re
from config import config
from functools import lru_cache
from collections import defaultdict
import math
import numpy as np
from PIL import Image
from typing import Dict, List
import json
try:
    import nltk
    from nltk.corpus import wordnet as wn
    nltk.download('wordnet', quiet=True)
except ImportError:
    wn = None
# Defer sentence-transformers usage
_embed_model = None
_embed_model_class = None
_embed_util = None

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _get_cache_dir():
    """Return the configured cache dir from config, handling missing attributes."""
    try:
        storage = getattr(config, "storage", None)
        cache_dir = None
        if storage is not None:
            cache_dir = getattr(storage, "cache_dir", None)
        cache_dir = cache_dir or getattr(config, "cache_dir", None)
        if cache_dir:
            import os
            try:
                os.makedirs(cache_dir, exist_ok=True)
            except Exception:
                logger.debug(f"Could not create cache_dir '{cache_dir}', continuing without creating it.")
        return cache_dir
    except Exception as e:
        logger.debug(f"Error while resolving cache_dir: {e}")
        return None

class EmotionClassifier:
    def __init__(self):
        self.tokenizer1 = None
        self.model1 = None
        self.tokenizer2 = None
        self.model2 = None
        self.clip_processor = None
        self.clip_model = None
        self._loaded = False

        # Emotion label sets - Expanded for deeper classification
        self.emotion_labels1 = [
            "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
            "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
            "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism",
            "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral",
            "helplessness", "guilt", "frustration", "powerlessness", "self_doubt", "hope", "courage"
        ]
        self.emotion_labels2 = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
        self.emotion_labels_clip = ["happy", "sad", "angry", "fearful", "neutral"]

        # Keyword clusters - Enhanced with more nuanced terms for deep detection
        self.keyword_map = {
            "grief": ["loss", "died", "passed", "funeral", "pet", "gone", "heartbroken", "mourning", "bereavement", "sorrow", "widowed", "orphan"],
            "anger": ["angry", "furious", "mad", "argument", "rage", "irritated", "offended", "betrayed", "frustrated", "upset", "resentful", "bitter"],
            "guilt": ["should have", "regret", "fault", "blame", "sorry", "ashamed", "remorse", "apologize", "failed", "wronged", "betrayed_self"],
            "sadness": ["lonely", "hopeless", "depressed", "cry", "down", "tired", "miss", "heartache", "despair", "melancholy", "blue"],
            "fear": ["afraid", "scared", "terrified", "worried", "panic", "nervous", "anxious", "dread", "frightened", "phobia", "terror"],
            "helplessness": ["powerless", "stuck", "can't move", "useless", "helpless", "lost", "trapped", "overwhelmed", "defeated", "resigned"],
            "joy": ["happy", "grateful", "excited", "hopeful", "content", "relieved", "proud", "triumph", "elated", "blissful", "ecstatic"],
            "love": ["love", "care", "attached", "cherish", "dear", "affection", "bond", "adoration", "tenderness", "passion", "attachment"],
            "admiration": ["inspired", "admire", "respect", "awe", "impressed", "amazed", "hero", "brave", "courageous", "idolize"],
            "frustration": ["annoyed", "irritated", "stuck", "blocked", "hindered", "exasperated", "aggravated", "impatient", "discontent"],
            "disappointment": ["let down", "disappointed", "failed expectation", "bummed", "deflated", "underwhelmed"],
            "relief": ["relieved", "thankful", "eased", "calmed", "soothed", "peaceful"],
            "surprise": ["surprised", "shocked", "astonished", "unexpected", "startled"],
            "confusion": ["confused", "puzzled", "bewildered", "lost", "uncertain", "dazed"],
            "pride": ["proud", "accomplished", "achieved", "honored", "victorious"],
            "self_doubt": ["doubt", "insecure", "unworthy", "imposter", "not good enough", "failure"],
            "hope": ["hope", "optimistic", "better tomorrow", "light at end", "possibility"],
            "courage": ["brave", "courage", "face fear", "stand up", "resilient"]
        }
        self.negative_emotions = [
            "anger", "annoyance", "disappointment", "disapproval", "disgust", "embarrassment",
            "fear", "grief", "nervousness", "remorse", "sadness", "guilt", "helplessness",
            "frustration", "powerlessness", "self_doubt"
        ]
        self.positive_emotions = [
            "joy", "love", "admiration", "excitement", "gratitude", "pride", "relief",
            "hope", "courage", "approval", "amusement", "curiosity", "desire"
        ]
        self._expanded_keywords = None

    def _load_models(self):
        if self._loaded:
            return
        try:
            logger.info("Loading emotion models...")
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            # import CLIP components lazily
            try:
                from transformers import CLIPProcessor, CLIPModel
            except Exception:
                CLIPProcessor = None
                CLIPModel = None
            cache_dir = _get_cache_dir()

            model1_name = getattr(config, "emotion_model_name", "SamLowe/roberta-base-go_emotions")
            model2_name = getattr(config, "emotion_model_name2", "j-hartmann/emotion-english-distilroberta-base")
            torch_dtype = getattr(config, "torch_dtype", None)
            if torch_dtype and isinstance(torch_dtype, str):
                torch_dtype = getattr(torch, torch_dtype, torch.float32)  # Fix str dtype error

            self.tokenizer1 = AutoTokenizer.from_pretrained(model1_name, cache_dir=cache_dir)
            if torch_dtype is not None:
                self.model1 = AutoModelForSequenceClassification.from_pretrained(
                    model1_name, torch_dtype=torch_dtype, cache_dir=cache_dir
                ).eval()
            else:
                self.model1 = AutoModelForSequenceClassification.from_pretrained(
                    model1_name, cache_dir=cache_dir
                ).eval()
            self.model1.to(getattr(config, "device", "cpu"))

            self.tokenizer2 = AutoTokenizer.from_pretrained(model2_name, cache_dir=cache_dir)
            if torch_dtype is not None:
                self.model2 = AutoModelForSequenceClassification.from_pretrained(
                    model2_name, torch_dtype=torch_dtype, cache_dir=cache_dir
                ).eval()
            else:
                self.model2 = AutoModelForSequenceClassification.from_pretrained(
                    model2_name, cache_dir=cache_dir
                ).eval()
            self.model2.to(getattr(config, "device", "cpu"))

            if getattr(config, "multimodal_enabled", False) and CLIPProcessor and CLIPModel:
                self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir=cache_dir)
                if torch_dtype is not None:
                    self.clip_model = CLIPModel.from_pretrained(
                        "openai/clip-vit-base-patch32", torch_dtype=torch_dtype, cache_dir=cache_dir
                    ).eval()
                else:
                    self.clip_model = CLIPModel.from_pretrained(
                        "openai/clip-vit-base-patch32", cache_dir=cache_dir
                    ).eval()
                self.clip_model.to(getattr(config, "device", "cpu"))
            self._loaded = True
            logger.info("Emotion models loaded successfully.")
        except Exception as e:
            logger.warning(f"⚠️ Model loading failed, fallback to heuristic mode: {e}")
            self._loaded = False

    def _expand_keywords(self):
        if self._expanded_keywords is not None:
            return self._expanded_keywords
        expanded = defaultdict(set)
        for emotion, words in self.keyword_map.items():
            for w in words:
                expanded[emotion].add(w)
                if wn:
                    for syn in wn.synsets(w):
                        for lemma in syn.lemma_names():
                            if len(lemma) > 2 and lemma.isalpha():
                                expanded[emotion].add(lemma.replace('_', ' '))
                        # Add hypernyms for broader matching
                        for hyper in syn.hypernyms():
                            for h_lemma in hyper.lemma_names():
                                expanded[emotion].add(h_lemma.replace('_', ' '))
        # If sentence-transformers is available, instantiate the model lazily
        global _embed_model, _embed_model_class, _embed_util
        if _embed_model is None:
            try:
                from sentence_transformers import SentenceTransformer as _ST, util as _util
                _embed_model_class = _ST
                _embed_util = _util
                cache_dir = _get_cache_dir()
                _embed_model = _ST("all-MiniLM-L6-v2", cache_folder=cache_dir) if cache_dir else _ST("all-MiniLM-L6-v2")
            except Exception:
                _embed_model = None

        if _embed_model is not None and _embed_util is not None:
            try:
                logger.info("Expanding keywords semantically...")
                all_words = list({w for ws in expanded.values() for w in ws})
                word_embs = _embed_model.encode(all_words, convert_to_tensor=True, device=getattr(config, "device", "cpu"))
                for i, w in enumerate(all_words):
                    sim_scores = _embed_util.cos_sim(word_embs[i], word_embs)[0]
                    for j, score in enumerate(sim_scores):
                        if score > 0.7 and i != j:
                            for emo, kws in expanded.items():
                                if w in kws:
                                    expanded[emo].add(all_words[j])
            except Exception:
                logger.debug("Sentence-transformers expansion failed; continuing without semantic expansion.")
        self._expanded_keywords = {e: list(ws) for e, ws in expanded.items()}
        return self._expanded_keywords

    def _contextual_boosts(self, text: str):
        text_lower = text.lower()
        boosts = defaultdict(float)
        expanded = self._expand_keywords()
        for emotion, keywords in expanded.items():
            for kw in keywords:
                if re.search(rf"\b{re.escape(kw)}\b", text_lower):
                    boosts[emotion] += min(0.2 + len(kw) / 25.0, 0.8)
        total = sum(boosts.values()) or 1.0
        return {k: round(math.pow(v / total, 0.7), 4) for k, v in boosts.items()}

    def _classify_image(self, image: Image.Image):
        if not self._loaded or not self.clip_processor or not self.clip_model:
            return {"neutral": 1.0}
        try:
            inputs = self.clip_processor(
                images=image, return_tensors="pt", padding=True
            ).to(config.device)
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
            emotion_texts = [f"A person feeling {emo}" for emo in self.emotion_labels_clip]
            text_inputs = self.clip_processor(text=emotion_texts, return_tensors="pt", padding=True).to(config.device)
            with torch.no_grad():
                text_features = self.clip_model.get_text_features(**text_inputs)
            similarities = torch.nn.functional.cosine_similarity(image_features, text_features)
            scores = torch.softmax(similarities, dim=-1).cpu().numpy()
            return {emo: float(score) for emo, score in zip(self.emotion_labels_clip, scores)}
        except Exception as e:
            logger.error(f"Image emotion classification failed: {e}")
            return {"neutral": 1.0}

    def classify_emotion(self, text: str, image: Image.Image = None):
        if not text or not isinstance(text, str):
            return {"neutral": 1.0}, 0.0
        self._load_models()
        try:
            if self._loaded:
                inputs1 = self.tokenizer1(text, return_tensors="pt", truncation=True, max_length=config.max_length).to(config.device)
                inputs2 = self.tokenizer2(text, return_tensors="pt", truncation=True, max_length=config.max_length).to(config.device)
                with torch.no_grad():
                    out1 = self.model1(**inputs1)
                    out2 = self.model2(**inputs2)
                    s1 = torch.softmax(out1.logits, dim=-1).squeeze().cpu().numpy()
                    s2 = torch.softmax(out2.logits, dim=-1).squeeze().cpu().numpy()
                scores1 = {l: float(v) for l, v in zip(self.emotion_labels1, s1)}
                scores2 = {l: float(v) for l, v in zip(self.emotion_labels2, s2)}
                merged = {k: 0.6 * scores1.get(k, 0) + 0.4 * scores2.get(k, 0) for k in set(scores1) | set(scores2)}
                boosts = self._contextual_boosts(text)
                for emo, val in boosts.items():
                    merged[emo] = merged.get(emo, 0) + val * 0.3
                if image:
                    img_scores = self._classify_image(image)
                    for emo, score in img_scores.items():
                        merged[emo] = merged.get(emo, 0) + 0.2 * score
            else:
                merged = self._heuristic(text)
            total = sum(merged.values()) or 1.0
            emotion_scores = {k: min(round(v / total, 4), 1.0) for k, v in merged.items() if v > 0.1}
            if not emotion_scores:
                emotion_scores = {"neutral": 1.0}
            # Risk score - Adjusted for positive emotions like love/joy to not trigger crisis
            risk_weights = {}
            for e in emotion_scores:
                if e in self.negative_emotions:
                    risk_weights[e] = 2.5 if e in ["grief", "suicidal", "despair"] else 2.0 if e in ["sadness", "guilt", "fear"] else 1.5
                else:
                    risk_weights[e] = 0.0  # Positive = 0 risk
            risk_score = sum(emotion_scores.get(e, 0) * risk_weights.get(e, 0) for e in emotion_scores)
            risk_score = min(round(risk_score, 3), 1.0)
            logger.info(f"Emotions: {emotion_scores}, Risk: {risk_score}")
            return emotion_scores, risk_score
        except Exception as e:
            logger.error(f"Emotion classification failed: {e}")
            return self._heuristic(text), 0.0

    def _heuristic(self, text: str):
        text_lower = text.lower()
        # Enhanced heuristic with positive emotion detection
        if any(kw in text_lower for kw in self.keyword_map["love"] + self.keyword_map["joy"] + ["butterfly", "freedom", "happy", "joy", "attached"]):
            return {"love": 0.5, "joy": 0.3, "sadness": 0.2}
        for emo, kws in self.keyword_map.items():
            if any(kw in text_lower for kw in kws):
                return {emo: 0.8, "neutral": 0.2}
        return {"neutral": 1.0}

    def is_low_emotion(self, emotions: Dict[str, float]) -> bool:
        low_score = sum(emotions.get(emo, 0) for emo in config.emotions_to_monitor)
        return low_score > config.low_emotion_threshold

    # New method: Export emotion history
    def export_emotion_history(self, emotions_list: List[Dict]) -> str:
        """Export emotion data to JSON string for backup"""
        return json.dumps(emotions_list, default=str)

    # New method: Detect emotion trends
    def detect_emotion_trends(self, emotions_history: List[Dict]) -> Dict:
        """Analyze trends in emotion history"""
        if not emotions_history:
            return {"trend": "none"}
        primaries = [e.get("primary", "neutral") for e in emotions_history]
        if len(set(primaries)) == 1:
            return {"trend": "stable", "dominant": primaries[0]}
        elif primaries[-1] in self.positive_emotions and primaries[0] in self.negative_emotions:
            return {"trend": "improving", "from": primaries[0], "to": primaries[-1]}
        elif primaries[-1] in self.negative_emotions and primaries[0] in self.positive_emotions:
            return {"trend": "declining", "from": primaries[0], "to": primaries[-1]}
        else:
            return {"trend": "fluctuating", "dominant": max(set(primaries), key=primaries.count)}

emotion_classifier = EmotionClassifier()