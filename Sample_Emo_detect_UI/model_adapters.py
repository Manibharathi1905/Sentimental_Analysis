# utils/model_adapters.py
from typing import Dict, Optional
from PIL import Image
from config import config
import logging
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextEmotionAdapter:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        if not config.multimodal_enabled:
            return
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
            self.tokenizer = AutoTokenizer.from_pretrained(config.emotion_model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(config.emotion_model_name)
            self.pipeline = pipeline(
                'text-classification',
                model=self.model,
                tokenizer=self.tokenizer,
                top_k=None,
                device=0 if config.device == "cuda" else -1
            )
            self._loaded = True
        except Exception as e:
            logger.error(f"Failed to load text emotion model: {e}")
            self._loaded = False

    def predict(self, text: str) -> Dict[str, float]:
        self._load()
        if not self._loaded or not text:
            # Fallback heuristic
            lower = text.lower() if text else ""
            emotions = {
                'anger': 0.0, 'disgust': 0.0, 'fear': 0.0, 'joy': 0.0,
                'neutral': 1.0, 'sadness': 0.0, 'surprise': 0.0
            }
            if any(word in lower for word in ['sad', 'lost', 'grief', 'cry']):
                emotions['sadness'] = 0.8
                emotions['neutral'] = 0.2
            elif any(word in lower for word in ['happy', 'joy', 'excited']):
                emotions['joy'] = 0.9
                emotions['neutral'] = 0.1
            elif any(word in lower for word in ['anxiety', 'nervous', 'worried']):
                emotions['fear'] = 0.7
                emotions['neutral'] = 0.3
            elif any(word in lower for word in ['angry', 'frustrated', 'mad']):
                emotions['anger'] = 0.7
                emotions['neutral'] = 0.3
            elif any(word in lower for word in ['fear', 'scared', 'afraid']):
                emotions['fear'] = 0.7
                emotions['neutral'] = 0.3
            return emotions
        try:
            out = self.pipeline([text])[0]
            return {item['label'].lower(): float(item['score']) for item in out}
        except Exception as e:
            logger.error(f"Error in text emotion prediction: {e}")
            return {'neutral': 1.0}

class DialogueAdapter:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        if not config.dialogue_enabled:
            return
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            self.tokenizer = AutoTokenizer.from_pretrained(config.dialogue_model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(
                config.dialogue_model_name,
                device_map="auto" if config.device == "cuda" else None,
                load_in_8bit=True if config.device == "cuda" else False
            )
            self._loaded = True
        except Exception as e:
            logger.error(f"Failed to load dialogue model: {e}")
            self._loaded = False

    def generate(self, prompt: str, max_length: int = 150) -> str:
        self._load()
        if not self._loaded:
            from utils.response_generator import response_generator
            return response_generator.generate_empathetic_response(
                prompt, [], {"neutral": 1.0}, verbosity=1, mode="therapeutic"
            )
        try:
            inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=config.max_length).to(config.device)
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=0.7,
                do_sample=True
            )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        except Exception as e:
            logger.error(f"Error in dialogue generation: {e}")
            return "I'm here to support you. What's on your mind?"

class VisionAdapter:
    def __init__(self):
        self.model = None
        self.processor = None
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        if not config.vision_enabled:
            return
        try:
            from transformers import CLIPProcessor, CLIPModel
            self.processor = CLIPProcessor.from_pretrained(config.vision_model_name)
            self.model = CLIPModel.from_pretrained(config.vision_model_name)
            self._loaded = True
        except Exception as e:
            logger.error(f"Failed to load vision model: {e}")
            self._loaded = False

    def predict_image_emotion(self, image: Image.Image) -> Dict[str, float]:
        self._load()
        if not self._loaded or not image:
            return {
                'anger': 0.0, 'disgust': 0.0, 'fear': 0.0, 'joy': 0.0,
                'neutral': 1.0, 'sadness': 0.0, 'surprise': 0.0
            }
        try:
            inputs = self.processor(images=image, return_tensors='pt').to(config.device)
            outputs = self.model(**inputs)
            # Placeholder: Requires fine-tuned CLIP for emotion classification
            return {'neutral': 1.0}  # Update with actual emotion mapping
        except Exception as e:
            logger.error(f"Error in image emotion prediction: {e}")
            return {'neutral': 1.0}

class StoryAdapter:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        if not config.story_generation_enabled:
            return
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            self.tokenizer = AutoTokenizer.from_pretrained(config.story_model_name)
            self.model = AutoModelForCausalLM.from_pretrained(config.story_model_name).to(config.device)
            self._loaded = True
        except Exception as e:
            logger.error(f"Failed to load story model: {e}")
            self._loaded = False

    def generate(self, prompt: str, max_length: int = 300) -> str:
        self._load()
        if not self._loaded:
            return "Once upon a time, there was hope..."
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(config.device)
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=0.8,
                do_sample=True
            )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Error in story generation: {e}")
            return "A motivational tale of perseverance..."

# Global instances
text_emotion_adapter = TextEmotionAdapter()
dialogue_adapter = DialogueAdapter()
vision_adapter = VisionAdapter()
story_adapter = StoryAdapter()