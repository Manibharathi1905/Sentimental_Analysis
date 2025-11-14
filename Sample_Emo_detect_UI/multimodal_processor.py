from transformers import pipeline
from utils.model_adapters import text_emotion_adapter, vision_adapter, dialogue_adapter

# Multimodal input handling
class MultimodalProcessor:
    def __init__(self):
        self.text_pipeline = pipeline("text-classification", model=text_emotion_adapter.model, tokenizer=text_emotion_adapter.tokenizer)
        self.vision_adapter = vision_adapter
        self.dialogue_adapter = dialogue_adapter

    def process_text(self, text):
        """Process text input for emotion detection."""
        return self.text_pipeline(text)

    def process_image(self, image):
        """Process image input for emotion detection."""
        return self.vision_adapter.predict_image_emotion(image)

    def process_dialogue(self, prompt):
        """Generate dialogue response."""
        return self.dialogue_adapter.generate(prompt)

# Global instance
multimodal_processor = MultimodalProcessor()