# config.py
import warnings
from types import SimpleNamespace
from datetime import timedelta
import random
import torch  # Ensure torch is imported for device detection

# -----------------------------
# Device Configuration - Enhanced
# -----------------------------
try:
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _torch_dtype = torch.float16 if _device == "cuda" else torch.float32
except Exception:
    warnings.warn("Torch not available; falling back to CPU", RuntimeWarning)
    _device = "cpu"
    _torch_dtype = torch.float32

# -----------------------------
# Core Model & Pipeline Config - Expanded for professional use, with smaller demo model
# -----------------------------
_cfg = {
    # LLMs - Switched to smaller DialoGPT for demo (avoids large downloads)
    "llm_model_name": "AliiaR/DialoGPT-medium-empathetic-dialogues",  # Smaller empathetic model (~1GB)
    "dialogue_model_name": "microsoft/DialoGPT-medium",  # Fallback dialogue (~500MB)
    "story_model_name": "gpt2-medium",  # Story generation (~500MB)

    # Emotion Detection - Deep models
    "emotion_model_name": "SamLowe/roberta-base-go_emotions",  # Primary deep classifier (~500MB)
    "emotion_model_name2": "j-hartmann/emotion-english-distilroberta-base",  # Secondary for nuance (~250MB)

    # Vision & Multimodal
    "vision_model_name": "openai/clip-vit-base-patch32",  # For image emotion (~130MB)
    "image_model_name": "runwayml/stable-diffusion-v1-5",  # For generative images (optional, disabled for demo)
    "vision_enabled": True,

    # Device & Runtime - Enhanced
    "device": _device,
    "torch_dtype": str(_torch_dtype),
    "batch_size": 8,
    "epochs": 3,
    "max_length": 512,
    "use_quantization": True,  # 8-bit for efficiency on smaller models
    "use_torch_compile": False,  # Experimental speed-up
    "cache_enabled": True,
    "image_generation_enabled": False,  # Toggle heavy diffusion models - Disabled for demo

    # Dialogue & Story Management - Dynamic thresholds
    "dialogue_enabled": True,
    "multimodal_enabled": True,  # Image/audio emotion support
    "story_generation_enabled": True,
    "story_emotion_threshold": 0.3,        # Min intensity for story
    "story_interaction_threshold": 5,      # After 5 messages
    "llm_generation_enabled": True,
    "max_story_length": 800,  # Tokens for detailed stories

    # Emotion Monitoring & Thresholds - Fine-tuned
    "emotions_to_monitor": ["sadness", "grief", "guilt", "anger", "fear", "helplessness", "powerlessness", "self_doubt", "disappointment", "frustration"],
    "low_emotion_threshold": 0.4,
    "high_emotion_threshold": 0.8,        # Strong interventions
    "risk_threshold": 0.75,               # Crisis flag
    "intensity_weighting": 0.6,           # For dynamic adaptation

    # UI & Frontend defaults - Professional theme
    "theme": "dark",
    "verbosity_level": 2,                 # Default balanced
    "session_timeout": timedelta(hours=24),  # Auto-expire inactive sessions
}

config = SimpleNamespace(**_cfg)

# -----------------------------
# Storage Configuration - Enhanced with more paths
# -----------------------------
_storage_cfg = {
    "db_path": "data/chat.db",
    "semantic_memory_limit": 50,           # Max interactions/session
    "cache_dir": "data/cache",             # Model/TTS cache
    "audio_dir": "data/temp_audio",        # TTS outputs
    "story_dir": "data/stories",           # Pre-gen stories
    "image_dir": "data/images",            # Generated visuals
    "video_dir": "data/videos",            # Motivational clips
    "backup_interval": timedelta(days=7),  # DB backups
}
storage_config = SimpleNamespace(**_storage_cfg)

# -----------------------------
# UI / Frontend Defaults - Enhanced modes
# -----------------------------
_ui_cfg = {
    "theme": "dark",
    "page_title": "EmoDude: Professional Emotional Support",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
    "enable_modes": ["empathetic_support", "therapeutic_guidance", "motivational_story", "crisis_support"],  # Expanded
    "max_chat_history": 50,  # UI limit for performance
}
ui_config = SimpleNamespace(**_ui_cfg)

# Backwards compatibility
config.ui_config = ui_config
config.storage = storage_config

# -----------------------------
# Dynamic Threshold Helper Functions - Enhanced logic
# -----------------------------
def should_trigger_story(emotion_intensity: float, user_message_count: int, risk_score: float = 0) -> bool:
    """Dynamic check whether to generate motivational story - Enhanced with risk"""
    threshold_intensity = config.story_emotion_threshold
    threshold_count = config.story_interaction_threshold
    # Dynamic variation based on randomness, intensity, and risk
    return (
        user_message_count >= threshold_count or
        emotion_intensity >= threshold_intensity or
        risk_score > 0.5 or  # High risk triggers sooner
        (emotion_intensity > 0.7 and random.random() > 0.3)
    )

def is_high_risk(risk_score: float, user_input: str) -> bool:
    """Check if user is high risk (suicidal/harm) - Expanded keywords"""
    crisis_keywords = ['suicide', 'harm', 'end it', 'kill myself', 'overdose', 'cut myself', 'no reason to live']
    return risk_score >= config.risk_threshold or any(word in user_input.lower() for word in crisis_keywords)

def get_adaptive_verbosity(current_emotion_intensity: float) -> int:
    """Dynamically adjust verbosity based on emotional state"""
    if current_emotion_intensity > 0.8:  # High intensity = more support
        return 3
    elif current_emotion_intensity < 0.3:  # Low = gentle probe
        return 1
    return 2  # Balanced default

# -----------------------------
# Model Loading Helpers - For lazy initialization
# -----------------------------
def load_emotion_models():
    """Explicitly load emotion models if needed for warm-up"""
    from utils.emotion_classifier import emotion_classifier
    return emotion_classifier

def load_llm_model():
    """Explicitly load LLM for warm-up - Now smaller model"""
    from utils.response_generator import response_generator
    return response_generator._model

# Example usage in app startup
if __name__ == "__main__":
    print(f"EmoDude Config Loaded - Device: {config.device}")
    print(f"Models Ready: Emotion={load_emotion_models()}, LLM={load_llm_model()}")