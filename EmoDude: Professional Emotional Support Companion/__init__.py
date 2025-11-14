# __init__.py (Full Version)
"""Lightweight convenience exports for the ``utils`` package.

Avoid importing heavy ML-related submodules at package import time.
This module provides lazy attribute access so callers can do
``from utils import chat_manager`` without importing large
dependencies until the attribute is actually used.
"""
from typing import Any
import importlib

__all__ = [
    "chat_manager",
    "response_generator",
    "motivational_story_generator",
    "emotion_classifier",
    "video_utils",
    "tts_utils",
    "multimodal_generator",
]


def _lazy_import(module_name: str) -> Any:
    """Import a submodule from the utils package and return it."""
    return importlib.import_module(f"utils.{module_name}")


def __getattr__(name: str) -> Any:
    """Lazily resolve commonly exported attributes.

    This avoids importing heavy submodules (transformers, torch, etc.)
    when someone only needs a small piece of functionality from the
    package. It keeps startup fast and reduces import-time side effects.
    """
    if name == "chat_manager":
        return _lazy_import("chat_manager").chat_manager
    if name == "response_generator":
        return _lazy_import("response_generator").response_generator
    if name == "motivational_story_generator":
        return _lazy_import("motivational_story_generator").generate_comprehensive_story
    if name == "emotion_classifier":
        return _lazy_import("emotion_classifier").emotion_classifier
    if name == "video_utils":
        return _lazy_import("video_utils")
    if name == "tts_utils":
        return _lazy_import("tts_utils")
    if name == "multimodal_generator":
        return _lazy_import("multimodal_generator")
    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)