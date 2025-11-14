# utils/tts_utils.py
import os
import logging
import uuid
from pathlib import Path
from typing import Optional, Dict
import hashlib
import asyncio

from gtts import gTTS

try:
    import edge_tts
except ImportError:
    edge_tts = None

try:
    import coqui_tts as coqui_ai_tts
except ImportError:
    coqui_ai_tts = None

from config import storage_config

logger = logging.getLogger(__name__)

AUDIO_DIR = Path(storage_config.audio_dir)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# Simple in-memory cache to avoid regenerating same text
_audio_cache: Dict[str, str] = {}

class TTSManager:
    def __init__(self, default_engine: str = "gtts"):
        self.default_engine = default_engine.lower()
        self.supported_engines = ["gtts", "edge", "coqui"]
        if self.default_engine not in self.supported_engines:
            logger.warning(f"Engine {self.default_engine} not supported. Falling back to gTTS.")
            self.default_engine = "gtts"

    def _hash_text(self, text: str) -> str:
        """Create a hash for caching purposes."""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    async def _generate_edge_audio(self, text: str, file_path: str, voice: str = "en-US-JennyNeural"):
        communicator = edge_tts.Communicate(text, voice)
        await communicator.save(str(file_path))

    def _generate_gtts_audio(self, text: str, file_path: str, language: str = "en"):
        tts = gTTS(text=text, lang=language)
        tts.save(str(file_path))

    def _generate_coqui_audio(self, text: str, file_path: str, voice: str = "alloy"):
        if coqui_ai_tts:
            # coqui_tts provides a high-level CLI/API; attempt common interface
            try:
                coqui_ai_tts.text_to_speech(text=text, voice=voice, output=str(file_path))
            except Exception:
                # fallback: try the top-level function name used by some builds
                coqui_ai_tts.tts(text=text, voice=voice, output=str(file_path))
        else:
            raise RuntimeError("Coqui TTS not available")

    async def generate_audio(self, text: str, voice: Optional[str] = None, language: str = "en") -> Dict[str, str]:
        """
        Dynamically generate audio using best TTS engine.
        Returns metadata: file path, engine, duration placeholder.
        """
        text_hash = self._hash_text(text)
        if text_hash in _audio_cache:
            logger.info("Using cached audio")
            return _audio_cache[text_hash]

        filename = f"{uuid.uuid4()}.mp3"
        file_path = AUDIO_DIR / filename
        engine_used = self.default_engine

        try:
            if self.default_engine == "gtts":
                self._generate_gtts_audio(text, file_path, language)
            elif self.default_engine == "edge" and edge_tts:
                engine_used = "edge"
                await self._generate_edge_audio(text, file_path, voice or "en-US-JennyNeural")
            elif self.default_engine == "coqui":
                engine_used = "coqui"
                self._generate_coqui_audio(text, file_path, voice or "alloy")
            else:
                logger.warning("Invalid TTS engine, falling back to gTTS")
                self._generate_gtts_audio(text, file_path, language)

            metadata = {
                "file_path": str(file_path),
                "engine": engine_used,
                "text_hash": text_hash,
                "duration_sec": None  # could integrate audio analysis if needed
            }

            _audio_cache[text_hash] = metadata
            logger.info(f"TTS generated: {file_path} using {engine_used}")
            return metadata

        except Exception as e:
            logger.error(f"TTS generation failed: {e}")
            return {"file_path": "", "engine": engine_used, "text_hash": text_hash, "duration_sec": None}

    async def generate_audio_url(self, text: str, voice: Optional[str] = None) -> str:
        """Generate audio and return a URL or local path placeholder."""
        meta = await self.generate_audio(text, voice)
        return f"/audio/{Path(meta['file_path']).name}" if meta["file_path"] else ""

# Global async instance
tts_manager = TTSManager(default_engine="gtts")


# Example async usage
if __name__ == "__main__":
    import asyncio
    async def test():
        text = "Hello! This is a dynamic test of EmoDude's motivational voice."
        url = await tts_manager.generate_audio_url(text)
        print(f"Generated audio URL: {url}")

    asyncio.run(test())

