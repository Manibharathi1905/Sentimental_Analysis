#empathy_responder.py
from typing import Dict, Optional
from config import config
from utils.response_generator import response_generator
from utils.model_adapters import dialogue_adapter


class EmpathyResponder:
    """
    Thin wrapper to centralize empathetic response generation and safety filtering.

    Future: plug in a high-quality LLM or API-backed generator.
    """

    def __init__(self):
        # Placeholder for model handles if we integrate an LLM later
        self._model = None

    def safety_check(self, text: str) -> Dict[str, bool]:
        """
        Basic safety check. Returns a dict with flags like 'self_harm', 'violence', 'abuse'.
        For now use keyword heuristics. Replace with a proper moderation model in prod.
        """
        lower = (text or "").lower()
        flags = {
            'self_harm': any(w in lower for w in ['kill myself', 'i want to die', 'suicid', 'harm myself']),
            'violence': any(w in lower for w in ['kill', 'stab', 'shoot']),
            'abuse': any(w in lower for w in ['abuse', 'harass', 'assault'])
        }
        return flags

    def generate_response(self, prompt: str, history: Optional[list] = None, emotions: Optional[Dict[str, float]] = None) -> Dict[str, str]:
        """
        Generate an empathetic reply. Returns: {'text': reply, 'safety': flags}
        If safety flags indicate crisis/self-harm, return a crisis intervention guide message instead.
        """
        flags = self.safety_check(prompt)
        if flags.get('self_harm'):
            # Crisis-safe response
            crisis_msg = (
                "I'm really sorry you're feeling this way. If you're thinking about harming yourself, please contact local emergency services or a crisis hotline immediately. "
                "If you're in the United States, you can call or text 988 to reach the Suicide & Crisis Lifeline. If you're elsewhere, please look up local resources or let me know your country so I can find hotlines near you."
            )
            return {'text': crisis_msg, 'safety': flags}

        # Otherwise, prefer the dialogue adapter if enabled, then fall back to response_generator
        try:
            if hasattr(dialogue_adapter, 'generate') and config.dialogue_enabled:
                # Build a compact prompt including recent history and detected emotion
                hist_text = ''
                if history:
                    for m in history[-6:]:
                        hist_text += f"{m.get('role','')}: {m.get('content','')}\n"
                tone_hint = ''
                if emotions:
                    dom = max(emotions, key=emotions.get)
                    tone_hint = f"Tone: respond with empathy to {dom}."
                prompt_full = f"{hist_text}\nUser: {prompt}\n{tone_hint}\nAssistant:"
                reply = dialogue_adapter.generate(prompt_full)
            else:
                reply = response_generator.generate_empathetic_response(prompt, history, detected_emotions=emotions)
        except Exception:
            reply = response_generator.generate_empathetic_response(prompt, history, detected_emotions=emotions)
        return {'text': reply, 'safety': flags}


# Global instance
empathy_responder = EmpathyResponder()
