# openai_client.py (Full Version)
from typing import Dict

try:
    import openai  # type: ignore
    OPENAI_AVAILABLE = True
except Exception:
    openai = None
    OPENAI_AVAILABLE = False


def generate_empathetic_response(api_key: str, user_input: str, history: str, emotions: Dict[str, float]) -> str:
    """Call OpenAI ChatCompletion to generate an empathetic response.

    If `openai` is not installed, return an informative fallback message.
    """
    if not OPENAI_AVAILABLE:
        return "(OpenAI not installed) To use OpenAI responses, install the openai package and provide an API key in the sidebar."

    openai.api_key = api_key
    dominant = max(emotions, key=emotions.get) if emotions else "neutral"
    system = (
        "You are an empathetic emotional support assistant. Respond compassionately, validate feelings, and offer gentle support. "
        "Keep responses concise, warm, and ask a follow-up question to continue the conversation."
    )
    user_msg = f"User said: {user_input}\nDetected emotions: {emotions}\nConversation history:\n{history}\nRespond empathetically and ask one open question."
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.8,
            max_tokens=300,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"(OpenAI error) I'm sorry — I couldn't generate a response right now. ({e})"


def generate_motivational_story(api_key: str, emotions: Dict[str, float]) -> Dict[str, str]:
    """Generate a motivational story via OpenAI.

    If `openai` is not installed, return a fallback indicating the package is missing.
    """
    if not OPENAI_AVAILABLE:
        return {"text": "(OpenAI not installed) Install the openai package to enable this feature.", "video_prompt": ""}

    openai.api_key = api_key
    dominant = max(emotions, key=emotions.get) if emotions else "resilience"
    prompt = (
        f"Write a short motivational story (200-300 words) based on a true incident about overcoming {dominant}. "
        "Make it inspiring, end with a positive message, and keep the tone gentle and hopeful."
    )
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=500,
        )
        text = resp.choices[0].message.content.strip()
        video_prompt = f"Animated short (30s) illustrating a story about {dominant}, gentle music, hopeful tone"
        return {"text": text, "video_prompt": video_prompt}
    except Exception as e:
        return {"text": f"(OpenAI error) Could not generate story: {e}", "video_prompt": ""}
