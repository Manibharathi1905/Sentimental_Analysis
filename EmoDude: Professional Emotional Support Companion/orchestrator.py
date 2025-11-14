# orchestrator.py
from typing import Dict, List, Optional
from datetime import datetime
from utils.response_generator import response_generator
from utils.motivational_story_generator import motivational_story_generator
from utils.emotion_classifier import emotion_classifier
from utils.chat_manager import chat_manager, semantic_memory
import logging
import random

logger = logging.getLogger(__name__)

def handle_user_input(session_id: str,
                      user_input: str,
                      mode: str,
                      history: List[Dict],
                      verbosity: int = 2,
                      user_profile: Optional[Dict] = None) -> Dict:
    """Dynamic orchestration: adaptive empathy, motivation, memory integration."""
    
    if user_profile is None:
        user_profile = {}

    # --- Step 1: Deep Emotion Classification ---
    emotions_data = emotion_classifier.classify(text=user_input)
    emotions = emotions_data.get('emotions', {})
    risk_score = emotions_data.get('risk_score', 0)
    intensity = max(emotions.values()) if emotions else 0

    # --- Step 2: Reference semantic memory for context-aware reply ---
    past_context = semantic_memory.retrieve_relevant_context(session_id, user_input, top_k=3)
    context_summary = " ".join([ctx["interaction"] for ctx in past_context]) if past_context else ""

    # --- Step 3: Dynamic empathetic/therapeutic response ---
    dynamic_mode = "therapeutic" if mode == "therapeutic guidance" else "empathetic"
    response = response_generator.generate_empathetic_response(
        input_text=user_input,
        history=history,
        context=context_summary,
        emotions=emotions,
        verbosity=verbosity,
        mode=dynamic_mode,
        intensity=intensity
    )

    result = {
        "assistant_text": response,
        "emotions": emotions_data,
        "context_summary": context_summary
    }

    # --- Step 4: Crisis Detection & Immediate Guidance ---
    crisis_keywords = ['suicide', 'harm', 'end it', 'kill myself']
    if risk_score > 0.7 or any(word in user_input.lower() for word in crisis_keywords):
        result["action"] = "crisis"
        result["crisis_resources"] = {
            'immediate': "🚨 **IF IN IMMEDIATE DANGER:** Call emergency services immediately",
            'us_988': "🇺🇸 Call/text 988 (24/7, confidential)",
            'global_hotline': "🌍 www.findahelpline.com",
            'text_support': "📱 Text HOME to 741741 (US)",
            'professional_help': "🏥 Contact a licensed mental health professional"
        }
        result["assistant_text"] += "\n\n**Safety First:** Your safety is my priority. Please reach out to one of these resources now."
        return result

    # --- Step 5: Generate Adaptive Micro-strategies ---
    solutions = response_generator.generate_therapeutic_solutions(user_input, emotions)
    if solutions:
        # Pick a random micro-strategy to keep it dynamic
        chosen = random.choice(list(solutions.items()))
        result["solutions"] = solutions
        result["assistant_text"] += f"\n\n**Quick Tip:** {chosen[0]} → {chosen[1]}"

    # --- Step 6: Dynamic Motivational Story Trigger ---
    user_interactions = len([m for m in history if m["role"] == "user"])
    # Dynamic thresholds: intensity, frequency, and mode combined
    story_trigger = (
        (mode == "therapeutic guidance") or
        (mode == "empathetic support" and user_interactions >= 3) or
        (intensity > 0.7 and random.random() > 0.3)
    )
    if story_trigger:
        story = motivational_story_generator.generate_comprehensive_story(
            emotions=emotions,
            context=user_input,
            history=history,
            user_profile=user_profile
        )
        result["story"] = story
        result["assistant_text"] += "\n\n**Here's an inspirational narrative that might resonate with your journey:**"

    # --- Step 7: Adaptive Learning & Pattern Awareness ---
    recent_emotions = [msg.get("emotions", {}).get("primary", "neutral") for msg in history[-4:] if msg["role"] == "user"]
    if recent_emotions and len(set(recent_emotions)) == 1:
        dominant_emotion = recent_emotions[0]
        if intensity > 0.6:
            result["assistant_text"] += f"\n\n**Notice:** Your feeling of '{dominant_emotion}' has been persistent recently. Exploring this could help."

    # --- Step 8: Personalization: CBT / Mindfulness / Eclectic ---
    preferences = user_profile.get("preferences", {})
    approach = preferences.get("preferred_approach")
    if approach == "cbt":
        result["assistant_text"] += "\n\n**CBT Prompt:** What evidence supports this feeling? Can we reframe it differently?"
    elif approach == "mindfulness":
        result["assistant_text"] += "\n\n**Mindfulness Prompt:** Observe where this emotion is felt in your body. Can you breathe into it?"
    elif approach == "story-driven":
        result["assistant_text"] += "\n\n**Reflection:** Remember how past stories and experiences can guide you here."

    # --- Step 9: Store Interaction in Memory & DB ---
    semantic_memory.store_interaction(session_id, user_input, metadata={"emotions": emotions})
    semantic_memory.store_interaction(session_id, response, metadata={"emotions": emotions})
    chat_manager.add_exchange(session_id, user_input, response, emotions)

    return result

def update_user_profile(session_id: str, new_preferences: Dict, learned_interaction: Optional[Dict] = None):
    """Update user profile with dynamic learning and emotional trends."""
    try:
        profile_data = chat_manager.update_user_profile(
            session_id=session_id,
            preferences=new_preferences,
            learned_patterns=[learned_interaction] if learned_interaction else []
        )
        return profile_data
    except Exception as e:
        logger.error(f"Profile update failed: {e}")
        return None
