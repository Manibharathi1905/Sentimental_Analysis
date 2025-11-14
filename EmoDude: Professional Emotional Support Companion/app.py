# app.py
import streamlit as st
from utils.emotion_classifier import emotion_classifier
from utils.response_generator import response_generator
from utils.motivational_story_generator import generate_comprehensive_story, motivational_story_generator
from utils.chat_manager import chat_manager
from config import config
import logging
import os
import json
from datetime import datetime
from PIL import Image

logging.basicConfig(level=logging.INFO, filename="emodude.log", filemode="a")
logger = logging.getLogger(__name__)

# Page Configuration
st.set_page_config(
    page_title=config.ui_config.page_title,
    layout=config.ui_config.layout,
    initial_sidebar_state=config.ui_config.initial_sidebar_state,
    page_icon="🤖"
)

# Custom CSS - Enhanced for professional look
st.markdown("""
    <style>
        .main { 
            background: linear-gradient(135deg, #0f1419 0%, #1a1a2e 50%, #16213e 100%); 
            color: #e2e8f0; 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
        }
        .stButton>button { 
            background: linear-gradient(135deg, #1d4ed8 0%, #3b82f6 100%); 
            color: #ffffff; 
            border: none; 
            border-radius: 12px; 
            font-weight: bold;
            padding: 10px 24px;
            transition: all 0.3s;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(59, 130, 246, 0.4);
        }
        .stTextArea textarea, .stTextInput input { 
            background-color: #1e293b; 
            color: #e2e8f0; 
            border: 1px solid #334155; 
            border-radius: 12px; 
            padding: 10px; 
        }
        .stSidebar { 
            background: linear-gradient(135deg, #111827 0%, #1f2937 100%); 
        }
        .emotion-card {
            background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
        }
        .solution-card {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            padding: 12px;
            border-radius: 8px;
            margin: 8px 0;
        }
        .story-card {
            background: linear-gradient(135deg, #7c3aed 0%, #a78bfa 100%);
            padding: 20px;
            border-radius: 12px;
            margin: 15px 0;
        }
        .crisis-alert {
            background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);
            padding: 15px;
            border-radius: 10px;
            border: 2px solid #fca5a5;
        }
        .video-section {
            background: linear-gradient(135deg, #f59e0b 0%, #fbbf24 100%);
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
        }
        .response-section {
            background: linear-gradient(135deg, #8b5cf6 0%, #a78bfa 100%);
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
        }
    </style>
""", unsafe_allow_html=True)

# Header - Enhanced with professional branding
st.markdown('<h1 style="text-align: center;">🤖 EmoDude: Professional Emotional Support Companion</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #94a3b8;">Your empathetic AI therapist powered by advanced emotional intelligence and deep learning models</p>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #64748b; font-style: italic;">Built for compassionate listening, validation, and transformative support</p>', unsafe_allow_html=True)
st.markdown("---")

# Initialize Session State - Enhanced with profile tracking
if "session_id" not in st.session_state:
    st.session_state.session_id = chat_manager.new_session()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "interaction_count" not in st.session_state:
    st.session_state.interaction_count = 0
if "user_profile" not in st.session_state:
    st.session_state.user_profile = {}

# Sidebar - Enhanced with more options
with st.sidebar:
    st.markdown('<h2 style="color: #3b82f6;">🎯 Mode Selection</h2>', unsafe_allow_html=True)
    mode = st.radio(
        "Choose Functionality:",
        ("🧠 Mental Health Post Analysis", "💬 Professional Chat Support"),
        index=1,
        help="Select how you want to interact with EmoDude"
    )
    
    st.markdown("---")
    st.markdown('<h3 style="color: #e2e8f0;">⚙️ Response Settings</h3>', unsafe_allow_html=True)
    verbosity = st.slider(
        "Response Depth",
        min_value=1,
        max_value=3,
        value=2,
        help="1=Concise, 2=Balanced, 3=Comprehensive"
    )
    therapy_style = st.selectbox(
        "Preferred Therapy Approach",
        ["Eclectic", "CBT", "Mindfulness", "Narrative", "DBT"],
        index=0,
        help="Tailors responses to your style"
    )
    st.session_state.user_profile["preferred_approach"] = therapy_style
    
    st.markdown("---")
    if st.button("🔄 New Session", use_container_width=True):
        st.session_state.session_id = chat_manager.new_session()
        st.session_state.messages = []
        st.session_state.interaction_count = 0
        st.session_state.user_profile = {}
        st.success("✅ New session started! Fresh emotional space created.")
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 📊 Session Info")
    st.info(f"**Messages:** {len(st.session_state.messages)}\n**Session ID:** {st.session_state.session_id[:20]}...\n**Dominant Emotion:** {st.session_state.get('last_emotion', 'Neutral')}\n**Therapy Style:** {therapy_style}")

# ===========================
# MODE 1: MENTAL HEALTH POST ANALYSIS - Enhanced professional flow
# ===========================
if mode == "🧠 Mental Health Post Analysis":
    st.markdown('<h2 style="color: #3b82f6;">🔍 Deep Emotional Analysis & Therapeutic Support</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color: #94a3b8;">Share a mental health forum post or personal reflection. EmoDude will provide deep analysis, empathetic validation, tailored therapeutic solutions, and an inspiring motivational story with video.</p>')
    
    # Input with placeholder example
    post = st.text_area(
        "📝 Share your thoughts or a mental health forum post:",
        placeholder="Example: Last Friday, I saved a little butterfly with a damaged wing... (your full sample post here for deep analysis)",
        height=250,
        key="post_input"
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button("🚀 Analyze & Generate Professional Support", use_container_width=True, type="primary")
    
    if analyze_button:
        if not post.strip():
            st.error("⚠️ Please enter some text to analyze.")
        else:
            with st.spinner("🔄 Conducting deep emotional analysis and generating comprehensive support..."):
                try:
                    # 1. DEEP EMOTION DETECTION - Enhanced output
                    emotions, risk = emotion_classifier.classify_emotion(post)
                    primary = max(emotions, key=emotions.get)
                    st.session_state.last_emotion = primary.capitalize()
                    
                    st.markdown("### 1️⃣ Detected Emotions - Deep Analysis")
                    emotion_display = []
                    for emo, score in sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:6]:
                        if score > 0.05:  # Lower threshold for tertiary
                            level = "Primary" if score > 0.4 else "Secondary" if score > 0.2 else "Tertiary"
                            emotion_display.append(f"**{emo.capitalize()}** ({score:.0%}) - {level}")
                    
                    st.markdown('<div class="emotion-card">', unsafe_allow_html=True)
                    st.markdown("From your words, I detect a rich emotional landscape:\n\n" + "\n\n".join(emotion_display))
                    st.markdown(f"\n\n**Risk Level:** {'High - Prioritizing Safety' if risk > 0.7 else 'Moderate - Gentle Support' if risk > 0.4 else 'Low - Building Resilience'}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # 2. PROFESSIONAL EMPATHETIC RESPONSE - Structured
                    st.markdown("### 2️⃣ Empathetic Validation & Response")
                    response = response_generator.generate_empathetic_response(
                        input_text=post,
                        emotions=emotions,
                        verbosity=verbosity
                    )
                    # Also include the dedicated empathetic response from the story generator for richer validation
                    primary_emotion = primary
                    empathetic_response = motivational_story_generator.generate_empathetic_response(primary_emotion, post)
                    st.markdown(f'<div class="response-section">{response}<br><br><em>{empathetic_response}</em></div>', unsafe_allow_html=True)
                    
                    # 3. ENHANCED THERAPEUTIC SOLUTIONS - Psychiatrist-like
                    st.markdown("### 3️⃣ Tailored Therapeutic Solutions")
                    resp_gen_solutions = response_generator.generate_therapeutic_solutions(post, emotions) or {}
                    # Prefer story-generator's curated practical solutions when available
                    sg_solutions_list = motivational_story_generator.therapeutic_solutions.get(primary, None)
                    final_solutions = {}
                    # Start with response_generator solutions (if dict)
                    if isinstance(resp_gen_solutions, dict):
                        final_solutions.update(resp_gen_solutions)
                    # Append story-generator list as numbered recommendations if present
                    if sg_solutions_list:
                        for i, s in enumerate(sg_solutions_list):
                            final_solutions[f"Recommended Action {i+1}"] = s

                    for challenge, solution in final_solutions.items():
                        st.markdown(f'<div class="solution-card"><strong>🎯 {challenge}</strong><br><br>{solution}</div>', unsafe_allow_html=True)
                    
                    # 4. CRISIS RESOURCES - Enhanced if needed
                    if risk > 0.75:
                        st.markdown('### 🚨 Priority: Immediate Safety Resources')
                        st.markdown('<div class="crisis-alert">', unsafe_allow_html=True)
                        crisis_resources = [
                            "🚨 **Emergency:** Call 911 (US) or local emergency services immediately",
                            "🇺🇸 **988 Suicide & Crisis Lifeline:** Call/text 988 (24/7, confidential)",
                            "🌍 **Global Hotlines:** Visit [findahelpline.com](https://findahelpline.com) for your country",
                            "📱 **Crisis Text Line:** Text HOME to 741741 (US/Canada)",
                            "🏥 **Professional Care:** Schedule with a licensed therapist via Psychology Today or local services",
                            "💙 **Veterans:** Call 988 then press 1 for Veterans Crisis Line"
                        ]
                        for resource in crisis_resources:
                            st.markdown(f"• {resource}")
                        st.markdown('</div>', unsafe_allow_html=True)
                        st.markdown("*Your safety is paramount. Please reach out now.*")
                    
                    # 5. MOTIVATIONAL STORY - Long, detailed
                    st.markdown("### 4️⃣ Inspirational Motivational Story")
                    story = generate_comprehensive_story(emotions, post)
                    
                    if story:
                        st.markdown(f'<div class="story-card">', unsafe_allow_html=True)
                        st.markdown(f"#### ✨ {story['title']}")
                        st.markdown(f"*Inspired by: {story.get('source', 'timeless tales of human resilience')}*")
                        st.markdown(story["story"])
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Enhanced Audio
                        if story.get("audio_path") and os.path.exists(story["audio_path"]):
                            st.audio(story["audio_path"], format="audio/mp3")
                            with open(story["audio_path"], "rb") as audio_file:
                                st.download_button(
                                    label="📥 Download Audio Narration",
                                    data=audio_file,
                                    file_name=f"{story['title'].replace(' ', '_')}.mp3",
                                    mime="audio/mp3"
                                )
                    
                    # 6. YOUTUBE VIDEO WITH THUMBNAIL - Professional embed
                    st.markdown("### 5️⃣ Curated YouTube Video for Inspiration")
                    st.markdown(f'<div class="video-section">', unsafe_allow_html=True)
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        if story.get("thumbnail"):
                            st.image(story["thumbnail"], use_column_width=True, caption="Video Thumbnail")
                        else:
                            st.image("https://www.youtube.com/s/desktop/4c86237f/img/favicon_144x144.png", caption="YouTube")
                    with col2:
                        st.markdown(f"**🎥 {story.get('video_title', 'Motivational Video')}**")
                        st.markdown(f"[Watch on YouTube]({story['youtube_link']})")
                        st.caption("*This video complements your story with visual motivation.*")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Save to database - Enhanced (include empathetic_response and therapeutic solutions)
                    chat_manager.add_exchange(
                        st.session_state.session_id,
                        post,
                        response,
                        emotions,
                        extra={
                            "empathetic_response": empathetic_response,
                            "therapeutic_solutions": final_solutions
                        }
                    )
                    st.session_state.interaction_count += 1
                    
                    st.success("✅ Analysis complete. Your emotional journey is honored here.")
                    
                except Exception as e:
                    logger.error(f"Post analysis error: {str(e)}", exc_info=True)
                    st.error(f"❌ An error occurred during analysis: {str(e)}. Please try again or check logs.")

# ===========================
# MODE 2: PROFESSIONAL CHAT SUPPORT - Enhanced conversation flow
# ===========================
else:
    st.markdown('<h2 style="color: #10b981;">💬 Professional Therapeutic Conversation</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color: #94a3b8;">Engage in empathetic, history-aware dialogue. EmoDude remembers your journey and adapts. After 5 interactions, receive a personalized motivational story.</p>')
    
    # Enhanced chat display with emotions and solutions
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if message["role"] == "assistant":
                # Emotion recap
                if "emotions" in message:
                    st.caption(f"🎭 Detected: {message['emotions']}")
                
                # Expandable solutions
                if "solutions" in message:
                    with st.expander("💡 Therapeutic Solutions", expanded=False):
                        for challenge, solution in message["solutions"].items():
                            st.markdown(f"**{challenge}:** {solution}")
                
                # Expandable story
                if "story" in message:
                    with st.expander("📖 Personalized Motivational Story", expanded=True):
                        st.markdown(f"### {message['story']['title']}")
                        st.markdown(message['story']['story'])
                        if message['story'].get('audio_path') and os.path.exists(message['story']['audio_path']):
                            st.audio(message['story']['audio_path'])
                        st.markdown(f"[🎥 Watch Related Video]({message['story']['youtube_link']})")
                        st.image(message['story']['thumbnail'], caption=message['story'].get('video_title', ''))
    
    # Progress bar for story trigger - Enhanced visual
    if 0 < st.session_state.interaction_count < 5:
        progress = st.session_state.interaction_count / 5
        st.progress(progress)
        remaining = 5 - st.session_state.interaction_count
        st.caption(f"💫 {remaining} more message(s) until your custom motivational story! Building emotional resonance...")
    
    # Chat input with placeholder
    prompt = st.chat_input("💭 Share what's on your mind... EmoDude is listening with compassion.")
    
    if prompt:
        try:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.interaction_count += 1
            
            with st.spinner("🤖 Processing with deep empathy and therapeutic insight..."):
                # Deep emotion detection
                emotions, risk = emotion_classifier.classify_emotion(prompt)
                primary = max(emotions, key=emotions.get)
                emotion_str = ", ".join([f"{emo.capitalize()} ({score:.0%})" for emo, score in sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:4] if score > 0.1])
                
                # Generate response with history
                history_for_gen = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
                response = response_generator.generate_empathetic_response(
                    input_text=prompt,
                    history=history_for_gen,
                    emotions=emotions,
                    verbosity=verbosity
                )

                # Build enhanced assistant response and include story-generator empathetic reply + curated solutions
                empathetic_response = motivational_story_generator.generate_empathetic_response(primary, prompt, history_for_gen)
                assistant_response = {
                    "role": "assistant",
                    "content": response,
                    "emotions": emotion_str,
                    "empathetic_response": empathetic_response
                }
                
                # Enhanced crisis integration
                if risk > 0.75:
                    crisis_add = "\n\n### 🚨 Safety First - Resources for You\n\nYour well-being matters most. Here's immediate support:\n\n"
                    crisis_add += "- 🚨 **Emergency:** 911 or local services\n"
                    crisis_add += "- 🇺🇸 **988 Lifeline:** Call/text 988\n"
                    crisis_add += "- 🌍 **Global:** [findahelpline.com](https://findahelpline.com)\n"
                    crisis_add += "- 📱 **Text:** HOME to 741741\n"
                    assistant_response["content"] += crisis_add
                
                # Therapeutic solutions (merge generator and story-generator suggestions)
                resp_gen_solutions = response_generator.generate_therapeutic_solutions(prompt, emotions) or {}
                sg_solutions_list = motivational_story_generator.therapeutic_solutions.get(primary, None)
                merged_solutions = {}
                if isinstance(resp_gen_solutions, dict):
                    merged_solutions.update(resp_gen_solutions)
                if sg_solutions_list:
                    for i, s in enumerate(sg_solutions_list):
                        merged_solutions[f"Recommended Action {i+1}"] = s
                if merged_solutions:
                    assistant_response["solutions"] = merged_solutions
                
                # Story trigger after 5 - Reset counter
                if st.session_state.interaction_count >= 5:
                    story = generate_comprehensive_story(emotions, prompt, st.session_state.messages)
                    if story:
                        assistant_response["story"] = story
                        st.session_state.interaction_count = 0  # Reset for next cycle
                        assistant_response["content"] += "\n\n✨ As we've journeyed together, here's a motivational story tailored to your emotional landscape:"
                
                # Add to messages
                st.session_state.messages.append(assistant_response)
                
                # Enhanced DB save
                chat_manager.add_exchange(
                    st.session_state.session_id,
                    prompt,
                    response,
                    emotions,
                    learned_insights={"therapy_style": st.session_state.user_profile.get("preferred_approach", "eclectic")}
                )
                # Update profile
                chat_manager.update_user_profile(st.session_state.session_id, st.session_state.user_profile, {"input": prompt, "emotions": emotions})
            
            # Rerun to update UI
            st.rerun()
            
        except Exception as e:
            logger.error(f"Chat processing error: {str(e)}", exc_info=True)
            st.error(f"❌ Unexpected error: {str(e)}. Session saved; try again or start new.")

# Footer - Enhanced disclaimer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px; color: #94a3b8;">
    <p><strong>🤖 EmoDude: Professional AI Therapeutic Companion</strong></p>
    <p>Powered by advanced emotional AI, deep learning models, and compassionate design • Built with ❤️ for mental health support</p>
    <p style="font-size: 12px;">⚠️ EmoDude provides empathetic support and resources but is not a replacement for professional mental health care. For crises, contact emergency services immediately.</p>
    <p style="font-size: 10px; color: #64748b;">© 2025 EmoDude - All rights reserved. Models: Mistral-7B, RoBERTa, CLIP</p>
</div>
""", unsafe_allow_html=True)