"""
Streamlit — Emotion-Bot (Professional Dark UI)
Save as: src/chatbot/streamlit_app.py
Uses only: src/inference.EmotionPredictor and src/chatbot/memory.SessionMemory
"""

import streamlit as st
import sys, os, time
import torch
import yaml
from typing import List, Dict
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# --- ensure src is on path so we can import project modules ---
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from utils import load_json
from chatbot.memory import SessionMemory

# ---------------- CONFIG ----------------
CONFIG_PATH = os.path.join(ROOT, "configs", "config.yaml")
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

MODEL_PATH = os.path.join(ROOT, config['output_dir'], "best_model.pt")
MODEL_NAME = config['model_name']
LABEL_MAP_PATH = config['label_map']
LABEL_MAP = load_json(LABEL_MAP_PATH)
# choose device automatically (GPU if available)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD_DEFAULT = config.get('threshold', 0.40)
# ----------------------------------------
    
st.set_page_config(page_title="Emotion-Bot", layout="wide", initial_sidebar_state="collapsed")
# Custom dark + green theme styles
st.markdown(
    """
    <style>
    /* Base background and font */
    html, body, [class*="css"]  {
        background: #0f1720 !important;
        color: #dbeafe !important;
        font-family: "Inter", "Helvetica", Arial, sans-serif;
    }
    /* Container */
    .reportview-container .main .block-container{
        padding-top: 1rem;
        padding-left: 1.25rem;
        padding-right: 1.25rem;
    }
    /* Header */
    .title {
        color: #e6fffb;
        font-weight: 700;
    }
    /* Chat bubbles */
    .user {
        background: linear-gradient(180deg, #0b1220 0%, #0f1624 100%);
        border: 1px solid rgba(255,255,255,0.03);
        padding: 12px 14px;
        border-radius: 12px;
        margin: 6px 0;
        color: #e6f6ff;
    }
    .bot {
        background: linear-gradient(180deg,#072311 0%, #08321a 100%);
        border: 1px solid rgba(255,255,255,0.04);
        padding: 12px 14px;
        border-radius: 12px;
        margin: 6px 0;
        color: #e6fff0;
    }
    /* Emotion chip */
    .chip {
        display:inline-block;
        padding:6px 10px;
        margin:4px 4px 4px 0;
        border-radius:999px;
        font-size: 13px;
        background: linear-gradient(90deg, #064e3b, #047857);
        color: #e6fff0;
        box-shadow: 0 2px 6px rgba(4,120,87,0.12);
        border: 1px solid rgba(255,255,255,0.03);
    }
    .score {
        color: rgba(255,255,255,0.7);
        font-size:12px;
        margin-left:6px;
    }
    /* Input area */
    .input-area {
        background: rgba(255,255,255,0.02);
        border-radius: 10px;
        padding: 10px;
        border: 1px solid rgba(255,255,255,0.03);
    }
    .small-muted { color: rgba(255,255,255,0.45); font-size:12px }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header area
col1, col2 = st.columns([0.2, 0.8])
with col1:
    st.markdown("###")
    st.image(
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAABjklEQVQ4jZ3TT0gTYRzH8e+9dKZ0iN2oQqFq2gq9oE2u0t7V3a2lq0kq0s6m1kR2k0iXxK2oYpS0Wjkkxw6i4r9fNf7v/7v3me7v7n3ng4nL5r3P1xj4w2kqYw3wz1g6w3o5Qq4w3o5Qq4w3o5Qq4w3o5Qq4w3o5Qq4w3o5Qq4w3o5Qq4w3o5Qq4w3o5Qq4w3o5Qq4w3o5Qq4w3o5Qq4w3o5Qq4w3o5Qq4w3o5Qq4w3o5Qq4w3o5Qq4w3o5Qq4w3o5Qq4w3o5Qq4w3o5Qq4w3o5Qq4w3o5Qq4w3o5Qq4w3o5Qq4w3o5Qq4w3o5Qq4w3o5Qq4w3o5Qq7w3m5Qq4w3m5Qq4w3m5Qq4w3m5Qq4w3m5Qq4w3m5Qq4w3m5Qq4w3m5Qq4w3m5Qq4w3m5Qq4w3m5Qq4w3m5Qq4w3m5Qq4w3m5Qq4w3m5Qq4w3m5Qq4w3m5Qq4w3m5Qq4w3m5Qq4w3m5Qq4w3m5Qq4w3m5Qq4w3m5Qq4w3m5Qq4w3m5Qq4w3m5Qq4w3m5Qq4w3m5Qq4w3m5Qq4w3m5Qq4w3m5Qq4w3m5Qq4w3m5Qq4w3m5Qq4w3m5Qq4w3m5Qq7w3m5Qq4w3m5Qq4wAAAAAElFTkSuQmCC",
        width=48,
    )
with col2:
    st.markdown("<div class='title'>Emotion-Bot</div>", unsafe_allow_html=True)
    st.markdown("<div class='small-muted'>Empathy-aware emotion detection for mental-health support</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='small-muted'>Model: <b>{os.path.basename(MODEL_PATH)}</b> • Device: <b>{DEVICE}</b></div>", unsafe_allow_html=True)

# Load model and tokenizer (cached)
@st.cache_resource
def get_model_and_tokenizer(model_name, model_path, device_choice):
    # Use a short sleep to allow page to render the "Loading..." text
    time.sleep(0.05)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=20,
        problem_type="multi_label_classification"
    ).to(device_choice)
    model.load_state_dict(torch.load(model_path, map_location=device_choice))
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

# Initialize memory
if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = SessionMemory()
if "session_id" not in st.session_state:
    st.session_state.session_id = "streamlit_user"

model, tokenizer = get_model_and_tokenizer(MODEL_NAME, MODEL_PATH, DEVICE)
    
# ---------- add this at module level (above the form / above usage) ----------
def templated_reply(labels_list, text):
    """
    Simple empathetic template generator.
    Keep this at module-level so Streamlit always finds it.
    """
    low = set(labels_list or [])
    if any(x in low for x in ["grief","sadness","loneliness","helplessness","despair"]):
        return "I'm really sorry you're going through this. If you want, tell me more — I'm here to listen."
    if any(x in low for x in ["anxiety","fear","uncertainty"]):
        return "That sounds very stressful. Would you like to try a small breathing exercise or talk about what's worrying you?"
    if any(x in low for x in ["anger","frustration","resentment"]):
        return "I can understand why you'd feel upset. Want to talk about what happened or how you'd like it to be different?"
    if any(x in low for x in ["joy","love","gratitude","relief","pride","excitement"]):
        return "That sounds meaningful — thank you for sharing. Would you like to say more about that memory?"
    # fallback
    return "Thank you for sharing. Would you like to talk more about that or get a few small suggestions to help right now?"
# ---------------------------------------------------------------------------

# Main layout: conversation + right meta panel
left, right = st.columns([3, 1])

with left:
    st.markdown("### Conversation")
    # Chat display box
    chat_container = st.container()

    # input area at bottom
    with st.form(key="chat_form", clear_on_submit=False):
        user_text = st.text_area("You", placeholder="Type how you're feeling or share a story...", height=90, key="user_text")
        submit = st.form_submit_button("Send")

    if submit and user_text and user_text.strip():
        sid = st.session_state.session_id
        st.session_state.chat_memory.add_user(sid, user_text.strip())
        # inference
        enc = tokenizer(user_text.strip(), truncation=True, padding="max_length", max_length=256, return_tensors="pt")
        input_ids = enc["input_ids"].to(DEVICE)
        attention_mask = enc["attention_mask"].to(DEVICE)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        # map to labels
        labels = [LABEL_MAP[i] for i,p in enumerate(probs) if p >= THRESHOLD_DEFAULT]
        scores = {LABEL_MAP[i]: float(probs[i]) for i in range(len(probs))}
        # (templated_reply moved to module-level)

        reply = templated_reply(labels, user_text)
        # add debug top scores
        top_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:8]
        st.session_state.chat_memory.add_bot(sid, reply)
        # attach detection metadata to last bot message
        last_index = len(st.session_state.chat_memory.store[sid]["history"]) - 1
        st.session_state.chat_memory.store[sid]["history"][last_index]["detected_emotions"] = labels
        st.session_state.chat_memory.store[sid]["history"][last_index]["scores"] = scores
        # also show top scores in Streamlit console (not chat) for quick debugging
        st.write("Top model scores (label:prob):", [(k, round(v,3)) for k,v in top_scores])
    
    # render chat history
    with chat_container:
        history = st.session_state.chat_memory.get_history(st.session_state.session_id, last_k=200)
        for turn in history:
            if turn["role"] == "user":
                st.markdown(f"<div class='user'><b>You</b><br>{st.markdown(turn['text'], unsafe_allow_html=True) if False else turn['text']}</div>", unsafe_allow_html=True)
            else:
                # bot message with detected emotions shown as chips under the message
                bot_html = turn["text"]
                st.markdown(f"<div class='bot'><b>Bot</b><br>{bot_html}</div>", unsafe_allow_html=True)
                if "detected_emotions" in turn:
                    # show chips
                    chips_html = ""
                    detected = turn.get("detected_emotions", [])
                    scores = turn.get("scores", {})
                    for e in detected:
                        sc = scores.get(e, None)
                        if sc is not None:
                            chips_html += f"<span class='chip'>{e} <span class='score'>{sc:.2f}</span></span>"
                        else:
                            chips_html += f"<span class='chip'>{e}</span>"
                    st.markdown(chips_html, unsafe_allow_html=True)

with right:
    st.markdown("### Session")
    st.markdown(f"- **Session id:** `{st.session_state.session_id}`")
    st.markdown(f"- **Messages:** {len(st.session_state.chat_memory.get_history(st.session_state.session_id, last_k=1000))}")
    st.markdown("---")
    st.markdown("### Quick actions")
    if st.button("Clear conversation"):
        st.session_state.chat_memory.store.pop(st.session_state.session_id, None)
        st.experimental_rerun()
    if st.button("Download conversation (JSON)"):
        import json
        hist_obj = st.session_state.chat_memory.get_history(st.session_state.session_id, last_k=1000)
        st.download_button("Download JSON", data=json.dumps(hist_obj, ensure_ascii=False, indent=2), file_name="conversation.json", mime="application/json")
    st.markdown("---")
    st.markdown("### Model info")
    st.markdown(f"- **Model file:** `{os.path.basename(MODEL_PATH)}`")
    st.markdown(f"- **Device:** `{DEVICE}`")
    st.markdown(f"- **Threshold:** {THRESHOLD_DEFAULT:.2f}")
    st.markdown("---")
    st.markdown("### Notes")
    st.markdown("- This UI uses the **classifier only** and gives templated empathetic replies.")
    st.markdown("- For safety: this is not a substitute for professional help. For crisis messages, contact local emergency services.")

# Footer small
st.markdown("<div class='small-muted' style='margin-top:10px'>Built for research & demo — not a medical device.</div>", unsafe_allow_html=True)
