# app_streamlit.py
import streamlit as st
import torch
import numpy as np
from streamlit_lottie import st_lottie
import requests

from src.config import cfg
from src.hybrid_model import HybridEmotionModel
from src.dataset import tokenizer

def load_lottie(url):
    return requests.get(url).json()

# ==========================================================
# 🌙 Dark Mode + Glassmorphism Styling
# ==========================================================
st.set_page_config(page_title="Mental Health Emotion Analyzer", page_icon="💬", layout="centered")

st.markdown("""
<style>
html, body, [class*="css"]  {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e) !important;
    color: #ffffff !important;
}

/* Hide ugly whitespace at sides */
.main {
    background: transparent !important;
}

/* Glass text area */
textarea {
    background: rgba(255,255,255,0.08) !important;
    backdrop-filter: blur(12px) !important;
    border-radius: 16px !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    padding: 15px !important;
    color: white !important;
    font-size: 16px !important;
}

/* Header */
.title {
    font-size: 40px;
    font-weight: 800;
    text-align: center;
    color: #f8f9fa;
    margin-bottom: 10px;
}

/* Emotion glass card */
.emotion-card {
    background: rgba(255,255,255,0.1);
    backdrop-filter: blur(10px);
    padding: 18px;
    border-radius: 14px;
    margin-bottom: 12px;
    border: 1px solid rgba(255,255,255,0.15);
}

/* Futuristic button */
button[kind="secondary"] {
    background: linear-gradient(90deg, #7f00ff, #e100ff) !important;
    border-radius: 12px !important;
    padding: 0.6rem 1rem !important;
    font-size: 17px !important;
    color: white !important;
    border: none !important;
}
    
button[kind="secondary"]:hover {
    transform: scale(1.03);
    transition: 0.15s ease-in-out;
    background: linear-gradient(90deg, #9d00ff, #ff00d4) !important;
}

/* Emotion progress bars */
progress {
    accent-color: #00eaff;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# 😄 Emoji Mapping
# ==========================================================
EMOJI = {
    "joy":"😊","sadness":"😢","neutral":"😐","anger":"😡",
    "love":"❤️","fear":"😨","disgust":"🤢","confusion":"❓",
    "surprise":"😲","shame":"😳","guilt":"😔"
}

# ==========================================================
# 🚀 Load Model + GTE (Cached)
# ==========================================================
@st.cache_resource
def load_model():
    model = HybridEmotionModel(cfg)
    ck = torch.load("outputs/checkpoints/best_model.pt", map_location=cfg.device)
    model.load_state_dict(ck["model_state_dict"])
    model.to(cfg.device)
    model.eval()
    return model

@st.cache_resource
def load_gte():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("thenlper/gte-large")

THRESH = np.load("outputs/thresholds.npy")

# ==========================================================
# UI Header
# ==========================================================
st.markdown("<div class='title'>💬 Emotion Analyzer</div>", unsafe_allow_html=True)
st.write("<p style='text-align:center; font-size:18px;'>AI-powered mental health emotion detection</p>", unsafe_allow_html=True)
st.write("")

# ==========================================================
# Input
# ==========================================================
text = st.text_area("Your message:", height=180, placeholder="Type how you feel...")

analyze_btn = st.button("🔍 Analyze Emotion", type="secondary", use_container_width=True)

# ==========================================================
# Prediction
# ==========================================================
if analyze_btn:
    if text.strip() == "":
        st.warning("⚠️ Please type a message.")
    else:
        model = load_model()
        gte_model = load_gte()

        enc = tokenizer(text, truncation=True, padding=True, return_tensors="pt").to(cfg.device)
        gte_emb = gte_model.encode([text], convert_to_tensor=True).to(cfg.device)

        with torch.no_grad():
            logits = model(
                enc["input_ids"],
                enc["attention_mask"],
                [text],
                gte_emb,
            )
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        binary = (probs >= THRESH).astype(int)
        results = list(zip(cfg.label_list, probs, binary))
        results.sort(key=lambda x: x[1], reverse=True)

        st.write("## 🎯 Top Emotions")

        # Display Top 5 Emotions
        for emo, score, flag in results[:5]:
            st.markdown(f"""
                <div class="emotion-card">
                    <h4 style="margin-bottom:5px;">{EMOJI.get(emo,'')} {emo.title()}</h4>
                    <progress value="{score:.3f}" max="1" style="width:100%; height:14px;"></progress>
                    <p style="margin:5px 0 0;"><b>{score:.3f}</b></p>
                </div>
            """, unsafe_allow_html=True)

        detected = [emo for emo, _, f in results if f == 1]

        st.write("## 🧠 Strong Emotions Detected")
        if detected:
            st.success(" ".join([f"{EMOJI[e]} **{e}**" for e in detected]))
        else:
            st.info("No strong emotional signals detected.")

# Footer
st.write("<br><hr><p style='text-align:center;'>Built with ❤️ using RoBERTa + GTE</p>", unsafe_allow_html=True)
