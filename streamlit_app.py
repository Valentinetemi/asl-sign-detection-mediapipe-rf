import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
from xgboost import XGBClassifier
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os
import base64
import io

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ọwọ AI",
    page_icon="🤚",
    layout="centered"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

:root {
  --bg: #07080f;
  --surface: #0e0f1c;
  --surface2: #161828;
  --accent: #7c3aff;
  --accent-bright: #a855f7;
  --accent2: #06d6a0;
  --text: #f0eeff;
  --muted: #6b6b8a;
  --border: rgba(124,58,255,0.2);
}

html, body, [class*="css"] {
  font-family: 'DM Sans', sans-serif;
  background-color: var(--bg) !important;
  color: var(--text) !important;
}

.stApp {
  background: var(--bg) !important;
  background-image:
    radial-gradient(ellipse 80% 50% at 50% -20%, rgba(124,58,255,0.12), transparent),
    radial-gradient(ellipse 60% 40% at 85% 85%, rgba(6,214,160,0.06), transparent);
}

/* Hide default Streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
  padding-top: 2rem;
  max-width: 760px;
  margin-left: auto !important;
  margin-right: auto !important;
  padding-left: 2rem !important;
  padding-right: 2rem !important;
}

/* ── Hero Header ── */
.hero {
  text-align: center;
  padding: 3rem 1rem 2rem;
  position: relative;
  width: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
}
.hero-badge {
  display: inline-block;
  background: rgba(124,58,255,0.1);
  border: 1px solid var(--border);
  color: var(--accent-bright);
  font-size: 0.72rem;
  font-weight: 500;
  letter-spacing: 0.15em;
  text-transform: uppercase;
  padding: 0.35rem 1rem;
  border-radius: 100px;
  margin-bottom: 1.5rem;
}
.hero-title {
  font-family: 'Syne', sans-serif;
  font-size: clamp(4.5rem, 18vw, 9rem);
  font-weight: 800;
  line-height: 0.88;
  margin: 0 0 1.4rem;
  letter-spacing: -0.04em;
}
.hero-title .line1 { color: var(--text); display: block; font-size: 7rem; font-weight: 800}
.hero-title .line2 {
  color: var(--accent-bright);
  display: block;
  font-size: 6rem
  text-shadow: 0 0 80px rgba(124,58,255,0.4);
}
.hero-subtitle {
  color: var(--muted);
  font-size: 1rem;
  font-weight: 300;
  max-width: 480px;
  margin: 0 auto 2rem;
  line-height: 1.7;
}
.hero-stats {
  display: flex;
  justify-content: center;
  gap: 2rem;
  flex-wrap: wrap;
}
.stat-pill {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 0.6rem 1.2rem;
  font-size: 0.85rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}
.stat-pill strong { color: var(--accent-bright); font-weight: 700; }

/* ── Upload Zone ── */
.upload-zone {
  background: var(--surface);
  border: 1.5px dashed rgba(124,58,255,0.35);
  border-radius: 20px;
  padding: 2.5rem;
  text-align: center;
  margin: 2rem 0;
  transition: border-color 0.3s;
}
.upload-zone:hover { border-color: rgba(124,58,255,0.65); }
.upload-icon { font-size: 2.5rem; margin-bottom: 0.8rem; }
.upload-label { color: var(--muted); font-size: 0.9rem; }

/* ── Result Card ── */
.result-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 2rem;
  margin: 1.5rem 0;
  position: relative;
  overflow: hidden;
}
.result-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
  background: linear-gradient(90deg, var(--accent), var(--accent2));
}
.sign-display {
  text-align: center;
  padding: 1.5rem 0;
}
.sign-letter {
  font-family: 'Syne', sans-serif;
  font-size: 7rem;
  font-weight: 800;
  color: var(--accent-bright);
  line-height: 1;
  text-shadow: 0 0 80px rgba(124,58,255,0.5), 0 0 30px rgba(168,85,247,0.3);
  display: block;
}
.sign-label {
  font-size: 0.8rem;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: var(--muted);
  margin-top: 0.5rem;
}
.confidence-bar-container {
  margin: 1.5rem 0 0.5rem;
}
.conf-label {
  display: flex;
  justify-content: space-between;
  font-size: 0.82rem;
  color: var(--muted);
  margin-bottom: 0.5rem;
}
.conf-val { color: var(--text); font-weight: 500; }
.confidence-bar {
  height: 6px;
  background: var(--surface2);
  border-radius: 100px;
  overflow: hidden;
}
.confidence-fill {
  height: 100%;
  border-radius: 100px;
  transition: width 0.8s ease;
}

/* ── Status Messages ── */
.status-high {
  background: rgba(6,214,160,0.06);
  border: 1px solid rgba(6,214,160,0.25);
  border-radius: 12px;
  padding: 0.9rem 1.2rem;
  display: flex;
  align-items: center;
  gap: 0.7rem;
  font-size: 0.9rem;
  margin-top: 1rem;
}
.status-med {
  background: rgba(255,165,0,0.06);
  border: 1px solid rgba(255,165,0,0.2);
  border-radius: 12px;
  padding: 0.9rem 1.2rem;
  display: flex;
  align-items: center;
  gap: 0.7rem;
  font-size: 0.9rem;
  margin-top: 1rem;
}
.status-low {
  background: rgba(255,70,70,0.06);
  border: 1px solid rgba(255,70,70,0.2);
  border-radius: 12px;
  padding: 0.9rem 1.2rem;
  display: flex;
  align-items: center;
  gap: 0.7rem;
  font-size: 0.9rem;
  margin-top: 1rem;
}

/* ── TTS Button ── */
.tts-container { text-align: center; margin-top: 1.5rem; }

/* ── Performance Table ── */
.perf-section {
  background: var(--surface);
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 20px;
  padding: 1.8rem;
  margin: 2rem 0;
}
.perf-title {
  font-family: 'Syne', sans-serif;
  font-size: 1rem;
  font-weight: 700;
  margin-bottom: 1.2rem;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.1em;
}
.perf-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.75rem 0;
  border-bottom: 1px solid rgba(255,255,255,0.04);
  font-size: 0.9rem;
}
.perf-row:last-child { border-bottom: none; }
.perf-row.winner { color: var(--accent2); font-weight: 500; }
.perf-badge {
  background: rgba(6,214,160,0.12);
  border: 1px solid rgba(6,214,160,0.3);
  border-radius: 6px;
  padding: 0.2rem 0.6rem;
  font-size: 0.75rem;
  color: var(--accent2);
}

/* ── Footer ── */
.footer {
  text-align: center;
  padding: 2rem 0 1rem;
  color: var(--muted);
  font-size: 0.82rem;
  border-top: 1px solid rgba(255,255,255,0.05);
  margin-top: 3rem;
}
.footer a { color: var(--accent-bright); text-decoration: none; }
.footer-links { display: flex; gap: 1.5rem; justify-content: center; flex-wrap: wrap; margin-top: 0.8rem; }

/* ── Streamlit Component Overrides ── */
.stFileUploader > div {
  background: transparent !important;
  border: none !important;
}
[data-testid="stFileUploadDropzone"] {
  background: var(--surface) !important;
  border: 1.5px dashed rgba(124,58,255,0.35) !important;
  border-radius: 16px !important;
  color: var(--muted) !important;
}
[data-testid="stFileUploadDropzone"]:hover {
  border-color: rgba(124,58,255,0.65) !important;
}
.stButton > button {
  background: linear-gradient(135deg, var(--accent), var(--accent-bright)) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 12px !important;
  font-family: 'Syne', sans-serif !important;
  font-weight: 700 !important;
  font-size: 0.95rem !important;
  padding: 0.7rem 2rem !important;
  letter-spacing: 0.03em !important;
  transition: all 0.2s !important;
  width: 100%;
}
.stButton > button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 30px rgba(124,58,255,0.4) !important;
}
[data-testid="stImage"] img {
  border-radius: 16px !important;
  border: 1px solid rgba(255,255,255,0.06) !important;
}
/* Divider */
hr { border-color: rgba(255,255,255,0.06) !important; }
</style>
""", unsafe_allow_html=True)


# ─── Hero ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-badge">✦ BUILT FOR NIGERIA'S DEAF COMMUNITY</div>
  <h1 class="hero-title">
    <span class="line1">Ọwọ</span>
    <span class="line2">AI</span>
  </h1>
  <p class="hero-subtitle">
    Upload a hand sign image. Get instant ASL letter detection with voice output — 
    powered by MediaPipe landmarks + XGBoost.
  </p>
  <div class="hero-stats">
    <div class="stat-pill">🏆 <strong>98.43%</strong> accuracy</div>
    <div class="stat-pill">✋ <strong>28</strong> signs</div>
    <div class="stat-pill">🤚🏾 <strong>Works on</strong> dark skin tones</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─── Model Loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_detector():
    if not os.path.exists("hand_landmarker.task"):
        with st.spinner("Downloading MediaPipe model…"):
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
                "hand_landmarker.task"
            )
    base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return vision.HandLandmarker.create_from_options(options)


@st.cache_resource
def load_model():
    clf = XGBClassifier()
    clf.load_model("asl_model_xgb.json")
    le = joblib.load("label_encoder.pkl")
    return clf, le


def normalize_landmarks(row):
    landmarks = np.array(row).reshape(21, 3)
    wrist = landmarks[0]
    landmarks = landmarks - wrist
    hand_size = np.linalg.norm(landmarks[12])
    if hand_size > 0:
        landmarks = landmarks / hand_size
    return landmarks.flatten().tolist()


def tts_html(text):
    """Inject a hidden autoplay TTS trigger via Web Speech API."""
    return f"""
    <script>
    (function() {{
        const synth = window.speechSynthesis;
        synth.cancel();
        const utt = new SpeechSynthesisUtterance("{text}");
        utt.rate = 0.85;
        utt.pitch = 1.1;
        utt.volume = 1;
        synth.speak(utt);
    }})();
    </script>
    """


detector = load_detector()
clf, le = load_model()


# ─── Upload ────────────────────────────────────────────────────────────────────
st.markdown("### Upload a hand sign photo")
uploaded_file = st.file_uploader(
    "Drop your image here or click to browse",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    st.image(image_rgb, caption="Your uploaded image", use_container_width=True)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = detector.detect(mp_image)

    if result.hand_landmarks:
        landmarks = result.hand_landmarks[0]

        # Draw landmark dots
        annotated = image_rgb.copy()
        for lm in landmarks:
            cx = int(lm.x * annotated.shape[1])
            cy = int(lm.y * annotated.shape[0])
            cv2.circle(annotated, (cx, cy), 5, (232, 255, 71), -1)
        # Draw connections (rough skeleton)
        connections = [
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (0,9),(9,10),(10,11),(11,12),
            (0,13),(13,14),(14,15),(15,16),
            (0,17),(17,18),(18,19),(19,20),
            (5,9),(9,13),(13,17)
        ]
        for a, b in connections:
            ax = int(landmarks[a].x * annotated.shape[1])
            ay = int(landmarks[a].y * annotated.shape[0])
            bx = int(landmarks[b].x * annotated.shape[1])
            by = int(landmarks[b].y * annotated.shape[0])
            cv2.line(annotated, (ax, ay), (bx, by), (232, 255, 71, 120), 1)

        row = []
        for lm in landmarks:
            row.extend([lm.x, lm.y, lm.z])
        row = normalize_landmarks(row)

        prediction_enc = clf.predict([row])[0]
        prediction = le.inverse_transform([prediction_enc])[0]
        confidence = clf.predict_proba([row]).max() * 100

        # Annotated image
        st.image(annotated, caption="Hand landmarks detected", use_container_width=True)

        # ─ The Result Card ─
        conf_color = (
            "#e8ff47" if confidence > 90 else
            "#ffa500" if confidence > 70 else
            "#ff4646"
        )
        st.markdown(f"""
        <div class="result-card">
          <div class="sign-display">
            <span class="sign-letter">{prediction}</span>
            <span class="sign-label">Detected ASL Sign</span>
          </div>
          <div class="confidence-bar-container">
            <div class="conf-label">
              <span>Confidence</span>
              <span class="conf-val">{confidence:.1f}%</span>
            </div>
            <div class="confidence-bar">
              <div class="confidence-fill" style="width:{confidence:.1f}%; background:{conf_color};"></div>
            </div>
          </div>
          {"<div class='status-high'>✅ <span>High confidence - the model is sure about this one.</span></div>" if confidence > 90 else
           "<div class='status-med'>⚠️ <span>Medium confidence — try a better-lit image for a stronger result.</span></div>" if confidence > 70 else
           "<div class='status-low'>😔  <span>Low confidence — please try again with a clearer hand photo.</span></div>"}
        </div>
        """, unsafe_allow_html=True)

        # ── TTS ──
        st.markdown("<div class='tts-container'>", unsafe_allow_html=True)
        if st.button(f"🔊 Hear it — speak \"{prediction}\""):
            st.components.v1.html(tts_html(f"The sign is {prediction}"), height=0)
        st.markdown("</div>", unsafe_allow_html=True)

        # Auto-speak on detection
        st.components.v1.html(tts_html(f"Detected sign: {prediction}. Confidence: {confidence:.0f} percent."), height=0)

    else:
        st.markdown("""
        <div class="status-low">
          😔  <span>No hand detected. Make sure your hand is clearly visible in the photo and well-lit.</span>
        </div>
        """, unsafe_allow_html=True)


# ─── Performance ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="perf-section">
  <div class="perf-title">Model Benchmarks</div>
  <div class="perf-row">
    <span>Random Forest — raw landmarks</span>
    <span>97.64%</span>
  </div>
  <div class="perf-row">
    <span>Random Forest — normalized</span>
    <span>98.17%</span>
  </div>
  <div class="perf-row winner">
    <span>⚡ XGBoost — normalized <span class="perf-badge">Live Model</span></span>
    <span>98.43%</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ─── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  <div>Built by <strong>Temiloluwa Valentine</strong> — Ọwọ AI 🇳🇬</div>
  <div class="footer-links">
    <a href="https://github.com/Valentinetemi/asl-sign-detection-mediapipe-rf" target="_blank">GitHub →</a>
    <a href="https://temiloluwaval.medium.com/i-replaced-87-000-images-with-63-numbers-heres-how-i-built-a-sign-language-detector-404f73b3c3aa" target="_blank">Beginner Article →</a>
    <a href="https://temiloluwaval.medium.com/why-i-chose-63-numbers-over-millions-of-pixels-landmarks-vs-cnn-for-sign-detection-14e0f75a92f7" target="_blank">Technical Deep-Dive →</a>
  </div>
</div>
""", unsafe_allow_html=True)