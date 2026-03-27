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

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ọwọ AI — ASL Sign Detection",
    page_icon="🖐🏾",
    layout="centered"
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Geist+Mono:wght@300;400;600&family=Cabinet+Grotesk:wght@400;500;700;800;900&display=swap');

:root {
  --ink:      #05080f;
  --ink2:     #0b1019;
  --ink3:     #111826;
  --glow:     #00e5c8;
  --glow-dim: #00b89e;
  --glow-bg:  rgba(0,229,200,0.07);
  --ember:    #ff8c42;
  --ember-bg: rgba(255,140,66,0.08);
  --text:     #e8f0ef;
  --muted:    #5a6e72;
  --muted2:   #2a3a3f;
  --line:     rgba(0,229,200,0.1);
  --r:        18px;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; }

html, body, [class*="css"], .stApp {
  font-family: 'Cabinet Grotesk', sans-serif !important;
  background: var(--ink) !important;
  color: var(--text) !important;
}

/* Atmospheric background */
.stApp::before {
  content: '';
  position: fixed;
  inset: 0;
  background:
    radial-gradient(ellipse 70% 55% at 15% 10%, rgba(0,229,200,0.055) 0%, transparent 65%),
    radial-gradient(ellipse 55% 45% at 88% 80%, rgba(255,140,66,0.045) 0%, transparent 60%),
    radial-gradient(ellipse 40% 35% at 50% 50%, rgba(0,100,90,0.03) 0%, transparent 70%);
  pointer-events: none;
  z-index: 0;
}

#MainMenu, footer, header { visibility: hidden; }

.block-container {
  padding-top: 0 !important;
  max-width: 720px !important;
  margin: 0 auto !important;
  padding-left: 1.5rem !important;
  padding-right: 1.5rem !important;
  position: relative;
  z-index: 1;
}

/* ── TOP NAV BAR ── */
.topbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1.4rem 0 1rem;
  border-bottom: 1px solid var(--line);
  margin-bottom: 0;
}
.topbar-brand {
  font-family: 'Geist Mono', monospace;
  font-size: 0.72rem;
  letter-spacing: 0.18em;
  color: var(--glow);
  text-transform: uppercase;
  font-weight: 600;
}
.topbar-tag {
  font-family: 'Geist Mono', monospace;
  font-size: 0.65rem;
  color: var(--muted);
  letter-spacing: 0.1em;
  text-transform: uppercase;
}
.topbar-dot {
  width: 7px; height: 7px;
  border-radius: 50%;
  background: var(--glow);
  box-shadow: 0 0 12px var(--glow), 0 0 24px rgba(0,229,200,0.4);
  animation: pulse 2.4s ease-in-out infinite;
}
@keyframes pulse {
  0%, 100% { opacity: 1; transform: scale(1); }
  50%       { opacity: 0.5; transform: scale(0.8); }
}

/* ── HERO ── */
.hero {
  padding: 4.5rem 0 3.5rem;
  position: relative;
}
.hero-eyebrow {
  font-family: 'Geist Mono', monospace;
  font-size: 0.68rem;
  letter-spacing: 0.22em;
  color: var(--glow-dim);
  text-transform: uppercase;
  margin-bottom: 1.6rem;
  display: flex;
  align-items: center;
  gap: 0.7rem;
}
.hero-eyebrow::before {
  content: '';
  display: inline-block;
  width: 28px; height: 1px;
  background: var(--glow);
  opacity: 0.5;
}
.hero-heading {
  font-family: 'Instrument Serif', serif;
  font-size: clamp(3.8rem, 10vw, 7.2rem);
  line-height: 0.92;
  letter-spacing: -0.02em;
  color: var(--text);
  margin-bottom: 0.15em;
}
.hero-heading em {
  font-style: italic;
  color: var(--glow);
  text-shadow: 0 0 60px rgba(0,229,200,0.35);
}
.hero-sub {
  margin-top: 1.8rem;
  font-size: 1rem;
  color: var(--muted);
  max-width: 460px;
  line-height: 1.75;
  font-weight: 400;
}
.hero-sub strong { color: var(--text); font-weight: 500; }

/* Floating accent number */
.hero-accent-num {
  position: absolute;
  right: 0; top: 4.5rem;
  font-family: 'Geist Mono', monospace;
  font-size: 7rem;
  font-weight: 300;
  color: rgba(0,229,200,0.05);
  line-height: 1;
  pointer-events: none;
  user-select: none;
}

/* ── STAT ROW ── */
.stats-row {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 1px;
  background: var(--line);
  border: 1px solid var(--line);
  border-radius: var(--r);
  overflow: hidden;
  margin: 2.5rem 0 3rem;
}
.stat-cell {
  background: var(--ink2);
  padding: 1.2rem 1.4rem;
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}
.stat-cell:hover { background: var(--ink3); }
.stat-num {
  font-family: 'Geist Mono', monospace;
  font-size: 1.5rem;
  font-weight: 600;
  color: var(--glow);
  letter-spacing: -0.03em;
}
.stat-desc {
  font-size: 0.75rem;
  color: var(--muted);
  line-height: 1.4;
}

/* ── SECTION HEADERS ── */
.section-head {
  display: flex;
  align-items: center;
  gap: 0.8rem;
  margin-bottom: 1.2rem;
}
.section-label {
  font-family: 'Geist Mono', monospace;
  font-size: 0.65rem;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: var(--muted);
}
.section-line {
  flex: 1;
  height: 1px;
  background: var(--line);
}

/* ── REFERENCE GRID ── */
.ref-notice {
  background: var(--glow-bg);
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 0.75rem 1rem;
  font-size: 0.82rem;
  color: var(--glow-dim);
  display: flex;
  align-items: center;
  gap: 0.6rem;
  margin-bottom: 1.2rem;
}

/* ── FILE UPLOADER OVERRIDES ── */
[data-testid="stFileUploadDropzone"] {
  background: var(--ink2) !important;
  border: 1px solid rgba(0,229,200,0.2) !important;
  border-radius: var(--r) !important;
  transition: border-color 0.25s, background 0.25s !important;
  min-height: 130px !important;
}
[data-testid="stFileUploadDropzone"]:hover {
  border-color: rgba(0,229,200,0.5) !important;
  background: var(--ink3) !important;
}
[data-testid="stFileUploadDropzone"] svg { display: none !important; }
[data-testid="stFileUploadDropzone"] p {
  color: var(--muted) !important;
  font-size: 0.88rem !important;
}
[data-testid="stFileUploadDropzone"]::before {
  content: '⬆';
  display: block;
  font-size: 1.6rem;
  text-align: center;
  color: rgba(0,229,200,0.3);
  margin-bottom: 0.4rem;
  padding-top: 0.5rem;
}
.stFileUploader > div { background: transparent !important; border: none !important; }
.stFileUploader label {
  font-family: 'Geist Mono', monospace !important;
  font-size: 0.65rem !important;
  letter-spacing: 0.15em !important;
  text-transform: uppercase !important;
  color: var(--muted) !important;
}

/* ── UPLOADED IMAGE ── */
[data-testid="stImage"] img {
  border-radius: var(--r) !important;
  border: 1px solid var(--line) !important;
}

/* ── RESULT CARD ── */
.result-wrap {
  margin: 1.8rem 0;
  position: relative;
}
.result-card {
  background: var(--ink2);
  border: 1px solid var(--line);
  border-radius: 24px;
  overflow: hidden;
  position: relative;
}
.result-glow-bar {
  height: 3px;
  background: linear-gradient(90deg, transparent, var(--glow), var(--ember), transparent);
  animation: shimmer 3s linear infinite;
  background-size: 200% 100%;
}
@keyframes shimmer {
  0%   { background-position: -200% 0; }
  100% { background-position: 200% 0; }
}
.result-inner {
  padding: 2.2rem 2rem 2rem;
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 2rem;
  align-items: center;
}
.result-letter-block {
  display: flex;
  flex-direction: column;
  align-items: center;
  min-width: 120px;
}
.result-letter {
  font-family: 'Instrument Serif', serif;
  font-size: 8rem;
  line-height: 0.9;
  color: var(--glow);
  text-shadow: 0 0 80px rgba(0,229,200,0.5), 0 0 160px rgba(0,229,200,0.2);
}
.result-letter-label {
  font-family: 'Geist Mono', monospace;
  font-size: 0.6rem;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: var(--muted);
  margin-top: 0.4rem;
}
.result-meta {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}
.result-conf-label {
  font-family: 'Geist Mono', monospace;
  font-size: 0.65rem;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--muted);
  display: flex;
  justify-content: space-between;
  margin-bottom: 0.45rem;
}
.result-conf-val {
  font-family: 'Geist Mono', monospace;
  font-size: 0.9rem;
  color: var(--text);
  font-weight: 600;
}
.conf-track {
  height: 4px;
  background: var(--muted2);
  border-radius: 100px;
  overflow: hidden;
}
.conf-fill {
  height: 100%;
  border-radius: 100px;
  transition: width 0.9s cubic-bezier(.23,1,.32,1);
}
.result-verdict {
  font-size: 0.85rem;
  color: var(--muted);
  line-height: 1.5;
  padding-top: 0.5rem;
  border-top: 1px solid var(--line);
}
.verdict-icon { font-size: 1rem; }

/* ── BUTTON ── */
.stButton > button {
  background: transparent !important;
  color: var(--glow) !important;
  border: 1px solid rgba(0,229,200,0.35) !important;
  border-radius: 10px !important;
  font-family: 'Geist Mono', monospace !important;
  font-size: 0.75rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.12em !important;
  text-transform: uppercase !important;
  padding: 0.75rem 1.8rem !important;
  transition: all 0.2s ease !important;
  width: auto !important;
}
.stButton > button:hover {
  background: var(--glow-bg) !important;
  border-color: var(--glow) !important;
  box-shadow: 0 0 24px rgba(0,229,200,0.2) !important;
  transform: translateY(-1px) !important;
}
.tts-col { display: flex; justify-content: flex-start; margin-top: 1rem; }

/* ── ERROR STATE ── */
.no-hand {
  background: var(--ink2);
  border: 1px solid rgba(255,140,66,0.2);
  border-radius: var(--r);
  padding: 1.6rem;
  display: flex;
  align-items: flex-start;
  gap: 1rem;
  margin: 1.5rem 0;
}
.no-hand-icon { font-size: 1.5rem; }
.no-hand-title { font-weight: 700; font-size: 0.9rem; margin-bottom: 0.25rem; }
.no-hand-body { font-size: 0.82rem; color: var(--muted); line-height: 1.6; }

/* ── BENCHMARK TABLE ── */
.bench {
  background: var(--ink2);
  border: 1px solid var(--line);
  border-radius: var(--r);
  overflow: hidden;
  margin: 1rem 0 3rem;
}
.bench-row {
  display: grid;
  grid-template-columns: 1fr auto auto;
  gap: 1rem;
  padding: 1rem 1.4rem;
  border-bottom: 1px solid var(--line);
  align-items: center;
  font-size: 0.85rem;
  transition: background 0.15s;
}
.bench-row:last-child { border-bottom: none; }
.bench-row:hover { background: var(--ink3); }
.bench-row.active { background: rgba(0,229,200,0.04); }
.bench-model { color: var(--muted); }
.bench-row.active .bench-model { color: var(--text); font-weight: 500; }
.bench-tag {
  font-family: 'Geist Mono', monospace;
  font-size: 0.62rem;
  letter-spacing: 0.1em;
  background: rgba(0,229,200,0.1);
  border: 1px solid rgba(0,229,200,0.25);
  color: var(--glow);
  padding: 0.2rem 0.6rem;
  border-radius: 5px;
}
.bench-score {
  font-family: 'Geist Mono', monospace;
  font-size: 0.88rem;
  color: var(--muted);
  text-align: right;
}
.bench-row.active .bench-score { color: var(--glow); font-weight: 600; }

/* ── FOOTER ── */
.footer {
  padding: 2rem 0 1.5rem;
  border-top: 1px solid var(--line);
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 1rem;
  font-size: 0.78rem;
  color: var(--muted);
}
.footer-brand { font-weight: 700; color: var(--text); }
.footer-links { display: flex; gap: 1.4rem; }
.footer-links a {
  color: var(--muted);
  text-decoration: none;
  transition: color 0.15s;
  font-family: 'Geist Mono', monospace;
  font-size: 0.68rem;
  letter-spacing: 0.08em;
}
.footer-links a:hover { color: var(--glow); }

/* ── IMAGE CAPTION FIX ── */
.stImage > figcaption, [data-testid="caption"] {
  font-family: 'Geist Mono', monospace !important;
  font-size: 0.65rem !important;
  letter-spacing: 0.12em !important;
  text-transform: uppercase !important;
  color: var(--muted) !important;
  text-align: center;
  margin-top: 0.5rem;
}

/* ── SPINNER ── */
.stSpinner > div > div { border-top-color: var(--glow) !important; }

/* ── INFO BOX OVERRIDE ── */
.stAlert {
  background: var(--ink2) !important;
  border: 1px solid var(--line) !important;
  border-radius: 10px !important;
  color: var(--muted) !important;
}

/* ── COLUMNS ── */
[data-testid="column"] { padding: 0 0.4rem !important; }

/* ── DIVIDER ── */
hr { border-color: var(--line) !important; margin: 2.5rem 0 !important; }
</style>
""", unsafe_allow_html=True)


# ─── TOP NAV ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
  <div class="topbar-brand">Ọwọ · AI</div>
  <div style="display:flex;align-items:center;gap:0.8rem;">
    <span class="topbar-tag">Live</span>
    <div class="topbar-dot"></div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─── HERO ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-accent-num">28</div>
  <div class="hero-eyebrow">ASL Sign Language Detection</div>
  <h1 class="hero-heading">Read every<br><em>hand.</em></h1>
  <p class="hero-sub">
    Upload a hand sign photo. Ọwọ detects the ASL letter and 
    speaks it aloud — built on <strong>MediaPipe landmarks</strong> + 
    <strong>XGBoost</strong> with 98.43% accuracy. For Nigeria's Deaf community.
  </p>
</div>
""", unsafe_allow_html=True)

# ─── STATS ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="stats-row">
  <div class="stat-cell">
    <div class="stat-num">98.43%</div>
    <div class="stat-desc">Model accuracy on test set</div>
  </div>
  <div class="stat-cell">
    <div class="stat-num">28</div>
    <div class="stat-desc">ASL signs recognised</div>
  </div>
  <div class="stat-cell">
    <div class="stat-num">63</div>
    <div class="stat-desc">landmark numbers per hand</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─── MODEL LOADING ────────────────────────────────────────────────────────────
@st.cache_resource
def load_detector():
    if not os.path.exists("hand_landmarker.task"):
        with st.spinner("Fetching MediaPipe model…"):
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
                "hand_landmarker.task"
            )
    base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
    options = vision.HandLandmarkerOptions(
        base_options=base_options, num_hands=1,
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
    return f"""<script>
    (function(){{
        const s = window.speechSynthesis;
        s.cancel();
        const u = new SpeechSynthesisUtterance("{text}");
        u.rate = 0.88; u.pitch = 1.05; u.volume = 1;
        s.speak(u);
    }})();
    </script>"""

detector = load_detector()
clf, le = load_model()


# ─── REFERENCE SIGNS ─────────────────────────────────────────────────────────
st.markdown("""
<div class="section-head">
  <span class="section-label">Reference signs</span>
  <span class="section-line"></span>
</div>
<div class="ref-notice">
  <span>✦</span>
  <span>These are example hand positions — try uploading a similar photo below</span>
</div>
""", unsafe_allow_html=True)

samples = {"B":"sampleb.png","C":"samplec.png","L":"samplel.png","V":"samplev.png","W":"samplew.png","I":"samplei.png"}
cols = st.columns(len(samples))
for i, (label, path) in enumerate(samples.items()):
    with cols[i]:
        st.image(path)
        st.caption(label)

st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)

# ─── UPLOAD ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="section-head">
  <span class="section-label">Upload your sign</span>
  <span class="section-line"></span>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "upload_sign",
    type=["jpg","jpeg","png"],
    label_visibility="collapsed"
)

# ─── INFERENCE ────────────────────────────────────────────────────────────────
if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    col_orig, col_ann = st.columns(2)
    with col_orig:
        st.image(image_rgb, caption="Original", use_container_width=True)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result  = detector.detect(mp_image)

    if result.hand_landmarks:
        landmarks = result.hand_landmarks[0]

        # Draw landmarks
        annotated = image_rgb.copy()
        for lm in landmarks:
            cx = int(lm.x * annotated.shape[1])
            cy = int(lm.y * annotated.shape[0])
            cv2.circle(annotated, (cx, cy), 5, (0, 229, 200), -1)
        connections = [
            (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
            (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
            (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)
        ]
        for a, b in connections:
            ax = int(landmarks[a].x * annotated.shape[1]); ay = int(landmarks[a].y * annotated.shape[0])
            bx = int(landmarks[b].x * annotated.shape[1]); by = int(landmarks[b].y * annotated.shape[0])
            cv2.line(annotated, (ax, ay), (bx, by), (0, 180, 160), 1)

        with col_ann:
            st.image(annotated, caption="Landmarks", use_container_width=True)

        row = []
        for lm in landmarks:
            row.extend([lm.x, lm.y, lm.z])
        row = normalize_landmarks(row)

        prediction_enc = clf.predict([row])[0]
        prediction     = le.inverse_transform([prediction_enc])[0]
        confidence     = clf.predict_proba([row]).max() * 100

        # Confidence colour
        if confidence > 90:
            fill_color = "var(--glow)"
            verdict_icon  = "✦"
            verdict_text  = "High confidence — strong landmark match."
            verdict_class = "status-high"
        elif confidence > 70:
            fill_color = "var(--ember)"
            verdict_icon  = "◈"
            verdict_text  = "Medium confidence — try better lighting for a sharper result."
            verdict_class = "status-med"
        else:
            fill_color = "#ff4646"
            verdict_icon  = "◇"
            verdict_text  = "Low confidence — clearer photo recommended."
            verdict_class = "status-low"

        st.markdown(f"""
        <div class="result-wrap">
          <div class="result-card">
            <div class="result-glow-bar"></div>
            <div class="result-inner">
              <div class="result-letter-block">
                <span class="result-letter">{prediction}</span>
                <span class="result-letter-label">ASL sign</span>
              </div>
              <div class="result-meta">
                <div>
                  <div class="result-conf-label">
                    <span>Confidence</span>
                    <span class="result-conf-val">{confidence:.1f}%</span>
                  </div>
                  <div class="conf-track">
                    <div class="conf-fill" style="width:{confidence:.1f}%;background:{fill_color};box-shadow:0 0 12px {fill_color}60;"></div>
                  </div>
                </div>
                <div class="result-verdict">
                  <span class="verdict-icon">{verdict_icon}</span> {verdict_text}
                </div>
              </div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # TTS button
        if st.button(f"◈ Speak  {prediction}"): 
            st.components.v1.html(tts_html(f"The sign is {prediction}"), height=0)

        # Auto-speak
        st.components.v1.html(
            tts_html(f"Detected sign: {prediction}. Confidence: {confidence:.0f} percent."),
            height=0
        )

    else:
        st.markdown("""
        <div class="no-hand">
          <div class="no-hand-icon">◇</div>
          <div>
            <div class="no-hand-title">No hand detected</div>
            <div class="no-hand-body">
              Make sure your hand fills most of the frame, with clear lighting and an uncluttered background. 
              Dark skin tones are supported — bright, diffused lighting works best.
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)


# ─── BENCHMARKS ───────────────────────────────────────────────────────────────
st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
st.markdown("""
<div class="section-head">
  <span class="section-label">Model benchmarks</span>
  <span class="section-line"></span>
</div>
<div class="bench">
  <div class="bench-row">
    <span class="bench-model">Random Forest — raw landmarks</span>
    <span></span>
    <span class="bench-score">97.64%</span>
  </div>
  <div class="bench-row">
    <span class="bench-model">Random Forest — normalized</span>
    <span></span>
    <span class="bench-score">98.17%</span>
  </div>
  <div class="bench-row active">
    <span class="bench-model">XGBoost — normalized</span>
    <span class="bench-tag">Live</span>
    <span class="bench-score">98.43%</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  <div>
    <span class="footer-brand">Ọwọ AI</span>
    &nbsp;·&nbsp; Built by Temiloluwa Valentine 🇳🇬
  </div>
  <div class="footer-links">
    <a href="https://github.com/Valentinetemi/asl-sign-detection-mediapipe-rf" target="_blank">GitHub ↗</a>
    <a href="https://temiloluwaval.medium.com/i-replaced-87-000-images-with-63-numbers-heres-how-i-built-a-sign-language-detector-404f73b3c3aa" target="_blank">1st Article ↗</a>
    <a href="https://temiloluwaval.medium.com/why-i-chose-63-numbers-over-millions-of-pixels-landmarks-vs-cnn-for-sign-detection-14e0f75a92f7" target="_blank">2nd Article ↗</a>
  </div>
</div>
""", unsafe_allow_html=True)