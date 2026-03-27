# Ọwọ AI — Hand Sign Detection for Nigeria's Deaf Community

![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.33-green?style=flat-square)
![XGBoost](https://img.shields.io/badge/XGBoost-98.43%25_accuracy-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)
![SDG](https://img.shields.io/badge/UN_SDG-10_%7C_3-red?style=flat-square)

> 🏆 **Submitted to the 2030 AI Challenge — Code for Change. Build the Future.**  
> Aligned with **UN SDG 10** (Reduced Inequalities) · **UN SDG 3** (Good Health and Well-being)

**[→ Live Demo](https://your-streamlit-url.streamlit.app)** · **[→ Beginner Article](https://temiloluwaval.medium.com/i-replaced-87-000-images-with-63-numbers-heres-how-i-built-a-sign-language-detector-404f73b3c3aa)** · **[→ Technical Deep-Dive](https://temiloluwaval.medium.com/why-i-chose-63-numbers-over-millions-of-pixels-landmarks-vs-cnn-for-sign-detection-14e0f75a92f7)**

---

## Overview

Ọwọ AI is an ASL alphabet detection system that achieves **98.43% accuracy across 28 sign classes** - without training on a single raw image. Instead of feeding pixels into a CNN, it extracts 21 hand keypoints via MediaPipe (63 features: x, y, z per landmark), normalizes them relative to the wrist, and classifies the resulting hand shape with XGBoost.

Built as an accessibility tool for Nigeria's ~3 million deaf people, using ASL as a practical proxy for Nigerian Sign Language (NSL) due to the absence of publicly available NSL datasets.

---

## The Problem

Most sign language AI is:
- Trained and tested on Western subjects with lighter skin tones
- Built on pixel-level CNNs that are brittle to lighting and background changes
- Not accessible to communities in low-bandwidth or low-resource environments

Ọwọ directly addresses all three. Landmark-based inference is skin-tone agnostic by design and it doesn't see pixels, only geometry.

---

## Technical Approach

### Why Landmarks Instead of Raw Images?

A CNN trained on 87,000 images learns to recognize pixels. A landmark model learns hand *shape*. That's a fundamentally more generalizable representation and one that's:

- **Faster** — inference on 63 numbers, not 224×224 pixels
- **Lighter** — XGBoost model is kilobytes, not megabytes
- **Fairer** — skin tone, lighting, and background are irrelevant to landmark geometry

### Pipeline

```
Raw Image (JPG/PNG)
      │
      ▼
MediaPipe HandLandmarker
  → 21 keypoints × (x, y, z) = 63 features
      │
      ▼
Wrist-Centered Normalization
  → subtract wrist position (keypoint 0)
  → scale by distance to middle finger MCP (keypoint 12)
  → output: scale-invariant, position-invariant hand shape
      │
      ▼
XGBoost Classifier
  → 28 output classes (A–Z + space + del)
  → trained on 69,273 samples (87,000 images − ~24% failed detections)
      │
      ▼
Predicted Sign + Confidence Score + TTS Output
```

### Normalization — The Critical Step

```python
def normalize_landmarks(row):
    landmarks = np.array(row).reshape(21, 3)
    wrist = landmarks[0]
    landmarks = landmarks - wrist          # translate to origin
    hand_size = np.linalg.norm(landmarks[12])
    if hand_size > 0:
        landmarks = landmarks / hand_size  # scale-normalize
    return landmarks.flatten().tolist()
```

This single transformation makes the model robust to:
- Hand distance from camera
- Hand scale differences between users
- Absolute position in frame

---

## Model Benchmarks

| Model | Normalization | Accuracy |
|---|---|---|
| Random Forest | ✗ Raw landmarks | 97.64% |
| Random Forest | ✓ Normalized | 98.17% |
| **XGBoost** | **✓ Normalized** | **98.43% ← live model** |

Each iteration was a deliberate, principled improvement — not hyperparameter lottery.

**XGBoost over Random Forest:** Boosting builds trees sequentially, each correcting the residual errors of the previous. For a 28-class problem where several signs are visually similar (M/N, A/X, E/S), this sequential error-correction gives measurable gains over bagging.

---

## Results

| Metric | Score |
|---|---|
| Accuracy | **98.43%** |
| Macro Avg Precision | 0.97 |
| Macro Avg Recall | 0.97 |
| Macro Avg F1 | 0.97 |
| Classes above 95% accuracy | **26 / 28** |

### Confusion Matrix

![Confusion Matrix](confusion_matrix.png)

### Notable Observations

- **Perfect accuracy:** Z (unique two-stroke motion path)
- **Hardest pair:** M and N — 3 vs. 2 fingers folded over the thumb; geometrically close in 3D landmark space
- **Skin-tone validation:** Confirmed working on dark skin tones via real-world webcam testing — the normalization pipeline is why

---

## Dataset

- **Source:** [ASL Alphabet Dataset — Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- **Raw images:** 87,000 across 28 classes (A–Z + space + del)
- **After landmark extraction:** 69,273 samples retained
- **Detection failure rate:** ~24% — primarily due to extreme angles, occluded hands, and low-contrast backgrounds
- **Train/test split:** 80/20 stratified

> **Large file note:** `landmarks_dataset.csv` (82MB) is hosted on Google Drive due to GitHub limits.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Hand Detection | MediaPipe HandLandmarker 0.10.33 |
| Classification | XGBoost |
| Preprocessing | Scikit-learn |
| Web App | Streamlit |
| Image Processing | OpenCV |
| Data | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Language | Python 3.12 |

---

## Project Structure

```
asl-sign-detection-mediapipe-rf/
├── asl_sign_detection.ipynb    # Full training pipeline with analysis
├── app.py                       # Streamlit web app (upload + TTS)
├── asl_model_xgb.json          # Trained XGBoost model
├── label_encoder.pkl            # Label encoder
├── confusion_matrix.png         # Evaluation output
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Limitations & Honest Assessment

| Limitation | Detail |
|---|---|
| ASL ≠ NSL | Trained on ASL — a practical proxy, not a native solution |
| Confusable pairs | M/N, A/X, E/S remain the hardest classes due to geometric similarity |
| Lighting sensitivity | Performance degrades in poor lighting conditions |
| Dataset diversity | Single-source dataset limits contributor diversity |

---

## Future Work

- **NSL dataset collection** — the highest-leverage next step for true localization
- **Joint angle features** — derived features (finger curl angles, relative distances) for richer representation beyond raw xyz
- **Dynamic sign support** — LSTM/Transformer over landmark sequences for J, Z, and motion-dependent signs
- **Mobile deployment** — TFLite export for offline-capable Android app
- **CNN baseline benchmark** — formal comparison to validate the landmark-first hypothesis

---

## SDG Alignment

| Goal | Connection |
|---|---|
| **SDG 10 — Reduced Inequalities** | Builds accessible communication tools for Nigeria's ~3M deaf people, a population historically excluded from AI development |
| **SDG 3 — Good Health and Well-being** | Reduces communication barriers in healthcare settings where sign language interpretation is unavailable |

---

## Author

**Temiloluwa Valentine**  

- GitHub: [@Valentinetemi](https://github.com/Valentinetemi)
- Medium: [@temiloluwaval](https://temiloluwaval.medium.com)

---

## License

MIT — free to use, adapt, and build on.
