readme = """# ASL Sign Detection using MediaPipe and Random Forest

![Python](https://img.shields.io/badge/Python-3.12-blue)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.33-green)
![Accuracy](https://img.shields.io/badge/Accuracy-97.64%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Overview
A real-time American Sign Language (ASL) alphabet detection system built using MediaPipe hand landmarks and a Random Forest classifier. Achieves **97.64% accuracy** across 28 classes (A-Z + space + del).

## Motivation
Built as a proxy for Nigerian Sign Language (NSL) alphabet detection due to the lack of publicly available NSL datasets. ASL and NSL share similar fingerspelling patterns, making this a valid and practical starting point for building accessible communication tools for the Nigerian deaf community.

## Approach
Instead of training a CNN directly on raw images, this project:
1. Extracts 21 hand keypoints per image using MediaPipe (63 features: x, y, z per keypoint)
2. Trains a lightweight Random Forest classifier on those landmarks
3. Runs real-time inference via webcam

This approach is faster, more generalizable, and works across different skin tones and backgrounds compared to pixel-based CNNs.

## Pipeline
![Pipeline](confusion_matrix.png)

1. **Data Collection** — ASL Alphabet Dataset (87,000 images, 28 classes)
2. **Landmark Extraction** — MediaPipe HandLandmarker extracts 21 keypoints per image
3. **Training** — Random Forest classifier trained on 63 landmark features
4. **Evaluation** — 97.64% accuracy on 20% held-out test set
5. **Inference** — Real-time webcam detection *(coming soon)*

## Results
| Metric | Score |
|--------|-------|
| Accuracy | 97.64% |
| Macro Avg Precision | 0.97 |
| Macro Avg Recall | 0.97 |
| Macro Avg F1 | 0.97 |

### Confusion Matrix
![Confusion Matrix](confusion_matrix.png)

### Notable Observations
- **Perfect accuracy**: Z (100% — unique hand shape)
- **Most confused**: M and N (visually similar in ASL — 3 vs 2 fingers over thumb)
- **Consistent performance**: 26/28 classes above 95% accuracy

## Dataset
- [ASL Alphabet Dataset - Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- 87,000 images across 28 classes (A-Z + space + del)
- 66,000+ samples retained after MediaPipe landmark extraction
- ~24% failed detections due to awkward angles and lighting

## Tech Stack
| Tool | Purpose |
|------|---------|
| MediaPipe 0.10.33 | Hand landmark extraction |
| Scikit-learn | Random Forest classifier |
| OpenCV | Image processing |
| Pandas / NumPy | Data manipulation |
| Matplotlib / Seaborn | Visualization |
| Python 3.12 | Core language |

## Project Structure
