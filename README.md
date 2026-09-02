# ♻️ Garbage Classifier — Max-Confidence Neural Ensemble

[![Status](https://img.shields.io/badge/Status-Completed-success?style=flat-square)](#)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg?style=flat-square)](LICENSE)

> **One-Line Value Proposition**: An intelligent deep learning waste classification engine combining dual EfficientNet vision models with a 60% confidence threshold guardrail to prevent misclassifications in recycling workflows.

---

## 📌 Project Overview

Single-model vision classifiers often suffer from high prediction variance when processing ambiguous or out-of-distribution waste images. **Garbage Classifier** solves this by feeding incoming image streams through a dual-model neural pipeline:
1. **Model V2 (EfficientNetB4)**: High-resolution feature extraction (380x380px) across **9 waste categories**.
2. **Model V3 (EfficientNetB3)**: Compact spatial feature extraction (300x300px) across **8 waste categories**.

A programmatic logical gate compares class confidence distributions in real time, selecting the top prediction while enforcing a **60% confidence threshold** to flag ambiguous inputs as "Unknown / Require Manual Inspection".

---

## 🏗️ Architecture & Pipeline

`	ext
               ┌─────────────────────────────────────────┐
               │         Input Image / Camera            │
               └────────────────────┬────────────────────┘
                                    │
                     ┌──────────────┴──────────────┐
                     ▼                             ▼
         ┌──────────────────────┐      ┌──────────────────────┐
         │ Model V2 Preprocess  │      │ Model V3 Preprocess  │
         │     (380 × 380)      │      │     (300 × 300)      │
         └───────────┬──────────┘      └───────────┬──────────┘
                     ▼                             ▼
         ┌──────────────────────┐      ┌──────────────────────┐
         │ EfficientNetB4 (9-cl)│      │ EfficientNetB3 (8-cl)│
         └───────────┬──────────┘      └───────────┬──────────┘
                     │                             │
                     └──────────────┬──────────────┘
                                    ▼
                     ┌─────────────────────────────┐
                     │   Max-Confidence Comparator │
                     │   np.max(P1) vs np.max(P2)  │
                     └──────────────┬──────────────┘
                                    │
                          [Confidence >= 60%]
                         /                   \
                       YES                    NO
                       /                       \
                      ▼                         ▼
          Winning Waste Category       Flagged as "Unknown State"
`

---

## ✨ Features

- **Dual EfficientNet Inference**: Evaluates images across two distinct network depth configurations simultaneously.
- **Fail-Safe Confidence Thresholding**: Enforces a 60% minimum probability gate to prevent confident sorting mistakes.
- **Dynamic Array Preprocessing**: Scales input frames on-the-fly into parallel RGB tensor formats matching each model's native resolution requirements.
- **Interactive Visual UI**: Displays probability distribution graphs, individual model votes, and confidence scores in Streamlit.

---

## 📊 Measured Model Specs & Metrics

| Pipeline Component | Model V2 Specifications | Model V3 Specifications |
|---|---|---|
| **Base Network** | EfficientNetB4 | EfficientNetB3 |
| **Input Shape** | 380 × 380 × 3 RGB | 300 × 300 × 3 RGB |
| **Evaluated Categories** | 9 Classes | 8 Classes |
| **Decision Logic** | Max-Confidence Comparison Gate | Max-Confidence Comparison Gate |
| **Safety Threshold** | 60.0% Minimum Peak Probability | 60.0% Minimum Peak Probability |

### Waste Classes Evaluated
Cardboard, Glass, Metal, Paper, Plastic, Trash (Non-recyclable), Biodegradable Organic, Clothes, Footwear.

---

## 🛠️ Tech Stack

- **Core Language**: Python 3.10
- **Deep Learning Framework**: TensorFlow / Keras
- **Image Processing**: Pillow (PIL), NumPy
- **User Interface**: Streamlit

---

## 💻 Installation & Setup

`ash
# Clone repository
git clone https://github.com/wajahat2005/garbage-classifier.git
cd garbage-classifier

# Install dependencies
pip install -r requirements.txt

# Launch web application
streamlit run app.py
`

---

## 📁 Project Structure

`	ext
garbage-classifier/
├── app.py              # Main Streamlit application & ensemble inference pipeline
├── requirements.txt    # Python dependencies (TensorFlow, Streamlit, NumPy, Pillow)
├── README.md           # Project documentation & specs
└── .github/
    └── workflows/
        └── ci.yml      # GitHub Actions CI workflow
`

---

## 🔮 Future Roadmap

- [ ] Quantize model weights to TFLite for edge execution on Raspberry Pi zero.
- [ ] Upgrade bounding-box multi-object localization using YOLOv8.

---

## 📄 License

Distributed under the [MIT License](LICENSE).