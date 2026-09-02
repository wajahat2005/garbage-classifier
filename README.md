# Garbage Classifier — Max-Confidence Neural Ensemble

[![Status](https://img.shields.io/badge/Status-Completed_Prototype-success?style=flat-square)](#)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg?style=flat-square)](LICENSE)

> Deep learning waste classification application evaluating input images across dual EfficientNet neural networks with a 60% confidence threshold guardrail.

---

## Overview

Single-model vision classifiers can suffer from high prediction variance on ambiguous images. **Garbage Classifier** addresses this by routing image streams through two parallel vision architectures:
- **Model V2 (EfficientNetB4):** Trained on 9 waste categories with a 380x380px input shape.
- **Model V3 (EfficientNetB3):** Trained on 8 waste categories with a 300x300px input shape.

A logical gate compares prediction probability distributions in real time, selecting the top prediction while enforcing a **60% confidence threshold** to flag low-confidence inputs as "Unknown".

---

## Architecture & Pipeline

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

## Implemented Features

- **Dual Model Inference:** Runs parallel predictions through EfficientNetB4 and EfficientNetB3 networks.
- **Confidence Threshold Gate:** Flags predictions below 60% confidence to prevent confident misclassifications.
- **Dynamic Resizing:** Preprocesses inputs into RGB tensors matching each model's native resolution.
- **Streamlit Web Dashboard:** Interactive UI displaying individual model predictions, confidence scores, and probability distribution charts.

---

## Model Specifications

| Pipeline Component | Model V2 Specs | Model V3 Specs |
|---|---|---|
| **Base Architecture** | EfficientNetB4 | EfficientNetB3 |
| **Input Shape** | 380 × 380 × 3 RGB | 300 × 300 × 3 RGB |
| **Classes Evaluated** | 9 Classes | 8 Classes |
| **Ensemble Logic** | Max-Confidence Comparison Gate | Max-Confidence Comparison Gate |
| **Safety Threshold** | 60.0% Minimum Peak Probability | 60.0% Minimum Peak Probability |

---

## Tech Stack

- **Language:** Python 3.10
- **Deep Learning:** TensorFlow / Keras
- **Image Processing:** Pillow (PIL), NumPy
- **Dashboard:** Streamlit

---

## Installation & Setup

`ash
# Clone repository
git clone https://github.com/wajahat2005/garbage-classifier.git
cd garbage-classifier

# Install dependencies
pip install -r requirements.txt

# Launch app
streamlit run app.py
`

---

## License

Distributed under the [MIT License](LICENSE).