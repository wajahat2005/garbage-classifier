# ♻️ AI Waste Classifier — Max-Confidence Neural Ensemble

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg?style=flat-square)](LICENSE)

An intelligent deep learning web application that classifies physical waste into recyclable and non-recyclable categories to promote sustainable waste management. The system employs a **Max-Confidence Neural Ensemble** combining two distinct EfficientNet vision architectures to maximize accuracy and minimize misclassifications.

---

## 📌 Project Overview

Single-model computer vision classifiers often suffer from high variance on ambiguous or out-of-distribution waste images. To solve this, the **AI Waste Classifier** passes uploaded images or live camera frames through a dual-model pipeline:
- **Model V2 (EfficientNetB4)**: Specialized in high-resolution feature extraction (380x380px) across **9 waste categories**.
- **Model V3 (EfficientNetB3)**: Optimized for compact spatial representation (300x300px) across **8 core waste categories**.

A programmatic logical gate compares prediction confidence outputs in real time, selecting the top prediction while enforcing a **60% confidence threshold guardrail** to prevent misclassifications.

---

## ⚙️ Model Architecture & Technical Specifications

| Parameter | Model V2 | Model V3 |
|---|---|---|
| **Base Architecture** | EfficientNetB4 | EfficientNetB3 |
| **Input Resolution** | 380 × 380 px | 300 × 300 px |
| **Classes Evaluated** | 9 Classes | 8 Classes |
| **Ensemble Logic** | Max-Confidence Comparison (
p.max(pred1) vs np.max(pred2)) |
| **Guardrail Threshold** | 60% minimum probability for deterministic classification |

### Target Categories Evaluated
1. 📦 **Cardboard**
2. 🍾 **Glass**
3. 🥫 **Metal**
4. 📰 **Paper**
5. 🥤 **Plastic**
6. 🗑️ **Trash / Non-recyclable**
7. 🍂 **Biodegradable / Organic**
8. 👕 **Clothes & Textiles**
9. 👟 **Footwear / Shoes**

---

## 🚀 Key Features

- **Dual EfficientNet Face-Off**: Dynamically runs dual inference and displays individual confidence scores for both models side-by-side.
- **On-the-Fly Dynamic Preprocessing**: Converts input frames into RGB, scaling them into parallel tensor pipelines matching each neural network's exact input requirements.
- **Fail-Safe Thresholding**: Categorizes inputs under 60% peak probability as "Unknown / Ambiguous Waste" to prevent erroneous sorting recommendations.
- **Interactive Streamlit Interface**: Offers real-time camera feed capture, file upload support, and interactive probability distribution visualizations.

---

## 🛠️ Tech Stack

- **Core Language**: Python 3.10
- **Deep Learning**: TensorFlow / Keras
- **Computer Vision & Image Processing**: Pillow (PIL), NumPy
- **Frontend & Deployment**: Streamlit

---

## 💻 Installation & Usage

### 1. Clone Repository
`ash
git clone https://github.com/wajahat2005/garabge-classifier-final-combined-.git
cd garabge-classifier-final-combined-
`

### 2. Install Dependencies
`ash
pip install -r requirements.txt
`

### 3. Launch Application
`ash
streamlit run app.py
`

---

## 🔮 Future Roadmap

- [ ] **Edge Deployment**: Quantize model weights to TFLite for deployment on Raspberry Pi / mobile edge devices.
- [ ] **Bounding Box Detection**: Upgrade from full-image classification to multi-object detection using YOLOv8.
- [ ] **Recycling Recommendation API**: Integrate automated municipal disposal and recycling location lookup.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).