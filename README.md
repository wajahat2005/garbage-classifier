# ♻️ AI Garbage Classifier (Max-Confidence Ensemble)

An intelligent, interactive web application built with Streamlit and TensorFlow that classifies waste to promote better recycling habits. This project utilizes a **Max-Confidence Ensemble** of two separate EfficientNet models, dynamically comparing their predictions to deliver the most accurate classification.

## 🌟 Features

* **Dual-Model Face-Off:** Instead of relying on a single neural network, the app runs the image through two distinct models (Model V2 and Model V3) simultaneously. It evaluates both outputs and intelligently selects the prediction with the highest confidence score.
* **Dynamic Preprocessing:** Seamlessly handles multi-model inference by dynamically resizing and preprocessing the input image to match the specific architecture requirements of each model (380x380px vs. 300x300px) on the fly.
* **Transparent UI:** The application features a "glass-box" approach, displaying the individual guesses and confidence scores of *both* models before revealing the ultimate winner, along with a detailed probability distribution chart.
* **Flexible Input:** Users can upload existing images or use their device's camera for real-time waste classification.

## 🧠 Technical Architecture

1. **Base Models:** * **Model V2:** EfficientNet architecture optimized for `380x380` inputs across 9 waste classes.
   * **Model V3:** EfficientNet architecture optimized for `300x300` inputs across 8 waste classes.
2. **Inference Pipeline:**
   * Images are converted to RGB and branched into two separate preprocessing pipelines.
   * Both arrays are fed to their respective models.
   * A programmatic logical gate compares `np.max(pred1)` against `np.max(pred2)`.
   * A confidence threshold (60%) acts as a final fail-safe to prevent confident misclassifications, outputting an "Unknown" state if the threshold is not met.

## 🛠️ Tech Stack

* **Frontend/Deployment:** Streamlit
* **Deep Learning:** TensorFlow / Keras
* **Data Processing:** NumPy, Pandas
* **Computer Vision:** Pillow (PIL)

## 🚀 How to Run Locally

### Prerequisites
Make sure you have Python installed, then clone this repository to your local machine.

### 1. Install Dependencies
Navigate to the project folder and install the required libraries:
```bash
pip install -r requirements.txt
