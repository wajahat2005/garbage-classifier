import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image

# ==========================================
# 1. PAGE CONFIG
# ==========================================
st.set_page_config(page_title="♻️ AI Garbage Classifier", layout="centered")

# ==========================================
# 2. CONSTANTS & CLASS NAMES
# ==========================================
# Model 1 (v2) - 380x380
MODEL1_URL = "https://huggingface.co/2005-wajahat/gaarbage-classifier-v2/resolve/main/model.keras"
CLASSES_1 = ['cardboard', 'e-waste', 'glass', 'metal', 'organic', 'paper', 'plastic', 'textile', 'trash']
SIZE_1 = (380, 380)

# Model 2 (v3) - 300x300
MODEL2_URL = "https://huggingface.co/2005-wajahat/garabage-classifier-v3/resolve/main/trash_classifier_final.keras"
CLASSES_2 = ['Unknown', 'glass', 'metal', 'organic_waste', 'paper_cardboard', 'plastic', 'textiles', 'trash']
SIZE_2 = (300, 300)

CONFIDENCE_THRESHOLD = 0.60

# ==========================================
# 3. CACHE & LOAD BOTH MODELS
# ==========================================
@st.cache_resource
def load_models():
    # Important: Give them different local filenames so they don't overwrite each other
    path1 = tf.keras.utils.get_file("model_v2.keras", MODEL1_URL)
    path2 = tf.keras.utils.get_file("model_v3.keras", MODEL2_URL)
    
    m1 = tf.keras.models.load_model(path1)
    m2 = tf.keras.models.load_model(path2)
    return m1, m2

with st.spinner("Downloading and loading AI models (this may take a minute)..."):
    model1, model2 = load_models()

# ==========================================
# 4. PREPROCESS FUNCTION
# ==========================================
def preprocess_image(image, target_size):
    """Resizes and preprocesses the image for the specific EfficientNet model."""
    img = image.convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ==========================================
# 5. USER INTERFACE
# ==========================================
st.title("♻️ AI Garbage Classifier (Ensemble)")
st.write("Upload an image or use your camera. Our dual-model AI will compare results and give you the most confident prediction.")

tab1, tab2 = st.tabs(["📸 Camera", "📁 Upload"])

image_data = None
with tab1:
    camera_file = st.camera_input("Take a picture")
    if camera_file: image_data = camera_file
with tab2:
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    if uploaded_file: image_data = uploaded_file

# ==========================================
# 6. ENSEMBLE LOGIC & PREDICTION
# ==========================================
if image_data is not None:
    image = Image.open(image_data)
    
    # Only display the uploaded image (camera automatically displays itself)
    if image_data == uploaded_file:
        st.image(image, caption="Input Image", use_container_width=True)
    
    if st.button("Analyze Waste"):
        with st.spinner("Analyzing with both models..."):
            
            # --- Model 1 Prediction ---
            img1 = preprocess_image(image, SIZE_1)
            pred1 = model1.predict(img1, verbose=0)[0]
            conf1 = np.max(pred1)
            class1 = CLASSES_1[np.argmax(pred1)]
            
            # --- Model 2 Prediction ---
            img2 = preprocess_image(image, SIZE_2)
            pred2 = model2.predict(img2, verbose=0)[0]
            conf2 = np.max(pred2)
            class2 = CLASSES_2[np.argmax(pred2)]
            
            st.markdown("---")
            
            # --- The "Face-Off" Logic ---
            # Compare which model is more confident
            if conf1 >= conf2:
                winner_class = class1
                winner_conf = conf1
                winner_name = "Model V2 (380px)"
                winning_probs = pred1
                winning_classes = CLASSES_1
            else:
                winner_class = class2
                winner_conf = conf2
                winner_name = "Model V3 (300px)"
                winning_probs = pred2
                winning_classes = CLASSES_2

            # --- Display Results ---
            st.subheader("🤖 AI Ensemble Results")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Model V2 guessed:** {class1.title()} ({conf1*100:.1f}%)")
            with col2:
                st.write(f"**Model V3 guessed:** {class2.title()} ({conf2*100:.1f}%)")

            st.write("") # Spacer
            
            # Final Threshold Check based on the winning model
            if winner_conf < CONFIDENCE_THRESHOLD or winner_class == "Unknown":
                st.warning("⚠️ **Not sure what this is (Low confidence)**")
                st.write("Neither model is confident enough to classify this item.")
            else:
                st.success(f"🏆 **Final Prediction:** {winner_class.upper()}")
                st.info(f"**Selected by:** {winner_name} with **{winner_conf*100:.1f}%** confidence")
            
            # Show the bar chart for the winning model's thought process
            st.write(f"**{winner_name} Probability Distribution:**")
            chart_data = pd.DataFrame({
                "Probability": winning_probs
            }, index=[c.title() for c in winning_classes])
            
            st.bar_chart(chart_data)
