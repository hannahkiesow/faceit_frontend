
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from theme import apply_theme
import os
from PIL import Image

apply_theme()

# -------------------
# PAGE CONFIG
# -------------------
st.set_page_config(page_title="ML Model Overview", layout="wide")

st.title("Face It : Model Insights")
st.markdown("Built on **FER-2013 Dataset** to classify 7 universal emotions (Ekman & Friesen, 1971).")

# -------------------
# MAIN TABS
# -------------------
tab_emotions, tab_dataset, tab_data_journey, tab_results, tab_explainability, tab_future = st.tabs(
    ["😊 Introduction", "📊 Dataset", "🛠️ Data Journey", "📈 Results", "🔍 Explainability", "🚀 Future Work"]
)

# -------------------
# TAB 1: EMOTIONS
# -------------------
with tab_emotions:
    st.header("😊 7 Universal Emotions")
    emotions = {
        "Happy": "😊",
        "Sad": "😢",
        "Fear": "😱",
        "Anger": "😡",
        "Disgust": "🤢",
        "Surprise": "😲",
        "Neutral": "😐"
    }
    cols = st.columns(4)
    for i, (emo, icon) in enumerate(emotions.items()):
        with cols[i % 4]:
            st.markdown(f"### {icon} {emo}")
    st.markdown("---")

    st.header("Project Overview")
    st.markdown("""
    **Basic Idea:**
    - Capture a face image from the user or a dataset.
    - Preprocess the image (grayscale, resize, normalize).
    - Feed the image to a trained **CNN** or **pretrained ResNet50**.
    - Predict emotion with confidence score.
    - Explain predictions with **Grad-CAM** & **SHAP**.

    **Goal:**
    - Build a system that is **accurate, fair, and explainable**.
    - Reduce misclassifications, especially subtle emotions like Fear & Disgust.
    - Deploy for real-time apps/web.
    """)
    st.markdown("---")

# -------------------
# TAB 2: DATASET
# -------------------
with tab_dataset:
    st.header("📊 Dataset Information")
    st.markdown("""
    **Dataset Name:** FER 2013 Dataset with balanced images for Disgust
    **Format:** PNG/JPG
    **Emotion Classes:** Happiness, Sadness, Fear, Anger, Disgust, Surprise, Neutral
    """)
    dataset_image_path = "media/Images/Dataset.png"
    if os.path.exists(dataset_image_path):
        img = Image.open(dataset_image_path)
        max_width = 1000
        w_percent = (max_width / float(img.size[0]))
        h_size = int((float(img.size[1]) * float(w_percent)))
        img = img.resize((max_width, h_size))
        st.image(img, caption="Sample Dataset Image", use_container_width=False)
    else:
        st.warning(f"Dataset image not found: {dataset_image_path}")
    st.info("✅ Balanced data ensures fairness and reduces misclassifications.")

# -------------------
# TAB 3: DATA JOURNEY
# -------------------
with tab_data_journey:
    st.header("🛠️ The Data Journey Blueprint")
    data_journey_img_path = "media/Images/Blueprint1.png"
    if os.path.exists(data_journey_img_path):
        img = Image.open(data_journey_img_path)

        # Resize image manually
        max_width = 1000  # desired width
        w_percent = (max_width / float(img.size[0]))
        h_size = int((float(img.size[1]) * float(w_percent)))
        img = img.resize((max_width, h_size))

        st.image(img, caption="Data Journey Overview", use_container_width=False)
    else:
        st.warning(f"Data Journey image not found: {data_journey_img_path}")
    st.markdown("---")

# -------------------
# TAB 4: RESULTS
# -------------------
with tab_results:
    st.header("📈 Model Performance")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Overall Accuracy", "69%")
    col2.metric("Disgust", "92%")
    col3.metric("Happy", "88%")
    col4.metric("Fear", "Low Recall")
    st.markdown("---")

# -------------------
# TAB 5: MODEL EXPLAINABILITY
# -------------------
with tab_explainability:
    st.header("🔍 Model Explainability")
    st.markdown("""
    We used a **pretrained ResNet50 model** to recognize facial emotions.
    Interpretability techniques help us see which facial regions influenced predictions.
    """)

    tab_grad, tab_shap, tab_cm = st.tabs(["Grad-CAM", "SHAP Values", "Confusion Matrix"])

    # Grad-CAM
    with tab_grad:
        st.markdown("**Grad-CAM** highlights the facial regions critical for the model’s prediction.")
        gradcam_img_path = "media/Images/Gradcam.png"
        if os.path.exists(gradcam_img_path):
            img = Image.open(gradcam_img_path)
            max_width = 1000
            w_percent = (max_width / float(img.size[0]))
            h_size = int((float(img.size[1]) * float(w_percent)))
            img = img.resize((max_width, h_size))
            st.image(img, caption="Grad-CAM Example", use_container_width=False)
        else:
            st.warning(f"Image not found: {gradcam_img_path}")

    # SHAP Values
    with tab_shap:
        st.markdown("**SHAP values** quantify pixel-level contributions to predictions.")
        shap_img_path = "media/Images/SHAP.png"
        if os.path.exists(shap_img_path):
            img = Image.open(shap_img_path)
            max_width = 300
            w_percent = (max_width / float(img.size[0]))
            h_size = int((float(img.size[1]) * float(w_percent)))
            img = img.resize((max_width, h_size))
            st.image(img, caption="SHAP Example", use_container_width=False)
        else:
            st.warning(f"Image not found: {shap_img_path}")

    # Confusion Matrix
    with tab_cm:
        st.markdown("**Confusion Matrix** shows misclassifications per emotion class.")
        cm_img_path = "media/Images/CM test.png"
        if os.path.exists(cm_img_path):
            img = Image.open(cm_img_path)
            max_width = 1000
            w_percent = (max_width / float(img.size[0]))
            h_size = int((float(img.size[1]) * float(w_percent)))
            img = img.resize((max_width, h_size))
            st.image(img, caption="Confusion Matrix", use_container_width=False)
        else:
            st.warning(f"Image not found: {cm_img_path}")

# -------------------
# TAB 6: FUTURE WORK
# -------------------
with tab_future:
    st.header("🚀 Future Work")
    st.markdown("""
    - Improve recall for **Fear** and other low-performing emotions
    - Explore **Vision Transformers**
    - Build **mobile/web app deployment**
    - Continue reducing bias & improving fairness
    """)
