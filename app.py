import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image
from model import load_mantranet_model, detect_forgery
import joblib

# Load models
mantra_model = load_mantranet_model("mantranet.pth")
fake_news_model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Page setup
st.set_page_config(page_title="Deception Detection App", layout="centered")
st.title("🔍 SPOTLIGHT ON DECEPTION")
st.markdown("Detect **image forgery** and **fake news articles** using deep learning.")

# Tabs for two types of detection
tab1, tab2 = st.tabs(["🖼️ Image Forgery Detection", "📰 Fake News Detection"])

# === Image Forgery Detection ===
with tab1:
    st.subheader("Upload an image to check for forgery")
    image_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    
    if image_file is not None:
        image = Image.open(image_file).convert("RGB")
        image_np = np.array(image)

        result = detect_forgery(mantra_model, image_np)

        st.image(image, caption="Original Image", use_column_width=True)
        st.image(result, caption="Forgery Localization Heatmap", use_column_width=True)

# === Fake News Detection ===
with tab2:
    st.subheader("Paste a news article to check for authenticity")
    user_input = st.text_area("Enter your news article here")

    if st.button("Detect Fake News"):
        if user_input.strip():
            input_vector = vectorizer.transform([user_input])
            prediction = fake_news_model.predict(input_vector)[0]
            label = "🟢 Real News" if prediction == 0 else "🔴 Fake News"
            st.success(f"Prediction: **{label}**")
        else:
            st.warning("Please enter some text.")
