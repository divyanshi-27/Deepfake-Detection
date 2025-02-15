import streamlit as st
import os
import cv2
import numpy as np
import tempfile
from PIL import Image
from text_misinformation import predict_text_misinformation  # Ensure this function works

# Streamlit UI
st.title("📰 Deepfake & Fake News Detector")

# File uploader for images and videos
uploaded_file = st.file_uploader("Upload an image or video", type=['png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'])

# User Input for Text
user_input = st.text_area("Enter a news article or statement:")

# Function to detect deepfake (Placeholder Logic)
def detect_deepfake(file):
    try:
        if file.type.startswith("image"):
            image = Image.open(file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            return "REAL", 85.0  # Placeholder

        elif file.type.startswith("video"):
            temp_video = tempfile.NamedTemporaryFile(delete=False)
            temp_video.write(file.read())
            temp_video.close()

            cap = cv2.VideoCapture(temp_video.name)
            frame_count, fake_frames = 0, 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                if np.mean(frame) < 100:
                    fake_frames += 1

            cap.release()
            fake_ratio = fake_frames / frame_count if frame_count else 0
            is_fake = fake_ratio > 0.5
            confidence = 95.0 if is_fake else 88.0
            return ("FAKE" if is_fake else "REAL"), confidence
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return "ERROR", 0.0

# Processing Uploaded Media
if uploaded_file is not None:
    st.write(f"✅ File uploaded: {uploaded_file.name}")
    if uploaded_file.type.startswith("video"):
        st.video(uploaded_file, format="video/mp4")

    if st.button("Check Authenticity (Media)"):
        result, confidence = detect_deepfake(uploaded_file)
        if result == "ERROR":
            st.error("❌ Error in processing file. Please try another format.")
        else:
            st.success(f"🖼 Detection Result: {result} with confidence: {confidence:.2f}%")

# Processing Text Input
if user_input.strip():
    if st.button("Check Authenticity (Text)"):
        try:
            prediction, confidence_score = predict_text_misinformation(user_input)

            if confidence_score < 70:
                st.warning("⚠ Uncertain Prediction: More training data may be required.")
            else:
                if prediction == "Real News":
                    st.success(f"✅ REAL News with confidence: {confidence_score:.2f}%")
                else:
                    st.error(f"❌ FAKE News with confidence: {confidence_score:.2f}%")
        except Exception as e:
            st.error(f"Error in text prediction: {str(e)}")

# Run Streamlit App on the Correct Port
if _name_ == "_main_":
    port = int(os.environ.get("PORT", 8501))  # Use Render's provided port
    st.run(port=port, host="0.0.0.0")
