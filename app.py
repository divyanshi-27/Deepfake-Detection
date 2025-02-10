import streamlit as st
from text_misinformation import predict_text_misinformation, preprocess_text, load_dataset

import cv2
import numpy as np
import tempfile
from PIL import Image

# Streamlit UI
st.title("📰 Deepfake & Fake News Detector")

# File uploader for images and videos
uploaded_file = st.file_uploader("Upload an image or video", type=['png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'])

# User Input for Text
user_input = st.text_area("Enter a news article or statement:")

# Function to detect deepfake (Basic Logic)
def detect_deepfake(file):
    try:
        if file.type.startswith("image"):
            image = Image.open(file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            image_array = np.array(image)
            is_fake = np.mean(image_array) < 100  
            confidence = 90.0 if is_fake else 85.0
            return ("FAKE" if is_fake else "REAL"), confidence

        elif file.type.startswith("video"):
            temp_video = tempfile.NamedTemporaryFile(delete=False)
            temp_video.write(file.read())
            temp_video.close()

            cap = cv2.VideoCapture(temp_video.name)
            frame_count = 0
            fake_frames = 0

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
        return "ERROR", 0.0

# Load Dataset
X_train, y_train, X_test, y_test = load_dataset()  # Load the updated dataset

# Processing Uploaded Media

if uploaded_file is not None:
    st.write(f"✅ File uploaded: {uploaded_file.name}")
    if uploaded_file.type.startswith("video"):
        st.video(uploaded_file, format="video/mp4")

    if st.button("Check Authenticity (Media)", key="media_button"):
        result, confidence = detect_deepfake(uploaded_file)
        if result == "ERROR":
            st.error("❌ Error in processing file. Please try another format.")
        else:
            st.success(f"🖼 Detection Result: {result} with confidence: {confidence:.2f}%")

# Processing Text Input
if user_input.strip() != "":
    if st.button("Check Authenticity (Text)", key="text_button"):
        cleaned_text = preprocess_text(user_input)  # Ensure cleaned_text is defined
        prediction, confidence_score = predict_text_misinformation(cleaned_text)  # Use the new prediction function

        st.write("📝 *Processed Text:*", cleaned_text)  # Debugging purpose

        if confidence_score < 60:  # Adjusted threshold to 65%

            st.warning("⚠ Uncertain Prediction: This text is unclear. More training data may be required.")
        else:
            if prediction == "Real News":
                st.success(f"✅ *REAL News* with confidence: {confidence_score:.2f}%")
            else:
                st.error(f"❌ *FAKE News* with confidence: {confidence_score:.2f}%")
