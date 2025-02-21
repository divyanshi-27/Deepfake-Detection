import streamlit as st
import tensorflow as tf
from text_misinformation import predict_text_misinformation
import logging
import cv2
import numpy as np
import tempfile
from PIL import Image
import os
import imghdr
import magic

from dotenv import load_dotenv  # Import dotenv to load environment variables

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("❌ API Key not found! Please check your .env file.")

# Configure logging
logging.basicConfig(level=logging.INFO)

# Streamlit UI
st.title("📰 Deepfake & Fake News Detector")

# File uploader for images and videos
uploaded_file = st.file_uploader("Upload an image or video", type=['png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'])

# User Input for Text
user_input = st.text_area("Enter a news article or statement:")

# Function to detect deepfake (Placeholder Logic)
def validate_file_signature(file):
    """Validate file signature using magic numbers"""
    file_signature = magic.from_buffer(file.read(1024), mime=True)
    file.seek(0)  # Reset file pointer
    return file_signature

def detect_deepfake(file):
    try:
        if file.type.startswith("image"):
            # Validate file signature first
            file_signature = validate_file_signature(file)
            if not file_signature.startswith(('image/png', 'image/jpeg')):
                raise ValueError(f"Invalid image format. Expected PNG or JPEG, got {file_signature}")
            
            # Verify image can be opened
            image = Image.open(file)
            image.verify()  # Verify image integrity
            image = Image.open(file)  # Reopen after verification

            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Load the trained deepfake detection model
            model = tf.keras.models.load_model("deepfake_detector.h5")
            
            # Convert image to RGB and preprocess for the model
            image = image.convert("RGB")  # Convert to RGB
            image = image.resize((128, 128))  # Resize to match model input
            image_array = np.array(image) / 255.0  # Normalize the image
            image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
            
            # Make prediction
            prediction = model.predict(image_array)
            confidence = prediction[0][0] * 100  # Assuming binary classification
            
            is_fake = prediction[0][0] < 0.4  # Adjusted threshold for better detection
            st.write(f"Prediction: {'FAKE' if is_fake else 'REAL'}, Confidence: {confidence:.2f}%")

            return ("FAKE" if is_fake else "REAL"), confidence

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

    except ValueError as ve:
        st.error(f"❌ Invalid file format: {ve}")
        logging.error(f"File format error: {ve}")
        return "ERROR", 0.0
    except Image.UnidentifiedImageError:
        st.error("❌ Unable to identify image file. Please upload a valid PNG or JPEG image.")
        logging.error("Unidentified image file")
        return "ERROR", 0.0
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        logging.error(f"File processing error: {e}")
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
        prediction, confidence_score = predict_text_misinformation(user_input)

        if confidence_score < 75:  # Adjusted confidence threshold
            st.warning("⚠ Uncertain Prediction: More training data may be required.")
        else:
            if prediction == "Real News":
                st.success(f"✅ REAL News with confidence: {confidence_score:.2f}%")
            else:
                st.error(f"❌ FAKE News with confidence: {confidence_score:.2f}%")

