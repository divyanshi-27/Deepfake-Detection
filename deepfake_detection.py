import os
import cv2  # OpenCV for video processing
import numpy as np
from PIL import Image
import tensorflow as tf

# Load trained deepfake detection model
model = tf.keras.models.load_model("deepfake_detector.h5")  # Ensure the model is in the same folder

def check_csv_files():
    files = [ 
        "FakeNewsNet-master/FakeNewsNet-master/dataset/politifact_fake.csv",
        "FakeNewsNet-master/FakeNewsNet-master/dataset/politifact_real.csv"
    ]

    for file in files:
        if os.path.exists(file):
            print(f"✅ {file} FOUND")
        else:
            print(f"❌ {file} NOT FOUND - Check file path!")

# Call the function to check CSV files
check_csv_files()


# Function to check the file type (image or video)
def check_file_type(file_path):
    """Returns the type of file: 'image' or 'video'."""
    if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        return "image"
    elif file_path.lower().endswith(('.mp4', '.avi', '.mov')):
        return "video"
    else:
        return "unsupported"

# Function to detect deepfake from an uploaded image or video
def detect_deepfake(image_file):
    file_type = check_file_type(image_file)
    
    if file_type == "image":
        # Handle image detection
        image = Image.open(image_file).convert("RGB")
        image = image.resize((128, 128))  # Resize to match model input size
        image = np.array(image) / 255.0  # Normalize pixel values
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        
        # Make prediction
        prediction = model.predict(image)[0][0]
        confidence = prediction * 100  # Convert to percentage
        return "Fake" if prediction > 0.6 else "Real", confidence  # Adjusted threshold
    
    elif file_type == "video":
        # Handle video detection by processing each frame
        cap = cv2.VideoCapture(image_file)
        if not cap.isOpened():
            raise FileNotFoundError(f"Error opening video file: {image_file}")
        
        # Initialize variables for video processing
        frame_count = 0
        fake_count = 0
        real_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # End of video
            
            frame_resized = cv2.resize(frame, (128, 128))  # Resize to model input size
            frame_normalized = frame_resized / 255.0  # Normalize pixel values
            frame_input = np.expand_dims(frame_normalized, axis=0)  # Add batch dimension
            
            # Log the shape of the frame before prediction
            print(f"Frame shape: {frame_input.shape}")  # Debugging statement

            # Make prediction on the frame
            prediction = model.predict(frame_input)[0][0]
            if prediction > 0.6:  # Adjusted threshold
                fake_count += 1
            else:
                real_count += 1
            
            frame_count += 1
        
        cap.release()  # Release video capture

        # Decide the overall result based on frame predictions
        result = "Fake" if fake_count > real_count else "Real"
        confidence = (max(fake_count, real_count) / frame_count) * 100
        
        print(f"Processed video: {image_file}, Fake frames: {fake_count}, Real frames: {real_count}, Result: {result}, Confidence: {confidence:.2f}%")  # Debugging statement
        
        return result, confidence
    
    else:
        raise ValueError("Unsupported file type!")
