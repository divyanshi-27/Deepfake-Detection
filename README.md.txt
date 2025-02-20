Scaling Trust: AI-Powered Detection of Deepfakes & Misinformation

📌 Project Overview

This project aims to detect and moderate harmful content at scale using Machine Learning (ML), Natural Language Processing (NLP), and Generative AI. It provides a solution for identifying deepfake images and videos and detecting misinformation in text-based content.

🚀 Key Features

- ✅ *Deepfake Detection* – Identifies manipulated images and videos using AI.
- ✅ *Misinformation Detection* – Classifies news articles as Fake or Real using NLP.
- ✅ *Confidence Score Visualization* – Displays how confident the model is in its predictions.
- ✅ *Streamlit-based UI* – Simple and interactive user interface for easy use.
- ✅ *Web Deployment* – Live on Render, making it accessible online.


🔗 Live Demo:
1--  VS Code (Everything Working : Image ,Video,Text)
https://drive.google.com/file/d/1J7s-0LDQ1QKvwmRgsnwDlAqTzl1m618A/view?usp=drivesdk

2-- Render Deployment (Only Image & Video Working , Text processing not working)
https://drive.google.com/file/d/1IyYZG0SrYWBYqYr_iiCAWjvvmsb6rbIh/view?usp=drivesdk



🔬 Tech Stack

- *Programming Language:* Python
- *Libraries:* OpenCV, TensorFlow, Scikit-learn, nltk, Streamlit
- *Deployment:* Render / GitHub


🤟 How to Run Locally?

1️⃣ Clone this repository:
sh
git clone https://github.com/divyanshi-27/Deepfake-Detection.git
cd Deepfake-Detection


2️⃣ Install Dependencies:
sh
pip install -r requirements.txt


3️⃣ Run the Streamlit App:
sh
streamlit run app.py

**Live Demo:**
 [Deepfake & Fake news Detector]
https://deepfake-detection-gwz4.onrender.com


---

📊 Dataset Used

- *Deepfake Detection:* Kaggle Cats vs Dogs dataset (used for initial testing)
- *Misinformation Detection:* FakeNewsNet dataset (Politifact Fake & Politifact Real)

---

 📉 Model Performance

- *Final Accuracy:* 80%
- *Misinformation Detection:* Confidence-based classification
- *Deepfake Detection:* Frame-based image classification

---

 🔥 Challenges Faced & Solutions

1️⃣ *Misinformation Detection Confusion*
   - *Issue:* The model sometimes misclassified real news as fake and vice versa.
   - *Solution:* Improved preprocessing, optimized confidence threshold, and experimented with different ML models.

2️⃣ *Deepfake Model Complexity*
   - *Issue:* The model initially had high processing time.
   - *Solution:* Used optimized architectures for faster and reliable predictions.

3️⃣ *UI Design Issue*
   - *Issue:* Initially, the file upload option was not user-friendly in the Streamlit UI.
   - *Solution:* Adjusted UI layout, ensuring the title appears before the upload section for better clarity.

4️⃣ *Render Deployment Errors*
   - *Issue:* Initially, the app was not processing images/videos correctly on Render but worked fine in VS Code.
   - *Solution:* Cleared build cache and re-deployed manually, fixing inconsistencies.

---

🎯 Future Improvements

- ✅ Extend to real-time video deepfake detection.
- ✅ Improve accuracy of misinformation detection.
- ✅ Add explainability to model predictions (e.g., highlighting fake news keywords).
- ✅ Optimize GPU usage for faster processing.

---

 🏆 Hackathon Details

- *Event:* Hack2Skill
- *Deadline:* 28th March
- *Project Status:* ✅ Ready for Submission
- *GitHub Repository:* [Deepfake Detection GitHub](https://github.com/divyanshi-27/Deepfake-Detection)

---

📌 Comments on Current Status

 ✅ What’s Working Well?
- Image-based deepfake detection is functional.
- Misinformation detection model is producing results.
- UI improvements have enhanced the user experience.
- Deployment on Render is successful.

❗ Areas for Improvement
- Confidence score needs better visualization.
- Some misclassifications still occur in misinformation detection.
- Real-time deepfake video detection is not yet implemented.
- Text more train data is needed.

 💪 Future Enhancements You Plan to Work On
- Improving misinformation detection accuracy.
- Enhancing the UI with better feedback on predictions.
- Implementing real-time video processing.
- Adding an API for integration with external applications.

 
 
