Scaling Trust: AI-Powered Detection of Deepfakes & Misinformation

📌 Project Overview

This project aims to detect and moderate harmful content at scale using Machine Learning (ML), Natural Language Processing (NLP), and Generative AI. It provides a solution for identifying deepfake images and videos and detecting misinformation in text-based content.

🚀 Key Features

✅ Deepfake Detection – Identifies manipulated images and videos using AI.
✅ Misinformation Detection – Classifies news articles as Fake or Real using NLP.
✅ Confidence Score Visualization – Displays how confident the model is in its predictions.
✅ Streamlit-based UI – Simple and interactive user interface for easy use.
✅ Web Deployment – Live on Render, making it accessible online.

🔗 Live Demo:

👉 Deepfake & Fake News Detector


---

🔬 Tech Stack

Programming Language: Python

Libraries: OpenCV, TensorFlow, Scikit-learn, nltk, Streamlit

Deployment: Render / GitHub



---

🛠 How to Run Locally?

1️⃣ Clone this repository

git clone https://github.com/divyanshi-27/Deepfake-Detection.git
cd Deepfake-Detection

2️⃣ Install Dependencies

pip install -r requirements.txt

3️⃣ Run the Streamlit App

streamlit run app.py


---

📊 Dataset Used

For Deepfake Detection: Kaggle Cats vs Dogs dataset (used for initial testing)

For Misinformation Detection: FakeNewsNet dataset (Politifact Fake & Politifact Real)



---

📉 Model Performance

Final Accuracy: 80%

Misinformation Detection: Confidence-based classification

Deepfake Detection: Frame-based image classification



---

🔥 Challenges Faced & Solutions

1️⃣ Misinformation Detection Confusion

Issue: The model sometimes misclassified real news as fake and vice versa.
Solution: Improved preprocessing, optimized confidence threshold, and experimented with different ML models.

2️⃣ Deepfake Model Complexity

Issue: The model initially had high processing time.
Solution: Used optimized architectures for faster and reliable predictions.

3️⃣ UI Design Issue

Issue: Initially, the file upload option was not user-friendly in the Streamlit UI.
Solution: Adjusted UI layout, ensuring the title appears before the upload section for better clarity.

4️⃣ Render Deployment Errors

Issue: Initially, the app was not processing images/videos correctly on Render but worked fine in VS Code.
Solution: Cleared build cache and re-deployed manually, fixing inconsistencies.


---

🎯 Future Improvements

✅ Extend to real-time video deepfake detection.
✅ Improve accuracy of misinformation detection.
✅ Add explainability to model predictions (e.g., highlighting fake news keywords).
✅ Optimize GPU usage for faster processing.


---

🏆 Hackathon Details

Event: Hack2Skill

Deadline: 28th March

Project Status: ✅ Ready for Submission

GitHub Repository: Deepfake Detection GitHub



 

---

Let me know if you need any modifications before submission! 🚀
 
 
