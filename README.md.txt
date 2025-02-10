Scaling Trust: AI-Powered Detection of Deepfakes & Misinformation

📌 Project Overview

This project focuses on AI-driven solutions to detect and moderate harmful content at scale, using Machine Learning, NLP, and Generative AI. It helps in identifying deepfake images and videos as well as detecting misinformation in text-based content.

🚀 Key Features

✅ Deepfake Detection – Identifies manipulated images and videos using AI.
✅ Misinformation Detection – Classifies news articles as Fake or Real using NLP.
✅ Confidence Score Visualization – Displays how confident the model is in its predictions.
✅ Streamlit-based UI – A simple and interactive user interface for easy usage.

🔬 Tech Stack

Programming Language: Python

Libraries: OpenCV, TensorFlow, Scikit-learn, NLTK, Streamlit

Deployment: Render / GitHub


🛠 How to Run Locally?

1️⃣ Clone this repository

git clone <your-repo-link>
cd Hackathon-Project

2️⃣ Install Dependencies

pip install -r requirements.txt

3️⃣ Run the Streamlit App

streamlit run app.py

📊 Dataset Used

For Deepfake Detection: Kaggle Cats vs Dogs dataset (used for initial testing).

For Misinformation Detection: FakeNewsNet dataset (Politifact Fake & Politifact Real).


📉 Model Performance

Final Accuracy: 84.23%

Misinformation Detection Performance: Confidence-based classification


🔥 Challenges Faced & Solutions

1️⃣ Misinformation Detection Confusion

Issue: The model sometimes misclassified real news as fake and vice versa.

Solution: Improved preprocessing, optimized confidence threshold, and experimented with different ML models.


2️⃣ Deepfake Model Complexity

Issue: The model initially had high processing time.

Solution: Used optimized architectures for faster and reliable predictions.


🎯 Future Improvements

✅ Extend to real-time video deepfake detection

✅ Improve accuracy of misinformation detection

✅ Add explainability to model predictions


🏆 Hackathon Details

Event: Hack2Skill

Deadline: 20th March

Project Status: ✅ Ready for Submission



 