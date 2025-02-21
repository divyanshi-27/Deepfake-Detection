import os
import pandas as pd
import numpy as np
import re
import nltk
import google.generativeai as genai
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
from dotenv import load_dotenv

# ✅ Load API Key from .env
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    print("❌ Google API Key not found! Using only local model.")

# ✅ Download NLTK resources
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")

# ✅ Preprocessing Function
def preprocess_text(text):
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # Remove URLs
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters

    words = word_tokenize(text.lower())
    filtered_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    return " ".join(filtered_words)

# ✅ Load Dataset
def load_dataset():
    data = [
         {"text": "Breaking: Scientists discover new exoplanet!", "label": 1},
        {"text": "Aliens have landed in California!", "label": 0},
        {"text": "NASA confirms moon mission for 2025", "label": 1},
        {"text": "Government secretly controlling weather!", "label": 0},
        {"text": "5G towers are spreading COVID-19 vir,us! ","label": 0},
        {"text": "Bill Gates implants microchips in vaccines!", "label": 0},
        {"text": "Moon landing was staged in a Hollywood studio!", "label": 0},
        {"text": "Earth is flat and NASA is hiding the truth!", "label": 0},
        {"text": "Drinking bleach cures COVID-19!", "label": 0},
        {"text": "Vaccines cause autism in children!", "label": 0},
        {"text": "Elon Musk plans to control human thoughts with Neuralink!", "label": 0},
        {"text": "NASA discovers alien city on Mars!", "label": 0},
        {"text": "New study shows coffee prevents all diseases!", "label": 0},
        {"text": "Scientists confirm chocolate is a health food!", "label": 1},
        {"text": "International Space Station completes 25 years in orbit", "label": 1},
        {"text": "WHO announces new guidelines for pandemic preparedness", "label": 1},
        {"text": "Global leaders agree on climate change action plan", "label": 1},
        {"text": "COVID-19 is a hoax created to control the population!", "label": 0},
        {"text": "The government is hiding evidence of time travel!", "label": 0},
        {"text": "Chemtrails are being used to control our minds!", "label": 0},
        {"text": "NASA successfully launches new Mars rover", "label": 1},
        {"text": "World Health Organization declares end of pandemic", "label": 1},
        {"text": "Scientists develop new vaccine for rare disease", "label": 1}
    ]
    

    df = pd.DataFrame(data)
    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test


# ✅ Train Model
def train_model():
    print("⏳ Training model...")

    X_train, X_test, y_train, y_test = load_dataset()

    X_train = X_train.apply(preprocess_text)
    X_test = X_test.apply(preprocess_text)

    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 2))
    classifier = MultinomialNB(alpha=0.2)

    model = make_pipeline(vectorizer, classifier)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ Model accuracy: {accuracy * 100:.2f}%")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, "misinformation_model.pkl")
    print("✅ Model trained and saved successfully!")

# ✅ Google Gemini AI Integration
def predict_with_gemini(text):
    try:
        response = genai.generate_text(f"Is this news real or fake? {text}")
        prediction = response.text.lower()

        if "fake" in prediction:
            return "Fake News", 90.0  # Assumed 90% confidence
        elif "real" in prediction:
            return "Real News", 90.0
        else:
            return "Unclear", 50.0  # If Gemini AI is not confident
    except Exception as e:
        print(f"❌ Gemini AI Error: {e}")
        return "Error", 0.0

# ✅ Prediction Function (Hybrid Model)
def predict_text_misinformation(text):
    try:
        # Step 1: Local ML Model
        model = joblib.load("misinformation_model.pkl")
        text_processed = preprocess_text(text)
        prediction = model.predict([text_processed])[0]
        confidence = max(model.predict_proba([text_processed])[0]) * 100
        ml_result = "Fake News" if prediction == 0 else "Real News"

        # Step 2: Google Gemini AI Verification
        if GOOGLE_API_KEY:
            gemini_result, gemini_confidence = predict_with_gemini(text)
            if gemini_confidence > confidence:
                return gemini_result, gemini_confidence

        return ml_result, confidence
    except Exception as e:
        return f"Error: {e}", 0.0

# ✅ Train model if script is run
if __name__ == "__main__":

    train_model()
