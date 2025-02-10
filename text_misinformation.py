import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ✅ Download NLTK resources (only once)
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")

# ✅ Preprocessing Function
def preprocess_text(text):
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    # Remove URLs, special characters, numbers
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # Remove URLs
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove numbers and special characters

    # Tokenize and lemmatize
    words = word_tokenize(text.lower())
    filtered_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    return " ".join(filtered_words)

# ✅ Load Dataset
def load_dataset():
    politifact_fake = pd.read_csv(r"FakeNewsNet-master/FakeNewsNet-master/dataset/politifact_fake.csv")
    politifact_real = pd.read_csv(r"FakeNewsNet-master/FakeNewsNet-master/dataset/politifact_real.csv")


    # Assign labels (0 = Fake, 1 = Real)
    politifact_fake["label"] = 0
    politifact_real["label"] = 1

    # Balance dataset (equal real and fake samples)
    min_samples = min(len(politifact_fake), len(politifact_real))
    politifact_fake = politifact_fake.sample(min_samples, random_state=42)
    politifact_real = politifact_real.sample(min_samples, random_state=42)

    # Merge datasets
    combined_df = pd.concat([politifact_fake, politifact_real], ignore_index=True)

    # Use 'content' column if available, otherwise 'title'
    if "content" in combined_df.columns:
        combined_df = combined_df.dropna(subset=["content"])
        X = combined_df["content"]
    else:
        combined_df = combined_df.dropna(subset=["title"])
        X = combined_df["title"]

    y = combined_df["label"]

    # Split dataset (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    return X_train, y_train, X_test, y_test

# ✅ Train Model
def train_model():
    print("⏳ Training model...")

    # Load and preprocess dataset
    X_train, y_train, X_test, y_test = load_dataset()
    X_train = X_train.apply(preprocess_text)
    X_test = X_test.apply(preprocess_text)

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 2))

    # Train Naïve Bayes Model
    classifier = MultinomialNB(alpha=0.5)  # Adjusted alpha for better performance
    # Tokenize, handle negations, and lemmatize
    text = re.sub(r"\bnot\b\s+(\w+)", r"not_\1", text)  # Handle negations

    words = word_tokenize(text.lower())
    filtered_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    return " ".join(filtered_words)

# ✅ Load Dataset
def load_dataset():
    politifact_fake = pd.read_csv(r"FakeNewsNet-master/FakeNewsNet-master/dataset/politifact_fake.csv")
    politifact_real = pd.read_csv(r"FakeNewsNet-master/FakeNewsNet-master/dataset/politifact_real.csv")


    # Assign labels (0 = Fake, 1 = Real)
    politifact_fake["label"] = 0
    politifact_real["label"] = 1

    # Balance dataset (equal real and fake samples)
    min_samples = min(len(politifact_fake), len(politifact_real))
    politifact_fake = politifact_fake.sample(min_samples, random_state=42)
    politifact_real = politifact_real.sample(min_samples, random_state=42)

    # Merge datasets
    combined_df = pd.concat([politifact_fake, politifact_real], ignore_index=True)

    # Use 'content' column if available, otherwise 'title'
    if "content" in combined_df.columns:
        combined_df = combined_df.dropna(subset=["content"])
        X = combined_df["content"]
    else:
        combined_df = combined_df.dropna(subset=["title"])
        X = combined_df["title"]

    y = combined_df["label"]

    # Split dataset (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    return X_train, y_train, X_test, y_test

# ✅ Train Model
def train_model():
    print("⏳ Training model...")

    # Load and preprocess dataset
    X_train, y_train, X_test, y_test = load_dataset()
    X_train = X_train.apply(preprocess_text)
    X_test = X_test.apply(preprocess_text)

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 2))

    # Train Naïve Bayes Model
    classifier = MultinomialNB(alpha=0.2)  # Lower alpha to improve generalization

    model = make_pipeline(vectorizer, classifier)
    model.fit(X_train, y_train)

    # Evaluate Model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ Model accuracy: {accuracy * 100:.2f}%")
    print(classification_report(y_test, y_pred))

    # Save model
    joblib.dump(model, "misinformation_model.pkl")
    print("✅ Model trained and saved successfully!")

# ✅ Prediction Function
def predict_text_misinformation(text):
    try:
        model = joblib.load("misinformation_model.pkl")
        text_processed = preprocess_text(text)
        prediction = model.predict([text_processed])[0]
        confidence = max(model.predict_proba([text_processed])[0]) * 100

        if confidence < 70:  # If model is not confident

            return "Unclear Prediction - More training data needed.", confidence
        return "Fake News" if prediction == 0 else "Real News", confidence
    except Exception as e:
        return f"Error: {e}", 0.0

# ✅ Train model if script is run
if __name__ == "__main__":

    train_model()
