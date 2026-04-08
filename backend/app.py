from fastapi import FastAPI, Query
import pickle
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import WordNetLemmatizer
import nltk

# Download NLTK data if needed
nltk.download('wordnet')
nltk.download('omw-1.4')

app = FastAPI()

# Load trained model and TF-IDF
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)

lemmatizer = WordNetLemmatizer()

# Function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in ENGLISH_STOP_WORDS]
    return " ".join(words)

# Home route
@app.get("/")
def home():
    return {"message": "API is running"}

# Predict route using query parameter
@app.post("/predict")
def predict(description: str = Query(..., description="Job description to classify")):
    cleaned = clean_text(description)
    vector = tfidf.transform([cleaned])
    
    # Model prediction
    prediction = model.predict(vector)[0]
    
    # Confidence score using predict_proba
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(vector)[0]  # [prob_real, prob_fake]
        confidence = max(proba) * 100
    else:
        # If model does not support predict_proba, default 100%
        confidence = 100.0
    
    # Result message
    if prediction == 1:
        result = "Fraud Job 🚨"
    else:
        result = "Real Job ✅"
    
    return {"result": result, "confidence": f"{confidence:.2f}%"}