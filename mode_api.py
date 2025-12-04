from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import urllib.request
import os
import re

app = Flask(__name__)
CORS(app)

MODEL_URL = "https://huggingface.co/iganesh07/email-classifier-model/resolve/main/email_classifier_model_compressed.pkl"
VECTORIZER_URL = "https://huggingface.co/iganesh07/email-classifier-model/resolve/main/tfidf_vectorizer_compressed.pkl"

MODEL_PATH = "email_classifier_model_compressed.pkl"
VECTOR_PATH = "tfidf_vectorizer_compressed.pkl"

def download_if_missing(url, path):
    if not os.path.exists(path):
        print(f"Downloading {path} from HuggingFace...")
        urllib.request.urlretrieve(url, path)
        print(f"Downloaded {path}")

try:
    # Download files if missing
    download_if_missing(MODEL_URL, MODEL_PATH)
    download_if_missing(VECTORIZER_URL, VECTOR_PATH)

    # Load using joblib
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTOR_PATH)

    MODEL_LOADED = True
    print("Model & vectorizer loaded successfully!")

except Exception as e:
    MODEL_LOADED = False
    print("MODEL LOADING ERROR:", e)


def simple_preprocess(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


@app.route('/')
def home():
    return jsonify({
        'status': 'online',
        'model_loaded': MODEL_LOADED,
        'model_type': 'Random Forest',
        'accuracy': '83%',
        'endpoints': {
            'predict': '/predict (POST)',
            'health': '/health (GET)',
            'info': '/ (GET)'
        }
    })


@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy' if MODEL_LOADED else 'error',
        'model_loaded': MODEL_LOADED,
        'model_type': 'Random Forest',
        'model_accuracy': 0.83,
        'classes': ['URGENT', 'NORMAL', 'INQUIRY'] if MODEL_LOADED else []
    })


@app.route('/predict', methods=['POST'])
def predict():
    if not MODEL_LOADED:
        return jsonify({
            'error': 'Model not loaded',
            'message': 'Model failed to load from HuggingFace.'
        }), 500

    try:
        data = request.get_json()

        if not data or 'text' not in data:
            return jsonify({
                'error': 'Missing required field',
                'message': 'Request must include "text" field'
            }), 400

        email_text = data['text']

        if not email_text or len(email_text.strip()) == 0:
            return jsonify({
                'error': 'Empty text',
                'message': 'Email text cannot be empty'
            }), 400

        processed_text = simple_preprocess(email_text)
        text_vectorized = vectorizer.transform([processed_text])

        prediction = model.predict(text_vectorized)[0]
        probabilities = model.predict_proba(text_vectorized)[0]
        confidence = float(max(probabilities) * 100)

        priority_map = {
            'URGENT': 'HIGH',
            'INQUIRY': 'MEDIUM',
            'NORMAL': 'LOW'
        }
        priority = priority_map.get(prediction, 'MEDIUM')

        class_probabilities = {
            class_name: round(float(probabilities[i] * 100), 2)
            for i, class_name in enumerate(model.classes_)
        }

        return jsonify({
            'classification': prediction,
            'confidence': round(confidence, 2),
            'priority': priority,
            'model': 'Random Forest',
            'model_accuracy': 83,
            'processed_text': processed_text[:100],
            'original_length': len(email_text),
            'processed_length': len(processed_text),
            'class_probabilities': class_probabilities
        }), 200

    except Exception as e:
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500


if __name__ == '__main__':
    print(f"Model loaded: {MODEL_LOADED}")
    print("Starting server on http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)

