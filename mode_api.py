from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import re
import urllib.request
import ssl

app = Flask(__name__)
CORS(app)

# Disable SSL verification for Drive
ssl._create_default_https_context = ssl._create_unverified_context

VECTORIZER_URL = "https://drive.google.com/uc?export=download&id=1AGWmLwFpn3mN1Mzr2JhHzBjpLIRLeiz9"
MODEL_URL = "https://drive.google.com/uc?export=download&id=19STWsSyoQv9BvdTewj3NBeK_JrGZh34r"

# Load model and vectorizer
try:
    vectorizer_response = urllib.request.urlopen(VECTORIZER_URL, timeout=120)
    vectorizer = pickle.loads(vectorizer_response.read())
    
    model_response = urllib.request.urlopen(MODEL_URL, timeout=120)
    model = pickle.loads(model_response.read())
    
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    print("Make sure Google Drive files are set to 'Anyone with the link can view'")

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
            'batch': '/batch_predict (POST)',
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
            'message': 'Model failed to load from Google Drive. Check logs.'
        }), 500
    
    try:
        # Get email text from request
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
        
        # Preprocess
        processed_text = simple_preprocess(email_text)
        
        # Vectorize
        text_vectorized = vectorizer.transform([processed_text])
        
        # Predict
        prediction = model.predict(text_vectorized)[0]
        probabilities = model.predict_proba(text_vectorized)[0]
        confidence = float(max(probabilities) * 100)
        
        # Map to priority
        priority_map = {
            'URGENT': 'HIGH',
            'INQUIRY': 'MEDIUM',
            'NORMAL': 'LOW'
        }
        priority = priority_map.get(prediction, 'MEDIUM')
        
        # Get all class probabilities
        class_probabilities = {}
        for i, class_name in enumerate(model.classes_):
            class_probabilities[class_name] = round(float(probabilities[i] * 100), 2)
        
        # Response
        response = {
            'classification': prediction,
            'confidence': round(confidence, 2),
            'priority': priority,
            'model': 'Random Forest',
            'model_accuracy': 83,
            'processed_text': processed_text[:100],
            'original_length': len(email_text),
            'processed_length': len(processed_text),
            'class_probabilities': class_probabilities
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    print(f"Model loaded: {MODEL_LOADED}")
    print("\nStarting server on http://0.0.0.0:5000")
    print("="*60)
    
    app.run(host='0.0.0.0', port=5000, debug=True)