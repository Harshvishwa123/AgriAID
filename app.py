from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import os
import json
import numpy as np
import pandas as pd
import pickle
from PIL import Image
import tensorflow as tf
from werkzeug.utils import secure_filename
import requests
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('models', exist_ok=True)

# Global variables for models
disease_model = None
crop_model = None
class_labels = {}
disease_treatments = {}

def load_models():
    """Load all ML models at startup"""
    global disease_model, crop_model, class_labels, disease_treatments
    
    try:
        # Load disease detection model
        if os.path.exists('models/plant_disease_model.h5'):
            disease_model = tf.keras.models.load_model('models/plant_disease_model.h5')
            print("✅ Disease detection model loaded")
        else:
            print("⚠️ Disease model not found. Please train the model first.")
        
        # Load class labels
        if os.path.exists('models/class_labels.json'):
            with open('models/class_labels.json', 'r') as f:
                class_labels = json.load(f)
            print("✅ Class labels loaded")
            
        # Load crop recommendation model (we'll create this)
        if os.path.exists('models/crop_model.pkl'):
            with open('models/crop_model.pkl', 'rb') as f:
                crop_model = pickle.load(f)
            print("✅ Crop recommendation model loaded")
        else:
            print("⚠️ Crop model not found. Training new model...")
            train_crop_model()
            
        # Load disease treatments
        load_disease_treatments()
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")

def load_disease_treatments():
    """Load disease treatment information"""
    global disease_treatments
    
    # This is a simplified treatment database
    # In production, you'd load this from a proper database
    disease_treatments = {
        "Tomato___Late_blight": {
            "treatment": "Apply copper-based fungicides, improve air circulation, avoid overhead watering",
            "prevention": "Use resistant varieties, proper spacing, drip irrigation"
        },
        "Tomato___Early_blight": {
            "treatment": "Apply fungicides containing chlorothalonil, remove infected leaves",
            "prevention": "Crop rotation, proper mulching, adequate plant spacing"
        },
        "Potato___Late_blight": {
            "treatment": "Apply metalaxyl or copper fungicides, destroy infected plants",
            "prevention": "Use certified seed potatoes, avoid overhead irrigation"
        },
        "Corn_(maize)___Common_rust": {
            "treatment": "Apply triazole fungicides, remove infected leaves",
            "prevention": "Plant resistant hybrids, proper field sanitation"
        },
        "Apple___Apple_scab": {
            "treatment": "Apply fungicides before infection, prune for air circulation",
            "prevention": "Choose resistant varieties, fall cleanup of leaves"
        }
        # Add more treatments as needed based on your classes
    }

def train_crop_model():
    """Train crop recommendation model"""
    global crop_model
    
    try:
        # Check if CSV exists
        if not os.path.exists('datasets/crop_recommendation.csv'):
            print("❌ crop_recommendation.csv not found in datasets folder")
            return False
            
        # Load data
        df = pd.read_csv('datasets/crop_recommendation.csv')
        print(f"📊 Loaded crop dataset with {len(df)} samples")
        
        # Prepare features and target
        feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        X = df[feature_cols]
        y = df['label']  # assuming 'label' is the crop name column
        
        # Train Random Forest model
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        crop_model = RandomForestClassifier(n_estimators=100, random_state=42)
        crop_model.fit(X_train, y_train)
        
        # Test accuracy
        y_pred = crop_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"✅ Crop model trained with accuracy: {accuracy:.3f}")
        
        # Save model
        with open('models/crop_model.pkl', 'wb') as f:
            pickle.dump(crop_model, f)
        print("💾 Crop model saved")
        
        return True
        
    except Exception as e:
        print(f"❌ Error training crop model: {e}")
        return False

def predict_disease(image_path):
    """Predict disease from image"""
    if disease_model is None:
        return None, "Disease model not loaded"
    
    try:
        # Load and preprocess image
        with Image.open(image_path) as img:
            img = img.convert('RGB')
            img = img.resize((224, 224))
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        prediction = disease_model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(prediction[0])
        confidence = float(prediction[0][predicted_class_idx])
        
        # Get class name
        predicted_class = class_labels.get(str(predicted_class_idx), "Unknown")
        
        return {
            'disease': predicted_class,
            'confidence': confidence,
            'treatment': disease_treatments.get(predicted_class, {
                'treatment': 'Consult with agricultural expert',
                'prevention': 'Follow good agricultural practices'
            })
        }, None
        
    except Exception as e:
        return None, f"Error predicting disease: {str(e)}"

def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    """Predict best crop based on soil and weather conditions"""
    if crop_model is None:
        return None, "Crop model not loaded"
    
    try:
        # Prepare input data
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        
        # Make prediction
        predicted_crop = crop_model.predict(input_data)[0]
        
        # Get prediction probabilities for top 3 crops
        probabilities = crop_model.predict_proba(input_data)[0]
        top_3_indices = np.argsort(probabilities)[-3:][::-1]
        
        recommendations = []
        for idx in top_3_indices:
            crop_name = crop_model.classes_[idx]
            probability = probabilities[idx]
            recommendations.append({
                'crop': crop_name,
                'suitability': f"{probability*100:.1f}%"
            })
        
        return recommendations, None
        
    except Exception as e:
        return None, f"Error predicting crop: {str(e)}"

def get_market_prices():
    """Fetch real-time market prices"""
    try:
        # Using a mock API for demonstration
        # In production, use actual mandi APIs like AgMarkNet
        
        # Mock data - replace with real API calls
        market_data = [
            {
                'crop': 'Wheat',
                'market': 'Delhi Mandi',
                'price': 2150,
                'unit': 'per quintal',
                'date': datetime.now().strftime('%Y-%m-%d')
            },
            {
                'crop': 'Rice',
                'market': 'Mumbai Mandi',
                'price': 2800,
                'unit': 'per quintal',
                'date': datetime.now().strftime('%Y-%m-%d')
            },
            {
                'crop': 'Tomato',
                'market': 'Bangalore Mandi',
                'price': 45,
                'unit': 'per kg',
                'date': datetime.now().strftime('%Y-%m-%d')
            },
            {
                'crop': 'Onion',
                'market': 'Pune Mandi',
                'price': 35,
                'unit': 'per kg',
                'date': datetime.now().strftime('%Y-%m-%d')
            },
            {
                'crop': 'Potato',
                'market': 'Agra Mandi',
                'price': 25,
                'unit': 'per kg',
                'date': datetime.now().strftime('%Y-%m-%d')
            }
        ]
        
        return market_data, None
        
    except Exception as e:
        return None, f"Error fetching market prices: {str(e)}"

# Routes
@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/disease-detection')
def disease_detection():
    """Disease detection page"""
    return render_template('disease_detection.html')

@app.route('/crop-recommendation')
def crop_recommendation():
    """Crop recommendation page"""
    return render_template('crop_recommendation.html')

@app.route('/market-prices')
def market_prices():
    """Market prices page"""
    return render_template('market_prices.html')

@app.route('/upload-image', methods=['POST'])
def upload_image():
    """Handle image upload for disease detection"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Predict disease
            result, error = predict_disease(filepath)
            
            # Clean up uploaded file
            os.remove(filepath)
            
            if error:
                return jsonify({'error': error})
            
            return jsonify(result)
            
        except Exception as e:
            return jsonify({'error': f'Error processing image: {str(e)}'})
    
    return jsonify({'error': 'Invalid file format. Please upload PNG, JPG, or JPEG files.'})

@app.route('/recommend-crop', methods=['POST'])
def recommend_crop():
    """Handle crop recommendation request"""
    try:
        data = request.get_json()
        
        # Extract parameters
        N = float(data.get('N', 0))
        P = float(data.get('P', 0))
        K = float(data.get('K', 0))
        temperature = float(data.get('temperature', 0))
        humidity = float(data.get('humidity', 0))
        ph = float(data.get('ph', 0))
        rainfall = float(data.get('rainfall', 0))
        
        # Get recommendations
        recommendations, error = predict_crop(N, P, K, temperature, humidity, ph, rainfall)
        
        if error:
            return jsonify({'error': error})
        
        return jsonify({'recommendations': recommendations})
        
    except Exception as e:
        return jsonify({'error': f'Error getting recommendations: {str(e)}'})

@app.route('/get-market-prices')
def get_market_prices_api():
    """API endpoint for market prices"""
    prices, error = get_market_prices()
    
    if error:
        return jsonify({'error': error})
    
    return jsonify({'prices': prices})

if __name__ == '__main__':
    print("🌱 Starting AgriAid Application...")
    load_models()
    print("🚀 AgriAid is ready!")
    app.run(debug=True, host='0.0.0.0', port=5000)