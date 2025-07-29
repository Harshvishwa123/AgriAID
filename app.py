import os
import json
import numpy as np
from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
MODEL_PATH = 'models/plant_disease_model.h5'
LABELS_PATH = 'models/class_labels.json'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('models', exist_ok=True)

# Load model and class labels
try:
    model = load_model(MODEL_PATH)
    with open(LABELS_PATH, 'r') as f:
        class_labels = json.load(f)
    print("✅ Model and labels loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    class_labels = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_path):
    """Preprocess image for model prediction"""
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def get_disease_info(disease_name):
    """Get detailed information about the disease"""
    disease_info = {
        'Pepper__bell___Bacterial_spot': {
            'description': 'Bacterial spot causes dark, greasy spots on leaves and fruits.',
            'symptoms': ['Dark spots on leaves', 'Yellowing around spots', 'Fruit lesions'],
            'treatment': ['Remove affected plants', 'Use copper-based fungicides', 'Improve air circulation'],
            'prevention': ['Avoid overhead watering', 'Rotate crops', 'Use disease-resistant varieties']
        },
        'Pepper__bell___healthy': {
            'description': 'Your pepper plant appears to be healthy!',
            'symptoms': ['Green, vibrant leaves', 'No visible disease signs'],
            'treatment': ['Continue current care routine'],
            'prevention': ['Regular watering', 'Proper fertilization', 'Monitor for pests']
        },
        'Potato___Early_blight': {
            'description': 'Early blight causes brown spots with concentric rings on leaves.',
            'symptoms': ['Brown spots with rings', 'Yellow halos around spots', 'Leaf drop'],
            'treatment': ['Apply fungicides', 'Remove affected foliage', 'Improve drainage'],
            'prevention': ['Crop rotation', 'Avoid overhead watering', 'Mulching']
        },
        'Potato___Late_blight': {
            'description': 'Late blight causes water-soaked lesions that turn brown.',
            'symptoms': ['Water-soaked spots', 'White fuzzy growth', 'Rapid spread'],
            'treatment': ['Emergency fungicide application', 'Remove infected plants', 'Destroy affected tubers'],
            'prevention': ['Choose resistant varieties', 'Ensure good air circulation', 'Monitor weather conditions']
        },
        'Potato___healthy': {
            'description': 'Your potato plant looks healthy!',
            'symptoms': ['Lush green foliage', 'No disease symptoms'],
            'treatment': ['Maintain current care'],
            'prevention': ['Regular monitoring', 'Proper nutrition', 'Adequate watering']
        },
        'Tomato_Bacterial_spot': {
            'description': 'Bacterial spot causes small, dark spots on leaves and fruits.',
            'symptoms': ['Small dark spots', 'Yellowing leaves', 'Fruit cracking'],
            'treatment': ['Copper sprays', 'Remove affected plants', 'Sanitize tools'],
            'prevention': ['Drip irrigation', 'Crop rotation', 'Certified seeds']
        },
        'Tomato_Early_blight': {
            'description': 'Early blight forms concentric ring patterns on lower leaves.',
            'symptoms': ['Bull\'s eye spots', 'Yellowing leaves', 'Defoliation'],
            'treatment': ['Fungicide applications', 'Remove lower leaves', 'Mulching'],
            'prevention': ['Proper spacing', 'Avoid wetting leaves', 'Resistant varieties']
        },
        'Tomato_Late_blight': {
            'description': 'Late blight causes rapid leaf and fruit decay.',
            'symptoms': ['Water-soaked lesions', 'White mold', 'Fruit rot'],
            'treatment': ['Immediate fungicide treatment', 'Remove infected plants', 'Improve ventilation'],
            'prevention': ['Weather monitoring', 'Preventive spraying', 'Greenhouse growing']
        },
        'Tomato_Leaf_Mold': {
            'description': 'Leaf mold causes yellow spots that develop fuzzy growth underneath.',
            'symptoms': ['Yellow spots on top', 'Fuzzy mold underneath', 'Leaf curling'],
            'treatment': ['Increase ventilation', 'Reduce humidity', 'Fungicide spray'],
            'prevention': ['Proper spacing', 'Greenhouse ventilation', 'Avoid overhead watering']
        },
        'Tomato_Septoria_leaf_spot': {
            'description': 'Septoria leaf spot creates small spots with dark borders.',
            'symptoms': ['Small circular spots', 'Dark borders', 'Yellowing leaves'],
            'treatment': ['Fungicide treatment', 'Remove affected leaves', 'Mulching'],
            'prevention': ['Crop rotation', 'Avoid splashing water', 'Clean garden debris']
        },
        'Tomato_Spider_mites_Two_spotted_spider_mite': {
            'description': 'Spider mites cause stippling and webbing on leaves.',
            'symptoms': ['Fine webbing', 'Stippled leaves', 'Yellow spots'],
            'treatment': ['Miticide application', 'Increase humidity', 'Beneficial insects'],
            'prevention': ['Regular monitoring', 'Avoid over-fertilizing', 'Maintain humidity']
        },
        'Tomato__Target_Spot': {
            'description': 'Target spot creates concentric ring patterns on leaves.',
            'symptoms': ['Concentric rings', 'Brown centers', 'Yellowing'],
            'treatment': ['Fungicide sprays', 'Remove debris', 'Improve air flow'],
            'prevention': ['Crop rotation', 'Mulching', 'Resistant varieties']
        },
        'Tomato__Tomato_YellowLeaf__Curl_Virus': {
            'description': 'Yellow leaf curl virus causes upward curling of leaves.',
            'symptoms': ['Upward leaf curling', 'Yellowing', 'Stunted growth'],
            'treatment': ['Remove infected plants', 'Control whiteflies', 'Use row covers'],
            'prevention': ['Resistant varieties', 'Whitefly management', 'Remove weeds']
        },
        'Tomato__Tomato_mosaic_virus': {
            'description': 'Mosaic virus causes mottled patterns on leaves.',
            'symptoms': ['Mottled leaves', 'Distorted growth', 'Reduced yield'],
            'treatment': ['Remove infected plants', 'Sanitize tools', 'Control aphids'],
            'prevention': ['Use certified seeds', 'Control vectors', 'Sanitation']
        },
        'Tomato_healthy': {
            'description': 'Your tomato plant is healthy!',
            'symptoms': ['Vibrant green leaves', 'Normal growth'],
            'treatment': ['Continue current care'],
            'prevention': ['Regular monitoring', 'Balanced nutrition', 'Proper watering']
        }
    }
    
    return disease_info.get(disease_name, {
        'description': 'Disease information not available.',
        'symptoms': ['Consult agricultural expert'],
        'treatment': ['Professional diagnosis recommended'],
        'prevention': ['General plant care practices']
    })

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Make prediction
        if model is not None:
            img_array = preprocess_image(filepath)
            if img_array is not None:
                predictions = model.predict(img_array)
                predicted_class_idx = np.argmax(predictions[0])
                confidence = float(predictions[0][predicted_class_idx])
                predicted_class = class_labels[str(predicted_class_idx)]
                
                # Get disease information
                disease_info = get_disease_info(predicted_class)
                
                return jsonify({
                    'success': True,
                    'prediction': predicted_class,
                    'confidence': f"{confidence:.2%}",
                    'image_url': f"/static/uploads/{filename}",
                    'disease_info': disease_info
                })
            else:
                return jsonify({'error': 'Error processing image'})
        else:
            return jsonify({'error': 'Model not loaded'})
    
    return jsonify({'error': 'Invalid file type'})

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for mobile app integration"""
    return upload_file()

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'version': '1.0.0'
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)