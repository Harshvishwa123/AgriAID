import tensorflow as tf
import os
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")

def load_and_preprocess_data():
    """Load images manually to avoid ImageDataGenerator issues"""
    dataset_path = r"C:\Users\hvish\Desktop\AGRIAID\PlantVillage"
    
    if not os.path.exists(dataset_path):
        raise Exception(f"Dataset path not found: {dataset_path}")
    
    print("📁 Loading dataset...")
    
    # Get all class directories
    class_dirs = [d for d in os.listdir(dataset_path) 
                  if os.path.isdir(os.path.join(dataset_path, d)) and not d.startswith('.')]
    class_names = sorted(class_dirs)
    
    print(f"Found {len(class_names)} classes:")
    for i, class_name in enumerate(class_names):
        class_path = os.path.join(dataset_path, class_name)
        img_files = [f for f in os.listdir(class_path) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"  {i}: {class_name} ({len(img_files)} images)")
    
    # Load images
    images = []
    labels = []
    
    for class_idx, class_name in enumerate(class_names):
        class_path = os.path.join(dataset_path, class_name)
        img_files = [f for f in os.listdir(class_path) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Limit to 300 images per class for faster training (adjust as needed)
        img_files = img_files[:300]
        
        print(f"Loading {len(img_files)} images from {class_name}...")
        
        for img_file in img_files:
            try:
                img_path = os.path.join(class_path, img_file)
                
                # Load and preprocess image
                with Image.open(img_path) as img:
                    img = img.convert('RGB')
                    img = img.resize((224, 224))
                    img_array = np.array(img, dtype=np.float32) / 255.0
                    
                    images.append(img_array)
                    labels.append(class_idx)
                    
            except Exception as e:
                print(f"Skipping {img_file}: {e}")
                continue
    
    print(f"✅ Loaded {len(images)} images total")
    
    # Convert to numpy arrays
    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    
    # Convert labels to categorical
    y_categorical = tf.keras.utils.to_categorical(y, num_classes=len(class_names))
    
    return X, y_categorical, class_names

def create_model(num_classes):
    """Create a simple but effective CNN model"""
    model = tf.keras.Sequential([
        # Input layer
        tf.keras.layers.Input(shape=(224, 224, 3)),
        
        # First convolutional block
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        # Second convolutional block
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        # Third convolutional block
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        # Fourth convolutional block
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        # Flatten and dense layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def train_disease_model():
    """Main training function"""
    print("🌱 AgriAid Plant Disease Detection Model Training")
    print("=" * 60)
    
    try:
        # Load and preprocess data
        X, y, class_names = load_and_preprocess_data()
        print(f"Data shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        
        # Split data into train and validation
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, 
            stratify=np.argmax(y, axis=1)
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        
        # Create model
        model = create_model(len(class_names))
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("\n📋 Model Architecture:")
        model.summary()
        
        # Create callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train model
        print("\n🚀 Starting training...")
        print("This will take some time. Grab a coffee! ☕")
        
        history = model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=20,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Create models directory
        os.makedirs('models', exist_ok=True)
        
        # Save model
        model.save('models/plant_disease_model.h5')
        print("💾 Model saved as 'models/plant_disease_model.h5'")
        
        # Save class labels
        class_labels = {str(i): name for i, name in enumerate(class_names)}
        with open('models/class_labels.json', 'w') as f:
            json.dump(class_labels, f, indent=2)
        print("💾 Class labels saved as 'models/class_labels.json'")
        
        # Save training history
        history_dict = {
            'accuracy': [float(x) for x in history.history['accuracy']],
            'val_accuracy': [float(x) for x in history.history['val_accuracy']],
            'loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']]
        }
        
        with open('models/training_history.json', 'w') as f:
            json.dump(history_dict, f, indent=2)
        
        # Print final results
        best_val_acc = max(history.history['val_accuracy'])
        final_val_acc = history.history['val_accuracy'][-1]
        
        print("\n" + "=" * 60)
        print("🎯 TRAINING COMPLETED!")
        print(f"✅ Best validation accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
        print(f"✅ Final validation accuracy: {final_val_acc:.4f} ({final_val_acc*100:.2f}%)")
        print("=" * 60)
        
        # Test model with random sample
        print("\n🧪 Testing model with validation sample...")
        test_idx = np.random.randint(0, len(X_val))
        test_image = X_val[test_idx:test_idx+1]
        test_label = y_val[test_idx:test_idx+1]
        
        prediction = model.predict(test_image, verbose=0)
        predicted_class = np.argmax(prediction[0])
        actual_class = np.argmax(test_label[0])
        confidence = prediction[0][predicted_class]
        
        print(f"Predicted: {class_names[predicted_class]} (confidence: {confidence:.3f})")
        print(f"Actual: {class_names[actual_class]}")
        print(f"Correct: {'✅' if predicted_class == actual_class else '❌'}")
        
        return model, class_labels, history
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_saved_model():
    """Test the saved model"""
    try:
        print("\n🔍 Testing saved model...")
        
        # Load model
        model = tf.keras.models.load_model('models/plant_disease_model.h5')
        
        # Load class labels
        with open('models/class_labels.json', 'r') as f:
            class_labels = json.load(f)
        
        print("✅ Model loaded successfully!")
        print(f"Model input shape: {model.input_shape}")
        print(f"Number of classes: {len(class_labels)}")
        
        # Test with dummy data
        dummy_image = np.random.rand(1, 224, 224, 3).astype(np.float32)
        prediction = model.predict(dummy_image, verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class]
        
        print(f"✅ Test prediction successful!")
        print(f"Predicted class: {class_labels[str(predicted_class)]}")
        print(f"Confidence: {confidence:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing saved model: {e}")
        return False

if __name__ == "__main__":
    # Train the model
    result = train_disease_model()
    
    if result is not None:
        print("\n✅ Training successful!")
        
        # Test the saved model
        if test_saved_model():
            print("🎉 Model is ready for deployment!")
        else:
            print("⚠️ Model training completed but testing failed")
    else:
        print("❌ Training failed!")
        print("\nTroubleshooting tips:")
        print("1. Make sure your dataset path is correct")
        print("2. Check if you have enough disk space")
        print("3. Ensure images are not corrupted")