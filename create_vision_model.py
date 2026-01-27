"""
Create a Compatible Vision Model for SafeHome

This script creates a MobileNetV2-based model compatible with TensorFlow 2.15.1
that can be used for concrete defect detection.

Options:
1. Create a new untrained model (for testing the upload feature)
2. If you have training data, you can modify this to train the model

Run this script to generate: models/concrete_cancer_model.keras
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Define the 8 classes for concrete defect detection
CLASS_NAMES = ['Wall', 'algae', 'major_crack', 'minor_crack', 'normal', 'peeling', 'spalling', 'stain']
NUM_CLASSES = len(CLASS_NAMES)
IMG_SIZE = 224

def create_model():
    """
    Create a MobileNetV2-based model for concrete defect classification
    Compatible with TensorFlow 2.15.1
    """
    print("Creating MobileNetV2 model...")
    
    # Use MobileNetV2 as base model
    base_model = keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'  # Use pre-trained weights
    )
    
    # Freeze the base model
    base_model.trainable = False
    
    # Create the model
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Pre-processing
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)
    
    # Base model
    x = base_model(x, training=False)
    
    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"Model created with {NUM_CLASSES} output classes")
    print(f"Classes: {CLASS_NAMES}")
    
    return model

def save_model(model, filepath='models/concrete_cancer_model.keras'):
    """Save the model in Keras format compatible with TensorFlow 2.15.1"""
    print(f"\nSaving model to {filepath}...")
    model.save(filepath)
    print("[OK] Model saved successfully!")
    print(f"\nModel summary:")
    model.summary()

def test_model(model):
    """Test the model with a random image"""
    print("\nTesting model with random image...")
    
    # Create a random test image
    test_image = np.random.rand(1, IMG_SIZE, IMG_SIZE, 3).astype(np.float32)
    
    # Make prediction
    predictions = model.predict(test_image, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100
    
    print(f"[OK] Test prediction: {CLASS_NAMES[predicted_class]} ({confidence:.2f}% confidence)")
    print("[OK] Model is working correctly!")

if __name__ == "__main__":
    print("=" * 60)
    print("SafeHome - Vision Model Generator")
    print("=" * 60)
    print("\nThis script creates a MobileNetV2 model compatible with")
    print("TensorFlow 2.15.1 for concrete defect detection.")
    print("\nNOTE: This model is NOT trained. It will give random predictions.")
    print("To get accurate results, you need to train it with your dataset.")
    print("=" * 60)
    
    # Create the model
    model = create_model()
    
    # Test it
    test_model(model)
    
    # Save it
    save_model(model)
    
    print("\n" + "=" * 60)
    print("[DONE] You can now:")
    print("1. Restart your Flask app: python app.py")
    print("2. Upload images for defect detection")
    print("\nTo train this model with your data, see: TRAIN_MODEL.md")
    print("=" * 60)
