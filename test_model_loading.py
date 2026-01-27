"""
Test script to verify the vision model can be loaded
"""
import tensorflow as tf
import os

model_path = 'models/concrete_cancer_model.keras'

print(f"Checking if model file exists: {model_path}")
print(f"File exists: {os.path.exists(model_path)}")
print(f"File size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")

print("\nAttempting to load model...")
try:
    model = tf.keras.models.load_model(model_path)
    print("[SUCCESS] Model loaded successfully!")
    print(f"\nModel summary:")
    model.summary()
    
    # Test prediction
    import numpy as np
    test_img = np.random.rand(1, 224, 224, 3).astype(np.float32)
    pred = model.predict(test_img, verbose=0)
    print(f"\n[SUCCESS] Test prediction works! Output shape: {pred.shape}")
    
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    import traceback
    traceback.print_exc()
