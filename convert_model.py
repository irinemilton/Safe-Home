import tensorflow as tf
import os
import json

# Path setup
model_path = 'models/bd3_model_modern.keras'
output_path = 'models/bd3_model_modern_fixed.keras'

def repair_model():
    print(f"[INFO] Repairing {model_path}...")

    # 1. Custom object to handle the 'quantization_config' error
    # We create a dummy class that accepts any arguments but does nothing
    class FixedDense(tf.keras.layers.Dense):
        def __init__(self, *args, **kwargs):
            kwargs.pop('quantization_config', None)
            super().__init__(*args, **kwargs)

    custom_objects = {"Dense": FixedDense}

    try:
        # 2. Load the model WITHOUT compiling to avoid metadata checks
        # We use compile=False to ignore the broken optimizer/loss config
        model = tf.keras.models.load_model(
            model_path, 
            custom_objects=custom_objects, 
            compile=False
        )
        
        print("[OK] Model architecture and weights loaded.")

        # 3. Re-save the model
        # Saving it again in your current environment will write a clean 
        # config file compatible with your local Keras version.
        model.save(output_path)
        
        print("-" * 50)
        print(f"SUCCESS! Cleaned model saved to: {output_path}")
        print("Now update app.py to use this new file.")
        print("-" * 50)
        model.summary()

    except Exception as e:
        print(f"[ERROR] Repair failed: {e}")

if __name__ == "__main__":
    repair_model()