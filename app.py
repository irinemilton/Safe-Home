import os
import numpy as np
import pickle
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import io

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("WARNING: TensorFlow not installed. Vision model will not be available.")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- GLOBAL MODEL VARIABLES ---
sensor_model = None
vision_model = None
CLASS_NAMES = ['Wall', 'algae', 'major_crack', 'minor_crack', 'normal', 'peeling', 'spalling', 'stain']

# --- LOAD MODELS ON STARTUP ---
def load_models():
    """Load both AI models with error handling"""
    global sensor_model, vision_model
    
    print("Loading AI Models...")
    
    # Load Sensor Model (Random Forest)
    sensor_path = 'models/sensor_model.pkl'
    if os.path.exists(sensor_path):
        try:
            with open(sensor_path, 'rb') as f:
                sensor_model = pickle.load(f)
            print("[OK] Sensor Model loaded successfully")
        except Exception as e:
            print(f"[ERROR] Error loading sensor model: {e}")
    else:
        print(f"[WARNING] Sensor model not found at {sensor_path}")
    
    # Load Vision Model (TensorFlow/Keras)
    if TF_AVAILABLE:
        vision_path_keras = 'models/concrete_cancer_model.keras'
        vision_path_h5 = 'models/concrete_cancer_model.h5'
        
        print(f"[DEBUG] Checking for vision model at: {vision_path_keras}")
        print(f"[DEBUG] File exists: {os.path.exists(vision_path_keras)}")
        
        if os.path.exists(vision_path_keras):
            try:
                print(f"[DEBUG] Attempting to load vision model...")
                vision_model = tf.keras.models.load_model(vision_path_keras)
                print(f"[OK] Vision Model (.keras) loaded successfully")
                print(f"[DEBUG] Model type: {type(vision_model)}")
            except Exception as e:
                print(f"[ERROR] Error loading vision model (.keras): {e}")
                import traceback
                traceback.print_exc()
        elif os.path.exists(vision_path_h5):
            try:
                vision_model = tf.keras.models.load_model(vision_path_h5)
                print("[OK] Vision Model (.h5) loaded successfully")
            except Exception as e:
                print(f"[ERROR] Error loading vision model (.h5): {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"[WARNING] Vision model not found at {vision_path_keras} or {vision_path_h5}")
    
    if sensor_model or vision_model:
        print("[READY] Server ready!")
    else:
        print("[WARNING] No models loaded. Please add model files to the 'models/' directory.")

# Load models when app starts
load_models()

# --- ROUTES ---

@app.route('/')
def home():
    """Render the main dashboard"""
    return render_template('index.html', 
                         sensor_available=sensor_model is not None,
                         vision_available=vision_model is not None)

@app.route('/predict_sensor', methods=['POST'])
def predict_sensor():
    """Handle sensor data prediction"""
    if sensor_model is None:
        return render_template('index.html', 
                             sensor_result="❌ Sensor model not loaded. Please add sensor_model.pkl to models/ folder.",
                             sensor_color="red",
                             sensor_available=False,
                             vision_available=vision_model is not None)
    
    try:
        # Get form data
        temp = float(request.form.get('temperature', 0))
        moisture = float(request.form.get('moisture', 0))
        
        # Validate inputs
        if temp < -50 or temp > 100:
            raise ValueError("Temperature must be between -50°C and 100°C")
        if moisture < 0 or moisture > 100:
            raise ValueError("Moisture must be between 0% and 100%")
        
        # Predict using the model
        input_data = np.array([[temp, moisture]])
        prediction = sensor_model.predict(input_data)
        signal_strength = float(prediction[0])
        
        # Determine risk level
        if signal_strength < -140:
            msg = f"🚨 HIGH RISK: Signal Strength {signal_strength:.2f} mV (Possible Internal Corrosion)"
            color = "red"
        elif signal_strength < -100:
            msg = f"⚠️ WARNING: Signal Strength {signal_strength:.2f} mV (Monitor Closely)"
            color = "orange"
        else:
            msg = f"✅ SAFE: Signal Strength {signal_strength:.2f} mV (Stable Conditions)"
            color = "green"
        
        return render_template('index.html', 
                             sensor_result=msg, 
                             sensor_color=color,
                             sensor_available=True,
                             vision_available=vision_model is not None)
    
    except ValueError as ve:
        return render_template('index.html', 
                             sensor_result=f"❌ Invalid Input: {str(ve)}", 
                             sensor_color="red",
                             sensor_available=sensor_model is not None,
                             vision_available=vision_model is not None)
    except Exception as e:
        return render_template('index.html', 
                             sensor_result=f"❌ Error: {str(e)}", 
                             sensor_color="red",
                             sensor_available=sensor_model is not None,
                             vision_available=vision_model is not None)

@app.route('/predict_image', methods=['POST'])
def predict_image():
    """Handle image upload and defect detection"""
    
    try:
        # Check if file was uploaded
        if 'image' not in request.files:
            raise ValueError("No file uploaded")
        
        file = request.files['image']
        if file.filename == '':
            raise ValueError("No file selected")
        
        # Validate file type
        allowed_extensions = {'png', 'jpg', 'jpeg', 'bmp', 'gif'}
        file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
        if file_ext not in allowed_extensions:
            raise ValueError(f"Invalid file type. Allowed: {', '.join(allowed_extensions)}")
        
        # If vision model is not available, return a helpful message
        if vision_model is None:
            msg = "INFO: Image uploaded successfully!\n\nVision model is currently offline. To enable defect detection:\n1. Retrain your model with TensorFlow 2.15.1\n2. Save as 'concrete_cancer_model.keras'\n3. Place in models/ folder\n4. Restart the server\n\nSee VISION_MODEL_FIX.md for details."
            color = "blue"
            return render_template('index.html', 
                                 image_result=msg, 
                                 image_color=color,
                                 sensor_available=sensor_model is not None,
                                 vision_available=False)
        
        # Process the image
        img = Image.open(file.stream).convert('RGB')
        img_resized = img.resize((224, 224))
        
        # Convert to array and normalize
        img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        # Predict
        predictions = vision_model.predict(img_array, verbose=0)
        score = tf.nn.softmax(predictions[0])
        
        # Get the winner
        class_idx = np.argmax(score)
        prediction = CLASS_NAMES[class_idx]
        confidence = 100 * np.max(score)
        
        # Determine severity and message
        if prediction in ['spalling', 'peeling']:
            msg = f"🚨 CRITICAL ALERT: {prediction.upper()} Detected ({confidence:.1f}% confidence)\n⚠️ Concrete Cancer - Immediate structural audit required!"
            color = "red"
        elif prediction == 'major_crack':
            msg = f"⚠️ WARNING: {prediction.upper().replace('_', ' ')} Detected ({confidence:.1f}% confidence)\n⚠️ Structural integrity compromised - Professional inspection recommended."
            color = "orange"
        elif prediction == 'minor_crack':
            msg = f"ℹ️ NOTICE: {prediction.upper().replace('_', ' ')} Detected ({confidence:.1f}% confidence)\nMonitor for progression."
            color = "yellow"
        elif prediction == 'normal':
            msg = f"✅ SAFE: Surface appears Healthy ({confidence:.1f}% confidence)\nNo defects detected."
            color = "green"
        else:
            msg = f"ℹ️ INFO: {prediction.upper()} Detected ({confidence:.1f}% confidence)\nLikely cosmetic - no structural concern."
            color = "blue"
        
        return render_template('index.html', 
                             image_result=msg, 
                             image_color=color,
                             sensor_available=sensor_model is not None,
                             vision_available=True)
    
    except ValueError as ve:
        return render_template('index.html', 
                             image_result=f"❌ {str(ve)}", 
                             image_color="red",
                             sensor_available=sensor_model is not None,
                             vision_available=vision_model is not None)
    except Exception as e:
        return render_template('index.html', 
                             image_result=f"❌ Error processing image: {str(e)}", 
                             image_color="red",
                             sensor_available=sensor_model is not None,
                             vision_available=vision_model is not None)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
