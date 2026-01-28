import os
import numpy as np
import pickle
import random
import json
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from PIL import Image
import io
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("WARNING: TensorFlow not installed. Vision model will not be available.")

# Try to import Florence-2
try:
    from florence_detector import get_florence_detector
    FLORENCE_AVAILABLE = True
    print("[INFO] Florence-2 detector available")
except ImportError as e:
    FLORENCE_AVAILABLE = False
    print(f"[WARNING] Florence-2 not available: {e}")
    # print("[INFO] Install with: pip install torch transformers")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Configure Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        GEMINI_AVAILABLE = True
        print("[INFO] Gemini Vision API configured successfully")
    except Exception as e:
        GEMINI_AVAILABLE = False
        print(f"[WARNING] Failed to configure Gemini API: {e}")
else:
    GEMINI_AVAILABLE = False
    print("[WARNING] GEMINI_API_KEY not found in environment variables")

# --- GLOBAL MODEL VARIABLES ---
sensor_model = None
vision_model = None
# Ensure these match the classes your bd3_model_modern.keras was trained on
# Model outputs 7 classes (confirmed via model.output_shape)
CLASS_NAMES = ['algae', 'major_crack', 'minor_crack', 'normal', 'peeling', 'spalling', 'stain']

# --- LOAD MODELS ON STARTUP ---
def load_models():
    """Load both AI models with error handling"""
    global sensor_model, vision_model
    
    print("Loading AI Models...")
    
    # 1. Load Sensor Model
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
    
    # 2. Load Vision Model (Prioritizing bd3_model_modern_fixed.keras)
    if TF_AVAILABLE:
        vision_path_modern = 'models/bd3_model_modern_fixed.keras'  # Using fixed model
        vision_path_fallback = 'models/bd3_final_trained.keras'
        
        # Check for the modern model first
        if os.path.exists(vision_path_modern):
            try:
                print(f"[INFO] Loading primary vision model: {vision_path_modern}...")
                vision_model = tf.keras.models.load_model(vision_path_modern)
                print(f"[OK] Vision Model loaded successfully!")
                
                # Verify input shape
                input_shape = vision_model.input_shape
                print(f"[INFO] Model Input Shape: {input_shape}")
                
            except Exception as e:
                print(f"[ERROR] Failed to load {vision_path_modern}: {e}")
                import traceback
                traceback.print_exc()

        # Fallback if primary not found
        elif os.path.exists(vision_path_fallback):
            try:
                print(f"[INFO] Loading fallback vision model: {vision_path_fallback}...")
                vision_model = tf.keras.models.load_model(vision_path_fallback)
                print(f"[OK] Fallback Vision Model loaded.")
            except Exception as e:
                print(f"[ERROR] Error loading fallback model: {e}")
        else:
            print(f"[WARNING] No vision model found. Please place 'bd3_model_modern.keras' in the 'models/' folder.")
    
    if sensor_model or vision_model:
        print("[READY] Server ready!")
    else:
        print("[WARNING] No models loaded.")

# Load models when app starts
load_models()

# --- HELPER FUNCTIONS ---

def simulate_sensor_data():
    """Simulate environmental sensor data for demo purposes"""
    temperature = round(random.uniform(15, 35), 1)
    moisture = round(random.uniform(5, 80), 1)
    return temperature, moisture

def get_sensor_prediction(temperature, moisture):
    """Get sensor model prediction for corrosion risk"""
    if sensor_model is None:
        return None, "Sensor model not available"
    
    try:
        input_data = np.array([[temperature, moisture]])
        prediction = sensor_model.predict(input_data)
        signal_strength = float(prediction[0])
        
        if signal_strength < -140:
            return "high", f"High corrosion risk detected (Signal: {signal_strength:.1f} mV)"
        elif signal_strength < -100:
            return "medium", f"Moderate corrosion risk (Signal: {signal_strength:.1f} mV)"
        else:
            return "low", f"Low corrosion risk (Signal: {signal_strength:.1f} mV)"
    except Exception as e:
        return None, f"Sensor error: {str(e)}"

def analyze_with_gemini(image_path):
    """
    Analyze concrete image with Gemini Vision API for defect localization
    Returns dict with problem, confidence, location (bounding box), and description
    """
    print(f"\n[GEMINI] Starting analysis for: {image_path}")
    print(f"[GEMINI] API Available: {GEMINI_AVAILABLE}")
    
    if not GEMINI_AVAILABLE:
        print("[GEMINI] Skipping - API not available")
        return None
    
    try:
        # Load and prepare image
        img = Image.open(image_path)
        
        # Create Gemini model
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Craft prompt for structured output
        prompt = """Analyze this concrete structure image for defects.

Identify any of these defect types:
- major_crack: Large, deep cracks
- minor_crack: Small, hairline cracks
- spalling: Surface deterioration with material loss
- peeling: Paint or surface layer peeling off
- stain: Discoloration or staining
- algae: Green/biological growth
- normal: No defects detected

Return ONLY a JSON object with this exact structure (no markdown, no code blocks):
{
  "problem": "defect_type_from_list_above",
  "confidence": 0.85,
  "location": {
    "x": 0.3,
    "y": 0.4,
    "width": 0.25,
    "height": 0.2
  },
  "description": "Brief description of the defect"
}

Location coordinates should be percentages (0.0 to 1.0) representing:
- x: horizontal position from left edge
- y: vertical position from top edge
- width: box width as fraction of image width
- height: box height as fraction of image height

If no defect or location is uncertain, set all location values to 0."""

        # Generate response
        response = model.generate_content([prompt, img])
        
        # Parse JSON response
        response_text = response.text.strip()
        
        # Remove markdown code blocks if present
        if response_text.startswith('```'):
            response_text = response_text.split('```')[1]
            if response_text.startswith('json'):
                response_text = response_text[4:]
            response_text = response_text.strip()
        
        result = json.loads(response_text)
        
        print(f"\n[GEMINI] Analysis complete:")
        print(f"  Problem: {result.get('problem', 'unknown')}")
        print(f"  Confidence: {result.get('confidence', 0):.2%}")
        print(f"  Location: {result.get('location', {})}\n")
        
        return result
        
    except Exception as e:
        print(f"[ERROR] Gemini analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_vision_prediction(image):
    """Get vision model prediction for surface defects"""
    
    # Priority 1: Use Florence-2 if available
    if FLORENCE_AVAILABLE:
        try:
            detector = get_florence_detector()
            defect_type, confidence, description, bboxes = detector.analyze_image(image)
            
            if defect_type:
                if defect_type in ['spalling', 'peeling', 'major_crack']:
                    severity = "critical"
                elif defect_type == 'minor_crack':
                    severity = "notice"
                elif defect_type == 'stain':
                    severity = "warning"
                elif defect_type == 'normal':
                    severity = "safe"
                else:
                    severity = "info"
                
                # Return bounding boxes and description from Florence-2
                print(f"[Florence-2] Detected: {defect_type}, Confidence: {confidence}%, Description: {description}")
                return defect_type, confidence, severity, bboxes, description
        except Exception as e:
            print(f"[ERROR] Florence-2 error: {e}")
    
    # Priority 2: Use your Keras Model (bd3_model_modern.keras)
    if vision_model is not None:
        try:
            # Preprocessing
            target_size = (224, 224) # Standard for most models, adjust if your model differs
            img_resized = image.resize(target_size)
            img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            # NOTE: Model has built-in Rescaling layer - do NOT normalize manually!
            # The Rescaling layer expects raw pixel values [0-255]
            
            # Predict
            predictions = vision_model.predict(img_array, verbose=0)
            
            # Debug: Print all class probabilities
            print(f"\n[DEBUG] Raw predictions: {predictions[0]}")
            print(f"[DEBUG] Class probabilities:")
            for i, class_name in enumerate(CLASS_NAMES):
                print(f"  {class_name}: {predictions[0][i]:.4f} ({predictions[0][i]*100:.2f}%)")
            
            # Get result
            class_idx = np.argmax(predictions[0])
            prediction = CLASS_NAMES[class_idx]
            confidence = 100 * np.max(predictions[0])
            
            print(f"[DEBUG] Predicted class: {prediction} (index {class_idx}) with confidence {confidence:.2f}%\n")
            
            # Map prediction to severity
            if prediction in ['spalling', 'peeling']:
                severity = "critical"
            elif prediction == 'major_crack':
                severity = "warning"
            elif prediction == 'minor_crack':
                severity = "notice"
            elif prediction == 'normal':
                severity = "safe"
            else:
                severity = "info"
            
            # Keras doesn't provide bounding boxes or detailed description
            return prediction, confidence, severity, None, None
        except Exception as e:
            return None, None, f"Vision error: {str(e)}", None, None
    
    return None, None, "No vision model available", None, None

def combine_predictions(sensor_risk, vision_severity, vision_defect, vision_confidence, temp, moisture):
    """Combine sensor and vision predictions safely"""
    
    # Calculate Risk Scores for Chart based on Model Predictions
    
    # 1. Visual Defect Risk (Dynamic based on Model Confidence)
    # Weight multipliers: Critical defects contribute 100% of their confidence to risk, minor ones less.
    severity_weights = {
        'critical': 1.0,  # Spalling/Major Cracks = Direct Risk
        'warning': 0.6,   # Minor Cracks/Peeling = Moderate Risk
        'notice': 0.2,    # Stains/Algae = Low Risk
        'safe': 0.0,      # Normal = No Risk
        'info': 0.0
    }
    
    # visual_risk = Confidence * Severity Weight
    # Example: 90% confidence Spalling -> 90 * 1.0 = 90% Risk
    # Example: 90% confidence Stain    -> 90 * 0.2 = 18% Risk
    weight = severity_weights.get(vision_severity, 0.0)
    visual_risk_score = round(vision_confidence * weight, 1)

    # 2. Environmental Risk (Sensor based)
    # Fixed contributions based on sensor risk levels
    env_risk_score = 0
    if sensor_risk == 'high': env_risk_score = 25
    elif sensor_risk == 'medium': env_risk_score = 10
    elif sensor_risk == 'low': env_risk_score = 0
    
    # 3. Calculate Structural Integrity (Remaining percentage)
    # Integrity is what's left after subtracting risks
    total_risk = visual_risk_score + env_risk_score
    structural_integrity = max(0, round(100 - total_risk, 1))

    risk_analysis = {
        'visual': visual_risk_score,
        'environmental': env_risk_score,
        'integrity': structural_integrity
    }

    # --- CRITICAL FIX FOR CRASH ---
    # Handle case where vision model returned None (failed or not loaded)
    if vision_defect is None:
        if sensor_risk is None:
            return {
                'severity': 'info',
                'title': 'Analysis Failed',
                'message': f'Both models unavailable. Error: {vision_severity}',
                'risk_analysis': {'visual': 0, 'environmental': 0, 'integrity': 100}
            }
        # Sensor worked, Vision failed
        return {
            'severity': 'warning',
            'title': 'Partial Analysis (Sensor Only)',
            'message': f'Vision unavailable: {vision_severity}\n\nEnvironment: {sensor_risk.upper()} risk (Temp: {temp}°C, Moisture: {moisture}%).',
            'risk_analysis': {'visual': 0, 'environmental': env_risk_score, 'integrity': 100 - env_risk_score}
        }
    # -----------------------------

    # Vision worked, Sensor failed
    if sensor_risk is None:
        titles = {
            "critical": f"CRITICAL: {vision_defect.upper()}",
            "warning": f"WARNING: {vision_defect.replace('_', ' ').upper()}",
            "notice": f"NOTICE: {vision_defect.replace('_', ' ').title()}",
            "safe": "SAFE: Surface Healthy",
            "info": f"INFO: {vision_defect.title()}"
        }
        return {
            'severity': vision_severity,
            'title': f"{titles.get(vision_severity, 'INFO')} ({vision_confidence:.1f}%)",
            'message': f"Defect detected: {vision_defect}. Sensor data unavailable.",
            'risk_analysis': {'visual': visual_risk_score, 'environmental': 0, 'integrity': 100 - visual_risk_score}
        }

    # Both models worked
    if sensor_risk == "high" or vision_severity == "critical":
        return {
            'severity': 'critical',
            'title': 'CRITICAL ALERT: Multiple Issues',
            'message': f'Surface: {vision_defect.upper()} ({vision_confidence:.1f}%)\nEnvironment: {sensor_risk.upper()} risk (Temp: {temp}, Moist: {moisture})',
            'risk_analysis': risk_analysis
        }
    
    if sensor_risk == "medium" or vision_severity == "warning":
        return {
            'severity': 'warning',
            'title': 'WARNING: Structural Concerns',
            'message': f'Surface: {vision_defect} ({vision_confidence:.1f}%)\nEnvironment: {sensor_risk} risk.',
            'risk_analysis': risk_analysis
        }

    return {
        'severity': 'safe',
        'title': 'SAFE: Structure Healthy',
        'message': f'Surface: {vision_defect} ({vision_confidence:.1f}%)\nEnvironment: Stable.',
        'risk_analysis': risk_analysis
    }

# --- ROUTES ---

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/dashboard')
def dashboard():
    return render_template('index.html', result=None)

@app.route('/predict_image', methods=['POST'])
def predict_image():
    try:
        if 'image' not in request.files:
            raise ValueError("No file uploaded")
        
        file = request.files['image']
        if file.filename == '':
            raise ValueError("No file selected")
        
        # Validate file
        allowed = {'png', 'jpg', 'jpeg', 'bmp', 'gif'}
        ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
        if ext not in allowed:
            raise ValueError("Invalid file type")
        
        # 1. SAVE THE FILE (Required for display)
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 2. Process
        img = Image.open(filepath).convert('RGB')
        
        # 3. Get Predictions
        temp, moist = simulate_sensor_data()
        s_risk, s_msg = get_sensor_prediction(temp, moist)
        v_defect, v_conf, v_sev, florence_bboxes, florence_desc = get_vision_prediction(img)
        
        # Convert Florence-2 bounding boxes to format compatible with frontend
        gemini_result = None
        if florence_bboxes or florence_desc:  # Check if we have Florence-2 data (bboxes or description)
            gemini_result = {
                'problem': v_defect,
                'confidence': v_conf / 100.0,  # Convert to 0-1 range
                'description': florence_desc if florence_desc else f"Florence-2 detected {v_defect} with {v_conf:.1f}% confidence"
            }
            
            # Add location if available
            if florence_bboxes and 'bboxes' in florence_bboxes and len(florence_bboxes['bboxes']) > 0:
                # Get first bounding box and convert to percentage format
                bbox = florence_bboxes['bboxes'][0]
                img_width, img_height = img.size
                
                gemini_result['location'] = {
                    'x': bbox[0] / img_width,
                    'y': bbox[1] / img_height,
                    'width': (bbox[2] - bbox[0]) / img_width,
                    'height': (bbox[3] - bbox[1]) / img_height
                }
            
            print(f"[Florence-2] Generated frontend result: {gemini_result}")
        
        
        # 4. Gemini is disabled (regional quota issue)
        # Florence-2 bounding boxes are used instead (see above)
        
        
        # 5. Combine
        result = combine_predictions(s_risk, v_sev, v_defect, v_conf, temp, moist)
        
        # 6. Return with Gemini data
        return render_template('index.html', 
                             result=result, 
                             filename=filename,
                             gemini_data=gemini_result)
    
    except Exception as e:
        return render_template('index.html', result={
            'severity': 'info',
            'title': 'Error',
            'message': str(e)
        })

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)