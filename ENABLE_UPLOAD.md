# Quick Fix: Enable Image Upload Without Vision Model

## Problem
The "Scan for Defects" button is disabled because the vision model failed to load.

## Solution 1: Enable Upload with Mock Predictions (Quick Test)

This allows you to test the upload feature immediately with placeholder results.

### Step 1: Modify `app.py`

Find the line around line 180 in `app.py` where it says:
```python
@app.route('/predict_image', methods=['POST'])
def predict_image():
    if vision_model is None:
        return render_template('index.html', 
                             image_result="❌ Vision model not loaded..."
```

Replace the entire `predict_image` function with this:

```python
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
            msg = "INFO: Image uploaded successfully!\n\nVision model is currently offline. To enable defect detection:\n1. Retrain your model with TensorFlow 2.15.1\n2. Save as 'concrete_cancer_model.keras'\n3. Place in models/ folder\n4. Restart the server"
            color = "blue"
            return render_template('index.html', 
                                 image_result=msg, 
                                 image_color=color,
                                 sensor_available=sensor_model is not None,
                                 vision_available=False)
        
        # Process the image (original code continues here)
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
            msg = f"CRITICAL ALERT: {prediction.upper()} Detected ({confidence:.1f}% confidence)\nConcrete Cancer - Immediate structural audit required!"
            color = "red"
        elif prediction == 'major_crack':
            msg = f"WARNING: {prediction.upper().replace('_', ' ')} Detected ({confidence:.1f}% confidence)\nStructural integrity compromised - Professional inspection recommended."
            color = "orange"
        elif prediction == 'minor_crack':
            msg = f"NOTICE: {prediction.upper().replace('_', ' ')} Detected ({confidence:.1f}% confidence)\nMonitor for progression."
            color = "yellow"
        elif prediction == 'normal':
            msg = f"SAFE: Surface appears Healthy ({confidence:.1f}% confidence)\nNo defects detected."
            color = "green"
        else:
            msg = f"INFO: {prediction.upper()} Detected ({confidence:.1f}% confidence)\nLikely cosmetic - no structural concern."
            color = "blue"
        
        return render_template('index.html', 
                             image_result=msg, 
                             image_color=color,
                             sensor_available=sensor_model is not None,
                             vision_available=True)
    
    except ValueError as ve:
        return render_template('index.html', 
                             image_result=f"ERROR: {str(ve)}", 
                             image_color="red",
                             sensor_available=sensor_model is not None,
                             vision_available=vision_model is not None)
    except Exception as e:
        return render_template('index.html', 
                             image_result=f"ERROR: Error processing image: {str(e)}", 
                             image_color="red",
                             sensor_available=sensor_model is not None,
                             vision_available=vision_model is not None)
```

### Step 2: Modify `templates/index.html`

Find the line with the disabled button (around line 280):
```html
<button type="submit" {% if not vision_available %}disabled{% endif %}
```

Change it to:
```html
<button type="submit"
```

This removes the `disabled` attribute completely, allowing uploads even without the model.

### Step 3: Restart the server
```bash
# Press Ctrl+C in the terminal running the app
python app.py
```

Now you can upload images and get a helpful message about enabling the vision model!

---

## Solution 2: Fix the Vision Model (Proper Fix)

See `VISION_MODEL_FIX.md` for instructions on retraining your model with the correct TensorFlow version.
