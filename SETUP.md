# SafeHome Setup Guide

## Step-by-Step Installation

### 1. Navigate to Project Directory
```bash
cd c:/Users/USER/OneDrive/Desktop/safehome
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv
```

### 3. Activate Virtual Environment

**Windows (PowerShell):**
```bash
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```bash
venv\Scripts\activate.bat
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

This will install:
- Flask 3.0.0
- TensorFlow 2.15.0
- Scikit-learn 1.3.2
- Pandas, NumPy, Pillow

### 5. Add Your Models

**IMPORTANT**: Place your trained models in the `models/` folder:

1. Copy `sensor_model.pkl` → `models/sensor_model.pkl`
2. Copy `concrete_cancer_model.keras` → `models/concrete_cancer_model.keras`

### 6. Run the Application
```bash
python app.py
```

You should see:
```
⏳ Loading AI Models...
✅ Sensor Model loaded successfully
✅ Vision Model (.keras) loaded successfully
🚀 Server ready!
 * Running on http://127.0.0.1:5000
```

### 7. Open in Browser
Navigate to: **http://127.0.0.1:5000**

## Testing the Application

### Test Sensor Analysis
1. Enter Temperature: `30`
2. Enter Moisture: `15`
3. Click "Run Analysis"
4. Expected: Should show risk prediction

### Test Visual Inspection
1. Upload a concrete image
2. Click "Scan for Defects"
3. Expected: Should show defect classification

## Troubleshooting

### "Models not found" Warning
- Ensure model files are in `models/` folder
- Check file names match exactly

### TensorFlow Installation Issues
For Windows users with older CPUs:
```bash
pip install tensorflow-cpu==2.15.0
```

### Port Already in Use
Change port in `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)
```

## Next Steps

Once the app is running:
1. Upload your model files to the `models/` folder
2. Test both analysis modes
3. Verify results display correctly
4. Check responsive design on mobile
