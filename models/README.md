# IMPORTANT: Place your trained models in this folder

## Required Files:

1. **sensor_model.pkl**
   - Your Random Forest sensor analysis model
   - Trained to predict signal strength from temperature and moisture

2. **concrete_cancer_model.keras** (or .h5)
   - Your MobileNetV2 vision model
   - Trained on 8 concrete defect classes

## How to Add Models:

Simply copy your model files into this `models/` directory.

The Flask app will automatically detect and load them on startup.

## Verification:

When you run `python app.py`, you should see:
```
✅ Sensor Model loaded successfully
✅ Vision Model (.keras) loaded successfully
🚀 Server ready!
```

If you see warnings, check that:
- File names match exactly
- Files are not corrupted
- Models were saved with compatible library versions
