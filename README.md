# SafeHome - Concrete Health Monitoring System

![SafeHome Banner](https://img.shields.io/badge/AI-Powered-blue) ![Flask](https://img.shields.io/badge/Flask-3.0.0-green) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange)

## 🏗️ Overview

SafeHome is an AI-powered concrete health monitoring system that uses **dual machine learning models** to detect structural defects:

1. **Sensor Analysis Model** (Random Forest) - Predicts internal corrosion risk from temperature and moisture data
2. **Visual Inspection Model** (MobileNetV2 CNN) - Detects concrete cancer, spalling, and cracks from images

## ✨ Features

- 🔬 **Predictive Maintenance**: Detect invisible defects before they become visible
- 📷 **Computer Vision**: Identify 8 types of concrete defects from photos
- 🎨 **Modern UI**: Beautiful Tailwind CSS interface with gradient backgrounds
- 🚀 **Real-time Analysis**: Instant predictions with confidence scores
- 📊 **Risk Classification**: Color-coded results (Safe/Warning/Critical)

## 📁 Project Structure

```
safehome/
├── app.py                          # Flask backend
├── requirements.txt                # Python dependencies
├── models/
│   ├── sensor_model.pkl           # Sensor analysis model (upload yours)
│   └── concrete_cancer_model.keras # Vision model (upload yours)
├── templates/
│   └── index.html                 # Tailwind CSS UI
└── static/
    └── uploads/                   # Temporary image storage
```

## 🚀 Quick Start

### 1. Clone or Download

```bash
cd safehome
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Add Your Models

Place your trained models in the `models/` folder:
- `sensor_model.pkl` - Your Random Forest sensor model
- `concrete_cancer_model.keras` (or `.h5`) - Your MobileNetV2 vision model

### 4. Run the Application

```bash
python app.py
```

The app will start at **http://127.0.0.1:5000**

## 🎯 Usage

### Sensor Analysis
1. Navigate to the left panel
2. Enter temperature (°C) and moisture (%)
3. Click "Run Analysis"
4. View risk assessment with signal strength prediction

### Visual Inspection
1. Navigate to the right panel
2. Upload a concrete image (PNG, JPG, JPEG)
3. Click "Scan for Defects"
4. View defect classification with confidence score

## 🧠 Model Details

### Sensor Model
- **Type**: Random Forest Regressor
- **Input**: Temperature, Moisture
- **Output**: Signal Strength (mV)
- **Thresholds**:
  - < -140 mV: HIGH RISK (Corrosion)
  - -140 to -100 mV: WARNING
  - > -100 mV: SAFE

### Vision Model
- **Type**: MobileNetV2 (Transfer Learning)
- **Input**: 224x224 RGB images
- **Output**: 8 classes
  - `spalling` (Critical)
  - `peeling` (Critical)
  - `major_crack` (Warning)
  - `minor_crack` (Notice)
  - `normal` (Safe)
  - `Wall`, `algae`, `stain` (Cosmetic)

## 🎨 Technology Stack

- **Backend**: Flask 3.0
- **ML Framework**: TensorFlow 2.15, Scikit-learn 1.3
- **Frontend**: Tailwind CSS (CDN)
- **Image Processing**: Pillow
- **Data Handling**: NumPy, Pandas

## 🔧 Troubleshooting

### Models Not Loading
- Ensure model files are in the `models/` folder
- Check file names match exactly: `sensor_model.pkl` and `concrete_cancer_model.keras`
- Verify models were trained with compatible library versions

### TensorFlow Errors
- For Windows: `pip install tensorflow-cpu` (lighter version)
- For GPU: Install CUDA-compatible TensorFlow

### Port Already in Use
```bash
python app.py --port 5001
```

## 📝 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main dashboard |
| `/predict_sensor` | POST | Sensor data analysis |
| `/predict_image` | POST | Image defect detection |

## 🤝 Contributing

This is a prototype application. To improve:
1. Add database integration for historical tracking
2. Implement user authentication
3. Add batch processing for multiple images
4. Create API for mobile app integration

## 📄 License

MIT License - Feel free to use for educational and commercial purposes

## 👨‍💻 Author

Built with ❤️ for infrastructure safety

---

**Note**: This application requires pre-trained models. Ensure you have completed the training phase before deployment.
