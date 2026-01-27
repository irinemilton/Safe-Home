# 🚀 Quick Start Guide - SafeHome

## ⚡ 3-Step Setup

### 1️⃣ Install Dependencies
```bash
cd c:/Users/USER/OneDrive/Desktop/safehome
pip install -r requirements.txt
```

### 2️⃣ Add Your Models
Place these files in the `models/` folder:
- `sensor_model.pkl` (your Random Forest model)
- `concrete_cancer_model.keras` (your MobileNetV2 model)

### 3️⃣ Run the App
```bash
python app.py
```

Then open: **http://127.0.0.1:5000**

---

## 📁 What You Got

```
safehome/
├── app.py                    ← Flask backend (ready to run)
├── templates/index.html      ← Tailwind CSS UI
├── models/                   ← PUT YOUR MODELS HERE
├── requirements.txt          ← Dependencies list
├── README.md                 ← Full documentation
└── SETUP.md                  ← Detailed setup guide
```

---

## 🎯 Features

### Left Panel: Sensor Analysis
- Input: Temperature (°C) + Moisture (%)
- Output: Risk prediction with signal strength

### Right Panel: Visual Inspection
- Input: Upload concrete image
- Output: Defect detection (8 classes)

---

## 🎨 Design Highlights

- **Modern UI**: Tailwind CSS with gradients
- **Responsive**: Works on mobile + desktop
- **Color-coded**: Green (safe), Yellow (warning), Red (critical)
- **Status Indicators**: Shows if models are loaded

---

## ❓ Troubleshooting

**Models not loading?**
- Check files are in `models/` folder
- Verify file names match exactly

**TensorFlow errors?**
```bash
pip install tensorflow-cpu==2.15.0
```

**Need help?**
- See [README.md](file:///c:/Users/USER/OneDrive/Desktop/safehome/README.md) for full docs
- See [SETUP.md](file:///c:/Users/USER/OneDrive/Desktop/safehome/SETUP.md) for detailed setup

---

## 📊 Next Steps

1. ✅ Upload your model files
2. ✅ Run `pip install -r requirements.txt`
3. ✅ Run `python app.py`
4. ✅ Test both analysis modes
5. 🚀 Deploy to production (optional)

---

**Built with**: Flask + TensorFlow + Tailwind CSS
