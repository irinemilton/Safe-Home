# Training Your Vision Model

## Quick Start: Create a Test Model

If you just want to test the upload feature, run this command:

```bash
python create_vision_model.py
```

This will create `models/concrete_cancer_model.keras` that's compatible with your Flask app.

**Note**: This model is NOT trained, so predictions will be random. It's just for testing the upload feature.

---

## Training With Your Dataset

If you have a dataset of concrete images organized like this:

```
dataset/
├── train/
│   ├── Wall/
│   ├── algae/
│   ├── major_crack/
│   ├── minor_crack/
│   ├── normal/
│   ├── peeling/
│   ├── spalling/
│   └── stain/
└── validation/
    ├── Wall/
    ├── algae/
    └── ... (same structure)
```

### Training Script

Create `train_model.py`:

```python
import tensorflow as tf
from tensorflow import keras
from create_vision_model import create_model, CLASS_NAMES

# Configuration
DATASET_PATH = 'path/to/your/dataset'
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10

# Load dataset
train_ds = keras.utils.image_dataset_from_directory(
    f'{DATASET_PATH}/train',
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='int'
)

val_ds = keras.utils.image_dataset_from_directory(
    f'{DATASET_PATH}/validation',
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='int'
)

# Create model
model = create_model()

# Train
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# Save
model.save('models/concrete_cancer_model.keras')
print("Model trained and saved!")
```

Then run:
```bash
python train_model.py
```

---

## Alternative: Convert Your Existing Model

If you have a model trained with a newer TensorFlow version:

```python
import tensorflow as tf

# Load your old model (use the newer TensorFlow version)
old_model = tf.keras.models.load_model('path/to/old_model.keras')

# Save in H5 format (more compatible)
old_model.save('models/concrete_cancer_model.h5', save_format='h5')
```

Then copy the `.h5` file to your SafeHome `models/` folder.

---

## Verify Model Works

After creating/training your model:

1. Restart Flask app:
   ```bash
   python app.py
   ```

2. Check the console output:
   ```
   [OK] Vision Model (.keras) loaded successfully
   [READY] Server ready!
   ```

3. Upload a test image at http://127.0.0.1:5000

---

## Troubleshooting

**Model still won't load?**
- Make sure file is named exactly: `concrete_cancer_model.keras` or `.h5`
- Check it's in the `models/` folder
- Verify TensorFlow version: `pip show tensorflow`

**Getting random predictions?**
- The untrained model will give random results
- You need to train it with your actual concrete images
- See training instructions above
