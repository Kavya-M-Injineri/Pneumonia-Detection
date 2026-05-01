# Medical AI — Chest X-ray Pneumonia Classifier 

A medical-grade web application that classifies chest X-rays for pneumonia using **transfer learning on MobileNetV2**, with pathogen-level sub-classification (Viral vs Bacterial), confidence scoring, dual treatment recommendations, and a role-based access Flask backend.

---

## Highlights  

- Fine-tuned **MobileNetV2** (ImageNet pre-trained) on the Kaggle Chest X-Ray dataset for binary pneumonia detection, leveraging transfer learning for high accuracy on a limited medical imaging dataset
- Built **pathogen sub-classification** — model distinguishes Viral vs Bacterial pneumonia patterns with specific pathogen identification (e.g., Streptococcus pneumoniae, Influenza) and per-prediction confidence scores
- Generated **dual treatment reports** per diagnosis — Modern Medicine recommendations (specific antibiotics or antivirals) alongside Ayurvedic alternatives (e.g., Turmeric Milk, Tulsi Tea) for holistic care context
- Implemented **role-based access control** with three user tiers — Admin, Radiologist, and Viewer — each with scoped dashboard permissions
- Designed an **image preprocessing pipeline** (`utils.py`) handling resizing, normalization, and augmentation consistent between training and inference to prevent train-serve skew

---

## Tech Stack  

| Layer | Technology |
|---|---|
| ML Model | TensorFlow, Keras — MobileNetV2 (transfer learning) |
| Preprocessing | Custom pipeline in `utils.py` |
| Backend | Flask, Python 3.8+ |
| Auth | Role-based access (Admin / Radiologist / Viewer) |
| Frontend | HTML, CSS, JavaScript (drag-and-drop upload) |
| Dataset | Kaggle Chest X-Ray Images (Pneumonia) |

---
  
## Model Design  
   
- **Base model:** MobileNetV2 pre-trained on ImageNet, top layers replaced and fine-tuned
- **Task:** Binary classification (Normal vs Pneumonia) + sub-classification (Viral vs Bacterial)
- **Input:** Preprocessed chest X-ray images (resized, normalized)
- **Output:** Class label + confidence score + pathogen attribution
- **Saved to:** `models/pneumonia_model.h5`

---
  
## Setup
  
### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Dataset
Place the Kaggle [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) dataset in the project root:
```
chest_xray/
├── train/
├── val/
└── test/
```

### 3. Train the Model
```bash
python train.py
# → Saves trained model to models/pneumonia_model.h5
```

### 4. Run the App
```bash
python app.py
# → http://127.0.0.1:5000
```

---

## Usage

1. Login with role credentials (Admin / Radiologist / Viewer)
2. Drag and drop a chest X-ray image onto the dashboard
3. Click **Analyze** — model returns:
   - Normal / Pneumonia classification
   - Viral vs Bacterial sub-type + pathogen identification
   - Confidence score
   - **Modern Medicine plan** — specific antibiotics or antivirals based on pathogen type
   - **Ayurvedic plan** — holistic care suggestions (e.g., Turmeric Milk, Tulsi Tea, Giloy)

---

## Project Structure

```
├── app.py              # Flask server — routes, auth, prediction API
├── train.py            # MobileNetV2 fine-tuning + model save
├── utils.py            # Image preprocessing pipeline (resize, normalize)
├── models/
│   └── pneumonia_model.h5   # Trained model weights
├── templates/          # Dashboard, login, report UI
└── static/             # CSS, JS, assets
```

---

> **Disclaimer:** For educational and research purposes only. This tool is not intended for primary clinical diagnosis. Always consult a certified radiologist or medical professional for healthcare decisions.
