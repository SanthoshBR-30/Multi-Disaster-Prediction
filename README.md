# 🌪️ SafeHaven: Multi-Disaster Prediction using Deep Convolutional Neural Networks

> **97.28% test accuracy** on satellite disaster image classification using a six-stage hierarchical deep ensemble framework.

---
## 📌 Overview

SafeHaven classifies satellite images into four disaster categories — **Cyclone, Earthquake, Flood, Wildfire** — using a six-stage ensemble pipeline that combines three CNN backbones, XGBoost, a Dense Neural Meta-Classifier, Test-Time Augmentation, and dual explainability (Grad-CAM++ + LIME).

---


## 📁 Dataset

- **Source:** Kaggle Natural Disaster Image Dataset
- **Total images:** 4,325 across 4 classes
- **Split:** 3,219 train / 444 validation / 662 test
- **Input size:** 224×224 pixels, normalised to [0, 1]

| Class | Train | Val | Test |
|---|---|---|---|
| Cyclone | 696 | 93 | 139 |
| Earthquake | 942 | 136 | 201 |
| Flood | 805 | 108 | 160 |
| Wildfire | 776 | 107 | 162 |

---

## 🗂️ Project Structure

```
disaster-project/
├── app.py                        # Flask backend (REST API)
├── index.html                    # Frontend UI
├── requirements.txt              # Python dependencies
├── class_names.json              
├── EfficientNetV2_final.keras    # Backbone 1 
├── ConvNeXt_final.keras          # Backbone 2 
├── DenseNet201_final.keras       # Backbone 3 
└── dense_classifier_final.keras  # Meta-classifier 
```


---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/tdsrusti/Multi-Disaster-CNN-Ensemble.git
cd Multi-Disaster-CNN-Ensemble
```

### 2. Install dependencies
```bash
pip3 install -r requirements.txt
```

### 3. Download model files
Due to Copyright reason, it is adviced to build your own model and download the model in .keras file.

### 4. Run the backend
```bash
python3 app.py
```
Server starts at `http://localhost:5001`

> **Note for macOS users:** Port 5000 is reserved by AirPlay — the app runs on port 5001.

### 5. Open the frontend
Open `index.html` in your browser, or use VS Code Live Server.

---


## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Backend | Python, Flask |
| Deep Learning | TensorFlow 2.21, Keras 3.13.2 |
| Backbones | EfficientNetV2B0, ConvNeXt-Tiny, DenseNet201 |
| Meta-Classifier | XGBoost, Dense MLP |
| Explainability | Grad-CAM++ (tf-keras-vis), LIME |
| Frontend | HTML, CSS, Vanilla JavaScript |
| Training Platform | Google Colab (T4 GPU) |

---

## 📄 Citation

If you use this work, please cite:

```
Santhosh B R  "SafeHaven: Multi-Disaster Prediction using Deep
Convolutional Neural Networks." Dayananda Sagar College of Engineering,
Bengaluru, India. 2025.
```