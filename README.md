🚀 Live Demo: https://deepfake-detector.up.railway.app

# 🔍 Deepfake Detector

A deep learning CNN model that classifies images as **Real or Deepfake** using computer vision techniques.

## 📊 Results
- **Accuracy:** 75%+
- **Dataset:** 140,000 images (Kaggle)
- **False Positive Reduction:** ~20% through model optimization

## 🛠️ Technologies
- Python, TensorFlow, OpenCV, MTCNN
- FastAPI, Docker
- scikit-learn, matplotlib, seaborn

## 🏗️ Project Structure

deepfake-detector/
├── train.py          # CNN model training
├── evaluate.py       # Model evaluation & metrics
├── preprocess.py     # Data preprocessing pipeline
├── api/
│   └── main.py       # FastAPI REST API
├── frontend/
│   └── index.html    # Web interface
├── model/            # Saved model & metrics
├── Dockerfile        # Container configuration
└── requirements.txt  # Dependencies

## 🚀 Run Locally
```bash
# Activate environment
source venv/bin/activate

# Train model
python train.py

# Launch app
uvicorn api.main:app --reload
```

## 🌐 API Endpoints
- `POST /predict` — Upload image, get REAL/FAKE result
- `GET /metrics` — Model performance metrics
- `GET /health` — API health check

## 🧠 Model Architecture
- 4 Convolutional layers with BatchNormalization
- MaxPooling for dimensionality reduction
- Dropout layers to reduce overfitting
- Binary classification output (Real/Fake)

## 📁 Dataset
- Source: Kaggle — 140k Real and Fake Faces
- Train: 100,000 images
- Validation: 20,000 images
- Test: 20,000 images
