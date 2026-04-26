import os
import io
import json
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import tensorflow as tf
from PIL import Image
from mtcnn import MTCNN
import gdown

app = FastAPI(title="Deepfake Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Google Drive model download ──────────────────────────────────────────────
MODEL_PATH = "model/deepfake_detector.h5"
GDRIVE_FILE_ID = "1nCQUy0u7qMwL843MKZfef9rhAtxrkrkg"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("⬇️  Model not found — downloading from Google Drive...")
        os.makedirs("model", exist_ok=True)
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
        print("✅ Model downloaded successfully!")
    else:
        print("✅ Model already exists — skipping download")

# ── Load model and face detector ─────────────────────────────────────────────
model = None
face_detector = MTCNN()

def load_model():
    global model
    download_model()
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded")
    else:
        print("⚠️ Model not found — download failed")

load_model()

# ── Serve frontend ────────────────────────────────────────────────────────────
if os.path.exists("frontend"):
    app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
def root():
    if os.path.exists("frontend/index.html"):
        return FileResponse("frontend/index.html")
    return {"status": "ok", "model_loaded": model is not None}

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.get("/metrics")
def get_metrics():
    metrics_path = "model/metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            return json.load(f)
    return {"error": "Metrics not available yet"}

def detect_face(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_array = np.array(img)
    faces = face_detector.detect_faces(img_array)
    return len(faces) > 0, faces

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((128, 128))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model not loaded."}

    contents = await file.read()

    try:
        has_face, faces = detect_face(contents)

        if not has_face:
            return {
                "label": "NO_FACE",
                "confidence": 0,
                "raw_score": 0,
                "message": "⚠️ No face detected. Please upload a face image.",
                "faces_found": 0
            }

        img_array = preprocess_image(contents)
        prediction = model.predict(img_array)[0][0]

        label = "REAL" if prediction > 0.7 else "FAKE"
        confidence = float(prediction) if label == "REAL" else float(1 - prediction)

        return {
            "label": label,
            "confidence": round(confidence * 100, 2),
            "raw_score": round(float(prediction), 4),
            "faces_found": len(faces),
            "message": f"Detected {len(faces)} face(s). This image is {label} with {round(confidence*100, 2)}% confidence"
        }

    except Exception as e:
        return {"error": str(e)}