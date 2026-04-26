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
import cv2
from mtcnn import MTCNN

app = FastAPI(title="Deepfake Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount frontend
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Load model and face detector
MODEL_PATH = "model/deepfake_detector.h5"
model = None
face_detector = MTCNN()

def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded")
    else:
        print("⚠️ Model not found — train first")

load_model()

def detect_face(image_bytes):
    """Check if image contains a face using MTCNN"""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_array = np.array(img)
    faces = face_detector.detect_faces(img_array)
    return len(faces) > 0, faces

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((128, 128))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

@app.get("/")
def root():
    return FileResponse("frontend/index.html")

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

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model not loaded. Please train first."}

    contents = await file.read()

    try:
        # Step 1 — Check for face
        has_face, faces = detect_face(contents)

        if not has_face:
            return {
                "label": "NO_FACE",
                "confidence": 0,
                "raw_score": 0,
                "message": "⚠️ No face detected. Please upload a face image for deepfake detection.",
                "faces_found": 0
            }

        # Step 2 — Run deepfake detection
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