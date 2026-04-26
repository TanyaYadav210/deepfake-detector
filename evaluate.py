import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json

# ── Paths ──────────────────────────────────────────────
BASE_DIR = "archive/real_vs_fake/real-vs-fake"
TEST_DIR = os.path.join(BASE_DIR, "test")
MODEL_PATH = "model/deepfake_detector.h5"

# ── Load Model ─────────────────────────────────────────
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

# ── Load Test Data ─────────────────────────────────────
test_datagen = ImageDataGenerator(rescale=1./255)
test_gen = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# ── Predictions ────────────────────────────────────────
print("Running predictions...")
y_pred_probs = model.predict(test_gen)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()
y_true = test_gen.classes

# ── Report ─────────────────────────────────────────────
print("\n📊 Classification Report:")
report = classification_report(y_true, y_pred, target_names=["Real", "Fake"])
print(report)

# ── Confusion Matrix ───────────────────────────────────
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Real', 'Fake'],
            yticklabels=['Real', 'Fake'])
plt.title('Confusion Matrix — Deepfake Detector')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig("model/confusion_matrix.png")
print("✅ Confusion matrix saved to model/confusion_matrix.png")

# ── False Positive Rate ────────────────────────────────
fp = cm[0][1]
tn = cm[0][0]
fpr = fp / (fp + tn) * 100
print(f"\n📉 False Positive Rate: {fpr:.2f}%")

# ── Save Metrics ───────────────────────────────────────
metrics = {
    "false_positive_rate": round(fpr, 2),
    "total_test_images": len(y_true),
    "correct_predictions": int(np.sum(y_pred == y_true))
}
with open("model/evaluation_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("✅ Evaluation complete!")