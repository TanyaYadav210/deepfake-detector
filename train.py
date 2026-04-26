import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import json

# ── Paths ──────────────────────────────────────────────
BASE_DIR   = "archive/real_vs_fake/real-vs-fake"
TRAIN_DIR  = os.path.join(BASE_DIR, "train")
VALID_DIR  = os.path.join(BASE_DIR, "valid")
TEST_DIR   = os.path.join(BASE_DIR, "test")
MODEL_DIR  = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Config ─────────────────────────────────────────────
IMG_SIZE    = (128, 128)
BATCH_SIZE  = 32
EPOCHS      = 20
LR          = 1e-4

# ── Data Augmentation ──────────────────────────────────
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE,
    batch_size=BATCH_SIZE, class_mode='binary'
)
val_gen = val_datagen.flow_from_directory(
    VALID_DIR, target_size=IMG_SIZE,
    batch_size=BATCH_SIZE, class_mode='binary'
)
test_gen = val_datagen.flow_from_directory(
    TEST_DIR, target_size=IMG_SIZE,
    batch_size=BATCH_SIZE, class_mode='binary', shuffle=False
)

# ── CNN Model ──────────────────────────────────────────
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(256, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=LR),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

model.summary()

# ── Callbacks ──────────────────────────────────────────
callbacks = [
    ModelCheckpoint("model/deepfake_detector.h5", save_best_only=True, monitor='val_accuracy', verbose=1),
    EarlyStopping(patience=5, restore_best_weights=True, monitor='val_accuracy'),
    ReduceLROnPlateau(factor=0.5, patience=3, monitor='val_loss', verbose=1)
]

# ── Train ──────────────────────────────────────────────
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ── Evaluate ───────────────────────────────────────────
print("\n📊 Evaluating on test set...")
results = model.evaluate(test_gen)
print(f"Test Accuracy : {results[1]*100:.2f}%")
print(f"Test Precision: {results[2]*100:.2f}%")
print(f"Test Recall   : {results[3]*100:.2f}%")

# Save metrics
metrics = {
    "test_accuracy":  round(results[1]*100, 2),
    "test_precision": round(results[2]*100, 2),
    "test_recall":    round(results[3]*100, 2)
}
with open("model/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# ── Plot ───────────────────────────────────────────────
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy'); plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss'); plt.legend()

plt.savefig("model/training_plot.png")
print("✅ Training complete! Model saved to model/deepfake_detector.h5")