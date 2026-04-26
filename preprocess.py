import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

BASE_DIR = "archive/real_vs_fake/real-vs-fake"
TRAIN_DIR = os.path.join(BASE_DIR, "train")

def visualize_samples():
    """Visualize sample real vs fake images from dataset"""
    real_dir = os.path.join(TRAIN_DIR, "real")
    fake_dir = os.path.join(TRAIN_DIR, "fake")

    real_images = os.listdir(real_dir)[:4]
    fake_images = os.listdir(fake_dir)[:4]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Sample Dataset Images', fontsize=16)

    for i, img_name in enumerate(real_images):
        img = cv2.imread(os.path.join(real_dir, img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128))
        axes[0][i].imshow(img)
        axes[0][i].set_title('REAL')
        axes[0][i].axis('off')

    for i, img_name in enumerate(fake_images):
        img = cv2.imread(os.path.join(fake_dir, img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128))
        axes[1][i].imshow(img)
        axes[1][i].set_title('FAKE')
        axes[1][i].axis('off')

    plt.savefig("model/sample_images.png")
    print("✅ Sample images saved to model/sample_images.png")

def check_dataset_stats():
    """Print dataset statistics"""
    for split in ['train', 'valid', 'test']:
        split_dir = os.path.join(BASE_DIR, split)
        real_count = len(os.listdir(os.path.join(split_dir, 'real')))
        fake_count = len(os.listdir(os.path.join(split_dir, 'fake')))
        print(f"{split.upper():6} → Real: {real_count:6,} | Fake: {fake_count:6,} | Total: {real_count+fake_count:6,}")

if __name__ == "__main__":
    print("📊 Dataset Statistics:")
    check_dataset_stats()
    print("\n🖼️  Saving sample images...")
    visualize_samples()