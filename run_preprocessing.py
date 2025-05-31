# run_preprocessing.py
import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set parameters
IMG_SIZE = (224, 224)
DATASET_PATH = "dataset1"

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    return img

def load_images(data_type):
    images = []
    labels = []
    for label, class_name in enumerate(["normal", "stroke"]):
        class_dir = os.path.join(DATASET_PATH, data_type, class_name)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            try:
                img = preprocess_image(img_path)
                images.append(img)
                labels.append(label)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
    return np.array(images), np.array(labels)

if __name__ == '__main__':
    for split in ["train", "test", "val"]:
        print(f"Loading {split} data...")
        X, y = load_images(split)
        np.save(f"dataset1/{split}_X.npy", X)
        np.save(f"dataset1/{split}_y.npy", y)
        print(f"Saved {split} data: {X.shape[0]} samples")

