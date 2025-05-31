import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = cv2.imread(image_path)
    img = cv2.resize(img, target_size)
    img = img.astype("float") / 255.0
    return img_to_array(img)

def load_dataset_from_folder(base_path, target_size=(224, 224)):
    data = []
    labels = []
    for label in ['Normal', 'Stroke']:
        path = os.path.join(base_path, label)
        for file in os.listdir(path):
            img_path = os.path.join(path, file)
            try:
                image = load_and_preprocess_image(img_path, target_size)
                data.append(image)
                labels.append(0 if label == 'Normal' else 1)
            except Exception as e:
                print(f"Could not process {img_path}: {e}")
    return np.array(data), np.array(labels)
