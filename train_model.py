# ========================
# 1. Install & Import Libraries
# ========================


import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os



# Set your dataset and model path (example assumes your dataset is in Drive)
DATA_DIR = 'dataset1'  # Adjust path accordingly
MODEL_DIR = 'model'
os.makedirs(MODEL_DIR, exist_ok=True)

# ========================
# 3. Load Pre-trained MobileNet
# ========================
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze base layers
for layer in base_model.layers[:-20]:
    layer.trainable = False

# Custom top layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# ========================
# 4. Data Preparation
# ========================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.05,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'train'),
    target_size=(224, 224),
    batch_size=16,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'val'),
    target_size=(224, 224),
    batch_size=16,
    class_mode='binary'
)

# ========================
# 5. Compute Class Weights
# ========================
classes = train_generator.classes
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(classes), y=classes)
class_weights = dict(zip(np.unique(classes), class_weights))

# ========================
# 6. Callbacks
# ========================
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint(os.path.join(MODEL_DIR, 'mobilenet.keras'), save_best_only=True)
]

# ========================
# 7. Train Model
# ========================
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=30,
    callbacks=callbacks,
    class_weight=class_weights
)

# ========================
# 8. Save Final Model
# ========================
model.save(os.path.join(MODEL_DIR, 'mobilenet.keras'))
