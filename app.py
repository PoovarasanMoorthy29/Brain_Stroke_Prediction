from flask import Flask, render_template, request
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.xception import preprocess_input

app = Flask(__name__)
model = load_model("model/mobilenet_final.keras")

IMG_SIZE = (224, 224)

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, IMG_SIZE)
    img = preprocess_input(img)
    return np.expand_dims(img, axis=0)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    filepath = "static/" + file.filename
    file.save(filepath)
    img = preprocess_image(filepath)
    pred = model.predict(img)[0][0]
    result = "Stroke" if pred > 0.5 else "Normal"
    confidence = round(float(pred) * 100, 2) if pred > 0.5 else round((1 - float(pred)) * 100, 2)
    return render_template("result.html", prediction=result, confidence=confidence, img_path=filepath)

if __name__ == '__main__':
    app.run(debug=True)
