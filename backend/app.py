import os
import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

MODEL_PATH = "model/tumor_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(150, 150))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image = request.files["image"]
    image_path = "uploaded.jpg"
    image.save(image_path)

    img_array = preprocess_image(image_path)
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = float(np.max(prediction) * 100)

    return jsonify({
        "predicted_class": class_labels[class_index],
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
