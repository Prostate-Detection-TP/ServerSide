from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import tensorflow as tf
import cv2
import numpy as np
import os
import logging

app = Flask(__name__)

# Configuración
MODEL_PATH = 'models/v1.1.h5'
PORT = 81

# Configurar logging
logging.basicConfig(level=logging.INFO)

# Cargar el modelo al inicio
try:
    model = load_model(MODEL_PATH, compile=False)
    logging.info(f"Model loaded from {MODEL_PATH}")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    model = None


@app.route('/')
def index():
    return 'Welcome to the Prostate Cancer Detection API!'


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model is not loaded'}), 500

    image_file = request.files.get('image')
    if not image_file:
        return jsonify({'error': 'No image provided'}), 400

    try:
        # Convertir la imagen a un formato que OpenCV pueda leer
        nparr = np.fromstring(image_file.read(), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        resize = tf.image.resize(img, (256, 256), method=tf.image.ResizeMethod.AREA)
        prediction = model.predict(np.expand_dims(resize / 255, 0))
        logging.info(f"Prediction: {prediction}")

        return jsonify({'prediction': str(prediction[0][0])})

    except Exception as e:
        logging.error(f"Error processing prediction: {e}")
        return jsonify({'error': 'Error processing prediction'}), 500


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'OK'}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT)
