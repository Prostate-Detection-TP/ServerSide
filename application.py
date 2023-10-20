from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import tensorflow as tf
import cv2
import numpy as np

import logging
from flask_cors import CORS, cross_origin

import io
from PIL import Image


application = Flask(__name__)
cors = CORS(application, resources={r"/foo": {"origins": "http://localhost:5173"}})

# Configuración
MODEL_PATH = 'models/v1.2.h5'
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


@application.route('/')
def index():
    return 'Welcome to the Prostate Cancer Detection API!'


@application.route('/predict', methods=['POST'])
@cross_origin(origin='localhost', headers=['Content-Type', 'Authorization'])
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
        prediction = model.predict(np.expand_dims(resize / 255, 0))[0][0]
        logging.info(f"Prediction: {prediction}")

        # Calcula predictionFormated, message, significant
        predictionFormated = f"{prediction * 100:.2f}%"
        if prediction >= 0.5:
            message = 'significant'
            significant = True
        else:
            message = 'not-significant'
            significant = False

        # Convierte la imagen en base64
        image_pil = Image.fromarray(cv2.cvtColor(resize.numpy().astype(np.uint8), cv2.COLOR_BGR2RGB))
        buffered = io.BytesIO()
        image_pil.save(buffered, format="JPEG")

        # Genera una URL Blob para la imagen

        response_data = {
            'prediction': str(prediction),
            'predictionFormated': predictionFormated,
            'message': message,
            'significant': significant,
        }

        return jsonify(response_data)

    except Exception as e:
        logging.error(f"Error processing prediction: {e}")
        return jsonify({'error': 'Error processing prediction'}), 500


@application.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'OK'}), 200


if __name__ == '__main__':
    application.run(host='0.0.0.0', port=PORT)
