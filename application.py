from flask import Flask, request, jsonify
import tensorflow.lite as tflite

import cv2
import numpy as np
import logging
from flask_cors import CORS, cross_origin
import io
from PIL import Image

application = Flask(__name__)
cors = CORS(application, resources={r"/foo": {"origins": "http://localhost:5173"}})

# Configuración
MODEL_PATH = 'models/v1.2.tflite'  # Cambiamos la ruta al modelo .tflite
PORT = 81

# Configurar logging
logging.basicConfig(level=logging.INFO)

# Cargar el modelo al inicio
try:
    # Cargamos el modelo TensorFlow Lite
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    logging.info(f"Model loaded from {MODEL_PATH}")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    interpreter = None


@application.route('/')
def index():
    return 'Welcome to the Prostate Cancer Detection API!'


@application.route('/predict', methods=['POST'])
@cross_origin(origins='localhost', allow_headers=['Content-Type', 'Authorization'])
def predict():
    if interpreter is None:
        return jsonify({'error': 'Model is not loaded'}), 500

    image_file = request.files.get('image')
    if not image_file:
        return jsonify({'error': 'No image provided'}), 400

    try:
        nparr = np.fromstring(image_file.read(), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        resize = cv2.resize(img, (256, 256))
        resized_normalized = np.expand_dims(resize / 255.0, 0).astype(np.float32)

        # Configuración de las entradas y salidas del intérprete
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]['index'], resized_normalized)

        # Realiza la predicción
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]
        logging.info(f"Prediction: {prediction}")

        predictionFormated = f"{prediction * 100:.2f}%"
        if prediction >= 0.5:
            message = 'significant'
            significant = True
        else:
            message = 'not-significant'
            significant = False

        image_pil = Image.fromarray(cv2.cvtColor(resize.astype(np.uint8), cv2.COLOR_BGR2RGB))
        buffered = io.BytesIO()
        image_pil.save(buffered, format="JPEG")

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
