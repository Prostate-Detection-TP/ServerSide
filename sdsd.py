import tensorflow as tf

# Cargar el modelo Keras
model = tf.keras.models.load_model("models/v1.2.h5")

# Convertir el modelo a formato TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Guardar el modelo convertido en el directorio de modelos
with open("models/v1.2.tflite", "wb") as f:
    f.write(tflite_model)
