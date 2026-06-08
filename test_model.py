import os
import tensorflow as tf

print("TensorFlow version:", tf.__version__)
model_path = "pneumonia_vgg16_model.h5"

if os.path.exists(model_path):
    print(f"Found model at {model_path}, size: {os.path.getsize(model_path)} bytes")
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!")
        model.summary()
    except Exception as e:
        print("Error loading model:", e)
else:
    print("Model file not found!")
