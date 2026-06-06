import tensorflow as tf
import numpy as np
from PIL import Image
import os

print("Starting script...")

# show working directory (VERY IMPORTANT DEBUG)
print("Current folder:", os.getcwd())

# load model
model = tf.keras.models.load_model(
    r"D:\DeepFake-Detection\outputs\best_model.keras",
    compile=False
)

print("Model loaded successfully!")

# load test image
img = Image.open("test.jpg").resize((224, 224))
img = np.array(img) / 255.0
img = np.expand_dims(img, axis=0)

print("Image loaded!")

# prediction
pred = model.predict(img)[0][0]

print("Prediction score:", pred)

if pred > 0.5:
    print("FAKE")
else:
    print("REAL")