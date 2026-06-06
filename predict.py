import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras


IMAGE_SIZE = (224, 224)


def parse_args():
    parser = argparse.ArgumentParser(description="Predict REAL vs FAKE for one image.")
    parser.add_argument("image_path", type=Path)
    parser.add_argument("--model-path", type=Path, default=Path("outputs/best_model.keras"))
    parser.add_argument("--metadata-path", type=Path, default=Path("outputs/metadata.json"))
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def load_class_names(metadata_path):
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as file:
            return json.load(file).get("class_names", ["FAKE", "REAL"])
    return ["FAKE", "REAL"]


def load_image(image_path):
    image = keras.utils.load_img(image_path, target_size=IMAGE_SIZE)
    array = keras.utils.img_to_array(image)
    return np.expand_dims(array, axis=0)


def main():
    args = parse_args()
    if not args.image_path.exists():
        raise FileNotFoundError(f"Image not found: {args.image_path}")

    class_names = load_class_names(args.metadata_path)
    model = keras.models.load_model(args.model_path)
    image = load_image(args.image_path)

    probability = float(model.predict(image, verbose=0).ravel()[0])
    predicted_index = int(probability >= args.threshold)
    predicted_label = class_names[predicted_index]
    confidence = probability if predicted_index == 1 else 1.0 - probability

    print(f"Prediction: {predicted_label}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Probability for {class_names[1]}: {probability:.4f}")


if __name__ == "__main__":
    main()
