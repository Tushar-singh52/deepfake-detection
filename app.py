import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from tensorflow import keras


IMAGE_SIZE = (224, 224)
DEFAULT_MODEL_PATH = Path("outputs/best_model.keras")
DEFAULT_METADATA_PATH = Path("outputs/metadata.json")


@st.cache_resource
def load_model(model_path):
    return keras.models.load_model(model_path)


@st.cache_data
def load_class_names(metadata_path):
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as file:
            return json.load(file).get("class_names", ["FAKE", "REAL"])
    return ["FAKE", "REAL"]


def prepare_image(image):
    image = image.convert("RGB").resize(IMAGE_SIZE)
    array = keras.utils.img_to_array(image)
    return np.expand_dims(array, axis=0)


def predict_image(model, image, class_names, threshold):
    batch = prepare_image(image)
    probability_real = float(model.predict(batch, verbose=0).ravel()[0])
    predicted_index = int(probability_real >= threshold)
    predicted_label = class_names[predicted_index]
    confidence = probability_real if predicted_index == 1 else 1.0 - probability_real
    return predicted_label, confidence, probability_real


def main():
    st.set_page_config(
        page_title="SentinelAI",
        layout="centered",
    )

    st.markdown(
        """
       <style>
    :root {
        /* Dark base */
        --surface: #0b0f19;
        --ink: #e6f1ff;
        --muted: #93a4b8;
        --line: #1f2a3a;

        /* Blue accents */
        --real: #3b82f6;
        --fake: #2563eb;

        /* Soft blue glows */
        --soft-real: rgba(59, 130, 246, 0.12);
        --soft-fake: rgba(37, 99, 235, 0.12);
    }

    .stApp {
        background: linear-gradient(180deg, #05070d 0%, #0b0f19 50%, #05070d 100%);
        color: var(--ink);
    }

    .block-container {
        max-width: 900px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    [data-testid="stSidebar"] {
        background: #070a12;
        border-right: 1px solid var(--line);
    }

    .SentinelAI-header {
        border-bottom: 1px solid var(--line);
        padding: 0.25rem 0 1.1rem;
        margin-bottom: 1.4rem;
    }

    .SentinelAI-name {
        font-size: 2.35rem;
        font-weight: 760;
        line-height: 1.05;
        margin: 0;
        color: var(--ink);
    }

    .SentinelAI-subtitle {
        color: var(--muted);
        font-size: 1rem;
        margin-top: 0.4rem;
    }

    .result-card {
        border: 1px solid var(--line);
        border-radius: 10px;
        background: var(--surface);
        padding: 1.1rem 1.2rem;
        margin-top: 1rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.6);
    }

    .result-label {
        color: var(--muted);
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }

    .result-value {
        font-size: 2rem;
        font-weight: 780;
        line-height: 1;
        color: var(--ink);
    }

    .real {
        color: var(--real);
    }

    .fake {
        color: var(--fake);
    }

    .confidence-row {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.75rem;
        margin-top: 0.9rem;
    }

    .mini-card {
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 0.85rem;
        background: #0d1424;
    }

    .mini-value {
        font-size: 1.25rem;
        font-weight: 720;
        color: var(--ink);
    }

    .note {
        border-radius: 10px;
        padding: 0.8rem 0.9rem;
        margin-top: 0.9rem;
        font-weight: 600;
        border: 1px solid var(--line);
    }

    .note-real {
        background: var(--soft-real);
        color: var(--real);
    }

    .note-fake {
        background: var(--soft-fake);
        color: var(--fake);
    }

    .note-uncertain {
        background: rgba(147, 164, 184, 0.08);
        color: var(--muted);
    }

    div[data-testid="stFileUploader"] {
        border: 1px dashed #2b3a55;
        border-radius: 10px;
        padding: 0.75rem;
        background: #0b0f19;
        color: var(--muted);
    }

    div[data-testid="stImage"] img {
        border-radius: 10px;
        border: 1px solid var(--line);
    }

    @media (max-width: 640px) {
        .SentinelAI-name {
            font-size: 1.85rem;
        }

        .confidence-row {
            grid-template-columns: 1fr;
        }
    }
</style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <section class="SentinelAI-header">
            <h1 class="SentinelAI-name">SentinelAI</h1>
            <div class="SentinelAI-subtitle">Deepfake image detection with confidence scoring.</div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Settings")
        model_path = Path(st.text_input("Model path", str(DEFAULT_MODEL_PATH)))
        metadata_path = Path(st.text_input("Metadata path", str(DEFAULT_METADATA_PATH)))
        threshold = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.01)

    if not model_path.exists():
        st.error(f"Model not found: {model_path}")
        st.stop()

    model = load_model(model_path)
    class_names = load_class_names(metadata_path)

    uploaded_file = st.file_uploader(
        "Image",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    if uploaded_file is None:
        st.info("Upload an image to begin.")
        return

    image = Image.open(uploaded_file)
    st.image(image, use_container_width=True)

    predicted_label, confidence, probability_real = predict_image(
        model,
        image,
        class_names,
        threshold,
    )
    result_class = "real" if predicted_label.upper() == "REAL" else "fake"
    note_class = "note-real" if predicted_label.upper() == "REAL" else "note-fake"
    note_text = f"Predicted as {predicted_label}."

    if confidence < 0.70:
        note_class = "note-uncertain"
        note_text = "Low-confidence prediction."

    st.markdown(
        f"""
        <section class="result-card">
            <div class="result-label">Prediction</div>
            <div class="result-value {result_class}">{predicted_label}</div>
            <div class="confidence-row">
                <div class="mini-card">
                    <div class="result-label">Confidence</div>
                    <div class="mini-value">{confidence * 100:.2f}%</div>
                </div>
                <div class="mini-card">
                    <div class="result-label">Probability for {class_names[1]}</div>
                    <div class="mini-value">{probability_real:.4f}</div>
                </div>
            </div>
            <div class="note {note_class}">{note_text}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.progress(probability_real)


if __name__ == "__main__":
    main()
