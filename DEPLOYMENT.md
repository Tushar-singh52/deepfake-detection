# Frontend and Deployment Guide

This project can be used as a complete deep learning app with a Streamlit frontend.

## 1. Local App

Use the local project folder that is not inside OneDrive:

```powershell
cd "C:\DeepFake-Detection"
.\.venv\Scripts\Activate.ps1
python -m pip install streamlit
streamlit run app.py
```

The browser will open at:

```text
http://localhost:8501
```

Upload a `.jpg`, `.jpeg`, `.png`, or `.webp` image. The app will show:

```text
REAL or FAKE prediction
Confidence score
Probability for REAL
Low-confidence warning when confidence is below 70%
```

## 2. Required Files for a Full Project

Keep these files in your deployable project:

```text
app.py
predict.py
evaluate.py
train.py
requirements.txt
README.md
outputs/best_model.keras
outputs/metadata.json
outputs/classification_report.txt
outputs/confusion_matrix.png
```

The dataset is not required for deployment. Deployment only needs the trained model and app files.

## 3. Recommended GitHub Structure

```text
DeepFake-Detection/
  app.py
  train.py
  evaluate.py
  predict.py
  requirements.txt
  README.md
  DEPLOYMENT.md
  outputs/
    best_model.keras
    metadata.json
    classification_report.txt
    confusion_matrix.png
```

Do not commit the full CIFAKE dataset to GitHub.

## 4. Deployment Options

### Option A: Streamlit Community Cloud

Good for simple Streamlit apps. Push the project to GitHub, then create a Streamlit app from the repository and set:

```text
Main file path: app.py
```

If the model file is too large for GitHub, store it with Git LFS or use another model hosting location.

### Option B: Hugging Face Spaces

Good for ML demo projects. Create a Streamlit Space and upload:

```text
app.py
requirements.txt
outputs/best_model.keras
outputs/metadata.json
```

Hugging Face Spaces is often convenient for resume demos because it is designed for ML applications.

### Option C: Render or Railway

Use these if you want a web-service style deployment. Streamlit can run there too, but you may need to configure the start command:

```text
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

## 5. Resume Project Description

Use a line like:

```text
Built and deployed an end-to-end deepfake image detection web app using TensorFlow/Keras, EfficientNetB0 transfer learning, and Streamlit; achieved 96.47% test accuracy on 20,000 CIFAKE test images.
```

