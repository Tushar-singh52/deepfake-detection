# CPU-Friendly CIFAKE Deepfake Detection

This project trains a binary deepfake image classifier on the CIFAKE dataset using TensorFlow/Keras and EfficientNetB0 with ImageNet pretrained weights.

The pipeline includes transfer learning, fine-tuning of the last 20 EfficientNetB0 layers, data augmentation, batch normalization, dropout, early stopping, model checkpointing, label smoothing, training plots, a confusion matrix, and a classification report.

## Project Structure

```text
DEEPFAKE_DETECT/
  data/
    CIFAKE/
      train/
        FAKE/
        REAL/
      test/
        FAKE/
        REAL/
  outputs/
  train.py
  evaluate.py
  predict.py
  app.py
  requirements.txt
  README.md
```

Keras assigns labels alphabetically by folder name. With `FAKE` and `REAL`, label `0` is `FAKE` and label `1` is `REAL`. The scripts save this class order in `outputs/metadata.json`.

## Setup in VS Code on CPU

1. Open this folder in VS Code.
2. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

4. Download CIFAKE from Kaggle and place it under:

```text
data/CIFAKE/train/REAL
data/CIFAKE/train/FAKE
data/CIFAKE/test/REAL
data/CIFAKE/test/FAKE
```

If your dataset is somewhere else, pass `--data-dir path/to/CIFAKE`.

## Train

For a CPU machine, start with a small batch size:

```powershell
python train.py --data-dir data/CIFAKE --batch-size 16
```

Useful CPU-friendly options:

```powershell
python train.py --batch-size 8 --initial-epochs 8 --fine-tune-epochs 12
```

Training writes:

```text
outputs/best_model.keras
outputs/final_model.keras
outputs/training_history.json
outputs/training_accuracy.png
outputs/training_loss.png
outputs/training_auc.png
outputs/metadata.json
```

## Evaluate

```powershell
python evaluate.py --data-dir data/CIFAKE --model-path outputs/best_model.keras
```

Evaluation writes:

```text
outputs/test_metrics.json
outputs/confusion_matrix.png
outputs/classification_report.txt
```

## Predict One Image

```powershell
python predict.py path/to/image.jpg --model-path outputs/best_model.keras
```

Example output:

```text
Prediction: REAL
Confidence: 0.9412
Probability for REAL: 0.9412
```

## Run the Frontend

After training and evaluation, start the Streamlit app:

```powershell
streamlit run app.py
```

Upload an image in the browser to get a `REAL` or `FAKE` prediction with confidence.

## Tips for Reaching 90%+ Accuracy

CIFAKE is usually achievable with EfficientNetB0, but exact accuracy depends on the split, CPU time, TensorFlow version, and training settings. These settings are a strong starting point:

```powershell
python train.py --batch-size 16 --initial-epochs 8 --fine-tune-epochs 12 --learning-rate 0.001 --fine-tune-learning-rate 0.00001 --label-smoothing 0.05
```

If validation accuracy stalls below 90%, try these changes one at a time:

- Increase `--fine-tune-epochs` to `20`.
- Use `--batch-size 8` if your CPU/RAM struggles.
- Try `--dropout 0.25` if the model underfits, or `--dropout 0.45` if validation accuracy is unstable.
- Keep fine-tuning learning rate small: `0.00001` or `0.000005`.
- Make sure the CIFAKE folders are not mixed up and that train/test directories contain both `REAL` and `FAKE`.

The best model is selected by validation accuracy and restored through early stopping. After training, always report final performance from `evaluate.py` on the held-out CIFAKE `test` folder.

## Notes

- Image size is fixed at `224x224`.
- The model uses binary classification with a sigmoid output.
- EfficientNetB0 expects standard image tensors; Keras EfficientNet includes its required preprocessing internally.
- The dataset is not cached by default to keep RAM usage lower on CPU systems. Add `--cache-dataset` only if you have enough memory.
