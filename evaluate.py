import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras


IMAGE_SIZE = (224, 224)
AUTOTUNE = tf.data.AUTOTUNE


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained CIFAKE detector.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/CIFAKE"))
    parser.add_argument("--model-path", type=Path, default=Path("outputs/best_model.keras"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def load_class_names(output_dir, fallback):
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as file:
            return json.load(file).get("class_names", fallback)
    return fallback


def get_test_dataset(data_dir, batch_size):
    test_dir = data_dir / "test"
    if not test_dir.exists():
        raise FileNotFoundError(
            f"Could not find {test_dir}. Expected CIFAKE folders like "
            "data/CIFAKE/test/REAL and data/CIFAKE/test/FAKE."
        )

    test_ds = keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="binary",
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        shuffle=False,
    )
    class_names = test_ds.class_names
    return test_ds.prefetch(AUTOTUNE), class_names


def collect_predictions(model, dataset, threshold):
    y_true_batches = []
    y_prob_batches = []
    for images, labels in dataset:
        y_prob = model.predict(images, verbose=0).ravel()
        y_true_batches.append(labels.numpy().ravel())
        y_prob_batches.append(y_prob)

    y_true = np.concatenate(y_true_batches).astype(int)
    y_prob = np.concatenate(y_prob_batches)
    y_pred = (y_prob >= threshold).astype(int)
    return y_true, y_prob, y_pred


def save_confusion_matrix(cm, class_names, output_dir):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=35, ha="right")
    plt.yticks(tick_marks, class_names)

    threshold = cm.max() / 2.0 if cm.size else 0
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            plt.text(
                col,
                row,
                format(cm[row, col], "d"),
                ha="center",
                va="center",
                color="white" if cm[row, col] > threshold else "black",
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=180)
    plt.close()


def compute_confusion_matrix(y_true, y_pred):
    cm = np.zeros((2, 2), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        cm[int(true_label), int(pred_label)] += 1
    return cm


def build_classification_report(cm, class_names):
    lines = [
        "Classification Report",
        "",
        f"{'class':>12} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}",
    ]

    precisions = []
    recalls = []
    f1_scores = []
    supports = []

    for idx, class_name in enumerate(class_names):
        tp = cm[idx, idx]
        fp = cm[:, idx].sum() - tp
        fn = cm[idx, :].sum() - tp
        support = cm[idx, :].sum()

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        supports.append(support)

        lines.append(
            f"{class_name:>12} {precision:>10.4f} {recall:>10.4f} "
            f"{f1:>10.4f} {support:>10d}"
        )

    total = int(sum(supports))
    accuracy = np.trace(cm) / total if total else 0.0
    macro_precision = float(np.mean(precisions))
    macro_recall = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1_scores))
    weighted_precision = float(np.average(precisions, weights=supports)) if total else 0.0
    weighted_recall = float(np.average(recalls, weights=supports)) if total else 0.0
    weighted_f1 = float(np.average(f1_scores, weights=supports)) if total else 0.0

    lines.extend(
        [
            "",
            f"{'accuracy':>12} {'':>10} {'':>10} {accuracy:>10.4f} {total:>10d}",
            f"{'macro avg':>12} {macro_precision:>10.4f} {macro_recall:>10.4f} "
            f"{macro_f1:>10.4f} {total:>10d}",
            f"{'weighted avg':>12} {weighted_precision:>10.4f} {weighted_recall:>10.4f} "
            f"{weighted_f1:>10.4f} {total:>10d}",
        ]
    )
    return "\n".join(lines)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    test_ds, inferred_class_names = get_test_dataset(args.data_dir, args.batch_size)
    class_names = load_class_names(args.output_dir, inferred_class_names)
    model = keras.models.load_model(args.model_path)

    print("Evaluating Keras metrics...")
    metrics = model.evaluate(test_ds, verbose=1, return_dict=True)
    with open(args.output_dir / "test_metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    y_true, _, y_pred = collect_predictions(model, test_ds, args.threshold)
    cm = compute_confusion_matrix(y_true, y_pred)
    report = build_classification_report(cm, class_names)

    save_confusion_matrix(cm, class_names, args.output_dir)
    with open(args.output_dir / "classification_report.txt", "w", encoding="utf-8") as file:
        file.write(report)

    print(report)
    print(f"Saved confusion matrix to {args.output_dir / 'confusion_matrix.png'}")
    print(f"Saved classification report to {args.output_dir / 'classification_report.txt'}")


if __name__ == "__main__":
    main()
