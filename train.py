import argparse
import json
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


IMAGE_SIZE = (224, 224)
DEFAULT_SEED = 42
AUTOTUNE = tf.data.AUTOTUNE


def parse_args():
    parser = argparse.ArgumentParser(description="Train an EfficientNetB0 CIFAKE detector.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/CIFAKE"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--initial-epochs", type=int, default=8)
    parser.add_argument("--fine-tune-epochs", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--fine-tune-learning-rate", type=float, default=1e-5)
    parser.add_argument("--validation-split", type=float, default=0.15)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--cache-dataset", action="store_true")
    return parser.parse_args()


def set_reproducibility(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def configure_cpu():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    try:
        tf.config.threading.set_inter_op_parallelism_threads(2)
        tf.config.threading.set_intra_op_parallelism_threads(max(2, os.cpu_count() or 2))
    except RuntimeError:
        pass


def get_train_val_datasets(data_dir, batch_size, validation_split, seed, cache_dataset):
    train_dir = data_dir / "train"
    if not train_dir.exists():
        raise FileNotFoundError(
            f"Could not find {train_dir}. Expected CIFAKE folders like "
            "data/CIFAKE/train/REAL and data/CIFAKE/train/FAKE."
        )

    train_ds = keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="binary",
        validation_split=validation_split,
        subset="training",
        seed=seed,
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        shuffle=True,
    )
    val_ds = keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="binary",
        validation_split=validation_split,
        subset="validation",
        seed=seed,
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        shuffle=False,
    )

    class_names = train_ds.class_names
    if cache_dataset:
        train_ds = train_ds.cache()
        val_ds = val_ds.cache()

    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    return train_ds, val_ds, class_names


def build_augmentation():
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.12),
            layers.RandomContrast(0.12),
            layers.RandomBrightness(0.10),
        ],
        name="data_augmentation",
    )


def build_model(dropout):
    inputs = keras.Input(shape=(*IMAGE_SIZE, 3), name="image")
    x = build_augmentation()(inputs)

    base_model = keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_tensor=x,
        pooling=None,
    )
    base_model.trainable = False

    x = base_model.output
    x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x = layers.BatchNormalization(name="head_batch_norm")(x)
    x = layers.Dropout(dropout, name="head_dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="real_fake")(x)

    model = keras.Model(inputs, outputs, name="cifake_efficientnetb0")
    return model, base_model


def compile_model(model, learning_rate, label_smoothing):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing),
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )


def get_callbacks(output_dir, monitor="val_accuracy", initial_value_threshold=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    return [
        keras.callbacks.ModelCheckpoint(
            filepath=output_dir / "best_model.keras",
            monitor=monitor,
            mode="max",
            save_best_only=True,
            initial_value_threshold=initial_value_threshold,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor=monitor,
            mode="max",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
    ]


def fine_tune_last_layers(base_model, layers_to_unfreeze=20):
    base_model.trainable = True
    for layer in base_model.layers[:-layers_to_unfreeze]:
        layer.trainable = False

    for layer in base_model.layers[-layers_to_unfreeze:]:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False


def merge_histories(*histories):
    merged = {}
    for history in histories:
        for key, values in history.history.items():
            merged.setdefault(key, []).extend(values)
    return merged


def save_training_plots(history, output_dir):
    plot_specs = [
        ("accuracy", "val_accuracy", "Accuracy", "training_accuracy.png"),
        ("loss", "val_loss", "Loss", "training_loss.png"),
        ("auc", "val_auc", "AUC", "training_auc.png"),
    ]

    for train_key, val_key, title, filename in plot_specs:
        if train_key not in history:
            continue
        plt.figure(figsize=(8, 5))
        plt.plot(history[train_key], label=f"train_{train_key}")
        if val_key in history:
            plt.plot(history[val_key], label=val_key)
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(title)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=160)
        plt.close()


def save_metadata(output_dir, class_names, args):
    metadata = {
        "class_names": class_names,
        "image_size": IMAGE_SIZE,
        "model": "EfficientNetB0",
        "weights": "imagenet",
        "positive_class_index": 1,
        "args": vars(args),
    }
    metadata["args"]["data_dir"] = str(metadata["args"]["data_dir"])
    metadata["args"]["output_dir"] = str(metadata["args"]["output_dir"])
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)


def main():
    args = parse_args()
    configure_cpu()
    set_reproducibility(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, class_names = get_train_val_datasets(
        args.data_dir,
        args.batch_size,
        args.validation_split,
        args.seed,
        args.cache_dataset,
    )
    print(f"Class order: {class_names}. Keras label 1 maps to {class_names[1]}.")

    model, base_model = build_model(args.dropout)
    compile_model(model, args.learning_rate, args.label_smoothing)

    print("Stage 1: transfer learning with frozen EfficientNetB0.")
    initial_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.initial_epochs,
        callbacks=get_callbacks(args.output_dir),
    )
    best_stage_one_accuracy = max(initial_history.history.get("val_accuracy", [0.0]))

    print("Stage 2: fine-tuning the last 20 EfficientNetB0 layers.")
    fine_tune_last_layers(base_model, layers_to_unfreeze=20)
    compile_model(model, args.fine_tune_learning_rate, args.label_smoothing)
    fine_tune_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.initial_epochs + args.fine_tune_epochs,
        initial_epoch=len(initial_history.history["loss"]),
        callbacks=get_callbacks(
            args.output_dir,
            initial_value_threshold=best_stage_one_accuracy,
        ),
    )

    history = merge_histories(initial_history, fine_tune_history)
    with open(args.output_dir / "training_history.json", "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)
    save_training_plots(history, args.output_dir)
    save_metadata(args.output_dir, class_names, args)

    model.save(args.output_dir / "final_model.keras")
    print(f"Saved best model to {args.output_dir / 'best_model.keras'}")
    print(f"Saved final model to {args.output_dir / 'final_model.keras'}")


if __name__ == "__main__":
    main()
