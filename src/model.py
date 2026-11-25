import datetime
import json
import os
import shutil
from typing import Iterable, Optional, Tuple

import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers
from tensorflow.keras.applications import MobileNetV2

from src.preprocessing import BATCH_SIZE, IMG_SIZE, build_retrain_dataset

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "models", "phishiris_model.h5")
DEFAULT_UPLOAD_DIR = os.path.join(BASE_DIR, "data", "retrain_uploads")
DEFAULT_BACKUP_DIR = os.path.join(BASE_DIR, "models", "backups")


class TrueDivide(tf.keras.layers.Layer):
    """Handles legacy Keras/TF 'TrueDivide' op nodes when deserializing .h5 models."""

    def __init__(self, y=127.5, **kwargs):
        super().__init__(**kwargs)
        self.y = y

    def call(self, inputs, y=None):
        divisor = y if y is not None else self.y
        return tf.math.truediv(
            tf.cast(inputs, tf.float32),
            tf.cast(divisor, tf.float32),
        )

    def get_config(self):
        cfg = super().get_config()
        cfg["y"] = self.y
        return cfg


def build_base_model(
    input_shape: Tuple[int, int, int] = IMG_SIZE + (3,),
    dropout_rate: float = 0.35,
    l2_reg: float = 1e-4,
) -> tf.keras.Model:
    """Builds a clean MobileNetV2 binary phishing screenshot classifier."""
    base_cnn = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
        name="mobilenetv2_1.00_224",
    )
    base_cnn.trainable = False

    inputs = layers.Input(shape=input_shape, name="screenshot_input")
    x = base_cnn(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="global_average_pooling2d")(x)
    x = layers.BatchNormalization(name="batch_normalization")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)
    x = layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=regularizers.l2(l2_reg),
        name="dense",
    )(x)
    x = layers.Dropout(dropout_rate * 0.5, name="dropout_1")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="phishing_prob")(x)

    return models.Model(inputs=inputs, outputs=outputs, name="phishiris_mobilenetv2")


def build_colab_compatible_model(
    input_shape: Tuple[int, int, int] = IMG_SIZE + (3,),
    dropout_rate: float = 0.35,
    l2_reg: float = 1e-4,
) -> tf.keras.Model:
    """Reconstructs the Colab MobileNetV2 preprocessing + head graph so weights from
    phishiris_model.h5 map cleanly by name.
    """
    base_cnn = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights=None,
        name="mobilenetv2_1.00_224",
    )
    base_cnn.trainable = False

    inputs = layers.Input(shape=input_shape, name="screenshot_input")
    x = TrueDivide(127.5, name="true_divide_norm")(inputs)
    x = layers.Subtract(name="subtract_one")([x, tf.ones_like(x)])
    x = base_cnn(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_average_pooling2d")(x)
    x = layers.BatchNormalization(name="batch_normalization")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)
    x = layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=regularizers.l2(l2_reg),
        name="dense",
    )(x)
    x = layers.Dropout(dropout_rate * 0.5, name="dropout_1")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="phishing_prob")(x)

    return models.Model(inputs=inputs, outputs=outputs, name="phishiris_mobilenetv2")


def _compile_model(
    model: tf.keras.Model, learning_rate: float = 1e-4
) -> tf.keras.Model:
    """Compiles the model with Adam optimizer, binary cross-entropy loss, and core metrics."""
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def _load_weights_by_name_from_h5(model: tf.keras.Model, h5_path: str) -> int:
    """Loads matching datasets from `model_weights/...` inside the HDF5 file by name & shape."""
    datasets = {}

    def _collect(name, obj):
        if isinstance(obj, h5py.Dataset) and name.startswith("model_weights/"):
            datasets[name] = np.array(obj)

    with h5py.File(h5_path, "r") as f:
        f.visititems(_collect)

    matched = 0
    for weight in model.weights:
        w_name = weight.name.split(":")[0]
        w_parts = w_name.split("/")
        w_leaf = w_parts[-1]
        w_group = w_parts[-2] if len(w_parts) >= 2 else ""
        w_shp = tuple(weight.shape)

        for h5_key, arr in list(datasets.items()):
            if arr is None or tuple(arr.shape) != w_shp:
                continue
            if h5_key.endswith("/" + w_leaf) and (not w_group or w_group in h5_key):
                weight.assign(arr)
                datasets[h5_key] = None
                matched += 1
                break
    return matched


def load_trained_model(
    model_path: str = DEFAULT_MODEL_PATH,
    learning_rate: float = 1e-4,
) -> tf.keras.Model:
    """Loads the saved .h5 model with custom_objects or exact weight-name mapping."""
    if os.path.exists(model_path):
        try:
            loaded = tf.keras.models.load_model(
                model_path,
                compile=False,
                custom_objects={"TrueDivide": TrueDivide},
            )
            return _compile_model(loaded, learning_rate=learning_rate)
        except Exception as exc:
            print(
                f"Direct load_model fallback ({exc}); "
                "loading weights by name into compatible graph..."
            )
            rebuilt = build_colab_compatible_model()
            count = _load_weights_by_name_from_h5(rebuilt, model_path)
            print(
                f"Transferred {count} named weight tensors from {os.path.basename(model_path)}."
            )
            return _compile_model(rebuilt, learning_rate=learning_rate)

    fresh = build_base_model()
    return _compile_model(fresh, learning_rate=learning_rate)


def _set_fine_tune_layers(
    model: tf.keras.Model, fine_tune_layers: int = 20
) -> None:
    """Unfreezes the top `fine_tune_layers` inside any nested MobileNetV2 backbone."""
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) or "mobilenet" in layer.name.lower():
            layer.trainable = True
            if fine_tune_layers > 0 and len(layer.layers) > fine_tune_layers:
                for sub_layer in layer.layers[:-fine_tune_layers]:
                    sub_layer.trainable = False
            else:
                for sub_layer in layer.layers:
                    sub_layer.trainable = False


def save_retrain_uploads(
    saved_file_paths: Iterable[str],
    class_label: str,
    upload_root: str = DEFAULT_UPLOAD_DIR,
) -> dict:
    """Persists uploaded image files into `upload_root/<class_label>/` and appends
    metadata to `retrain_log.jsonl` for retraining traceability.
    """
    normalized_label = class_label.strip().lower()
    if normalized_label in ("0", "legitimate", "other"):
        folder_label = "legitimate"
        numeric_label = 0
    elif normalized_label in ("1", "phishing"):
        folder_label = "phishing"
        numeric_label = 1
    else:
        raise ValueError(
            f"Unsupported class_label '{class_label}'. Use 'legitimate' or 'phishing'."
        )

    target_dir = os.path.join(upload_root, folder_label)
    os.makedirs(target_dir, exist_ok=True)

    stored_files = []
    for src_path in saved_file_paths:
        if not os.path.isfile(src_path):
            continue
        fname = os.path.basename(src_path)
        dst_path = os.path.join(target_dir, fname)
        if os.path.abspath(src_path) != os.path.abspath(dst_path):
            shutil.copy2(src_path, dst_path)
        stored_files.append(dst_path)

    log_entry = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "class_label": folder_label,
        "numeric_label": numeric_label,
        "file_count": len(stored_files),
        "files": stored_files,
    }
    os.makedirs(upload_root, exist_ok=True)
    log_path = os.path.join(upload_root, "retrain_log.jsonl")
    with open(log_path, "a", encoding="utf-8") as lf:
        lf.write(json.dumps(log_entry) + "\n")

    return log_entry


def retrain_model(
    upload_dir: str = DEFAULT_UPLOAD_DIR,
    model_path: str = DEFAULT_MODEL_PATH,
    epochs: int = 3,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = 5e-5,
    fine_tune_layers: int = 20,
    backup_dir: str = DEFAULT_BACKUP_DIR,
) -> dict:
    """Preprocesses uploaded retraining data, fine-tunes the existing model,
    backs up the previous weights, and saves the updated model to `model_path`.
    """
    train_ds, sample_count = build_retrain_dataset(upload_dir, batch_size=batch_size)

    model = load_trained_model(model_path=model_path, learning_rate=learning_rate)
    if fine_tune_layers > 0:
        _set_fine_tune_layers(model, fine_tune_layers=fine_tune_layers)
        model = _compile_model(model, learning_rate=learning_rate)

    backup_path: Optional[str] = None
    if os.path.exists(model_path):
        os.makedirs(backup_dir, exist_ok=True)
        stamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(backup_dir, f"phishiris_model_{stamp}.h5")
        shutil.copy2(model_path, backup_path)

    history = model.fit(train_ds, epochs=epochs, verbose=1)

    os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
    model.save(model_path)

    last_metrics = {
        key: float(vals[-1]) for key, vals in history.history.items() if len(vals) > 0
    }

    return {
        "status": "retrained",
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "upload_dir": upload_dir,
        "unique_samples_used": int(sample_count),
        "epochs": int(epochs),
        "model_path": model_path,
        "backup_path": backup_path,
        "metrics": last_metrics,
    }


if __name__ == "__main__":
    m = load_trained_model()
    print("Loaded model successfully:", m.name, "| Input shape:", m.input_shape)