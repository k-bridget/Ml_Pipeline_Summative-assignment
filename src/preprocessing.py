import hashlib
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = False

IMG_SIZE = (224, 224)  # (height, width)
BATCH_SIZE = 32
IMAGE_EXTS = (".jpg", ".jpeg", ".png")

def load_rgb_image(src, size=IMG_SIZE) -> Image.Image:
    """Open an image and return an RGB PIL image resized to `size` (h, w). """
    im = Image.open(src)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[-1])
        im = bg
    else:
        im = im.convert("RGB")
    return im.resize((size[1], size[0]), Image.LANCZOS)


def validate_and_preprocess_image(image_path: str) -> np.ndarray:
    """Validates an image file and returns a (1, 224, 224, 3) float32 array
    with RAW pixel values (0-255)."""
    with Image.open(image_path) as im:
        im.verify()
    arr = np.asarray(load_rgb_image(image_path), dtype=np.float32)
    return np.expand_dims(arr, axis=0)


def extract_image_features(image_path: str) -> dict:
    """Extracts visual/structural features from a webpage screenshot."""
    with Image.open(image_path) as im:
        width, height = im.size
    rgb = load_rgb_image(image_path, size=(128, 128))
    arr = np.asarray(rgb, dtype=np.float32) / 255.0

    aspect_ratio = float(width / height) if height else 0.0
    brightness = float(arr.mean())
    contrast = float(arr.std())
    r_mean = float(arr[:, :, 0].mean())
    g_mean = float(arr[:, :, 1].mean())
    b_mean = float(arr[:, :, 2].mean())
    blue_dominance = b_mean - r_mean

    gray = arr.mean(axis=2)
    grad_x = np.abs(np.diff(gray, axis=1)).mean()
    grad_y = np.abs(np.diff(gray, axis=0)).mean()
    edge_density = float((grad_x + grad_y) / 2.0)

    return {
        "width": width,
        "height": height,
        "aspect_ratio": round(aspect_ratio, 4),
        "brightness": round(brightness, 4),
        "contrast": round(contrast, 4),
        "blue_dominance": round(blue_dominance, 4),
        "edge_density": round(edge_density, 4),
    }


def build_retrain_dataset(upload_dir: str, batch_size: int = BATCH_SIZE):
    """Scans uploaded class folders, removes duplicates"""
    if not os.path.isdir(upload_dir):
        raise FileNotFoundError(f"Retraining directory does not exist: {upload_dir}")

    records = []
    seen_hashes = set()

    for folder_name in sorted(os.listdir(upload_dir)):
        folder_path = os.path.join(upload_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        label = 0 if folder_name.lower() in ("other", "legitimate", "0") else 1

        for root, _, files in os.walk(folder_path):
            for fname in sorted(files):
                if not fname.lower().endswith(IMAGE_EXTS):
                    continue
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, "rb") as fh:
                        md5 = hashlib.md5(fh.read()).hexdigest()
                    if md5 in seen_hashes:
                        continue
                    with Image.open(fpath) as im:
                        im.verify()
                    seen_hashes.add(md5)
                    records.append({"path": fpath, "label": label})
                except Exception:
                    continue

    if not records:
        raise ValueError("No valid images found")

    df = pd.DataFrame(records)
    paths = df["path"].values
    labels = df["label"].values.astype(np.float32)

    def _load_py(path_bytes):
        path = path_bytes.decode("utf-8") if isinstance(path_bytes, bytes) else str(path_bytes)
        return np.asarray(load_rgb_image(path), dtype=np.float32)

    def _load(path, label):
        img = tf.numpy_function(_load_py, [path], tf.float32)
        img.set_shape([IMG_SIZE[0], IMG_SIZE[1], 3])
        return img, label  # raw 0-255, same as training

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.shuffle(len(df), seed=42).map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, len(df)


def resolve_data_dirs(base_dir: str):
    """Resolves train and test"""
    data_root = os.environ.get("PHISHIRIS_DATA_DIR", os.path.join(base_dir, "data"))
    train_dir = os.path.join(data_root, "train")
    test_dir = os.path.join(data_root, "test")
    return data_root, train_dir, test_dir


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_ROOT, TRAIN_DIR, TEST_DIR = resolve_data_dirs(BASE_DIR)

    print("\n PHISH-IRIS PREPROCESSING ")
    print(f"Using data root: {DATA_ROOT}")

    for dataset_name, dataset_dir in [("TRAINING", TRAIN_DIR), ("TESTING", TEST_DIR)]:
        print(f"\nProcessing {dataset_name} dataset...")

        for class_name in ["legitimate", "phishing"]:
            class_dir = os.path.join(dataset_dir, class_name)

            if not os.path.isdir(class_dir):
                print(f"WARNING: Missing directory: {class_dir}")
                continue

            image_files = [
                f for f in os.listdir(class_dir) if f.lower().endswith(IMAGE_EXTS)
            ]

            valid_count = 0
            invalid_count = 0

            for filename in image_files:
                image_path = os.path.join(class_dir, filename)
                try:
                    validate_and_preprocess_image(image_path)
                    valid_count += 1
                except Exception as e:
                    invalid_count += 1
                    print(f"Invalid image: {filename} - {e}")

            print(f"\nClass: {class_name}")
            print(f"Total images: {len(image_files)}")
            print(f"Valid images: {valid_count}")
            print(f"Invalid images: {invalid_count}")

    print("\n PREPROCESSING COMPLETED")