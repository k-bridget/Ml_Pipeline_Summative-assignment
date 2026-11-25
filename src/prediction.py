import os
from typing import Optional

import numpy as np
import tensorflow as tf

from src.model import DEFAULT_MODEL_PATH, load_trained_model
from src.preprocessing import extract_image_features, validate_and_preprocess_image


class PhishingPredictor:
    """Wraps the trained Phish-IRIS model for single-image inference."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, threshold: float = 0.5):
        self.model_path = model_path
        self.threshold = float(threshold)
        self.model: Optional[tf.keras.Model] = None
        self.reload()

    def reload(self) -> None:
        """Reloads model weights from disk."""
        self.model = load_trained_model(self.model_path)

    def predict(self, image_path: str, include_features: bool = True) -> dict:
        """Runs validation, preprocessing, and binary classification on a single image."""
        if self.model is None:
            self.reload()

        batch_tensor = validate_and_preprocess_image(image_path)
        prob = float(self.model.predict(batch_tensor, verbose=0)[0][0])
        is_phishing = prob >= self.threshold
        label_id = 1 if is_phishing else 0
        label_name = "Phishing" if is_phishing else "Legitimate"
        confidence = prob if is_phishing else (1.0 - prob)

        result = {
            "prediction": label_name,
            "label": label_id,
            "phishing_probability": round(prob, 4),
            "confidence": round(confidence, 4),
            "threshold": self.threshold,
        }
        if include_features:
            result["features"] = extract_image_features(image_path)
        return result


_DEFAULT_PREDICTOR: Optional[PhishingPredictor] = None


def predict_image(
    image_path: str,
    model_path: str = DEFAULT_MODEL_PATH,
    threshold: float = 0.5,
    include_features: bool = True,
) -> dict:
    global _DEFAULT_PREDICTOR
    if _DEFAULT_PREDICTOR is None or _DEFAULT_PREDICTOR.model_path != model_path:
        _DEFAULT_PREDICTOR = PhishingPredictor(model_path=model_path, threshold=threshold)
    _DEFAULT_PREDICTOR.threshold = float(threshold)
    return _DEFAULT_PREDICTOR.predict(image_path=image_path, include_features=include_features)


if __name__ == "__main__":
    sample_root = os.environ.get(
        "PHISHIRIS_DATA_DIR",
        r"C:\Users\Administrator\Desktop\ALU-projects\phishiris_prepared",
    )
    sample_dir = os.path.join(sample_root, "test", "phishing")
    if os.path.isdir(sample_dir):
        candidates = [
            os.path.join(sample_dir, f)
            for f in sorted(os.listdir(sample_dir))
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        if candidates:
            print("Testing prediction on:", candidates[0])
            print(predict_image(candidates[0]))
        else:
            print("No images found in:", sample_dir)
    else:
        print("Sample directory not found:", sample_dir)