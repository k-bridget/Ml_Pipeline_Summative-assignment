import io
import os
import random
from locust import HttpUser, between, task
from PIL import Image


def _make_fallback_png_bytes() -> bytes:
    """Creates a lightweight valid 224x224 RGB PNG"""
    img = Image.new("RGB", (224, 224), color=(240, 244, 248))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class PhishIrisLoadUser(HttpUser):
    wait_time = between(0.2, 1.0)

    def on_start(self):
        self.sample_images = []
        search_roots = []

        env_data_dir = os.environ.get("PHISHIRIS_DATA_DIR")
        if env_data_dir:
            search_roots.append(env_data_dir)

        search_roots.extend([
            "data/test",
            "data/train",
            "data/retrain_uploads",
            "data",
        ])

        seen = set()
        for root_dir in search_roots:
            if not root_dir or not os.path.isdir(root_dir):
                continue
            for dirpath, _, filenames in os.walk(root_dir):
                for fname in filenames:
                    if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                        full_path = os.path.abspath(os.path.join(dirpath, fname))
                        if full_path not in seen:
                            seen.add(full_path)
                            self.sample_images.append(full_path)

        self.fallback_png = _make_fallback_png_bytes()

    @task(4)
    def predict_screenshot(self):
        if self.sample_images:
            img_path = random.choice(self.sample_images)
            ext = os.path.splitext(img_path)[1].lower()
            mime = "image/png" if ext == ".png" else "image/jpeg"
            with open(img_path, "rb") as f:
                self.client.post(
                    "/predict",
                    files={"file": (os.path.basename(img_path), f, mime)},
                    params={"include_features": "true"},
                    name="/predict",
                )
        else:
            self.client.post(
                "/predict",
                files={"file": ("synthetic_login.png", self.fallback_png, "image/png")},
                params={"include_features": "true"},
                name="/predict",
            )

    @task(1)
    def check_health(self):
        self.client.get("/health", name="/health")