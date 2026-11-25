import os
import shutil
import tempfile
import time
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.model import (
    DEFAULT_MODEL_PATH,
    DEFAULT_UPLOAD_DIR,
    retrain_model,
    save_retrain_uploads,
)
from src.prediction import PhishingPredictor
from src.preprocessing import IMAGE_EXTS

START_TIME = time.time()
MODEL_PATH = os.environ.get("PHISHIRIS_MODEL_PATH", DEFAULT_MODEL_PATH)
UPLOAD_DIR = os.environ.get("PHISHIRIS_UPLOAD_DIR", DEFAULT_UPLOAD_DIR)

app = FastAPI(
    title="Phishing Detection",
    version="1.0.0",
    description="Prediction, bulk upload, and retraining API for webpage screenshot classification.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = PhishingPredictor(model_path=MODEL_PATH, threshold=0.5)


@app.get("/health")
def health():
    uptime_seconds = round(time.time() - START_TIME, 2)
    return {
        "status": "ok",
        "model_loaded": predictor.model is not None,
        "model_path": MODEL_PATH,
        "uptime_seconds": uptime_seconds,
    }


@app.post("/predict")
async def predict_endpoint(
    file: UploadFile = File(...),
    include_features: bool = True,
):
    if not file.filename or not file.filename.lower().endswith(IMAGE_EXTS):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed extensions: {IMAGE_EXTS}",
        )

    suffix = os.path.splitext(file.filename)[1].lower()
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        result = predictor.predict(tmp_path, include_features=include_features)
        result["filename"] = file.filename
        return result
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Prediction failed: {exc}") from exc
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/upload-retrain-data")
async def upload_retrain_data(
    class_label: str = Form(...),
    files: List[UploadFile] = File(...),
):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided for upload.")

    temp_paths = []
    tmp_dir = tempfile.mkdtemp(prefix="phish_upload_")
    try:
        for up in files:
            if not up.filename:
                continue
            ext = os.path.splitext(up.filename)[1].lower()
            if ext not in IMAGE_EXTS:
                continue
            dst = os.path.join(tmp_dir, os.path.basename(up.filename))
            with open(dst, "wb") as out_f:
                shutil.copyfileobj(up.file, out_f)
            temp_paths.append(dst)

        if not temp_paths:
            raise HTTPException(
                status_code=400,
                detail=f"No valid image files ({IMAGE_EXTS}) found in upload.",
            )

        log_entry = save_retrain_uploads(
            saved_file_paths=temp_paths,
            class_label=class_label,
            upload_root=UPLOAD_DIR,
        )
        return {
            "status": "uploaded",
            "saved_count": log_entry["file_count"],
            "class_label": log_entry["class_label"],
            "upload_dir": UPLOAD_DIR,
            "timestamp": log_entry["timestamp"],
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/retrain")
def retrain_endpoint(
    epochs: int = 3,
    learning_rate: float = 5e-5,
    fine_tune_layers: int = 20,
):
    try:
        summary = retrain_model(
            upload_dir=UPLOAD_DIR,
            model_path=MODEL_PATH,
            epochs=epochs,
            learning_rate=learning_rate,
            fine_tune_layers=fine_tune_layers,
        )
        predictor.reload()
        return summary
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Retraining failed: {exc}") from exc