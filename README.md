# Webpage Phishing Detection

An end-to-end machine learning application that identifies whether a webpage screenshot is **phishing** or **legitimate**. The project combines a MobileNetV2 image classifier, a FastAPI inference/retraining backend, and a Streamlit web interface.

**Live App URL:** https://phishing-recognition-app.politesea-e9169312.uaenorth.azurecontainerapps.io/

**Repository:** https://github.com/k-bridget/Ml_Pipeline_Summative-assignment

## Features

- Upload a screenshot and receive a prediction with confidence.
- Inspect extracted features such as aspect ratio, brightness, contrast, edge density, and blue dominance.
- View dataset and evaluation visualizations.
- Upload multiple screenshots and retrain the active model.
- Automatically back up the existing model before retraining.
- Persist uploaded-file metadata and retraining history in SQLite.
- Check API and model status through a health endpoint.
- Run HTTP load tests with Locust.

## How It Works

The Phish-IRIS dataset contains 15 webpage-brand classes. The project maps the 14 branded/phishing classes to label `1` and the `other`/legitimate class to label `0`.

Images are converted to RGB, composited over a white background when transparency is present, resized to `224 x 224`, and passed to MobileNetV2 with pixel values in the `0-255` range. The model applies internal rescaling to `-1 to 1`.

The classifier uses a sigmoid output with a `0.5` decision threshold:

| Phishing probability | Prediction |
|---:|---|
| >= 0.50 | Phishing |
| < 0.50 | Legitimate |

## Technology Stack

- **TensorFlow / Keras:** MobileNetV2 transfer learning
- **FastAPI:** Prediction, upload, retraining, and health endpoints
- **Streamlit:** User Interface
- **Pillow / OpenCV:** Image validation and preprocessing
- **Locust:** Load testing
- **Docker:** Containerized deployment
- **Azure Container Apps:** Hosted deployment

## Project Structure

```text
.
├── app/
│   └── app.py                  
├── src/
│   ├── api.py                  
│   ├── model.py                
│   ├── prediction.py           
│   └── preprocessing.py         
├── models/
│   ├── phishiris_model.keras   
│   ├── phishiris.weights.h5    
│   └── backups/                
├── data/
│   ├── raw/phishiris/          
│   └── retrain_uploads/        
├── notebook/
│   ├── Ml_Pipeline_Summative_assignment.ipynb
│   ├── feature_visualizations.png
│   └── evaluation_metrics.png
├── results/                    
├── Dockerfile
├── start.sh                    
├── locust.py                   
└── requirements.txt
```

## Dataset

The source dataset is [Phish-IRIS on Kaggle](https://www.kaggle.com/datasets/saurabhshahane/phishiris).

Preprocessing and data-quality checks include:

- Conversion of RGBA, LA, and palette images to RGB on a white background.
- Resizing to `224 x 224` with Lanczos interpolation.
- Corrupt-image validation and duplicate detection.
- Stratified train/validation/test splitting for the original training workflow.
- Training-only augmentation and class weighting.
- A shared RGB image loader for training, prediction, and retraining.

## Running Locally

### Prerequisites

- Python 3.11
- A virtual environment
- The model files under `models/`

### 1. Install dependencies

Windows PowerShell:

```powershell
python -m venv312 .venv321
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Start the FastAPI backend

```powershell
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

FastAPI documentation is available at:

- Swagger UI: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health

### 3. Start Streamlit

Open a second terminal and activate the same virtual environment:

```powershell
streamlit run app/app.py
```

The interface is available at http://localhost:8501.

```

## Docker

Build and run the complete application:

```powershell
docker build -t phishiris-app .
docker run --rm -p 8000:8000 -p 8501:8501 phishiris-app
```

- API: http://localhost:8000
- Streamlit: http://localhost:8501

The container starts both services with `start.sh`.

Use a Docker volume if uploaded files and database records must survive container restarts:

```powershell
docker run --rm -p 8000:8000 -p 8501:8501 `
  -v phishiris-data:/app/data/retrain_uploads `
  phishiris-app
```

### Docker environment variables

| Variable | Default | Purpose |
|---|---|---|
| `PHISHIRIS_API_URL` | `http://127.0.0.1:8000` | Streamlit API address |
| `PHISHIRIS_MODEL_PATH` | `models/phishiris_model.keras` | Active `.keras` model |
| `PHISHIRIS_UPLOAD_DIR` | `data/retrain_uploads` | Retraining image directory |
| `PHISHIRIS_DATABASE_PATH` | `<PHISHIRIS_UPLOAD_DIR>/phishiris.db` | SQLite database location |

## API Reference

### `GET /health`

Returns service uptime and model-loading status.

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_path": "models/phishiris_model.keras",
  "uptime_seconds": 42.1
}
```

### `GET /database/status`

Returns database counts, upload totals by class, recent upload batches, and recent retraining runs.

### `POST /predict`

Accepts a JPG, JPEG, or PNG file. `include_features` defaults to `true`.

PowerShell example:

```powershell
curl.exe -F "file=@C:\path\to\screenshot.png" `
  "http://127.0.0.1:8000/predict?include_features=true"
```

Example response:

```json
{
  "prediction": "Phishing",
  "label": 1,
  "phishing_probability": 0.9432,
  "confidence": 0.9432,
  "threshold": 0.5,
  "filename": "screenshot.png",
  "features": {
    "width": 1365,
    "height": 768,
    "aspect_ratio": 1.7773,
    "brightness": 0.7614,
    "contrast": 0.1982,
    "blue_dominance": -0.012,
    "edge_density": 0.0841
  }
}
```

### `POST /upload-retrain-data`

Uploads multiple images and assigns them to `phishing` or `legitimate`.

```powershell
curl.exe -F "class_label=phishing" `
  -F "files=@C:\path\to\one.png" `
  -F "files=@C:\path\to\two.jpg" `
  "http://127.0.0.1:8000/upload-retrain-data"
```

Accepted labels are `phishing`, `legitimate`, `0`, and `other`.

### `POST /retrain`

Trains on data in the retraining upload directory, backs up the active model, saves the updated model, and reloads it.

```powershell
curl.exe -X POST `
  "http://127.0.0.1:8000/retrain?epochs=3&learning_rate=0.00005&fine_tune_layers=20"
```

Retraining:

- Requires at least one valid, unique uploaded image.
- Removes duplicate files by MD5 hash.
- Uses up to 20 unfrozen backbone layers by default.
- Backs up the previous model to `models/backups/`.
- Reuses the same `224 x 224` preprocessing pipeline used for inference.

### SQLite database

The application stores retraining metadata in `data/retrain_uploads/phishiris.db`. The database contains:

- `retrain_upload_batches`: each bulk-upload operation and its class.
- `retrain_upload_files`: saved filenames, paths, class labels, MD5 hashes, and sizes.
- `retraining_runs`: successful and failed retraining attempts, parameters, metrics, backups, and model paths.

Use `GET /database/status` or the database panel in the Streamlit retraining tab to inspect the stored records.

## Load Testing

Start the API first, then run Locust from the project directory:

```powershell
locust -f locust.py --host http://127.0.0.1:8000
```

Open http://localhost:8089 to configure users and spawn a load test. Locust alternates prediction requests with health checks and will use local images when available, falling back to a generated screenshot.

## Pipeline Modules

The preprocessing, model, and prediction modules can also be invoked directly:

```powershell
python -m src.preprocessing
python -m src.model
python -m src.prediction
```

`src.preprocessing` validates the expected `data/train` and `data/test` layout. `src.prediction` may need an existing test-image directory to demonstrate inference from the command line.

## Model Details

- Base: MobileNetV2 pretrained on ImageNet
- Input: `224 x 224 x 3`
- Head: global average pooling, batch normalization, dense layer with 128 units, dropout, and sigmoid output
- Loss: binary cross-entropy
- Primary optimizer: Adam
- Metrics: accuracy, precision, and recall
- Regularization: L2 and dropout
