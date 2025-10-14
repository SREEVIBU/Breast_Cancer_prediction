# Breast Cancer Prediction API

This project provides a simple Flask-based API to serve a trained SVM model for breast cancer prediction.

Files
- `app.py` - Flask application that loads `breast_cancer.pkl` and exposes endpoints:
  - `GET /` - health check and model status
  - `POST /predict` - accepts JSON with features and returns prediction
- `breast_cancer.pkl` - expected pickled model file (dictionary with keys `model`, `scaler`, optionally `label_encoder` and `feature_names`).
- `data.csv` - dataset used for training (kept for reference)

Setup
1. Create and activate a Python virtual environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run the API

```powershell
python .\app.py
```

The app listens on port 5000 by default. You can change the host/port using environment variables:

```powershell
$env:APP_HOST = '127.0.0.1'
$env:APP_PORT = '8080'
$env:APP_DEBUG = 'False'
python .\app.py
```

Example requests (PowerShell)

- Health check:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:5000/
```

- Predict (features as list):

```powershell
$payload = @{ features = @(14.2, 20.3, 90.1, 600.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6) }
Invoke-RestMethod -Uri http://127.0.0.1:5000/predict -Method POST -Body (ConvertTo-Json $payload) -ContentType 'application/json'
```

- Predict (features as dict) — only if `feature_names` were saved into the pickle:

```powershell
$payload = @{ features = @{ radius_mean = 14.2; texture_mean = 20.3; /* ... */ } }
Invoke-RestMethod -Uri http://127.0.0.1:5000/predict -Method POST -Body (ConvertTo-Json $payload) -ContentType 'application/json'
```

Notes
- Ensure `breast_cancer.pkl` is in the same folder as `app.py`.
- If your pickle only contains the trained model object, the app will still work but `scaler` and `feature_names` will be unavailable; supply features as a list in the correct order.
- For production deployments consider using a WSGI server (gunicorn/uvicorn) behind a reverse proxy instead of Flask's built-in server.

If you want, I can:
- Add a small test script `test_predict.py` that calls the API with a sample payload and asserts the response structure.
- Add the `feature_names` automatically to the pickle by reading `data.csv` and saving them if missing.

## Conclusion

This repository provides a lightweight, easy-to-run API for serving a trained breast cancer SVM model. The `app.py` file loads a pickled model and optional preprocessing artifacts (scaler and label encoder) and exposes a simple `/predict` endpoint suitable for quick integration or prototyping.

To get started, install the dependencies, ensure `breast_cancer.pkl` is in the project root, and run the application. For production use, consider saving complete metadata (feature names and preprocessors) with the model, adding tests, and deploying behind a proper WSGI server.

If you'd like, I can implement any of those next steps (tests, auto-saving feature names, or a Streamlit UI) — tell me which and I'll add it.
# Breast_Cancer_prediction
