# Retainr

Random Forest churn prediction API for Salesforce, served via Flask + gunicorn.

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Retrain

```bash
python data/generate_training_data.py
python model/train_model.py
```

## Run locally

```bash
python app.py
```

## API

- `GET /health` — service and model status
- `POST /predict` — churn score for one account
- `POST /predict/batch` — churn scores for multiple accounts

See docstrings in [app.py](app.py) for request/response payloads.

## Deploy to Render

- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120` (see `Procfile`)
- **Health Check Path**: `/health`

## Salesforce integration

Call the deployed Render URL from Apex via a Named Credential (server-side callout), passing the six account features to `/predict`.
