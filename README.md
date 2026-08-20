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

### Features

Eleven, in the order `FEATURE_COLUMNS` declares them in [model/train_model.py](model/train_model.py). Six come from monthly engagement metrics; five are derived from interaction history:

| Feature | Range | Notes |
| --- | --- | --- |
| `total_interactions_30d` | 0–40 | Rolling 30-day count |
| `negative_sentiment_ratio` | 0–100 | Percent, not a 0–1 fraction |
| `avg_interaction_duration` | 0–120 | Minutes, one decimal |
| `days_since_last_interaction` | 0–90 | All-time recency, capped at 90 |
| `support_interaction_ratio` | 0–100 | Percent, not a 0–1 fraction |

An account with nothing in the 30-day window sends `0` for the counts and ratios and `90` for recency. That convention is deliberate: a zero negative-sentiment ratio would otherwise read as "perfectly happy" when it actually means "no data", so recency is left carrying the signal. The same rule is encoded in three places — `_apply_empty_window_convention` in the generator, `FEATURE_DEFAULTS` in `app.py`, and `ChurnFeatureBuilder` in Apex — and they must not drift apart.

## Deploy to Render

- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120` (see `Procfile`)
- **Health Check Path**: `/health`

## Salesforce integration

Call the deployed Render URL from Apex via a Named Credential (server-side callout), passing all eleven account features to `/predict`. `ChurnPredictionService` builds the six metric features from `Engagement_Metric__c` and delegates the five interaction features to `ChurnFeatureBuilder`.
