# Furniture Recommender Backend

FastAPI backend that serves furniture recommendations, generates creative product descriptions, and exposes dataset analytics.

## Prerequisites

- Python 3.10+
- The processed assets in `../data_prep_output/` (already generated).

## Setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the API

```bash
uvicorn app.main:app --reload --port 8000
```

The API exposes:

- `POST /api/chat` — conversational recommendations with generated descriptions.
- `GET /api/analytics/summary` — aggregated insights about the catalogue.
- `GET /api/health` — basic health check.

Static product thumbnails are served from `/static/images/...`.
