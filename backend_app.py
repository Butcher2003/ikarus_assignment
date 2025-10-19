# file: backend_app.py
# FastAPI app that loads FAISS index + embeddings + ranker + resnet classifier and exposes endpoints for recommend and generate_description.
# Run: uvicorn backend_app:app --reload --port 8000
"""Backward-compatible entrypoint that exposes the new FastAPI app."""

from backend.app.main import app  # noqa: F401
