# Ikarus Product Discovery Platform

An end-to-end product discovery and analytics stack that marries pretrained embeddings, clustering, and a lightweight ranker with an interactive React frontend. Users can explore category clusters, inspect analytics, chat about products, and generate marketing-ready descriptions for items surfaced by the recommendation engine.

## Features
- **Multimodal retrieval** combining FAISS-powered nearest-neighbor search with LightGBM reranking and business rules.
- **Cluster analytics** including UMAP projections, aggregate metrics, and product lists per cluster.
- **Conversational insights** through a chat-style interface backed by curated product context.
- **Description generation** that produces human-friendly marketing blurbs for up to three recommended items.
- **Single-port deployment** via Docker Compose with Nginx reverse-proxying the FastAPI backend.

## Tech Stack
- **Backend**: Python 3.11, FastAPI, SentenceTransformers, FAISS, LightGBM, pandas, PyTorch.
- **Frontend**: React 18, Vite, Tailwind CSS, Axios, Plotly.js.
- **Infrastructure**: Docker, Nginx, Uvicorn.

## Repository Layout
```
backend/
  app/
    main.py                # FastAPI entrypoint and route definitions
    api/                   # Pydantic schemas and route wiring
    services/              # Resource manager, recommender, description logic
  requirements.txt         # Backend Python dependencies
backend_app.py             # Legacy entrypoint (imported by uvicorn)
frontend/
  src/                     # React application (pages, components, hooks, services)
  public/
  package.json
  nginx.conf               # Reverse proxy rules served in the runtime container
train_cv.py                # Cross-validation pipeline for the ranker
train_ranker.py            # LightGBM training script
clustering_nlp.py          # Embedding and clustering routine
gen_description.py         # Batch description generator utilities
backend/Dockerfile         # Python application image
frontend/Dockerfile        # Vite build + Nginx runtime image
docker-compose.yml         # Orchestrates backend + frontend on a shared network
data_prep_output/          # Precomputed assets (embeddings, FAISS index, metadata)
```

## Data Artifacts
The `data_prep_output/` directory ships with all artifacts required to run inference:
- `faiss.index`, `text_embeddings.npy`, `image_embeddings.npy`: nearest-neighbor index and vectors.
- `meta.json`, `meta_with_clusters.json`, `products_clean.jsonl`: product metadata, cluster assignments, and cleaned records.
- `clusters.npy`, `umap_projection.npy`: clustering results and 2D/3D projections for visualization.
- `resnet50_prod.pth`: vision backbone weights used when refreshing embeddings.

> **Note:** Training scripts assume these files exist; replace them with regenerated artifacts if you rerun preprocessing.

## Local Development
### Prerequisites
- Python 3.11+
- Node.js 20+
- FAISS-compatible platform (Linux/macOS recommended)

### Backend
```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.app.main:app --reload --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```
Visit `http://localhost:5173` during development; the frontend proxies API calls to `http://localhost:8000` (configure in `frontend/src/services/api.js` if needed).

## Docker Deployment
To build and launch the unified stack on `http://localhost:8080`:
```bash
docker compose up --build
```
- `frontend` serves static assets through Nginx and forwards `/api/*` and `/static/images/*` requests to `backend`.
- Adjust env vars or volume mounts in `docker-compose.yml` if you need to override model assets or configuration.

## API Surface (selected)
| Method | Route | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Basic service heartbeat |
| `POST` | `/api/recommend` | Retrieve ranked recommendations keyed by product IDs |
| `POST` | `/api/description/generate` | Produce marketing blurbs for selected product IDs |
| `GET` | `/api/analytics/clusters` | Global cluster summary for analytics page |
| `GET` | `/api/analytics/cluster/{cluster_id}` | Detailed metrics and products for a cluster |
| `POST` | `/api/chat` | Retrieve conversational responses based on product context |

Refer to the Pydantic schemas in `backend/app/api/schemas` for request/response contracts.

## Frontend Pages
- `/` – Chat experience for conversational insights.
- `/analytics` – Cluster scatter plot, cluster detail pane, and product exploration.
- `/description` – Workflow to fetch recommendations, select items, and generate summaries.

Routing is handled by React Router with lazy-loaded pages to keep bundles small.

## Testing & Quality
- ML training scripts (`train_cv.py`, `train_ranker.py`) include deterministic seeds for reproducibility.
- FastAPI endpoints rely on pydantic validation; add `pytest` suites under `backend/tests/` if you extend the API.
- Frontend ships with ESLint and TypeScript-ready configs; run `npm run lint` and `npm run build` before committing.

## Troubleshooting
- **Large model loads**: Ensure the container or local environment has enough RAM (~4 GB) for all embeddings.
- **GPU acceleration**: Current stack runs on CPU; extend Dockerfiles with CUDA base images if GPU inference is needed.
- **Hot reload in Docker**: Mount source directories as volumes and override the `CMD` to use `uvicorn --reload` for backend development containers.

## License
This repository contains proprietary assets (product metadata, embeddings) and is intended for internal assessment. Do not redistribute without explicit authorization.
