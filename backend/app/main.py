import logging
from typing import List

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

try:  # pragma: no cover - runtime import safety
    from .schemas import (
        AnalyticsSummary,
        ChatMessage,
        ChatRequest,
        ChatResponse,
        DescriptionRequest,
        DescriptionResponse,
        RecommendationRequest,
        ClusterDetail,
        RecommendationOut,
    )
    from .services.recommender import Recommendation, ResourceManager
except ImportError:  # Support running `python main.py`
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent))
    from schemas import (  # type: ignore  # noqa: E402
        AnalyticsSummary,
        ChatMessage,
        ChatRequest,
        ChatResponse,
        DescriptionRequest,
        DescriptionResponse,
        RecommendationRequest,
        ClusterDetail,
        RecommendationOut,
    )
    from services.recommender import (  # type: ignore  # noqa: E402
        Recommendation,
        ResourceManager,
    )

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

resource_manager = ResourceManager()

app = FastAPI(title="Furniture Recommender API", version="1.0.0")


# Mount static images if available so the frontend can display thumbnails.
images_dir = resource_manager.data_dir / "images"
if images_dir.exists():
    app.mount("/static/images", StaticFiles(directory=images_dir), name="images")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust in production
    allow_credentials=True,
    allow_methods=["*"] ,
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event() -> None:
    resource_manager.load()


def get_resource_manager() -> ResourceManager:
    if not resource_manager._loaded:
        raise HTTPException(status_code=503, detail="Resources are still loading")
    return resource_manager


@app.get("/api/health")
def health_check() -> dict:
    return {"status": "ok"}


def _extract_user_prompt(messages: List[ChatMessage]) -> str:
    for message in reversed(messages):
        if message.role.lower() == "user":
            return message.content.strip()
    raise HTTPException(status_code=400, detail="Missing user message in payload")


def _serialize_recommendations(recommendations: List[Recommendation]) -> List[RecommendationOut]:
    return [
        RecommendationOut(
            uniq_id=rec.uniq_id,
            title=rec.title,
            brand=rec.brand,
            categories=rec.categories,
            similarity=rec.similarity,
            rank_score=rec.rank_score,
            price=rec.price,
            image_url=rec.image_url,
            generated_description=rec.generated_description,
            source_description=rec.source_description,
        )
        for rec in recommendations
    ]


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest, resources: ResourceManager = Depends(get_resource_manager)) -> ChatResponse:
    query = _extract_user_prompt(request.messages)
    recommendations: List[Recommendation] = resources.recommend(query, top_k=request.top_k)
    if not recommendations:
        reply = "I could not find matching furniture right now. Try another prompt or refine your request."
        return ChatResponse(reply=reply, recommendations=[])

    titles = ", ".join(rec.title for rec in recommendations[:3])
    reply = (
        f"Here are {len(recommendations)} furniture picks that fit '{query}': {titles}. "
        "Let me know if you want more options or details!"
    )
    payload = _serialize_recommendations(recommendations)
    return ChatResponse(reply=reply, recommendations=payload)


@app.get("/api/analytics/summary", response_model=AnalyticsSummary)
def analytics(resources: ResourceManager = Depends(get_resource_manager)) -> AnalyticsSummary:
    data = resources.analytics_summary()
    if not data:
        raise HTTPException(status_code=404, detail="Analytics unavailable")
    return AnalyticsSummary.parse_obj(data)


@app.get("/api/analytics/cluster/{cluster_id}", response_model=ClusterDetail)
def analytics_cluster(
    cluster_id: int,
    limit: int = 30,
    resources: ResourceManager = Depends(get_resource_manager),
) -> ClusterDetail:
    detail = resources.cluster_detail(cluster_id=cluster_id, limit=limit)
    if detail is None:
        raise HTTPException(status_code=404, detail="Cluster not found")
    return ClusterDetail.parse_obj(detail)


@app.post("/api/recommend", response_model=List[RecommendationOut])
def direct_recommend(
    request: RecommendationRequest,
    resources: ResourceManager = Depends(get_resource_manager),
) -> List[RecommendationOut]:
    recommendations = resources.recommend(request.query, top_k=request.top_k)
    if not recommendations:
        return []
    return _serialize_recommendations(recommendations)


@app.post("/api/description/generate", response_model=DescriptionResponse)
def generate_description(
    request: DescriptionRequest,
    resources: ResourceManager = Depends(get_resource_manager),
) -> DescriptionResponse:
    try:
        data = resources.generate_description_summary(
            query=request.query,
            product_ids=request.product_ids,
            top_k=request.top_k,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if data is None:
        raise HTTPException(status_code=404, detail="No products found to describe")

    return DescriptionResponse.parse_obj(data)


if __name__ == "__main__":  # pragma: no cover - CLI convenience
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
