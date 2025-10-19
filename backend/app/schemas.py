from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the speaker: user or assistant")
    content: str = Field(..., description="Message text")


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., description="Full conversation history")
    top_k: int = Field(3, ge=1, le=10, description="Number of recommendations to return")


class RecommendationOut(BaseModel):
    uniq_id: str
    title: str
    brand: Optional[str]
    categories: List[str]
    similarity: float
    rank_score: float
    price: Optional[float]
    image_url: Optional[str]
    generated_description: str
    source_description: Optional[str]


class ChatResponse(BaseModel):
    reply: str
    recommendations: List[RecommendationOut]


class NamedCount(BaseModel):
    name: str
    count: int


class ClusterCount(BaseModel):
    cluster: int
    count: int


class PriceSummary(BaseModel):
    count: int
    min: Optional[float]
    max: Optional[float]
    mean: Optional[float]
    median: Optional[float]


class ImageCoverage(BaseModel):
    with_: int = Field(..., alias="with")
    without: int


class AnalyticsSummary(BaseModel):
    total_products: int
    categories: List[NamedCount]
    brands: List[NamedCount]
    price: PriceSummary
    images: ImageCoverage
    clusters: List[ClusterCount]

    model_config = ConfigDict(populate_by_name=True)


class ClusterTerm(BaseModel):
    term: str
    count: int


class ClusterProduct(BaseModel):
    uniq_id: str
    title: str
    brand: Optional[str]
    price: Optional[float]
    categories: List[str]
    image_url: Optional[str]
    description: Optional[str]
    coordinates: Optional[List[float]]


class ClusterDetail(BaseModel):
    cluster: int
    total_products: int
    top_terms: List[ClusterTerm]
    centroid: Optional[List[float]]
    products: List[ClusterProduct]


class RecommendationRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(6, ge=1, le=20)


class DescriptionProduct(BaseModel):
    uniq_id: str
    title: str
    brand: Optional[str]
    price: Optional[float]
    categories: List[str]
    image_url: Optional[str]
    description: Optional[str]


class DescriptionRequest(BaseModel):
    query: str = Field(..., min_length=1)
    product_ids: List[str] = Field(default_factory=list)
    top_k: int = Field(6, ge=1, le=20)


class DescriptionResponse(BaseModel):
    description: str
    products: List[DescriptionProduct]
    used_product_ids: List[str]
