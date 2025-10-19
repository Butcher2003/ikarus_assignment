import json
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import lightgbm as lgb
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from .description import DescriptionGenerator

LOGGER = logging.getLogger(__name__)

TOKEN_PATTERN = re.compile(r"[A-Za-z]{3,}")


@dataclass
class Recommendation:
    uniq_id: str
    title: str
    brand: Optional[str]
    similarity: float
    rank_score: float
    categories: List[str]
    price: Optional[float]
    image_url: Optional[str]
    generated_description: str
    source_description: Optional[str]


class ResourceManager:
    def __init__(self, base_dir: Optional[Path] = None) -> None:
        self.base_dir = base_dir or Path(__file__).resolve().parents[3]
        self.data_dir = self.base_dir / "data_prep_output"
        self._loaded = False
        self.text_model: Optional[SentenceTransformer] = None
        self.faiss_index: Optional[faiss.Index] = None
        self.text_embeddings: Optional[np.ndarray] = None
        self.meta: Optional[List[Dict[str, object]]] = None
        self.meta_idx: Dict[str, int] = {}
        self.products_df: Optional[pd.DataFrame] = None
        self.products_by_id: Dict[str, Dict[str, object]] = {}
        self.image_map: Dict[str, List[str]] = {}
        self.cluster_assignments: Dict[str, int] = {}
        self.umap_projection: Optional[np.ndarray] = None
        self.ranker: Optional[lgb.Booster] = None
        self.description_generator: Optional[DescriptionGenerator] = None
        self.analytics_cache: Optional[Dict[str, object]] = None

    def load(self) -> None:
        if self._loaded:
            return
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Expected data directory at {self.data_dir}")

        LOGGER.info("Loading models and data from %s", self.data_dir)
        self.text_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.faiss_index = faiss.read_index(str(self.data_dir / "faiss.index"))
        self.text_embeddings = np.load(self.data_dir / "text_embeddings.npy").astype("float32")

        with open(self.data_dir / "meta.json", "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        self.meta_idx = {item["uniq_id"]: idx for idx, item in enumerate(self.meta)}

        products_path = self.data_dir / "products_clean.jsonl"
        self.products_df = pd.read_json(products_path, lines=True)
        self.products_by_id = {
            row["uniq_id"]: row.to_dict()
            for _, row in self.products_df.iterrows()
        }

        self.cluster_assignments = {}
        meta_clusters_path = self.data_dir / "meta_with_clusters.json"
        if meta_clusters_path.exists():
            with open(meta_clusters_path, "r", encoding="utf-8") as f:
                meta_clusters = json.load(f)
            for item in meta_clusters:
                uniq_id = item.get("uniq_id")
                if not uniq_id:
                    continue
                cluster_val = item.get("cluster")
                try:
                    cluster_int = int(cluster_val)
                except (TypeError, ValueError):
                    continue
                self.cluster_assignments[uniq_id] = cluster_int

        if self.products_df is not None:
            clusters = self.products_df["uniq_id"].map(self.cluster_assignments)
            clusters = clusters.fillna(-1).astype(int)
            self.products_df["cluster"] = clusters

        self.image_map = self._build_image_map()

        umap_path = self.data_dir / "umap_projection.npy"
        if umap_path.exists():
            projection = np.load(umap_path).astype("float32")
            if projection.ndim == 2 and projection.shape[1] == 2:
                zeros = np.zeros((projection.shape[0], 1), dtype="float32")
                projection = np.concatenate([projection, zeros], axis=1)
            if self.meta is not None and projection.shape[0] != len(self.meta):
                LOGGER.warning(
                    "UMAP projection length %s does not match metadata length %s",
                    projection.shape[0],
                    len(self.meta),
                )
            self.umap_projection = projection
        else:
            self.umap_projection = None

        ranker_path = self.data_dir / "lightgbm_ranker.txt"
        self.ranker = lgb.Booster(model_file=str(ranker_path)) if ranker_path.exists() else None

        self.description_generator = DescriptionGenerator()
        self.analytics_cache = self._build_analytics_cache()
        self._loaded = True
        LOGGER.info("Finished loading resources")

    def _build_image_map(self) -> Dict[str, List[str]]:
        image_dir = self.data_dir / "images"
        mapping: Dict[str, List[str]] = {}
        if not image_dir.exists():
            return mapping
        for path in image_dir.glob("*.jpg"):
            uniq_id = "_".join(path.stem.split("_")[:-1])
            mapping.setdefault(uniq_id, []).append(f"/static/images/{path.name}")
        return mapping

    def _build_analytics_cache(self) -> Dict[str, object]:
        if self.products_df is None:
            return {}
        df = self.products_df.copy()
        total_products = int(len(df))

        categories_series = (
            df["categories"].explode().dropna().astype(str)
            if "categories" in df.columns
            else pd.Series(dtype=str)
        )
        if not categories_series.empty:
            category_counts = categories_series.value_counts().head(10).reset_index()
            category_counts.columns = ["name", "count"]
            category_stats = [
                {"name": str(row["name"]), "count": int(row["count"])}
                for _, row in category_counts.iterrows()
            ]
        else:
            category_stats = []

        brand_series = (
            df["brand"].dropna().astype(str)
            if "brand" in df.columns
            else pd.Series(dtype=str)
        )
        brand_series = brand_series[brand_series.str.len() > 0]
        if not brand_series.empty:
            brand_counts = brand_series.value_counts().head(10).reset_index()
            brand_counts.columns = ["name", "count"]
            brand_stats = [
                {"name": str(row["name"]), "count": int(row["count"])}
                for _, row in brand_counts.iterrows()
            ]
        else:
            brand_stats = []

        price_series = df["price"].dropna() if "price" in df.columns else pd.Series(dtype=float)
        price_summary = {
            "count": int(price_series.count()),
            "min": float(price_series.min()) if not price_series.empty else None,
            "max": float(price_series.max()) if not price_series.empty else None,
            "mean": float(price_series.mean()) if not price_series.empty else None,
            "median": float(price_series.median()) if not price_series.empty else None,
        }

        image_with = sum(1 for images in self.image_map.values() if images)
        image_without = total_products - image_with

        cluster_series = df["cluster"] if "cluster" in df.columns else pd.Series(dtype=int)
        if not cluster_series.empty:
            cluster_counts = cluster_series.value_counts().head(10).reset_index()
            cluster_counts.columns = ["cluster", "count"]
            cluster_stats = [
                {"cluster": int(row["cluster"]), "count": int(row["count"])}
                for _, row in cluster_counts.iterrows()
            ]
        else:
            cluster_stats = []

        return {
            "total_products": total_products,
            "categories": category_stats,
            "brands": brand_stats,
            "price": price_summary,
            "images": {"with": int(image_with), "without": int(image_without)},
            "clusters": cluster_stats,
        }

    def recommend(self, query: str, top_k: int = 5) -> List[Recommendation]:
        if not self._loaded or self.text_model is None or self.faiss_index is None:
            raise RuntimeError("Resources not loaded")

        query_vec = self.text_model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        search_k = max(top_k * 3, top_k)
        distances, indices = self.faiss_index.search(np.array([query_vec]), search_k)

        candidates = []
        for idx, score in zip(indices[0], distances[0]):
            if idx < 0 or self.meta is None:
                continue
            meta_item = self.meta[idx]
            uniq_id = meta_item.get("uniq_id")
            if uniq_id not in self.products_by_id:
                continue
            prod = self.products_by_id[uniq_id]
            price = prod.get("price")
            if isinstance(price, float) and math.isnan(price):
                price = None
            candidate = {
                "uniq_id": uniq_id,
                "title": meta_item.get("title") or prod.get("title"),
                "brand": prod.get("brand"),
                "categories": prod.get("categories") or [],
                "price": price,
                "description": prod.get("description"),
                "image_url": self.image_map.get(uniq_id, [None])[0],
                "similarity": float(score),
                "meta_index": int(idx),
            }
            candidates.append(candidate)

        if not candidates:
            return []

        if self.ranker is not None and self.text_embeddings is not None:
            feature_mat = np.array([
                [float(np.dot(query_vec, self.text_embeddings[cand["meta_index"]]))]
                for cand in candidates
            ], dtype="float32")
            preds = self.ranker.predict(feature_mat)
            for cand, pred in zip(candidates, preds):
                cand["rank_score"] = float(pred)
        else:
            for cand in candidates:
                cand["rank_score"] = float(cand["similarity"])

        candidates = sorted(candidates, key=lambda c: c["rank_score"], reverse=True)[:top_k]

        generator = self.description_generator or DescriptionGenerator()
        descriptions = generator.generate_batch(query, candidates)
        results = []
        for cand, desc in zip(candidates, descriptions):
            results.append(
                Recommendation(
                    uniq_id=str(cand["uniq_id"]),
                    title=str(cand.get("title") or ""),
                    brand=str(cand.get("brand")) if cand.get("brand") else None,
                    similarity=float(cand.get("similarity", 0.0)),
                    rank_score=float(cand.get("rank_score", 0.0)),
                    categories=list(cand.get("categories") or []),
                    price=float(cand["price"]) if cand.get("price") is not None else None,
                    image_url=cand.get("image_url"),
                    generated_description=desc,
                    source_description=str(cand.get("description")) if cand.get("description") else None,
                )
            )
        return results

    def cluster_detail(self, cluster_id: int, limit: int = 30) -> Optional[Dict[str, object]]:
        if self.products_df is None:
            return None
        if limit <= 0:
            limit = 1
        limit = min(limit, 100)

        cluster_df = self.products_df[self.products_df["cluster"] == cluster_id]
        if cluster_df.empty:
            return None

        token_counts: Counter = Counter()
        for title in cluster_df["title"].dropna().astype(str):
            token_counts.update(token.lower() for token in TOKEN_PATTERN.findall(title))
        for brand in cluster_df["brand"].dropna().astype(str):
            token_counts.update([brand.lower()])
        if "categories" in cluster_df.columns:
            for cats in cluster_df["categories"].dropna():
                if isinstance(cats, list):
                    token_counts.update(cat.lower() for cat in cats if cat)
                else:
                    token_counts.update([str(cats).lower()])

        top_terms = [
            {"term": term, "count": int(count)}
            for term, count in token_counts.most_common(30)
        ]

        coordinates_all: List[List[float]] = []
        if self.umap_projection is not None:
            for uniq_id in cluster_df["uniq_id"]:
                idx = self.meta_idx.get(uniq_id)
                if idx is None or idx >= self.umap_projection.shape[0]:
                    continue
                vector = self.umap_projection[idx]
                coords = [float(v) for v in vector[:3]]
                coordinates_all.append(coords)

        centroid = None
        if coordinates_all:
            centroid = [
                float(np.mean([coord[i] for coord in coordinates_all]))
                for i in range(3)
            ]

        products: List[Dict[str, object]] = []
        for _, row in cluster_df.head(limit).iterrows():
            uniq_id = row["uniq_id"]
            idx = self.meta_idx.get(uniq_id)
            coords = None
            if self.umap_projection is not None and idx is not None and idx < self.umap_projection.shape[0]:
                vector = self.umap_projection[idx]
                coords = [float(v) for v in vector[:3]]

            categories = row.get("categories") or []
            if isinstance(categories, str):
                categories = [categories]
            elif not isinstance(categories, list):
                categories = [str(categories)] if categories else []

            price = row.get("price")
            if isinstance(price, float) and math.isnan(price):
                price = None

            description = row.get("description")
            if isinstance(description, float) and math.isnan(description):
                description = None
            if description is not None:
                description = str(description)
                if len(description) > 240:
                    description = description[:240] + "…"

            images = self.image_map.get(uniq_id)
            image_url = images[0] if images else None

            products.append(
                {
                    "uniq_id": str(uniq_id),
                    "title": str(row.get("title") or ""),
                    "brand": str(row.get("brand")) if row.get("brand") else None,
                    "price": float(price) if price is not None else None,
                    "categories": [str(cat) for cat in categories],
                    "image_url": image_url,
                    "description": description,
                    "coordinates": coords,
                }
            )

        return {
            "cluster": int(cluster_id),
            "total_products": int(cluster_df.shape[0]),
            "top_terms": top_terms,
            "centroid": centroid,
            "products": products,
        }

    @staticmethod
    def _normalize_categories(raw: object) -> List[str]:
        if raw is None:
            return []
        if isinstance(raw, list):
            return [str(item) for item in raw if item]
        if isinstance(raw, str):
            parts = [part.strip() for part in raw.split("|") if part.strip()]
            if parts:
                return parts
            return [raw.strip()] if raw.strip() else []
        if isinstance(raw, float) and math.isnan(raw):
            return []
        return [str(raw)]

    def _prepare_product_record(self, uniq_id: str) -> Optional[Dict[str, object]]:
        base = self.products_by_id.get(uniq_id)
        if base is None:
            return None

        meta_idx = self.meta_idx.get(uniq_id)
        title = base.get("title")
        if not title and meta_idx is not None and self.meta is not None and 0 <= meta_idx < len(self.meta):
            title = self.meta[meta_idx].get("title")

        brand = base.get("brand")
        if (not brand) and meta_idx is not None and self.meta is not None and 0 <= meta_idx < len(self.meta):
            brand = self.meta[meta_idx].get("brand")
        description = base.get("description")
        if isinstance(description, float) and math.isnan(description):
            description = None
        price = base.get("price")
        if isinstance(price, float) and math.isnan(price):
            price = None
        categories = self._normalize_categories(base.get("categories"))
        image_url = None
        if uniq_id in self.image_map and self.image_map[uniq_id]:
            image_url = self.image_map[uniq_id][0]

        record = {
            "uniq_id": str(uniq_id),
            "title": str(title) if title else "",
            "brand": str(brand) if brand else None,
            "description": description if description is None else str(description),
            "price": float(price) if price is not None else None,
            "categories": categories,
            "image_url": image_url,
            "meta_index": meta_idx,
        }
        return record

    def _format_description_product(self, record: Dict[str, object]) -> Dict[str, object]:
        description = record.get("description")
        if description and len(description) > 240:
            description = description[:240] + "…"
        return {
            "uniq_id": record.get("uniq_id"),
            "title": record.get("title") or "",
            "brand": record.get("brand"),
            "price": record.get("price"),
            "categories": record.get("categories") or [],
            "image_url": record.get("image_url"),
            "description": description,
        }

    def generate_description_summary(
        self,
        query: str,
        product_ids: Optional[List[str]] = None,
        top_k: int = 6,
    ) -> Optional[Dict[str, object]]:
        if not query or not query.strip():
            raise ValueError("Query text is required to generate a description.")

        products: List[Dict[str, object]] = []
        used_ids: List[str] = []

        if product_ids:
            for pid in product_ids:
                record = self._prepare_product_record(pid)
                if record is None:
                    continue
                products.append(record)
                used_ids.append(record["uniq_id"])

        if not products:
            recommendations = self.recommend(query, top_k)
            for rec in recommendations:
                record = self._prepare_product_record(rec.uniq_id)
                if record is None:
                    continue
                products.append(record)
                used_ids.append(record["uniq_id"])

        if not products:
            return None

        generator = self.description_generator or DescriptionGenerator()
        summary = generator.generate_summary(query, products)
        response_products = [self._format_description_product(record) for record in products]

        return {
            "description": summary,
            "products": response_products,
            "used_product_ids": used_ids,
        }

    def analytics_summary(self) -> Dict[str, object]:
        if self.analytics_cache is None:
            return {}
        return self.analytics_cache
