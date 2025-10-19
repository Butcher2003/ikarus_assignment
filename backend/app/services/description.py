import logging
from typing import Iterable, List, Mapping

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class DescriptionGenerator:
    """Wraps a seq2seq model to craft short marketing blurbs per product."""

    def __init__(self, model_name: str = "google/flan-t5-small") -> None:
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._available = False
        self._logger = logging.getLogger(__name__)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
            self._available = True
        except Exception as exc:  # pragma: no cover - defensive path
            prompt = (
                "You are a creative retail copywriter. Craft a concise two-paragraph product story for the query "
                f"'{query}'. Highlight how the following items work together:\n"
            )
            prompt += "\n".join(ctx_lines)
            prompt += "\nTone: friendly, modern, and focused on interior styling. End with a gentle call-to-action."
    def available(self) -> bool:
        return self._available

    def _fallback(self, query: str, product: Mapping[str, object]) -> str:
        title = product.get("title") or "This product"
        brand = product.get("brand") or "a trusted brand"
        categories: Iterable[str] = product.get("categories") or []
        category_text = ", ".join(categories) if categories else "home essentials"
        base_desc = product.get("description") or "Thoughtfully designed to elevate your space."
        return (
            f"{title} by {brand} complements your '{query}' search with its {category_text.lower()} focus. "
            f"{str(base_desc)[:140]}"
        )

    def generate_batch(self, query: str, products: List[Mapping[str, object]]) -> List[str]:
        if not products:
            return []
        if not self.available:
            return [self._fallback(query, p) for p in products]

        prompts = []
        for product in products:
            title = product.get("title") or "Furniture piece"
            brand = product.get("brand") or "Unknown brand"
            categories: Iterable[str] = product.get("categories") or []
            category_text = ", ".join(categories) if categories else "general furniture"
            base_desc = product.get("description") or "No additional details provided."
            prompt = (
                "Write a warm, two sentence product blurb tailored for someone searching for "
                f"'{query}'. Product title: {title}. Brand: {brand}. Category: {category_text}. "
                "Key details: "
                f"{str(base_desc)[:240]}"
            )
            prompts.append(prompt)

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=True,
                top_p=0.9,
                temperature=0.8,
            )
        texts = [self.tokenizer.decode(out, skip_special_tokens=True).strip() for out in outputs]
        return texts

    def generate_summary(self, query: str, products: List[Mapping[str, object]]) -> str:
        if not products:
            return "I need at least one product to craft a description."

        sample = products[:6]
        if not self.available:
            highlights = "; ".join(p.get("title") or "a featured piece" for p in sample)
            return (
                f"For '{query}', spotlight {highlights}. Each selection complements the theme with practical, "
                "style-forward details."
            )

        ctx_lines = []
        for product in sample:
            title = product.get("title") or "Furniture piece"
            brand = product.get("brand") or "Unknown brand"
            categories: Iterable[str] = product.get("categories") or []
            category_text = ", ".join(categories) if categories else "general furniture"
            base_desc = product.get("description") or "No additional details provided."
            ctx_lines.append(f"- {title[:140]} (brand: {brand}, category: {category_text}) :: {str(base_desc)[:240]}")

        prompt = (
            "You are a creative retail copywriter. Craft a concise two-paragraph product story for the query "
            f"'{query}'. Highlight how the following items work together:\n"
        )
        prompt += "\n".join(ctx_lines)
        prompt += "\nTone: friendly, modern, and focused on interior styling. End with a gentle call-to-action."

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=160,
                do_sample=True,
                top_p=0.9,
                temperature=0.8,
            )
        text = self.tokenizer.decode(output[0], skip_special_tokens=True).strip()
        return text
