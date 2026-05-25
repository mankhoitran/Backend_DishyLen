"""DuckDuckGo-backed search and extraction service using vLLM."""

from __future__ import annotations

import logging
from typing import Any

from duckduckgo_search import DDGS

from .vllm_client import VLLMClient

logger = logging.getLogger(__name__)

EXTRACTION_SYSTEM_PROMPT = (
    "You are a culinary data extractor. Use the provided sources when possible. "
    "Return strict JSON only."
)


class DuckDuckGoSearchService:
    """Service that fetches and structures dish information from the web."""

    def __init__(self, vllm_client: VLLMClient, max_results: int = 5) -> None:
        self.vllm_client = vllm_client
        self.max_results = max_results

    def search_dish(self, dish_name: str) -> dict[str, Any]:
        """Retrieve structured dish information using DuckDuckGo sources."""

        sources = self._search_text(
            f"{dish_name} dish ingredients spicy level nutrition calories macros"
        )
        instruction = (
            "Return ONLY JSON with keys: dish, spicy_level, macros, summary, image_url. "
            "spicy_level must be one of: not_spicy, mild, medium, hot, very_hot, unknown. "
            "macros must be an object with keys calories_kcal, protein_g, carbs_g, fat_g "
            "(numbers or null). "
            "summary should be factual, 2-3 sentences, and under 80 words. "
            "Avoid mentioning sources or search. "
            "If sources are empty, summary should be 'No sources found.'. "
            "image_url should be empty if no reliable image appears in sources."
        )
        fallback = {
            "dish": dish_name,
            "spicy_level": "unknown",
            "macros": {
                "calories_kcal": None,
                "protein_g": None,
                "carbs_g": None,
                "fat_g": None,
            },
            "summary": "No sources found.",
            "image_url": "",
        }
        payload = self._ask_json(instruction, sources, fallback)
        payload.setdefault("dish", dish_name)
        payload.setdefault("spicy_level", "unknown")
        payload.setdefault("macros", {})
        payload.setdefault("summary", "No summary found.")
        payload.setdefault("image_url", "")
        return payload

    def search_sources(self, query: str) -> list[dict[str, Any]]:
        """Return raw DuckDuckGo text sources for a query."""

        return self._search_text(query)

    def get_spicy_level(self, dish_name: str) -> dict[str, Any]:
        """Extract likely spicy level for a dish."""

        sources = self._search_text(f"{dish_name} spicy level heat level")
        instruction = (
            "Return ONLY JSON with keys dish and spicy_level. "
            "spicy_level must be one of: not_spicy, mild, medium, hot, very_hot, unknown."
        )
        fallback = {"dish": dish_name, "spicy_level": "unknown"}
        payload = self._ask_json(instruction, sources, fallback)
        payload.setdefault("dish", dish_name)
        payload.setdefault("spicy_level", "unknown")
        return payload

    def get_dish_macro(self, dish_name: str) -> dict[str, Any]:
        """Estimate dish macro nutrients."""

        sources = self._search_text(f"{dish_name} nutrition calories protein carbs fat")
        instruction = (
            "Return ONLY JSON with keys dish and macros. "
            "macros object keys: calories_kcal, protein_g, carbs_g, fat_g. "
            "Values numeric or null."
        )
        fallback = {"dish": dish_name, "macros": {}}
        payload = self._ask_json(instruction, sources, fallback)
        payload.setdefault("dish", dish_name)
        payload.setdefault("macros", {})
        return payload

    def get_dish_summary(self, dish_name: str) -> dict[str, Any]:
        """Generate concise dish summary."""

        sources = self._search_text(f"{dish_name} dish description ingredients")
        instruction = (
            "Return ONLY JSON with keys dish and summary. "
            "summary should be factual, 2-3 sentences, and under 80 words. "
            "Avoid mentioning sources or search. "
            "If sources are empty, summary should be 'No sources found.'."
        )
        fallback = {"dish": dish_name, "summary": "No sources found."}
        payload = self._ask_json(instruction, sources, fallback)
        payload.setdefault("dish", dish_name)
        payload.setdefault("summary", "")
        return payload

    def get_dish_ingredients(self, dish_name: str) -> dict[str, Any]:
        """Extract common ingredients for a dish."""

        sources = self._search_text(f"{dish_name} ingredients list")
        instruction = (
            "Return ONLY JSON with keys dish and ingredients. "
            "ingredients must be an array of strings. "
            "If sources are empty, ingredients should be an empty array."
        )
        fallback = {"dish": dish_name, "ingredients": []}
        payload = self._ask_json(instruction, sources, fallback)
        payload.setdefault("dish", dish_name)
        ingredients = payload.get("ingredients")
        if not isinstance(ingredients, list):
            payload["ingredients"] = []
        return payload

    def get_dish_image_url(self, dish_name: str) -> dict[str, Any]:
        """Pick one best representative image URL for a dish."""

        images = self._search_images(f"{dish_name} dish")
        image_url = ""
        if images:
            first = images[0]
            image_url = first.get("image") or first.get("thumbnail") or ""

        return {"dish": dish_name, "image_url": image_url}

    def translate(self, text: str, target_language: str) -> dict[str, Any]:
        """Translate text into target language."""

        instruction = (
            "Return ONLY JSON with keys target_language and translated_text. "
            f"target_language: {target_language}."
        )
        user_prompt = f"text: {text}"
        payload = self.vllm_client.generate_json(
            system_prompt="You are a translation engine.",
            user_prompt=f"{instruction}\n{user_prompt}",
            fallback={"target_language": target_language, "translated_text": text},
        )
        payload.setdefault("target_language", target_language)
        payload.setdefault("translated_text", text)
        return payload

    def _ask_json(
        self,
        instruction: str,
        sources: list[dict[str, Any]],
        fallback: dict[str, Any],
    ) -> dict[str, Any]:
        source_block = self._format_sources(sources)
        user_prompt = f"{instruction}\nSources:\n{source_block}"
        return self.vllm_client.generate_json(
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            fallback=fallback,
        )

    def _search_text(self, query: str) -> list[dict[str, Any]]:
        try:
            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=self.max_results))
        except Exception as exc:  # pragma: no cover - external dependency errors
            logger.warning("DuckDuckGo text search failed: %s", exc)
            return []

    def _search_images(self, query: str) -> list[dict[str, Any]]:
        try:
            with DDGS() as ddgs:
                return list(ddgs.images(query, max_results=5))
        except Exception as exc:  # pragma: no cover - external dependency errors
            logger.warning("DuckDuckGo image search failed: %s", exc)
            return []

    @staticmethod
    def _format_sources(sources: list[dict[str, Any]]) -> str:
        if not sources:
            return "No sources found."

        lines: list[str] = []
        for item in sources:
            title = item.get("title") or "Untitled"
            snippet = item.get("body") or item.get("snippet") or ""
            href = item.get("href") or item.get("url") or ""
            lines.append(f"- {title}: {snippet} ({href})")

        return "\n".join(lines)
