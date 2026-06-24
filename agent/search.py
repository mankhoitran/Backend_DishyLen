"""DuckDuckGo-backed search and extraction service using vLLM."""

from __future__ import annotations

import logging
from typing import Any

from duckduckgo_search import DDGS

from prompt.prompts import DUCKDUCKGO as DUCKDUCKGO_PROMPTS
from prompt.prompts import ALLERGY as ALLERGY_PROMPTS

from agent.clients.vllm_client import VLLMClient

logger = logging.getLogger(__name__)

# System prompt for extraction is now in prompt/prompts.py (DUCKDUCKGO.EXTRACTION_SYSTEM)


class DuckDuckGoSearchService:
    """Service that fetches and structures dish information from the web."""

    def __init__(self, vllm_client: VLLMClient, max_results: int = 5) -> None:
        self.vllm_client = vllm_client
        self.max_results = max_results

    def search_dish(self, dish_name: str) -> dict[str, Any]:
        """Search dish information and structured attributes."""

        sources = self._search_text(
            f"{dish_name} dish ingredients spicy level nutrition calories carbs protein fat"
        )
        instruction = DUCKDUCKGO_PROMPTS.SEARCH_DISH
        fallback = {
            "dish": dish_name,
            "description": "No sources found.",
            "summary": "No sources found.",
            "calories": None,
            "protein": None,
            "carbs": None,
            "fats": None,
            "ingredients": [],
            "allergens": [],
            "spicy_level": "unknown",
            "image_url": "",
        }
        payload = self._ask_json(instruction, sources, fallback)
        payload.setdefault("dish", dish_name)
        payload.setdefault("description", "No summary found.")
        payload.setdefault("summary", "No summary found.")
        payload.setdefault("calories", None)
        payload.setdefault("protein", None)
        payload.setdefault("carbs", None)
        payload.setdefault("fats", None)
        payload.setdefault("ingredients", [])
        payload.setdefault("allergens", [])
        payload.setdefault("spicy_level", "unknown")
        payload.setdefault("image_url", "")
        return payload

    def check_allergy(self, dish_name: str, ingredients: list[str], user_allergies: str) -> dict[str, Any]:
        """Check if ingredients conflict with user allergies."""
        fallback = {"allergyWarning": False, "allergens": []}
        if not user_allergies or not ingredients:
            return fallback

        instruction = ALLERGY_PROMPTS.CHECK.format(
            dish_name=dish_name,
            ingredients=", ".join(ingredients),
            user_allergies=user_allergies
        )
        return self.vllm_client.generate_json(
            system_prompt=ALLERGY_PROMPTS.SYSTEM,
            user_prompt=instruction,
            fallback=fallback
        )

    def search_sources(self, query: str) -> list[dict[str, Any]]:
        """Return raw DuckDuckGo text sources for a query."""

        return self._search_text(query)

    def get_spicy_level(self, dish_name: str) -> dict[str, Any]:
        """Extract likely spicy level for a dish."""

        sources = self._search_text(f"{dish_name} spicy level heat level")
        instruction = DUCKDUCKGO_PROMPTS.GET_SPICY_LEVEL
        fallback = {"dish": dish_name, "spicy_level": "unknown"}
        payload = self._ask_json(instruction, sources, fallback)
        payload.setdefault("dish", dish_name)
        payload.setdefault("spicy_level", "unknown")
        return payload

    def get_dish_macro(self, dish_name: str) -> dict[str, Any]:
        """Estimate dish macro nutrients."""

        sources = self._search_text(f"{dish_name} nutrition calories protein carbs fat")
        instruction = DUCKDUCKGO_PROMPTS.GET_DISH_MACRO
        fallback = {
            "dish": dish_name,
            "calories": None,
            "protein": None,
            "carbs": None,
            "fats": None,
        }
        payload = self._ask_json(instruction, sources, fallback)
        payload.setdefault("dish", dish_name)
        payload.setdefault("calories", None)
        payload.setdefault("protein", None)
        payload.setdefault("carbs", None)
        payload.setdefault("fats", None)
        return payload

    def get_dish_summary(self, dish_name: str) -> dict[str, Any]:
        """Generate concise dish summary."""

        sources = self._search_text(f"{dish_name} dish description ingredients")
        instruction = DUCKDUCKGO_PROMPTS.GET_DISH_SUMMARY
        fallback = {"dish": dish_name, "summary": "No sources found."}
        payload = self._ask_json(instruction, sources, fallback)
        payload.setdefault("dish", dish_name)
        payload.setdefault("summary", "")
        return payload

    def get_dish_ingredients(self, dish_name: str) -> dict[str, Any]:
        """Extract common ingredients for a dish."""

        sources = self._search_text(f"{dish_name} ingredients list")
        instruction = DUCKDUCKGO_PROMPTS.GET_DISH_INGREDIENTS
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

        instruction = DUCKDUCKGO_PROMPTS.TRANSLATE.format(target_language=target_language)
        user_prompt = f"text: {text}"
        payload = self.vllm_client.generate_json(
            system_prompt=DUCKDUCKGO_PROMPTS.TRANSLATE_SYSTEM,
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
            system_prompt=DUCKDUCKGO_PROMPTS.EXTRACTION_SYSTEM,
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
