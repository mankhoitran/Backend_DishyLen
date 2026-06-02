"""Search and extraction service powered by Gemini + Google Search tool."""

from __future__ import annotations

import logging
from typing import Any

from google.genai import types

from services.gemini_client import GeminiClient

logger = logging.getLogger(__name__)


class SearchService:
    """Service that fetches and structures dish information from the web."""

    def __init__(self, gemini_client: GeminiClient) -> None:
        self.gemini_client = gemini_client

    def search_dish(self, dish_name: str) -> dict[str, Any]:
        """Retrieve structured dish information using Google's search grounding tool."""

        prompt = (
            "You are a culinary data extractor. Use Google Search tool to find reliable dish information. "
            "Return ONLY strict JSON with keys: dish, spicy_level, macros, summary, image_url. "
            "Macros must be an object with keys calories_kcal, protein_g, carbs_g, fat_g (numbers or null). "
            "image_url must be one best direct image URL that clearly represents the dish, or empty string if unavailable. "
            f"Dish to research: {dish_name}"
        )

        tools = [types.Tool(google_search=types.GoogleSearch())]
        result = self.gemini_client.generate(prompt=prompt, tools=tools, response_mime_type="application/json")

        parsed = GeminiClient.safe_json_loads(
            result.get("text", ""),
            fallback={
                "dish": dish_name,
                "spicy_level": "unknown",
                "macros": {},
                "summary": "No summary found.",
                "image_url": "",
            },
        )

        parsed.setdefault("dish", dish_name)
        parsed.setdefault("spicy_level", "unknown")
        parsed.setdefault("macros", {})
        parsed.setdefault("summary", "No summary found.")
        parsed.setdefault("image_url", "")
        return parsed

    def get_spicy_level(self, dish_name: str) -> dict[str, Any]:
        """Extract likely spicy level for a dish."""

        prompt = (
            "Return ONLY JSON with keys dish and spicy_level. "
            "spicy_level must be one of: not_spicy, mild, medium, hot, very_hot, unknown. "
            f"Dish: {dish_name}"
        )
        result = self.gemini_client.generate(prompt=prompt, response_mime_type="application/json")
        payload = GeminiClient.safe_json_loads(result.get("text", ""), fallback={"dish": dish_name, "spicy_level": "unknown"})
        payload.setdefault("dish", dish_name)
        payload.setdefault("spicy_level", "unknown")
        return payload

    def get_dish_macro(self, dish_name: str) -> dict[str, Any]:
        """Estimate dish macro nutrients."""

        prompt = (
            "Return ONLY JSON with keys dish and macros. "
            "macros object keys: calories_kcal, protein_g, carbs_g, fat_g. Values numeric or null. "
            f"Dish: {dish_name}"
        )
        result = self.gemini_client.generate(prompt=prompt, response_mime_type="application/json")
        payload = GeminiClient.safe_json_loads(result.get("text", ""), fallback={"dish": dish_name, "macros": {}})
        payload.setdefault("dish", dish_name)
        payload.setdefault("macros", {})
        return payload

    def get_dish_summary(self, dish_name: str) -> dict[str, Any]:
        """Generate concise dish summary."""

        prompt = (
            "Return ONLY JSON with keys dish and summary. "
            "summary should be factual and under 60 words. "
            f"Dish: {dish_name}"
        )
        result = self.gemini_client.generate(prompt=prompt, response_mime_type="application/json")
        payload = GeminiClient.safe_json_loads(result.get("text", ""), fallback={"dish": dish_name, "summary": ""})
        payload.setdefault("dish", dish_name)
        payload.setdefault("summary", "")
        return payload

    def get_dish_image_url(self, dish_name: str) -> dict[str, Any]:
        """Pick one best representative image URL for a dish."""

        prompt = (
            "Use Google Search tool and return ONLY JSON with keys dish and image_url. "
            "Select one best direct image URL that visually represents the dish, "
            "prefer stable sources and avoid logo/thumbnail sprites. "
            "If no reliable image can be found, return an empty string for image_url. "
            f"Dish: {dish_name}"
        )
        tools = [types.Tool(google_search=types.GoogleSearch())]
        result = self.gemini_client.generate(prompt=prompt, tools=tools, response_mime_type="application/json")
        payload = GeminiClient.safe_json_loads(result.get("text", ""), fallback={"dish": dish_name, "image_url": ""})
        payload.setdefault("dish", dish_name)
        payload.setdefault("image_url", "")
        return payload

    def translate(self, text: str, target_language: str) -> dict[str, Any]:
        """Translate text into target language."""

        prompt = (
            "Return ONLY JSON with keys target_language and translated_text. "
            f"target_language: {target_language}. text: {text}"
        )
        result = self.gemini_client.generate(prompt=prompt, response_mime_type="application/json")
        payload = GeminiClient.safe_json_loads(
            result.get("text", ""),
            fallback={"target_language": target_language, "translated_text": text},
        )
        payload.setdefault("target_language", target_language)
        payload.setdefault("translated_text", text)
        return payload
