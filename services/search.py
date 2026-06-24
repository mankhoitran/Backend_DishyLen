"""Search and extraction service powered by Gemini + Google Search tool."""

from __future__ import annotations

import logging
from typing import Any

from google.genai import types

from prompt.prompts import SEARCH as SEARCH_PROMPTS
from services.gemini_client import GeminiClient

logger = logging.getLogger(__name__)


class SearchService:
    """Service that fetches and structures dish information from the web."""

    def __init__(self, gemini_client: GeminiClient) -> None:
        self.gemini_client = gemini_client

    def search_dish(self, dish_name: str, user_allergies: str | None = None) -> dict[str, Any]:
        """Retrieve structured dish information using Google's search grounding tool."""

        allergy_context = ""
        if user_allergies:
            allergy_context = f"\nUser allergies: {user_allergies}. If there are allergies listed here, add an explicit 'Allergy Warning:' in the summary if the dish likely contains them."
        prompt = SEARCH_PROMPTS.SEARCH_DISH.format(dish_name=dish_name, allergy_context=allergy_context)

        tools = [types.Tool(google_search=types.GoogleSearch())]
        result = self.gemini_client.generate(prompt=prompt, tools=tools, response_mime_type="application/json")

        parsed = GeminiClient.safe_json_loads(
            result.get("text", ""),
            fallback={
                "dish": dish_name,
                "spicy_level": "unknown",
                "calories": None,
                "protein": None,
                "carbs": None,
                "fats": None,
                "summary": "No summary found.",
                "image_url": "",
            },
        )

        parsed.setdefault("dish", dish_name)
        parsed.setdefault("spicy_level", "unknown")
        parsed.setdefault("calories", None)
        parsed.setdefault("protein", None)
        parsed.setdefault("carbs", None)
        parsed.setdefault("fats", None)
        parsed.setdefault("summary", "No summary found.")
        parsed.setdefault("image_url", "")
        return parsed

    def get_spicy_level(self, dish_name: str) -> dict[str, Any]:
        """Extract likely spicy level for a dish."""

        prompt = SEARCH_PROMPTS.GET_SPICY_LEVEL.format(dish_name=dish_name)
        result = self.gemini_client.generate(prompt=prompt, response_mime_type="application/json")
        payload = GeminiClient.safe_json_loads(result.get("text", ""), fallback={"dish": dish_name, "spicy_level": "unknown"})
        payload.setdefault("dish", dish_name)
        payload.setdefault("spicy_level", "unknown")
        return payload

    def get_dish_macro(self, dish_name: str) -> dict[str, Any]:
        """Estimate dish macro nutrients."""

        prompt = SEARCH_PROMPTS.GET_DISH_MACRO.format(dish_name=dish_name)
        result = self.gemini_client.generate(prompt=prompt, response_mime_type="application/json")
        payload = GeminiClient.safe_json_loads(result.get("text", ""), fallback={
            "dish": dish_name,
            "calories": None,
            "protein": None,
            "carbs": None,
            "fats": None,
        })
        payload.setdefault("dish", dish_name)
        payload.setdefault("calories", None)
        payload.setdefault("protein", None)
        payload.setdefault("carbs", None)
        payload.setdefault("fats", None)
        return payload

    def get_dish_summary(self, dish_name: str) -> dict[str, Any]:
        """Generate concise dish summary."""

        prompt = SEARCH_PROMPTS.GET_DISH_SUMMARY.format(dish_name=dish_name)
        result = self.gemini_client.generate(prompt=prompt, response_mime_type="application/json")
        payload = GeminiClient.safe_json_loads(result.get("text", ""), fallback={"dish": dish_name, "summary": ""})
        payload.setdefault("dish", dish_name)
        payload.setdefault("summary", "")
        return payload

    def get_dish_image_url(self, dish_name: str) -> dict[str, Any]:
        """Pick one best representative image URL for a dish."""

        prompt = SEARCH_PROMPTS.GET_DISH_IMAGE_URL.format(dish_name=dish_name)
        tools = [types.Tool(google_search=types.GoogleSearch())]
        result = self.gemini_client.generate(prompt=prompt, tools=tools, response_mime_type="application/json")
        payload = GeminiClient.safe_json_loads(result.get("text", ""), fallback={"dish": dish_name, "image_url": ""})
        payload.setdefault("dish", dish_name)
        payload.setdefault("image_url", "")
        return payload

    def translate(self, text: str, target_language: str) -> dict[str, Any]:
        """Translate text into target language."""

        prompt = SEARCH_PROMPTS.TRANSLATE.format(target_language=target_language, text=text)
        result = self.gemini_client.generate(prompt=prompt, response_mime_type="application/json")
        payload = GeminiClient.safe_json_loads(
            result.get("text", ""),
            fallback={"target_language": target_language, "translated_text": text},
        )
        payload.setdefault("target_language", target_language)
        payload.setdefault("translated_text", text)
        return payload
