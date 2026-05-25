"""vLLM-driven agent loop using DuckDuckGo search."""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy.orm import Session

from agent.parser import parse_input
from config import get_settings
from .search import DuckDuckGoSearchService
from .tools import VLLMAgentTools
from .vllm_client import VLLMClient

logger = logging.getLogger(__name__)


class VLLMFoodAgent:
    """Tool-driven food information agent powered by vLLM and DuckDuckGo."""

    def __init__(self, db: Session) -> None:
        self.settings = get_settings()
        self.vllm_client = VLLMClient()
        self.search_service = DuckDuckGoSearchService(
            self.vllm_client,
            max_results=self.settings.duckduckgo_max_results,
        )
        self.tools = VLLMAgentTools(db=db, search_service=self.search_service)

    def run(self, query: str, target_language: str | None = None) -> dict[str, Any]:
        """Execute end-to-end dish information workflow."""

        parsed = parse_input(query)
        logger.info("Parsed input type=%s dish=%s", parsed.input_type, parsed.dish_name)

        collected: dict[str, Any] = {
            "dish": parsed.dish_name,
            "description": "",
            "spicy_level": "unknown",
            "macros": {},
            "summary": "",
            "image_url": "",
            "calories": None,
            "protein": None,
            "carbs": None,
            "fats": None,
            "ingredients": [],
            "allergens": [],
            "sources": [],
            "source": "search",
        }

        existing = self.tools.get_processed_dish(parsed.dish_name)
        collected = self._merge_collected(collected, existing)
        should_search = existing.get("found") is not True or self._needs_refresh(existing)

        if should_search:
            if existing.get("found") is True:
                logger.info("Refreshing dish info from search due to incomplete data.")
            search_payload = self.tools.search_dish(parsed.dish_name)
            collected = self._merge_collected(collected, search_payload)

            if collected.get("spicy_level") in (None, "", "unknown"):
                collected = self._merge_collected(
                    collected,
                    self.tools.get_spicy_level(parsed.dish_name),
                )

            if self._macros_missing(collected.get("macros")):
                collected = self._merge_collected(
                    collected,
                    self.tools.get_dish_macro(parsed.dish_name),
                )

            if not collected.get("summary"):
                collected = self._merge_collected(
                    collected,
                    self.tools.get_dish_summary(parsed.dish_name),
                )

            if not collected.get("image_url"):
                collected = self._merge_collected(
                    collected,
                    self.tools.get_dish_image_url(parsed.dish_name),
                )

        if target_language and collected.get("summary"):
            translated = self.tools.translate(collected["summary"], target_language)
            collected["summary"] = translated.get("translated_text", collected["summary"])

        if collected.get("source") == "search":
            self.tools.persist_dish(
                dish_name=collected.get("dish", parsed.dish_name),
                spicy_level=collected.get("spicy_level", "unknown"),
                macros=collected.get("macros", {}),
                summary=collected.get("summary", ""),
            )

        return {
            "dish": collected.get("dish", parsed.dish_name),
            "description": collected.get("description", ""),
            "spicy_level": collected.get("spicy_level", "unknown"),
            "macros": collected.get("macros", {}),
            "calories": collected.get("calories"),
            "protein": collected.get("protein"),
            "carbs": collected.get("carbs"),
            "fats": collected.get("fats"),
            "summary": collected.get("summary", ""),
            "image_url": collected.get("image_url", ""),
            "ingredients": collected.get("ingredients", []),
            "allergens": collected.get("allergens", []),
            "sources": collected.get("sources", []),
            "source": collected.get("source", "search"),
        }

    @staticmethod
    def _merge_collected(current: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
        """Merge non-empty tool outputs into collected state."""

        merged = dict(current)
        for key in ("dish", "description", "spicy_level", "summary", "image_url", "source"):
            value = incoming.get(key)
            if value:
                merged[key] = value

        for key in ("calories", "protein", "carbs", "fats"):
            value = incoming.get(key)
            if value not in (None, "", "unknown"):
                merged[key] = value

        for key in ("ingredients", "allergens", "sources"):
            value = incoming.get(key)
            if isinstance(value, list) and value:
                merged[key] = value

        incoming_macros = incoming.get("macros")
        if isinstance(incoming_macros, dict) and incoming_macros:
            merged["macros"] = incoming_macros

        if incoming.get("found") is True:
            merged["source"] = "database"

        return merged

    @staticmethod
    def _macros_missing(macros: dict[str, Any] | None) -> bool:
        if not isinstance(macros, dict) or not macros:
            return True

        required = ["calories_kcal", "protein_g", "carbs_g", "fat_g"]
        values = [macros.get(key) for key in required]
        return all(value in (None, "", "unknown") for value in values)

    @staticmethod
    def _needs_refresh(existing: dict[str, Any]) -> bool:
        if existing.get("found") is not True:
            return True

        summary = existing.get("summary")
        spicy_level = existing.get("spicy_level")
        macros = existing.get("macros")

        if not summary or summary.strip() in ("No summary found.", "No sources found."):
            return True

        if spicy_level in (None, "", "unknown"):
            return True

        return VLLMFoodAgent._macros_missing(macros)
