"""Tool implementations callable by the vLLM-backed agent."""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy.orm import Session

from db import crud
from agent.search import DuckDuckGoSearchService

logger = logging.getLogger(__name__)


class VLLMAgentTools:
    """Collection of reusable tool functions for dish processing."""

    def __init__(self, db: Session, search_service: DuckDuckGoSearchService) -> None:
        self.db = db
        self.search_service = search_service

    def get_processed_dish(self, dish_name: str) -> dict[str, Any]:
        """Retrieve dish data from the local database."""

        dish = crud.get_dish_by_name(self.db, dish_name)
        if not dish:
            return {"found": False, "dish": dish_name}

        return {
            "found": True,
            "dish": dish.name,
            "spicy_level": dish.spicy_level,
            "macros": dish.macros or {},
            "summary": dish.summary,
            "image_url": "",
            "source": "database",
        }

    def search_dish(self, dish_name: str) -> dict[str, Any]:
        """Search dish information via DuckDuckGo sources."""

        payload = self.search_service.search_dish(dish_name)
        payload["source"] = "search"
        return payload

    def check_allergy(self, dish_name: str, ingredients: list[str], user_allergies: str) -> dict[str, Any]:
        """Check if dish ingredients conflict with user allergies."""

        return self.search_service.check_allergy(dish_name, ingredients, user_allergies)

    def get_spicy_level(self, dish_name: str) -> dict[str, Any]:
        """Get dish spicy level."""

        return self.search_service.get_spicy_level(dish_name)

    def get_dish_macro(self, dish_name: str) -> dict[str, Any]:
        """Get dish macro nutrient estimates."""

        return self.search_service.get_dish_macro(dish_name)

    def get_dish_summary(self, dish_name: str) -> dict[str, Any]:
        """Get concise dish summary."""

        return self.search_service.get_dish_summary(dish_name)

    def get_dish_image_url(self, dish_name: str) -> dict[str, Any]:
        """Get one best representative image URL for a dish."""

        return self.search_service.get_dish_image_url(dish_name)

    def translate(self, text: str, target_language: str) -> dict[str, Any]:
        """Translate text into a target language."""

        return self.search_service.translate(text, target_language)

    def persist_dish(
        self,
        dish_name: str,
        spicy_level: str,
        macros: dict[str, Any],
        summary: str,
    ) -> dict[str, Any]:
        """Persist dish data in local DB."""

        dish = crud.upsert_dish(
            self.db,
            name=dish_name,
            spicy_level=spicy_level,
            macros=macros,
            summary=summary,
        )
        return {
            "saved": True,
            "dish": dish.name,
            "source": "database",
        }
