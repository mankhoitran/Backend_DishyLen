"""Standalone summary service."""

import logging
from typing import Any

from agent.clients.vllm_client import VLLMClient
from prompt.prompts import SUMMARY as SUMMARY_PROMPTS
from prompt.prompts import ALLERGY as ALLERGY_PROMPTS

logger = logging.getLogger(__name__)

class SummaryService:
    """Service dedicated to summarizing food-related text."""

    def __init__(self, vllm_client: VLLMClient | None = None) -> None:
        self.vllm_client = vllm_client or VLLMClient()

    def summarize_text(self, text: str, max_words: int, user_allergies: str | None = None) -> dict[str, Any]:
        """Summarize text using vLLM."""
        fallback = {
            "description": "",
            "summary": "",
            "calories": None,
            "protein": None,
            "carbs": None,
            "fats": None,
            "ingredients": [],
            "allergens": [],
        }
        try:
            payload = self.vllm_client.generate_json(
                system_prompt=SUMMARY_PROMPTS.SYSTEM,
                user_prompt=SUMMARY_PROMPTS.USER.format(max_words=max_words, text=text),
                fallback=fallback,
            )
            
            if user_allergies and payload.get("ingredients"):
                allergy_instruction = ALLERGY_PROMPTS.CHECK.format(
                    dish_name="Unknown Dish",
                    ingredients=", ".join(payload["ingredients"]),
                    user_allergies=user_allergies
                )
                allergy_check = self.vllm_client.generate_json(
                    system_prompt=ALLERGY_PROMPTS.SYSTEM,
                    user_prompt=allergy_instruction,
                    fallback={"allergyWarning": False, "allergens": []}
                )
                payload["allergyWarning"] = allergy_check.get("allergyWarning", False)
                payload["allergens"] = allergy_check.get("allergens", [])
                
            return payload
        except Exception as exc:
            logger.warning("Summarization failed: %s", exc)
            return fallback
