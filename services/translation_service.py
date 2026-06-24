"""Standalone translation service."""

import logging
from typing import Any

from agent.clients.vllm_client import VLLMClient
from prompt.prompts import DUCKDUCKGO as DUCKDUCKGO_PROMPTS

logger = logging.getLogger(__name__)


class TranslationService:
    """Service dedicated to translating text."""

    def __init__(self) -> None:
        self.vllm_client = VLLMClient()

    def translate_text(self, text: str, target_language: str) -> dict[str, Any]:
        """Translate text to the target language using vLLM."""
        
        instruction = DUCKDUCKGO_PROMPTS.TRANSLATE.format(target_language=target_language)
        user_prompt = f"text: {text}"
        
        try:
            payload = self.vllm_client.generate_json(
                system_prompt=DUCKDUCKGO_PROMPTS.TRANSLATE_SYSTEM,
                user_prompt=f"{instruction}\n{user_prompt}",
                fallback={"target_language": target_language, "translated_text": text},
            )
            payload.setdefault("target_language", target_language)
            payload.setdefault("translated_text", text)
            return payload
        except Exception as exc:
            logger.warning("Translation failed: %s", exc)
            return {"target_language": target_language, "translated_text": text}
