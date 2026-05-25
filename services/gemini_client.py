"""Gemini API client wrapper with tool-calling support."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from google import genai
from google.genai import types
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class GeminiResult:
    """Normalized Gemini response payload."""

    text: str
    function_calls: list[dict[str, Any]]
    raw: Any


class GeminiClient:
    """Wrapper around Gemini content generation and function-calling APIs."""

    def __init__(self) -> None:
        settings = get_settings()
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is missing. Set it in your environment or .env file.")

        self.model = settings.gemini_model
        self.timeout = settings.gemini_timeout_seconds
        self.client = genai.Client(api_key=settings.gemini_api_key)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def generate(
        self,
        prompt: str,
        tools: list[types.Tool] | None = None,
        response_mime_type: str | None = None,
    ) -> dict[str, Any]:
        """Generate content for a single prompt with optional tools."""

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=tools or None,
                response_mime_type=response_mime_type,
                temperature=0.2,
            ),
        )
        result = self._parse_response(response)
        return {
            "text": result.text,
            "function_calls": result.function_calls,
            "raw": response,
        }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def generate_with_contents(
        self,
        contents: list[types.Content],
        tools: list[types.Tool] | None = None,
    ) -> GeminiResult:
        """Generate content from a conversation state including function responses."""

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                tools=tools or None,
                temperature=0.2,
            ),
        )
        return self._parse_response(response)

    def _parse_response(self, response: Any) -> GeminiResult:
        """Extract plain text and function call objects from Gemini response."""

        function_calls: list[dict[str, Any]] = []
        text_chunks: list[str] = []

        candidates = getattr(response, "candidates", []) or []
        if not candidates:
            logger.warning("Gemini returned no candidates")
            return GeminiResult(text="", function_calls=[], raw=response)

        first_candidate = candidates[0]
        content = getattr(first_candidate, "content", None)
        parts = getattr(content, "parts", []) if content else []

        for part in parts:
            function_call = getattr(part, "function_call", None)
            if function_call:
                args = dict(getattr(function_call, "args", {}) or {})
                function_calls.append({"name": function_call.name, "args": args})

            part_text = getattr(part, "text", None)
            if part_text:
                text_chunks.append(part_text)

        text = "\n".join(chunk.strip() for chunk in text_chunks if chunk and chunk.strip())
        return GeminiResult(text=text, function_calls=function_calls, raw=response)

    @staticmethod
    def safe_json_loads(raw_text: str, fallback: dict[str, Any] | None = None) -> dict[str, Any]:
        """Parse JSON safely from model output, stripping markdown wrappers if present."""

        fallback = fallback or {}
        text = (raw_text or "").strip()
        if not text:
            return fallback

        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Could not decode JSON from Gemini output")
            return fallback
