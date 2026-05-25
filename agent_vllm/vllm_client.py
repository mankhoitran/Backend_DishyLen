"""vLLM API client wrapper using the OpenAI-compatible endpoint."""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class VLLMResult:
    """Normalized vLLM response payload."""

    text: str
    raw: dict[str, Any]


class VLLMClient:
    """Wrapper around a vLLM OpenAI-compatible chat API."""

    def __init__(self) -> None:
        settings = get_settings()
        if not settings.vllm_base_url:
            raise ValueError("VLLM_BASE_URL is missing. Set it in your environment or .env file.")

        self.base_url = settings.vllm_base_url.rstrip("/")
        self.model = settings.vllm_model
        self.api_key = settings.vllm_api_key
        self.timeout = settings.vllm_timeout_seconds

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        response_format: dict[str, str] | None = None,
        max_tokens: int = 1024,
    ) -> VLLMResult:
        """Call vLLM chat completions endpoint."""

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            payload["response_format"] = response_format

        data = self._post_json("/chat/completions", payload)
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        text = message.get("content") or ""
        return VLLMResult(text=text, raw=data)

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1024,
        fallback: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate JSON output with best-effort parsing."""

        fallback = fallback or {}
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            result = self.chat(
                messages,
                response_format={"type": "json_object"},
                max_tokens=max_tokens,
            )
        except Exception as exc:
            logger.warning("vLLM response_format failed, retrying without it: %s", exc)
            result = self.chat(messages, max_tokens=max_tokens)

        return self.safe_json_loads(result.text, fallback=fallback)

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode("utf-8")

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        request = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
            return json.loads(body)
        except urllib.error.HTTPError as exc:
            body = ""
            if exc.fp:
                body = exc.fp.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"vLLM request failed: {exc.code} {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"vLLM request failed: {exc.reason}") from exc

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
            value = json.loads(text)
            return value if isinstance(value, dict) else fallback
        except json.JSONDecodeError:
            logger.warning("Could not decode JSON from vLLM output")
            return fallback
