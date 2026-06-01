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
from services.logging_utils import get_llm_response_logger

logger = logging.getLogger(__name__)
llm_response_logger = get_llm_response_logger()


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
        self.api_key = settings.vllm_api_key
        self.timeout = settings.vllm_timeout_seconds
        self.model = self._resolve_model(settings.vllm_model)

    def _resolve_model(self, configured_model: str | None) -> str:
        model = (configured_model or "").strip()
        if model and model.lower() != "auto":
            try:
                available = self._fetch_model_ids()
                if available:
                    if model in available:
                        return model
                    logger.warning(
                        "Configured vLLM model '%s' not found; using '%s' from server.",
                        model,
                        available[0],
                    )
                    return available[0]
            except Exception as exc:
                logger.warning(
                    "Failed to fetch vLLM models; using configured model '%s': %s",
                    model,
                    exc,
                )
                return model

            return model

        available = self._fetch_model_ids()
        if available:
            logger.info("Auto-detected vLLM model: %s", available[0])
            return available[0]

        raise ValueError("VLLM_MODEL is missing and the vLLM server returned no models.")

    def _fetch_model_ids(self) -> list[str]:
        data = self._get_json("/models")
        items = data.get("data") or []
        return [
            item.get("id")
            for item in items
            if isinstance(item, dict) and item.get("id")
        ]

    def _get_json(self, path: str) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        request = urllib.request.Request(url, headers=headers, method="GET")
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
        self._log_response(text, data)
        return VLLMResult(text=text, raw=data)

    def _log_response(self, text: str, raw: dict[str, Any]) -> None:
        payload = {
            "provider": "vllm",
            "model": self.model,
            "text": text,
        }
        usage = raw.get("usage") if isinstance(raw, dict) else None
        if usage:
            payload["usage"] = usage
        try:
            llm_response_logger.info(json.dumps(payload, ensure_ascii=True))
        except Exception:
            logger.exception("Failed to write LLM response log")

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
