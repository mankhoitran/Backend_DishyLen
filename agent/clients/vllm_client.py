"""vLLM API client wrapper using the OpenAI-compatible endpoint."""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from configs.configs import get_settings
from services.logging_utils import get_llm_response_logger
from agent.clients.openrouter_client import OpenRouterClient

logger = logging.getLogger(__name__)
llm_response_logger = get_llm_response_logger()


@dataclass
class VLLMResult:
    """Normalized vLLM response payload."""

    text: str
    raw: dict[str, Any]


class VLLMClient:
    """Wrapper around a vLLM OpenAI-compatible chat API.

    Automatically falls back to OpenRouter when vLLM is unavailable.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._fallback_client: OpenRouterClient | None = None
        self._fallback_mode = False

        self.base_url = settings.vllm_base_url.rstrip("/") if settings.vllm_base_url else ""
        self.api_key = settings.vllm_api_key
        self.timeout = settings.vllm_timeout_seconds
        self.model = ""

        if not self.base_url:
            logger.warning("VLLM_BASE_URL is missing; falling back to OpenRouter.")
            self._init_fallback()
            return

        try:
            self.model = self._resolve_model(settings.vllm_model)
        except Exception as exc:
            logger.exception("vLLM not available; falling back to OpenRouter: %s", exc)
            self._init_fallback()

    def _init_fallback(self) -> None:
        try:
            self._fallback_client = OpenRouterClient()
            self._fallback_mode = True
            self.model = self._fallback_client.model
        except Exception as exc:
            raise ValueError(
                "vLLM is unavailable and OpenRouter is not configured. "
                "Set OPENROUTER_API_KEY and OPENROUTER_MODEL."
            ) from exc

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

        if self._fallback_mode and self._fallback_client:
            result = self._fallback_client.chat(
                messages,
                temperature=temperature,
                response_format=response_format,
                max_tokens=max_tokens,
            )
            return VLLMResult(text=result.text, raw=result.raw)

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            payload["response_format"] = response_format
        try:
            data = self._post_json("/chat/completions", payload)
            choice = (data.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            text = message.get("content") or ""
            self._log_response(text, data)
            return VLLMResult(text=text, raw=data)
        except Exception as exc:
            logger.exception("vLLM chat failed; falling back to OpenRouter: %s", exc)
            if not self._fallback_client:
                self._init_fallback()
            result = self._fallback_client.chat(
                messages,
                temperature=temperature,
                response_format=response_format,
                max_tokens=max_tokens,
            )
            return VLLMResult(text=result.text, raw=result.raw)

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

        if self._fallback_mode and self._fallback_client:
            return self._fallback_client.generate_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=max_tokens,
                fallback=fallback,
            )

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
            return self.safe_json_loads(result.text, fallback=fallback)
        except Exception as exc:
            logger.warning("vLLM response_format failed, retrying without it: %s", exc)
            try:
                result = self.chat(messages, max_tokens=max_tokens)
                return self.safe_json_loads(result.text, fallback=fallback)
            except Exception as chat_exc:
                logger.exception("vLLM generate_json failed; falling back to OpenRouter: %s", chat_exc)
                if not self._fallback_client:
                    self._init_fallback()
                return self._fallback_client.generate_json(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_tokens=max_tokens,
                    fallback=fallback,
                )

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
