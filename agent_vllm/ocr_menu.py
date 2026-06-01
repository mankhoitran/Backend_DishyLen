"""OCR helpers for extracting menu items from images."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .parser import parse_input

logger = logging.getLogger(__name__)

CATEGORY_HEADINGS = {
    "menu",
    "appetizers",
    "entrees",
    "mains",
    "main",
    "sides",
    "salads",
    "soups",
    "desserts",
    "drinks",
    "beverages",
    "specials",
    "hot",
    "cold",
    "prices",
}

PRICE_PATTERN = re.compile(
    r"\s*(?:[$€£]|vnd|usd)?\s*\d+(?:[\.,]\d{1,2})?\s*$",
    flags=re.IGNORECASE,
)

SEPARATOR_PATTERN = re.compile(r"\s*\|\s*|\s*/\s*|\s{2,}")
CURRENCY_PATTERN = re.compile(r"[$€£]|vnd|usd", flags=re.IGNORECASE)

OCR_PROMPT_TEMPLATE = (
    "You are an OCR post-processor for menu images. Use the OCR text provided below as the only source.\n"
    "Return ONLY a strict JSON object with exactly one key: items. "
    "No extra keys, no markdown, no commentary.\n"
    "Step 1: Extract ONLY dish names and food/drink item names. Completely ignore prices, currency symbols, "
    "numbers, and headers like 'Menu' or 'Beverages'. Preserve the original reading order (top-to-bottom, "
    "left-to-right).\n"
    "Step 2: Correct the extracted items. Fix spelling errors, Vietnamese diacritics where applicable, and "
    "obvious OCR errors (e.g., 'Bow!' -> 'Bowl', 'eed Tea' -> 'Iced Tea'). Do NOT add or remove items.\n"
    "Each item must be a separate array element. Never combine multiple items into one string.\n"
    "Output schema example: {{\"items\": [\"Dish A\", \"Dish B\"]}}\n"
    "OCR text:\n{ocr_text}"
)


@dataclass
class OcrResult:
    """OCR output payload."""

    text: str
    status: str
    image_path: str


def ocr_menu_image(image_path: str | Path) -> OcrResult:
    """Run OCR on a menu image and return extracted text."""

    path = Path(image_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    try:
        from PIL import Image
        import pytesseract
    except Exception as exc:  # pragma: no cover - optional deps
        logger.warning("OCR dependencies missing: %s", exc)
        return OcrResult(text="", status="missing_deps", image_path=str(path))

    try:
        text = pytesseract.image_to_string(Image.open(path))
        cleaned = (text or "").strip()
        return OcrResult(text=cleaned, status="ocr", image_path=str(path))
    except Exception as exc:  # pragma: no cover - optional deps
        logger.warning("OCR failed: %s", exc)
        return OcrResult(text="", status="ocr_failed", image_path=str(path))


def extract_menu_items(ocr_text: str, max_items: int = 40) -> list[str]:
    """Extract normalized menu items from OCR text."""

    candidates: list[str] = []
    for line in _split_lines(ocr_text):
        for part in _split_candidates(line):
            normalized = _normalize_line(part)
            if not normalized:
                continue
            if _is_heading(normalized):
                continue
            parsed = parse_input(normalized)
            dish_name = parsed.dish_name.strip()
            if not dish_name or dish_name.lower() in CATEGORY_HEADINGS:
                continue
            candidates.append(dish_name)

    deduped: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
        if len(deduped) >= max_items:
            break

    return deduped


def select_menu_item(items: list[str], item_name: str | None, item_index: int | None) -> str:
    """Resolve a chosen menu item by name or index."""

    if not items:
        raise ValueError("No menu items were extracted.")

    if item_name:
        needle = item_name.strip().lower()
        for item in items:
            if item.lower() == needle:
                return item
        raise ValueError("Selected item was not found in extracted menu items.")

    if item_index is None:
        raise ValueError("Provide item_name or item_index to select a dish.")

    if item_index < 0 or item_index >= len(items):
        raise ValueError("item_index is out of range for extracted menu items.")

    return items[item_index]


def apply_ocr_prompt(
    ocr_text: str,
    fallback_items: Iterable[str] | None = None,
    prefer_backend: str = "auto",
) -> dict[str, str]:
    """Use an LLM to format OCR output with normalized menu items."""

    fallback = {
        "raw_text": _normalize_ocr_text(ocr_text),
        "corrected_text": _items_to_lines(fallback_items),
    }
    prompt = _build_ocr_prompt(ocr_text)

    if prefer_backend in {"auto", "vllm"}:
        try:
            from .vllm_client import VLLMClient

            vllm_client = VLLMClient()
            payload = vllm_client.generate_json(
                system_prompt="You are a strict OCR post-processor.",
                user_prompt=prompt,
                max_tokens=1200,
                fallback=fallback,
            )
            return _sanitize_ocr_payload(payload, fallback)
        except Exception as exc:
            logger.warning("vLLM OCR post-process failed: %s", exc)
            if prefer_backend == "vllm":
                return fallback

    if prefer_backend in {"auto", "openrouter"}:
        try:
            from services.openrouter_client import OpenRouterClient

            openrouter_client = OpenRouterClient()
            payload = openrouter_client.generate_json(
                system_prompt="You are a strict OCR post-processor.",
                user_prompt=prompt,
                max_tokens=1200,
                fallback=fallback,
            )
            return _sanitize_ocr_payload(payload, fallback)
        except Exception as exc:
            logger.warning("OpenRouter OCR post-process failed: %s", exc)
            if prefer_backend == "openrouter":
                return fallback

    if prefer_backend in {"auto", "gemini"}:
        try:
            from services.gemini_client import GeminiClient

            gemini_client = GeminiClient()
            result = gemini_client.generate(prompt=prompt, response_mime_type="application/json")
            payload = GeminiClient.safe_json_loads(result.get("text", ""), fallback=fallback)
            return _sanitize_ocr_payload(payload, fallback)
        except Exception as exc:
            logger.warning("Gemini OCR post-process failed: %s", exc)

    return fallback


def corrected_items_from_text(text: str, max_items: int | None = None) -> list[str]:
    """Split corrected OCR output into a deduped list of dish names."""

    items = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if not items:
        return []

    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
        if max_items is not None and len(deduped) >= max_items:
            break

    return deduped


def _split_lines(text: str) -> Iterable[str]:
    for raw in (text or "").splitlines():
        cleaned = raw.strip()
        if cleaned:
            yield cleaned


def _split_candidates(line: str) -> list[str]:
    parts = [p.strip() for p in SEPARATOR_PATTERN.split(line) if p.strip()]
    if len(parts) == 1:
        return parts

    flattened: list[str] = []
    for part in parts:
        if "," in part and len(part.split(",")) <= 3:
            flattened.extend([p.strip() for p in part.split(",") if p.strip()])
        else:
            flattened.append(part)
    return flattened


def _normalize_line(line: str) -> str:
    cleaned = PRICE_PATTERN.sub("", line).strip()
    cleaned = cleaned.strip("-•*")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def _is_heading(line: str) -> bool:
    cleaned = line.strip().rstrip(":").lower()
    if cleaned in CATEGORY_HEADINGS:
        return True
    if cleaned.endswith("menu") and len(cleaned.split()) <= 2:
        return True
    return False


def _build_ocr_prompt(ocr_text: str) -> str:
    return OCR_PROMPT_TEMPLATE.format(ocr_text=(ocr_text or "").strip())


def _normalize_ocr_text(text: str) -> str:
    cleaned = (text or "")
    cleaned = CURRENCY_PATTERN.sub(" ", cleaned)
    cleaned = re.sub(r"\d+", " ", cleaned)
    cleaned = re.sub(r"[^\w\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _items_to_lines(items: Iterable[str] | None) -> str:
    if not items:
        return ""
    return "\n".join([str(item).strip() for item in items if str(item).strip()])


def _sanitize_ocr_payload(payload: dict[str, str], fallback: dict[str, str]) -> dict[str, str]:
    items = _coerce_items_payload(payload.get("items"))
    raw_text = payload.get("raw_text")
    corrected_text = payload.get("corrected_text")

    if not isinstance(raw_text, str):
        raw_text = fallback.get("raw_text", "")
    if not isinstance(corrected_text, str):
        corrected_text = fallback.get("corrected_text", "")

    if items:
        corrected_text = "\n".join(items)

    raw_text = _normalize_ocr_text(raw_text)
    corrected_text = _normalize_corrected_text(corrected_text)

    if not raw_text and fallback.get("raw_text"):
        raw_text = fallback["raw_text"]
    if not corrected_text and fallback.get("corrected_text"):
        corrected_text = fallback["corrected_text"]

    return {
        "raw_text": raw_text.strip(),
        "corrected_text": corrected_text.strip(),
    }


def _normalize_corrected_text(text: str) -> str:
    lines = []
    for line in (text or "").splitlines():
        cleaned = _normalize_item_line(line)
        if cleaned and not _is_heading_like(cleaned):
            lines.append(cleaned)
    return "\n".join(lines)


def _coerce_items_payload(items: object) -> list[str]:
    if isinstance(items, list):
        return _normalize_items_list(items)

    if isinstance(items, str):
        if _looks_like_header_blob(items):
            return []
        parts = re.split(r"\s*[,;/|]\s*", items)
        if len(parts) > 1:
            return _normalize_items_list(parts)

    return []


def _normalize_items_list(items: Iterable[object]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, str):
            continue
        cleaned = _normalize_item_line(item)
        if not cleaned or _is_heading_like(cleaned):
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(cleaned)
    return normalized


def _normalize_item_line(text: str) -> str:
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^[-•*\d\.)\]]+\s*", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def _is_heading_like(text: str) -> bool:
    cleaned = re.sub(r"[^a-zA-Z\s]", " ", text).strip().lower()
    if not cleaned:
        return True
    tokens = cleaned.split()
    if cleaned in CATEGORY_HEADINGS:
        return True
    if cleaned.endswith("menu") and len(tokens) <= 3:
        return True
    if "menu" in tokens or "prices" in tokens or "price" in tokens or "beverages" in tokens:
        return len(tokens) <= 3
    if len(tokens) <= 2 and any(token in CATEGORY_HEADINGS for token in tokens):
        return True
    return False


def _looks_like_header_blob(text: str) -> bool:
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if len(cleaned) < 80:
        return False
    return _is_heading_like(cleaned)
