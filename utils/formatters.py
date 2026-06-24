import json
import re
from typing import Any

from db.models import Dish, HistoryEntry, User
import schemas.response as response_schemas


def dish_to_response(dish: Dish) -> response_schemas.DishResponse:
    return response_schemas.DishResponse(
        dish=dish.name,
        spicy_level=dish.spicy_level or "unknown",
        macros=dish.macros or {},
        summary=dish.summary or "",
        image_url="",
        source="database",
    )


def user_to_response(user: User) -> response_schemas.UserResponse:
    return response_schemas.UserResponse(
        id=user.id,
        email=user.email,
        name=user.name,
        picture_url=user.picture_url,
        allergies=user.allergies,
    )


def history_to_response(
    entry: HistoryEntry,
    user: User,
) -> response_schemas.HistoryEntryResponse:
    return response_schemas.HistoryEntryResponse(
        id=entry.id,
        type=entry.type,
        title=entry.title,
        payload=entry.payload or {},
        created_at=entry.created_at.isoformat(),
        user_id=user.id,
        user_email=user.email,
    )


_RANGE_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)")
_NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?")


def strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    return stripped


def maybe_parse_json_text(text: str) -> Any | None:
    cleaned = strip_code_fences(text)
    if not cleaned:
        return None
    if not (cleaned.startswith("{") or cleaned.startswith("[")):
        return None
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


def extract_text(value: Any, preferred_keys: tuple[str, ...]) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        parsed = maybe_parse_json_text(value)
        if isinstance(parsed, dict):
            return extract_text(parsed, preferred_keys)
        if isinstance(parsed, list):
            return ", ".join(str(item).strip() for item in parsed if str(item).strip())
        return value.strip()
    if isinstance(value, dict):
        for key in preferred_keys:
            if key in value:
                text = extract_text(value.get(key), preferred_keys)
                if text:
                    return text
        for item in value.values():
            text = extract_text(item, preferred_keys)
            if text:
                return text
        return ""
    if isinstance(value, list):
        return ", ".join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip()


def to_number(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        for key in ("value", "amount", "kcal", "g"):
            if key in value:
                return to_number(value.get(key))
        return 0.0

    text = str(value).strip().lower()
    if not text or text in ("unknown", "n/a", "na"):
        return 0.0
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    range_match = _RANGE_PATTERN.search(text)
    if range_match:
        low = float(range_match.group(1))
        high = float(range_match.group(2))
        return (low + high) / 2.0
    numbers = _NUMBER_PATTERN.findall(text)
    if not numbers:
        return 0.0
    if len(numbers) >= 2 and (" to " in text or "-" in text):
        low = float(numbers[0])
        high = float(numbers[1])
        if low != high:
            return (low + high) / 2.0
    return float(numbers[0])


def dedupe_list(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def to_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items = [str(item).strip() for item in value if str(item).strip()]
        return dedupe_list(items)
    if isinstance(value, str):
        parsed = maybe_parse_json_text(value)
        if isinstance(parsed, list):
            items = [str(item).strip() for item in parsed if str(item).strip()]
            return dedupe_list(items)
        if "," in value:
            items = [part.strip() for part in value.split(",") if part.strip()]
            return dedupe_list(items)
    return []


def short_summary(text: str, max_words: int = 30) -> str:
    words = (text or "").split()
    if not words:
        return ""
    if len(words) <= max_words:
        return " ".join(words)
    trimmed = " ".join(words[:max_words]).rstrip(" ,;:")
    if trimmed.endswith("."):
        return trimmed
    return f"{trimmed}."


def normalize_summary_fields(payload: dict[str, Any]) -> response_schemas.SummaryFields:
    description = extract_text(payload.get("description") or payload.get("summary"), ("description", "summary"))
    summary = extract_text(payload.get("summary"), ("summary", "description"))
    if not summary:
        summary = short_summary(description)
    if not description:
        description = summary

    return response_schemas.SummaryFields(
        description=description,
        summary=summary,
        calories=to_number(payload.get("calories")),
        protein=to_number(payload.get("protein")),
        carbs=to_number(payload.get("carbs")),
        fats=to_number(payload.get("fats")),
        ingredients=to_str_list(payload.get("ingredients")),
        allergens=to_str_list(payload.get("allergens")),
    )


_ALLERGEN_KEYWORDS: dict[str, list[str]] = {
    "shellfish": ["shrimp", "prawn", "crab", "lobster", "scallop", "mussel", "clam", "oyster"],
    "fish": ["salmon", "tuna", "cod", "tilapia", "sardine", "anchovy", "trout"],
    "dairy": ["milk", "cheese", "butter", "cream", "yogurt"],
    "egg": ["egg", "eggs"],
    "wheat": ["wheat", "flour", "bread", "pasta", "noodle", "barley", "rye"],
    "soy": ["soy", "tofu", "soybean", "edamame"],
    "peanut": ["peanut", "peanuts"],
    "tree nuts": ["almond", "cashew", "walnut", "pecan", "hazelnut", "pistachio"],
    "sesame": ["sesame"],
}


def normalize_sources(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items = [str(item).strip() for item in value if str(item).strip()]
        return dedupe_list(items)
    if isinstance(value, str):
        parsed = maybe_parse_json_text(value)
        if isinstance(parsed, list):
            items = [str(item).strip() for item in parsed if str(item).strip()]
            return dedupe_list(items)
        text = value.strip()
        if text.startswith("http"):
            return [text]
    return []


def infer_allergens(ingredients: list[str]) -> list[str]:
    if not ingredients:
        return []
    lowered = [item.lower() for item in ingredients]
    found: list[str] = []
    for allergen, keywords in _ALLERGEN_KEYWORDS.items():
        if any(keyword in ingredient for ingredient in lowered for keyword in keywords):
            found.append(allergen)
    return found


def normalize_dish_detail(
    payload: dict[str, Any],
    fallback_name: str,
    ingredients: list[str] | None,
    sources: list[str] | None,
) -> response_schemas.DishDetailResponse:
    name = extract_text(payload.get("name") or payload.get("dish") or fallback_name, ("name", "dish"))
    if not name:
        name = fallback_name

    description = extract_text(payload.get("description") or payload.get("summary"), ("description", "summary"))
    summary = extract_text(payload.get("summary"), ("summary", "description"))
    if not summary:
        summary = short_summary(description)
    if not description:
        description = summary

    macros = payload.get("macros") if isinstance(payload.get("macros"), dict) else {}
    calories = to_number(payload.get("calories") or macros.get("calories_kcal"))
    protein = to_number(payload.get("protein") or macros.get("protein_g"))
    carbs = to_number(payload.get("carbs") or macros.get("carbs_g"))
    fats = to_number(payload.get("fats") or macros.get("fat_g"))

    ingredient_list = to_str_list(payload.get("ingredients"))
    if not ingredient_list and ingredients:
        ingredient_list = dedupe_list([item for item in ingredients if item])

    allergen_list = to_str_list(payload.get("allergens"))
    if not allergen_list:
        allergen_list = infer_allergens(ingredient_list)

    source_list = normalize_sources(payload.get("sources")) or normalize_sources(sources or [])

    return response_schemas.DishDetailResponse(
        name=name,
        description=description,
        calories=calories,
        protein=protein,
        carbs=carbs,
        fats=fats,
        ingredients=ingredient_list,
        allergens=allergen_list,
        summary=summary,
        sources=source_list,
    )


def build_sources(raw_sources: list[dict[str, str]], limit: int = 5) -> list[str]:
    sources: list[str] = []
    for item in raw_sources[:limit]:
        url = item.get("href") or item.get("url") or ""
        if url:
            sources.append(url)
    return dedupe_list(sources)


def sources_to_text(raw_sources: list[dict[str, str]], limit: int = 5) -> str:
    if not raw_sources:
        return ""

    lines: list[str] = []
    for item in raw_sources[:limit]:
        title = item.get("title") or "Untitled"
        snippet = item.get("body") or item.get("snippet") or ""
        line = f"{title}: {snippet}".strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def sanitize_bytes(obj: Any) -> Any:
    if isinstance(obj, bytes):
        try:
            return obj.decode("utf-8")
        except UnicodeDecodeError:
            return "<binary_data>"
    elif isinstance(obj, dict):
        return {k: sanitize_bytes(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_bytes(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(sanitize_bytes(v) for v in obj)
    return obj
