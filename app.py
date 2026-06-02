"""FastAPI demo app for the vLLM-backed agent and summarization tests."""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from agent_vllm.agent import VLLMFoodAgent
from agent_vllm.ocr_menu import (
    apply_ocr_prompt,
    corrected_items_from_text,
    extract_menu_items,
    ocr_menu_image,
    select_menu_item,
)
from agent_vllm.search import DuckDuckGoSearchService
from agent_vllm.vllm_client import VLLMClient
from config import get_settings
from db import crud
from db.database import Base, engine, get_db
from db.models import Dish, HistoryEntry, User
import schemas.request as request_schemas
import schemas.response as response_schemas
from prompt.loader import format_prompt, load_prompt
from schemas.request import QueryRequest
from schemas.response import DishListResponse, DishResponse, OCRUploadResponse
from services.auth import AuthError, create_access_token, decode_access_token, verify_google_id_token
from services.schema_logger import collect_pydantic_models, log_schema_snapshot


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

settings = get_settings()

app = FastAPI(title=f"{settings.app_name} (vLLM demo)", version=settings.app_version)

origins = [origin.strip() for origin in settings.cors_origins.split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path(__file__).resolve().parent / settings.uploads_dir
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


class SummaryRequest(BaseModel):
    """Request payload for summarization tests."""

    query: str | None = Field(default=None, description="Search query for sources")
    text: str | None = Field(default=None, description="Raw text to summarize")
    max_words: int = Field(default=80, ge=20, le=200)
    include_sources: bool = Field(
        default=True,
        description="Include source snippets when using query-based summaries",
    )
    target_language: str | None = Field(
        default=None,
        description="Optional language code for translation, e.g. 'en', 'vi', 'th'",
    )


class SummaryFields(BaseModel):
    """Structured summary fields extracted from LLM output."""

    description: str = Field(default="")
    summary: str = Field(default="")
    calories: float | None = Field(default=None)
    protein: float | None = Field(default=None)
    carbs: float | None = Field(default=None)
    fats: float | None = Field(default=None)
    ingredients: list[str] = Field(default_factory=list)
    allergens: list[str] = Field(default_factory=list)


class SummaryResponse(BaseModel):
    """Response payload for summarization tests."""

    summary: str = Field(default="")
    description: str = Field(default="")
    calories: float | None = Field(default=None)
    protein: float | None = Field(default=None)
    carbs: float | None = Field(default=None)
    fats: float | None = Field(default=None)
    ingredients: list[str] = Field(default_factory=list)
    allergens: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    input_type: Literal["text", "search"]
    target_language: str | None = Field(default=None)


class DishDetailResponse(BaseModel):
    """Frontend-friendly dish detail payload."""

    name: str = Field(default="")
    description: str = Field(default="")
    calories: float | None = Field(default=None)
    protein: float | None = Field(default=None)
    carbs: float | None = Field(default=None)
    fats: float | None = Field(default=None)
    ingredients: list[str] = Field(default_factory=list)
    allergens: list[str] = Field(default_factory=list)
    summary: str = Field(default="")
    sources: list[str] = Field(default_factory=list)


class OCRMenuRequest(BaseModel):
    """Request payload for OCR menu extraction."""

    image_path: str = Field(..., description="Path to the menu image")
    max_items: int = Field(default=40, ge=1, le=200)
    ocr_backend: Literal["auto", "vllm", "openrouter", "gemini"] = Field(
        default="auto",
        description="LLM backend for OCR post-processing",
    )


class OCRMenuResponse(BaseModel):
    """Response payload for OCR menu extraction."""

    image_path: str
    image_url: str = Field(default="")
    ocr_status: str
    ocr_text: str
    raw_text: str = Field(default="")
    corrected_text: str = Field(default="")
    items: list[str] = Field(default_factory=list)


class OCRSelectRequest(BaseModel):
    """Request payload for selecting a menu item from OCR output."""

    image_path: str = Field(..., description="Path to the menu image")
    item_name: str | None = Field(default=None, description="Exact item name to select")
    item_index: int | None = Field(default=None, description="Index from extracted items")
    max_items: int = Field(default=40, ge=1, le=200)
    ocr_backend: Literal["auto", "vllm", "openrouter", "gemini"] = Field(
        default="auto",
        description="LLM backend for OCR post-processing",
    )
    include_ingredients: bool = Field(default=True)
    target_language: str | None = Field(
        default=None,
        description="Optional language code for translation, e.g. 'en', 'vi', 'th'",
    )


class OCRSelectResponse(BaseModel):
    """Response payload for OCR menu selection."""

    image_path: str
    image_url: str = Field(default="")
    ocr_status: str
    raw_text: str = Field(default="")
    corrected_text: str = Field(default="")
    selected_item: str
    dish_info: DishDetailResponse
    ingredients: list[str] = Field(default_factory=list)
    items: list[str] = Field(default_factory=list)


@app.on_event("startup")
def on_startup() -> None:
    """Initialize required resources."""

    Base.metadata.create_all(bind=engine)
    try:
        current_module = sys.modules[__name__]
        models = collect_pydantic_models(request_schemas, response_schemas, current_module)
        log_schema_snapshot(Base, models)
    except Exception as exc:
        logging.getLogger(__name__).warning("Failed to log schema snapshot: %s", exc)


def _current_user(
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
) -> User:
    """Resolve the current user from a bearer token."""

    if not authorization:
        raise HTTPException(status_code=401, detail="Missing authorization header")

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=401, detail="Invalid authorization header")

    try:
        claims = decode_access_token(token)
    except AuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    user_id = claims.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid access token")

    try:
        user = crud.get_user_by_id(db, int(user_id))
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=401, detail="Invalid access token") from exc
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


@app.get("/health")
def health_check() -> dict[str, str]:
    """Health endpoint."""

    return {"status": "ok"}


@app.post("/auth/google", response_model=response_schemas.AuthResponse)
def google_login(
    payload: request_schemas.GoogleAuthRequest,
    db: Session = Depends(get_db),
) -> response_schemas.AuthResponse:
    """Authenticate with a Google ID token and return an app access token."""

    try:
        claims = verify_google_id_token(payload.id_token)
        user = crud.upsert_user(
            db,
            google_sub=str(claims["sub"]),
            email=str(claims["email"]),
            name=str(claims.get("name") or ""),
            picture_url=str(claims.get("picture") or ""),
        )
        token = create_access_token(
            user_id=user.id,
            email=user.email,
            name=user.name,
            picture_url=user.picture_url,
        )
    except AuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    return response_schemas.AuthResponse(
        access_token=token,
        user=_user_to_response(user),
    )


@app.get("/auth/me", response_model=response_schemas.UserResponse)
def get_me(current_user: User = Depends(_current_user)) -> response_schemas.UserResponse:
    """Return the current authenticated user."""

    return _user_to_response(current_user)


@app.post("/history", response_model=response_schemas.HistoryEntryResponse)
def create_history(
    payload: request_schemas.HistoryCreateRequest,
    current_user: User = Depends(_current_user),
    db: Session = Depends(get_db),
) -> response_schemas.HistoryEntryResponse:
    """Persist one history entry for the current user."""

    entry = crud.create_history_entry(
        db,
        user_id=current_user.id,
        entry_type=payload.type,
        title=payload.title,
        payload=payload.payload,
    )
    return _history_to_response(entry, current_user)


@app.get("/history", response_model=response_schemas.HistoryListResponse)
def list_history(
    type: Literal["query", "ocr", "summary"] | None = None,
    limit: int = 100,
    offset: int = 0,
    current_user: User = Depends(_current_user),
    db: Session = Depends(get_db),
) -> response_schemas.HistoryListResponse:
    """List history entries for the current user."""

    if limit < 1 or limit > 100:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 100")
    if offset < 0:
        raise HTTPException(status_code=400, detail="offset must be >= 0")

    items, total = crud.list_history_entries(
        db,
        user_id=current_user.id,
        entry_type=type,
        limit=limit,
        offset=offset,
    )
    return response_schemas.HistoryListResponse(
        items=[_history_to_response(item, current_user) for item in items],
        total=total,
    )


@app.get("/dishes", response_model=DishListResponse)
def list_dishes(
    q: str | None = None,
    limit: int = 20,
    offset: int = 0,
    db: Session = Depends(get_db),
) -> DishListResponse:
    """List dishes from the local database."""

    if limit < 1 or limit > 100:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 100")
    if offset < 0:
        raise HTTPException(status_code=400, detail="offset must be >= 0")

    items, total = crud.list_dishes(db, query=q, limit=limit, offset=offset)
    payload = [_dish_to_response(item) for item in items]
    return DishListResponse(items=payload, total=total)


@app.post("/vllm/query", response_model=DishDetailResponse)
def query_dish(payload: QueryRequest, db: Session = Depends(get_db)) -> DishDetailResponse:
    """Process food query through the vLLM-backed agent."""

    try:
        agent = VLLMFoodAgent(db=db)
        result = agent.run(query=payload.query, target_language=payload.target_language)
        ingredients: list[str] = []
        try:
            vllm_client = VLLMClient()
            search_service = DuckDuckGoSearchService(
                vllm_client,
                max_results=settings.duckduckgo_max_results,
            )
            ingredient_payload = search_service.get_dish_ingredients(
                result.get("dish", payload.query)
            )
            ingredients = _to_str_list(ingredient_payload.get("ingredients"))
        except Exception:
            ingredients = []

        return _normalize_dish_detail(
            result,
            fallback_name=payload.query,
            ingredients=ingredients,
            sources=[],
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - runtime guard for external API failures
        raise HTTPException(status_code=500, detail=f"Failed to process query: {exc}") from exc


@app.post("/vllm/summary", response_model=SummaryResponse)
def summarize(payload: SummaryRequest) -> SummaryResponse:
    """Summarize either raw text or search-backed sources."""

    if not payload.text and not payload.query:
        raise HTTPException(status_code=400, detail="Provide either 'text' or 'query'.")

    try:
        vllm_client = VLLMClient()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    summary_fields = SummaryFields()
    sources: list[str] = []
    input_type: Literal["text", "search"] = "text"

    try:
        if payload.text:
            summary_fields = _summarize_text(vllm_client, payload.text, payload.max_words)
        else:
            query = (payload.query or "").strip()
            if not query:
                raise HTTPException(
                    status_code=400,
                    detail="Query is required when text is empty.",
                )
            input_type = "search"
            search_service = DuckDuckGoSearchService(
                vllm_client,
                max_results=settings.duckduckgo_max_results,
            )
            raw_sources = search_service.search_sources(query)
            source_text = _sources_to_text(raw_sources)
            if source_text:
                summary_fields = _summarize_text(vllm_client, source_text, payload.max_words)
            else:
                summary_fields = SummaryFields(
                    description="No sources found.",
                    summary="No sources found.",
                )
            if payload.include_sources:
                sources = _build_sources(raw_sources)

        description_text = summary_fields.description or summary_fields.summary
        summary_text = summary_fields.summary or _short_summary(description_text)
        if payload.target_language and description_text:
            translated = _translate_text(vllm_client, description_text, payload.target_language)
            description_text = translated
            summary_text = _short_summary(translated)
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - runtime guard for external API failures
        raise HTTPException(status_code=500, detail=f"Failed to summarize: {exc}") from exc

    return SummaryResponse(
        summary=summary_text,
        description=description_text,
        calories=summary_fields.calories,
        protein=summary_fields.protein,
        carbs=summary_fields.carbs,
        fats=summary_fields.fats,
        ingredients=summary_fields.ingredients,
        allergens=summary_fields.allergens,
        sources=sources,
        input_type=input_type,
        target_language=payload.target_language,
    )


@app.post("/vllm/ocr/upload", response_model=OCRUploadResponse)
async def upload_menu_image(file: UploadFile = File(...)) -> OCRUploadResponse:
    """Upload a menu image and return a path reference for OCR."""

    image_path = await _save_upload(file)
    return OCRUploadResponse(
        image_path=str(image_path),
        image_url=_make_image_url(image_path),
    )


@app.post("/vllm/ocr/items", response_model=OCRMenuResponse)
def ocr_menu_items(payload: OCRMenuRequest) -> OCRMenuResponse:
    """Extract menu items from an OCR image."""

    try:
        ocr_result = ocr_menu_image(payload.image_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    items = extract_menu_items(ocr_result.text, max_items=payload.max_items)
    ocr_payload = apply_ocr_prompt(
        ocr_result.text,
        fallback_items=items,
        prefer_backend=payload.ocr_backend,
    )
    corrected_items = corrected_items_from_text(
        ocr_payload.get("corrected_text", ""),
        max_items=payload.max_items,
    )
    if corrected_items:
        items = corrected_items

    return OCRMenuResponse(
        image_path=ocr_result.image_path,
        image_url=_make_image_url(Path(ocr_result.image_path)),
        ocr_status=ocr_result.status,
        ocr_text=ocr_result.text,
        raw_text=ocr_payload.get("raw_text", ""),
        corrected_text=ocr_payload.get("corrected_text", ""),
        items=items,
    )


@app.post("/vllm/ocr/select", response_model=OCRSelectResponse)
def ocr_menu_select(payload: OCRSelectRequest, db: Session = Depends(get_db)) -> OCRSelectResponse:
    """Select a menu item from OCR output and retrieve dish information."""

    try:
        ocr_result = ocr_menu_image(payload.image_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    items = extract_menu_items(ocr_result.text, max_items=payload.max_items)
    ocr_payload = apply_ocr_prompt(
        ocr_result.text,
        fallback_items=items,
        prefer_backend=payload.ocr_backend,
    )
    corrected_items = corrected_items_from_text(
        ocr_payload.get("corrected_text", ""),
        max_items=payload.max_items,
    )
    if corrected_items:
        items = corrected_items

    try:
        selected_item = select_menu_item(items, payload.item_name, payload.item_index)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        vllm_client = VLLMClient()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        agent = VLLMFoodAgent(db=db)
        dish_payload = agent.run(query=selected_item, target_language=payload.target_language)

        ingredients: list[str] = []
        if payload.include_ingredients:
            search_service = DuckDuckGoSearchService(
                vllm_client,
                max_results=settings.duckduckgo_max_results,
            )
            ingredient_payload = search_service.get_dish_ingredients(selected_item)
            raw_ingredients = ingredient_payload.get("ingredients")
            ingredients = _to_str_list(raw_ingredients)
        dish_info = _normalize_dish_detail(
            dish_payload,
            fallback_name=selected_item,
            ingredients=ingredients,
            sources=[],
        )
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - runtime guard for external API failures
        raise HTTPException(status_code=500, detail=f"Failed to retrieve dish info: {exc}") from exc

    return OCRSelectResponse(
        image_path=ocr_result.image_path,
        image_url=_make_image_url(Path(ocr_result.image_path)),
        ocr_status=ocr_result.status,
        raw_text=ocr_payload.get("raw_text", ""),
        corrected_text=ocr_payload.get("corrected_text", ""),
        selected_item=selected_item,
        dish_info=dish_info,
        ingredients=ingredients,
        items=items,
    )


def _dish_to_response(dish: Dish) -> DishResponse:
    return DishResponse(
        dish=dish.name,
        spicy_level=dish.spicy_level or "unknown",
        macros=dish.macros or {},
        summary=dish.summary or "",
        image_url="",
        source="database",
    )


def _user_to_response(user: User) -> response_schemas.UserResponse:
    return response_schemas.UserResponse(
        id=user.id,
        email=user.email,
        name=user.name,
        picture_url=user.picture_url,
    )


def _history_to_response(
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


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    return stripped


def _maybe_parse_json_text(text: str) -> Any | None:
    cleaned = _strip_code_fences(text)
    if not cleaned:
        return None
    if not (cleaned.startswith("{") or cleaned.startswith("[")):
        return None
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


def _extract_text(value: Any, preferred_keys: tuple[str, ...]) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        parsed = _maybe_parse_json_text(value)
        if isinstance(parsed, dict):
            return _extract_text(parsed, preferred_keys)
        if isinstance(parsed, list):
            return ", ".join(str(item).strip() for item in parsed if str(item).strip())
        return value.strip()
    if isinstance(value, dict):
        for key in preferred_keys:
            if key in value:
                text = _extract_text(value.get(key), preferred_keys)
                if text:
                    return text
        for item in value.values():
            text = _extract_text(item, preferred_keys)
            if text:
                return text
        return ""
    if isinstance(value, list):
        return ", ".join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip()


def _to_number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        for key in ("value", "amount"):
            if key in value:
                return _to_number(value.get(key))
        return None

    text = str(value).strip().lower()
    if not text or text in ("unknown", "n/a", "na"):
        return None
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    range_match = _RANGE_PATTERN.search(text)
    if range_match:
        low = float(range_match.group(1))
        high = float(range_match.group(2))
        return (low + high) / 2.0
    numbers = _NUMBER_PATTERN.findall(text)
    if not numbers:
        return None
    if len(numbers) >= 2 and (" to " in text or "-" in text):
        low = float(numbers[0])
        high = float(numbers[1])
        if low != high:
            return (low + high) / 2.0
    return float(numbers[0])


def _to_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items = [str(item).strip() for item in value if str(item).strip()]
        return _dedupe_list(items)
    if isinstance(value, str):
        parsed = _maybe_parse_json_text(value)
        if isinstance(parsed, list):
            items = [str(item).strip() for item in parsed if str(item).strip()]
            return _dedupe_list(items)
        if "," in value:
            items = [part.strip() for part in value.split(",") if part.strip()]
            return _dedupe_list(items)
    return []


def _dedupe_list(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _short_summary(text: str, max_words: int = 30) -> str:
    words = (text or "").split()
    if not words:
        return ""
    if len(words) <= max_words:
        return " ".join(words)
    trimmed = " ".join(words[:max_words]).rstrip(" ,;:")
    if trimmed.endswith("."):
        return trimmed
    return f"{trimmed}."


def _normalize_summary_fields(payload: dict[str, Any]) -> SummaryFields:
    description = _extract_text(payload.get("description") or payload.get("summary"), ("description", "summary"))
    summary = _extract_text(payload.get("summary"), ("summary", "description"))
    if not summary:
        summary = _short_summary(description)
    if not description:
        description = summary

    return SummaryFields(
        description=description,
        summary=summary,
        calories=_to_number(payload.get("calories")),
        protein=_to_number(payload.get("protein")),
        carbs=_to_number(payload.get("carbs")),
        fats=_to_number(payload.get("fats")),
        ingredients=_to_str_list(payload.get("ingredients")),
        allergens=_to_str_list(payload.get("allergens")),
    )


def _summarize_text(vllm_client: VLLMClient, text: str, max_words: int) -> SummaryFields:
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
    payload = vllm_client.generate_json(
        system_prompt=load_prompt("summary_system.txt").strip(),
        user_prompt=format_prompt("summary_user.txt", max_words=max_words, text=text),
        fallback=fallback,
    )
    return _normalize_summary_fields(payload)


def _translate_text(vllm_client: VLLMClient, text: str, target_language: str) -> str:
    instruction = (
        "Return ONLY JSON with keys target_language and translated_text. "
        f"target_language: {target_language}."
    )
    payload = vllm_client.generate_json(
        system_prompt="You are a translation engine.",
        user_prompt=f"{instruction}\ntext: {text}",
        fallback={"target_language": target_language, "translated_text": text},
    )
    return (payload.get("translated_text") or text).strip()


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


def _normalize_sources(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items = [str(item).strip() for item in value if str(item).strip()]
        return _dedupe_list(items)
    if isinstance(value, str):
        parsed = _maybe_parse_json_text(value)
        if isinstance(parsed, list):
            items = [str(item).strip() for item in parsed if str(item).strip()]
            return _dedupe_list(items)
        text = value.strip()
        if text.startswith("http"):
            return [text]
    return []


def _infer_allergens(ingredients: list[str]) -> list[str]:
    if not ingredients:
        return []
    lowered = [item.lower() for item in ingredients]
    found: list[str] = []
    for allergen, keywords in _ALLERGEN_KEYWORDS.items():
        if any(keyword in ingredient for ingredient in lowered for keyword in keywords):
            found.append(allergen)
    return found


def _normalize_dish_detail(
    payload: dict[str, Any],
    fallback_name: str,
    ingredients: list[str] | None,
    sources: list[str] | None,
) -> DishDetailResponse:
    name = _extract_text(payload.get("name") or payload.get("dish") or fallback_name, ("name", "dish"))
    if not name:
        name = fallback_name

    description = _extract_text(payload.get("description") or payload.get("summary"), ("description", "summary"))
    summary = _extract_text(payload.get("summary"), ("summary", "description"))
    if not summary:
        summary = _short_summary(description)
    if not description:
        description = summary

    macros = payload.get("macros") if isinstance(payload.get("macros"), dict) else {}
    calories = _to_number(payload.get("calories") or macros.get("calories_kcal"))
    protein = _to_number(payload.get("protein") or macros.get("protein_g"))
    carbs = _to_number(payload.get("carbs") or macros.get("carbs_g"))
    fats = _to_number(payload.get("fats") or macros.get("fat_g"))

    ingredient_list = _to_str_list(payload.get("ingredients"))
    if not ingredient_list and ingredients:
        ingredient_list = _dedupe_list([item for item in ingredients if item])

    allergen_list = _to_str_list(payload.get("allergens"))
    if not allergen_list:
        allergen_list = _infer_allergens(ingredient_list)

    source_list = _normalize_sources(payload.get("sources")) or _normalize_sources(sources or [])

    return DishDetailResponse(
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


def _build_sources(raw_sources: list[dict[str, str]], limit: int = 5) -> list[str]:
    sources: list[str] = []
    for item in raw_sources[:limit]:
        url = item.get("href") or item.get("url") or ""
        if url:
            sources.append(url)
    return _dedupe_list(sources)


def _sources_to_text(raw_sources: list[dict[str, str]], limit: int = 5) -> str:
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


async def _save_upload(file: UploadFile) -> Path:
    suffix = Path(file.filename or "").suffix
    filename = f"{uuid4().hex}{suffix}"
    image_path = UPLOAD_DIR / filename
    try:
        image_path.write_bytes(await file.read())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {exc}") from exc
    return image_path


def _make_image_url(image_path: Path) -> str:
    return f"/uploads/{image_path.name}"
