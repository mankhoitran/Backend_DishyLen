"""FastAPI entrypoint for food agent system."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
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
from agent.agent import FoodAgent
from config import get_settings
from db import crud
from db.database import Base, engine, get_db
from db.models import Dish
from schemas.request import OCRItemsRequest, OCRSelectRequest, QueryRequest
from schemas.response import (
    DishListResponse,
    DishResponse,
    MenuScanResponse,
    OCRItemsResponse,
    OCRSelectResponse,
    OCRUploadResponse,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

settings = get_settings()

LOG_DIR = Path(__file__).resolve().parent / "logs"
OCR_LOG_PATH = LOG_DIR / "ocr.log"


def _configure_ocr_logger() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ocr_logger = logging.getLogger("agent_vllm.ocr")
    if not any(
        isinstance(handler, logging.FileHandler)
        and getattr(handler, "baseFilename", "") == str(OCR_LOG_PATH)
        for handler in ocr_logger.handlers
    ):
        handler = logging.FileHandler(OCR_LOG_PATH)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        )
        ocr_logger.addHandler(handler)
    ocr_logger.setLevel(logging.INFO)
    ocr_logger.propagate = False
    return ocr_logger


def _one_line(value: str) -> str:
    return " ".join((value or "").split())


ocr_logger = _configure_ocr_logger()

app = FastAPI(title=settings.app_name, version=settings.app_version)

UPLOAD_DIR = Path(__file__).resolve().parent / settings.uploads_dir
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

origins = [origin.strip() for origin in settings.cors_origins.split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    """Initialize required resources."""

    Base.metadata.create_all(bind=engine)


@app.get("/health")
def health_check() -> dict[str, str]:
    """Health endpoint."""

    return {"status": "ok"}


@app.post("/query", response_model=DishResponse)
def query_dish(payload: QueryRequest, db: Session = Depends(get_db)) -> DishResponse:
    """Process food query through tool-based agent."""

    try:
        agent = FoodAgent(db=db)
        result = agent.run(query=payload.query, target_language=payload.target_language)
        return DishResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - runtime guard for external API failures
        raise HTTPException(status_code=500, detail=f"Failed to process query: {exc}") from exc


@app.post("/vllm/ocr/upload", response_model=OCRUploadResponse)
async def upload_menu_image(file: UploadFile = File(...)) -> OCRUploadResponse:
    """Upload a menu image and return a path reference for OCR."""

    image_path = await _save_upload(file)
    return OCRUploadResponse(
        image_path=str(image_path),
        image_url=_make_image_url(image_path),
    )


@app.post("/vllm/ocr/items", response_model=OCRItemsResponse)
def ocr_menu_items(payload: OCRItemsRequest) -> OCRItemsResponse:
    """Extract menu items from an uploaded image path using vLLM post-processing."""

    try:
        ocr_result = ocr_menu_image(payload.image_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    items = extract_menu_items(ocr_result.text, max_items=payload.max_items)
    raw_items = list(items)
    corrected_items: list[str] = []
    corrected_text = ""
    if ocr_result.text:
        ocr_payload = apply_ocr_prompt(
            ocr_result.text,
            fallback_items=items,
            prefer_backend="vllm",
        )
        corrected_text = ocr_payload.get("corrected_text", "")
        corrected_items = corrected_items_from_text(
            corrected_text,
            max_items=payload.max_items,
        )
        if corrected_items:
            items = corrected_items

    fallback_used = False
    if not items and settings.scan_fallback_items:
        items = [
            item.strip()
            for item in settings.scan_fallback_items.split(",")
            if item.strip()
        ][: payload.max_items]
        fallback_used = True

    ocr_logger.info(
        "ocr_items | image_path=%s | status=%s | text=%s | raw_items=%s | corrected_text=%s | "
        "corrected_items=%s | fallback_used=%s",
        ocr_result.image_path,
        ocr_result.status,
        _one_line(ocr_result.text),
        json.dumps(raw_items, ensure_ascii=True),
        _one_line(corrected_text),
        json.dumps(corrected_items, ensure_ascii=True),
        fallback_used,
    )

    return OCRItemsResponse(
        image_path=ocr_result.image_path,
        image_url=_make_image_url(Path(ocr_result.image_path)),
        ocr_status=ocr_result.status,
        ocr_text=ocr_result.text,
        items=items,
    )


@app.post("/vllm/ocr/select", response_model=OCRSelectResponse)
def ocr_menu_select(payload: OCRSelectRequest, db: Session = Depends(get_db)) -> OCRSelectResponse:
    """Select a menu item and retrieve dish details via vLLM."""

    try:
        ocr_result = ocr_menu_image(payload.image_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    items = extract_menu_items(ocr_result.text, max_items=payload.max_items)
    raw_items = list(items)
    corrected_items: list[str] = []
    corrected_text = ""
    if ocr_result.text:
        ocr_payload = apply_ocr_prompt(
            ocr_result.text,
            fallback_items=items,
            prefer_backend="vllm",
        )
        corrected_text = ocr_payload.get("corrected_text", "")
        corrected_items = corrected_items_from_text(
            corrected_text,
            max_items=payload.max_items,
        )
        if corrected_items:
            items = corrected_items

    if not items and payload.item_name:
        items = [payload.item_name]

    try:
        selected_item = select_menu_item(items, payload.item_name, payload.item_index)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    ocr_logger.info(
        "ocr_select | image_path=%s | status=%s | text=%s | raw_items=%s | corrected_text=%s | "
        "corrected_items=%s | item_name=%s | item_index=%s | selected_item=%s",
        ocr_result.image_path,
        ocr_result.status,
        _one_line(ocr_result.text),
        json.dumps(raw_items, ensure_ascii=True),
        _one_line(corrected_text),
        json.dumps(corrected_items, ensure_ascii=True),
        payload.item_name,
        payload.item_index,
        selected_item,
    )

    try:
        agent = VLLMFoodAgent(db=db)
        dish_payload = agent.run(query=selected_item, target_language=payload.target_language)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - runtime guard for external API failures
        raise HTTPException(status_code=500, detail=f"Failed to retrieve dish info: {exc}") from exc

    ingredients: list[str] = []
    try:
        vllm_client = VLLMClient()
        search_service = DuckDuckGoSearchService(
            vllm_client,
            max_results=settings.duckduckgo_max_results,
        )
        ingredient_payload = search_service.get_dish_ingredients(selected_item)
        raw_ingredients = ingredient_payload.get("ingredients")
        if isinstance(raw_ingredients, list):
            ingredients = [str(item).strip() for item in raw_ingredients if str(item).strip()]
    except Exception:
        ingredients = []

    return OCRSelectResponse(
        image_path=ocr_result.image_path,
        image_url=_make_image_url(Path(ocr_result.image_path)),
        selected_item=selected_item,
        dish_info=DishResponse(**dish_payload),
        ingredients=ingredients,
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


@app.post("/menu/scan", response_model=MenuScanResponse)
async def scan_menu(
    file: UploadFile = File(...),
    max_items: int = Form(default=settings.scan_max_items),
    target_language: str | None = Form(default=None),
    db: Session = Depends(get_db),
) -> MenuScanResponse:
    """Upload a menu image, extract dish names, and fetch dish details."""

    if max_items < 1 or max_items > 20:
        raise HTTPException(status_code=400, detail="max_items must be between 1 and 20")

    suffix = Path(file.filename or "").suffix
    filename = f"{uuid4().hex}{suffix}"
    image_path = UPLOAD_DIR / filename

    try:
        image_path.write_bytes(await file.read())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {exc}") from exc

    try:
        ocr_result = ocr_menu_image(image_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    items = extract_menu_items(ocr_result.text, max_items=max_items)
    raw_items = list(items)
    corrected_items: list[str] = []
    corrected_text = ""
    if ocr_result.text:
        ocr_payload = apply_ocr_prompt(ocr_result.text, fallback_items=items)
        corrected_text = ocr_payload.get("corrected_text", "")
        corrected_items = corrected_items_from_text(corrected_text, max_items=max_items)
        if corrected_items:
            items = corrected_items

    fallback_used = False
    if not items and settings.scan_fallback_items:
        items = [
            item.strip()
            for item in settings.scan_fallback_items.split(",")
            if item.strip()
        ]
        fallback_used = True

    ocr_logger.info(
        "scan_menu | image_path=%s | status=%s | text=%s | raw_items=%s | corrected_text=%s | "
        "corrected_items=%s | fallback_used=%s",
        str(image_path),
        ocr_result.status,
        _one_line(ocr_result.text),
        json.dumps(raw_items, ensure_ascii=True),
        _one_line(corrected_text),
        json.dumps(corrected_items, ensure_ascii=True),
        fallback_used,
    )

    agent: FoodAgent | None
    try:
        agent = FoodAgent(db=db)
    except ValueError:
        agent = None
    results: list[DishResponse] = []
    for item in items[:max_items]:
        cached = crud.get_dish_by_name(db, item)
        if cached:
            results.append(_dish_to_response(cached))
            continue

        try:
            if agent is None:
                raise RuntimeError("Gemini API key not configured")
            dish_payload = agent.run(query=item, target_language=target_language)
            results.append(DishResponse(**dish_payload))
        except Exception as exc:  # pragma: no cover - external API errors
            results.append(
                DishResponse(
                    dish=item,
                    spicy_level="unknown",
                    macros={},
                    summary=f"Failed to fetch details: {exc}",
                    image_url="",
                    source="search",
                )
            )

    return MenuScanResponse(
        image_path=str(image_path),
        ocr_status=ocr_result.status,
        ocr_text=ocr_result.text,
        items=results,
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
