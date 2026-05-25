"""FastAPI entrypoint for food agent system."""

from __future__ import annotations

import logging
from pathlib import Path
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from agent_vllm.ocr_menu import apply_ocr_prompt, corrected_items_from_text, extract_menu_items, ocr_menu_image
from agent.agent import FoodAgent
from config import get_settings
from db import crud
from db.database import Base, engine, get_db
from db.models import Dish
from schemas.request import QueryRequest
from schemas.response import DishListResponse, DishResponse, MenuScanResponse


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

settings = get_settings()

app = FastAPI(title=settings.app_name, version=settings.app_version)

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

    upload_dir = Path(__file__).resolve().parent / settings.uploads_dir
    upload_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "").suffix
    filename = f"{uuid4().hex}{suffix}"
    image_path = upload_dir / filename

    try:
        image_path.write_bytes(await file.read())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {exc}") from exc

    try:
        ocr_result = ocr_menu_image(image_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    items = extract_menu_items(ocr_result.text, max_items=max_items)
    if ocr_result.text:
        ocr_payload = apply_ocr_prompt(ocr_result.text, fallback_items=items)
        corrected_items = corrected_items_from_text(ocr_payload.get("corrected_text", ""), max_items=max_items)
        if corrected_items:
            items = corrected_items

    if not items and settings.scan_fallback_items:
        items = [
            item.strip()
            for item in settings.scan_fallback_items.split(",")
            if item.strip()
        ]

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
