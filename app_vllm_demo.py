"""FastAPI demo app for the vLLM-backed agent and summarization tests."""

from __future__ import annotations

import logging
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException
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
from db.database import Base, engine, get_db
from schemas.request import QueryRequest
from schemas.response import DishResponse


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

settings = get_settings()

app = FastAPI(title=f"{settings.app_name} (vLLM demo)", version=settings.app_version)


class SourceItem(BaseModel):
    """Readable source snippet for summary verification."""

    title: str = Field(default="")
    snippet: str = Field(default="")
    url: str = Field(default="")


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


class SummaryResponse(BaseModel):
    """Response payload for summarization tests."""

    summary: str = Field(default="")
    input_type: Literal["text", "search"]
    sources: list[SourceItem] = Field(default_factory=list)
    target_language: str | None = Field(default=None)


class OCRMenuRequest(BaseModel):
    """Request payload for OCR menu extraction."""

    image_path: str = Field(..., description="Path to the menu image")
    max_items: int = Field(default=40, ge=1, le=200)
    ocr_backend: Literal["auto", "vllm", "gemini"] = Field(
        default="auto",
        description="LLM backend for OCR post-processing",
    )


class OCRMenuResponse(BaseModel):
    """Response payload for OCR menu extraction."""

    image_path: str
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
    ocr_backend: Literal["auto", "vllm", "gemini"] = Field(
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
    ocr_status: str
    raw_text: str = Field(default="")
    corrected_text: str = Field(default="")
    selected_item: str
    dish_info: DishResponse
    ingredients: list[str] = Field(default_factory=list)
    items: list[str] = Field(default_factory=list)


@app.on_event("startup")
def on_startup() -> None:
    """Initialize required resources."""

    Base.metadata.create_all(bind=engine)


@app.get("/health")
def health_check() -> dict[str, str]:
    """Health endpoint."""

    return {"status": "ok"}


@app.post("/vllm/query", response_model=DishResponse)
def query_dish(payload: QueryRequest, db: Session = Depends(get_db)) -> DishResponse:
    """Process food query through the vLLM-backed agent."""

    try:
        agent = VLLMFoodAgent(db=db)
        result = agent.run(query=payload.query, target_language=payload.target_language)
        return DishResponse(**result)
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

    summary = ""
    sources: list[SourceItem] = []
    input_type: Literal["text", "search"] = "text"

    try:
        if payload.text:
            summary = _summarize_text(vllm_client, payload.text, payload.max_words)
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
            summary_payload = search_service.get_dish_summary(query)
            summary = summary_payload.get("summary", "")
            if payload.include_sources:
                sources = _build_sources(raw_sources)

        if payload.target_language and summary:
            summary = _translate_text(vllm_client, summary, payload.target_language)
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - runtime guard for external API failures
        raise HTTPException(status_code=500, detail=f"Failed to summarize: {exc}") from exc

    return SummaryResponse(
        summary=summary,
        input_type=input_type,
        sources=sources,
        target_language=payload.target_language,
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
        dish_info = DishResponse(**dish_payload)

        ingredients: list[str] = []
        if payload.include_ingredients:
            search_service = DuckDuckGoSearchService(
                vllm_client,
                max_results=settings.duckduckgo_max_results,
            )
            ingredient_payload = search_service.get_dish_ingredients(selected_item)
            raw_ingredients = ingredient_payload.get("ingredients")
            if isinstance(raw_ingredients, list):
                ingredients = [str(item).strip() for item in raw_ingredients if str(item).strip()]
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - runtime guard for external API failures
        raise HTTPException(status_code=500, detail=f"Failed to retrieve dish info: {exc}") from exc

    return OCRSelectResponse(
        image_path=ocr_result.image_path,
        ocr_status=ocr_result.status,
        raw_text=ocr_payload.get("raw_text", ""),
        corrected_text=ocr_payload.get("corrected_text", ""),
        selected_item=selected_item,
        dish_info=dish_info,
        ingredients=ingredients,
        items=items,
    )


def _summarize_text(vllm_client: VLLMClient, text: str, max_words: int) -> str:
    instruction = f"Return ONLY JSON with key summary. summary under {max_words} words."
    payload = vllm_client.generate_json(
        system_prompt="You are a concise summarization engine.",
        user_prompt=f"{instruction}\nText:\n{text}",
        fallback={"summary": ""},
    )
    return (payload.get("summary") or "").strip()


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


def _build_sources(raw_sources: list[dict[str, str]], limit: int = 5) -> list[SourceItem]:
    sources: list[SourceItem] = []
    for item in raw_sources[:limit]:
        sources.append(
            SourceItem(
                title=item.get("title") or "Untitled",
                snippet=item.get("body") or item.get("snippet") or "",
                url=item.get("href") or item.get("url") or "",
            )
        )
    return sources
