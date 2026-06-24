"""API request schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    """Incoming query payload from clients."""
    query: str = Field(..., description="User question about a dish or menu item")
    target_language: str | None = Field(
        default=None,
        description="Optional language code to translate summary output, e.g. 'en', 'vi', 'th'",
    )

class OCRMenuRequest(BaseModel):
    """Request payload for OCR menu extraction."""
    image_path: str = Field(..., description="Path to the menu image")
    max_items: int = Field(default=40, ge=1, le=200)
    ocr_backend: Literal["auto", "vllm", "openrouter", "gemini"] = Field(
        default="auto",
        description="LLM backend for OCR post-processing",
    )

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

class DishInfoRequest(BaseModel):
    """Lightweight request payload for fetching dish info by name (no OCR required)."""
    item_name: str = Field(..., description="Exact dish name already extracted from the menu")
    include_ingredients: bool = Field(default=True)
    target_language: str | None = Field(
        default=None,
        description="Optional language code for translation, e.g. 'en', 'vi', 'th'",
    )

class HistoryCreateRequest(BaseModel):
    """Incoming history entry payload."""
    type: Literal["query", "ocr", "summary"]
    title: str = Field(..., min_length=1, max_length=255)
    payload: dict[str, Any] = Field(default_factory=dict)

class TranslateRequest(BaseModel):
    """Payload for text translation."""
    text: str = Field(..., description="Text to translate")
    target_language: str | None = Field(default=None, description="Language to translate to")
    language: str | None = Field(default=None, description="Alias for target_language")

class UserProfileUpdateRequest(BaseModel):
    """Payload for updating user profile."""
    allergies: str | None = Field(default=None, description="Free-form text describing user allergies")

class AddAllergyRequest(BaseModel):
    """Payload for adding new allergies to a user's profile."""
    text: str | None = Field(default=None, description="Free-form text containing multiple ingredients")
    allergies: str | None = Field(default=None)
    description: str | None = Field(default=None)

class RegisterRequest(BaseModel):
    """Payload for traditional email/password registration."""
    email: str = Field(..., description="User's email address")
    password: str = Field(..., min_length=6, description="User's password")
    name: str | None = Field(default=None, description="User's name")

class LoginRequest(BaseModel):
    """Payload for traditional email/password login."""
    email: str = Field(..., description="User's email address")
    password: str = Field(..., description="User's password")
