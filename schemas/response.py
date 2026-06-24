"""API response schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

class DishDetailResponse(BaseModel):
    """Frontend-friendly dish detail payload."""
    name: str = Field(default="")
    description: str = Field(default="")
    calories: float = Field(default=0.0)
    protein: float = Field(default=0.0)
    carbs: float = Field(default=0.0)
    fats: float = Field(default=0.0)
    ingredients: list[str] = Field(default_factory=list)
    allergens: list[str] = Field(default_factory=list)
    allergyWarning: bool = Field(default=False)
    summary: str = Field(default="")
    sources: list[str] = Field(default_factory=list)

class DishResponse(BaseModel):
    """Response payload for processed dish information."""
    dish: str = Field(..., description="Normalized dish name")
    spicy_level: str = Field(default="unknown")
    macros: dict[str, Any] = Field(default_factory=dict)
    summary: str = Field(default="")
    image_url: str = Field(default="")
    source: Literal["database", "search"]
    allergyWarning: bool = Field(default=False)

class UserResponse(BaseModel):
    """Response payload for a logged-in user."""
    model_config = ConfigDict(from_attributes=True)
    id: int
    email: str
    name: str
    picture_url: str
    allergies: str = ""

class AuthResponse(BaseModel):
    """Response payload for authentication tokens."""
    access_token: str
    token_type: str = Field(default="bearer")
    user: UserResponse

class ProfileUpdateResponse(BaseModel):
    """Response payload for updated user profile."""
    user: UserResponse

class HistoryEntryResponse(BaseModel):
    """Response payload for one user history entry."""
    id: int
    type: Literal["query", "ocr", "summary"]
    title: str
    payload: dict[str, Any]
    created_at: str
    user_id: int
    user_email: str

class HistoryListResponse(BaseModel):
    """Response payload for user history."""
    items: list[HistoryEntryResponse] = Field(default_factory=list)
    total: int = Field(default=0)

class OCRUploadResponse(BaseModel):
    """Response payload for OCR upload."""
    image_path: str
    image_url: str

class OCRMenuResponse(BaseModel):
    """Response payload for OCR menu extraction."""
    image_path: str
    image_url: str = Field(default="")
    ocr_status: str
    ocr_text: str
    raw_text: str = Field(default="")
    corrected_text: str = Field(default="")
    items: list[str] = Field(default_factory=list)

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

class DishListResponse(BaseModel):
    """Response payload for listing dishes from the database."""
    items: list[DishResponse] = Field(default_factory=list)
    total: int = Field(default=0)

class MenuScanResponse(BaseModel):
    """Response payload for menu scan results."""
    image_path: str
    ocr_status: str
    ocr_text: str
    items: list[DishResponse] = Field(default_factory=list)

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

class TranslationResponse(BaseModel):
    """Response payload for text translation."""
    original_text: str
    translated_text: str
    target_language: str
