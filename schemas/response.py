"""API response schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class DishResponse(BaseModel):
    """Response payload for processed dish information."""

    dish: str = Field(..., description="Normalized dish name")
    spicy_level: str = Field(default="unknown")
    macros: dict[str, Any] = Field(default_factory=dict)
    summary: str = Field(default="")
    image_url: str = Field(default="")
    source: Literal["database", "search"]


class UserResponse(BaseModel):
    """Response payload for a logged-in user."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    email: str
    name: str
    picture_url: str


class AuthResponse(BaseModel):
    """Response payload for authentication tokens."""

    access_token: str
    token_type: str = Field(default="bearer")
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


class OCRItemsResponse(BaseModel):
    """Response payload for OCR items."""

    image_path: str
    image_url: str
    ocr_status: str
    ocr_text: str
    items: list[str] = Field(default_factory=list)


class OCRSelectResponse(BaseModel):
    """Response payload for OCR selection and dish lookup."""

    image_path: str
    image_url: str
    selected_item: str
    dish_info: DishResponse
    ingredients: list[str] = Field(default_factory=list)


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
