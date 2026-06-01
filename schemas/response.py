"""API response schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class DishResponse(BaseModel):
    """Response payload for processed dish information."""

    dish: str = Field(..., description="Normalized dish name")
    spicy_level: str = Field(default="unknown")
    macros: dict[str, Any] = Field(default_factory=dict)
    summary: str = Field(default="")
    image_url: str = Field(default="")
    source: Literal["database", "search"]


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
