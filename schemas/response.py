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
