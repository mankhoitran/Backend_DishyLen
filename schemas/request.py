"""API request schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Incoming query payload from clients."""

    query: str = Field(..., description="User question about a dish or menu item")
    target_language: str | None = Field(
        default=None,
        description="Optional language code to translate summary output, e.g. 'en', 'vi', 'th'",
    )


class OCRItemsRequest(BaseModel):
    """Request payload for OCR menu extraction."""

    image_path: str = Field(..., description="Path to the menu image on the server")
    max_items: int = Field(default=40, ge=1, le=200)


class OCRSelectRequest(BaseModel):
    """Request payload for selecting a menu item from OCR output."""

    image_path: str = Field(..., description="Path to the menu image on the server")
    item_name: str | None = Field(default=None, description="Exact item name to select")
    item_index: int | None = Field(default=None, description="Index from extracted items")
    max_items: int = Field(default=40, ge=1, le=200)
    target_language: str | None = Field(
        default=None,
        description="Optional language code to translate summary output",
    )
