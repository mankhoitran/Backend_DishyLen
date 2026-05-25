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
