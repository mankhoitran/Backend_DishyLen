from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.deps import get_current_user
from db import crud
from db.database import get_db
from db.models import User
from schemas import request as request_schemas
from schemas import response as response_schemas
from utils.formatters import history_to_response

router = APIRouter()

@router.post("/", response_model=response_schemas.HistoryEntryResponse)
def create_history(
    payload: request_schemas.HistoryCreateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> response_schemas.HistoryEntryResponse:
    """Persist one history entry for the current user."""
    entry = crud.create_history_entry(
        db,
        user_id=current_user.id,
        entry_type=payload.type,
        title=payload.title,
        payload=payload.payload,
    )
    return history_to_response(entry, current_user)


@router.get("/", response_model=response_schemas.HistoryListResponse)
def list_history(
    type: Literal["query", "ocr", "summary"] | None = None,
    limit: int = 100,
    offset: int = 0,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> response_schemas.HistoryListResponse:
    """List history entries for the current user."""
    if limit < 1 or limit > 100:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 100")
    if offset < 0:
        raise HTTPException(status_code=400, detail="offset must be >= 0")

    items, total = crud.list_history_entries(
        db,
        user_id=current_user.id,
        entry_type=type,
        limit=limit,
        offset=offset,
    )
    return response_schemas.HistoryListResponse(
        items=[history_to_response(item, current_user) for item in items],
        total=total,
    )
