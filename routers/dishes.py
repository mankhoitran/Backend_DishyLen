from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from db import crud
from db.database import get_db
from schemas import response as response_schemas
from utils.formatters import dish_to_response

router = APIRouter()

@router.get("/", response_model=response_schemas.DishListResponse)
def list_dishes(
    q: str | None = None,
    limit: int = 20,
    offset: int = 0,
    db: Session = Depends(get_db),
) -> response_schemas.DishListResponse:
    """List dishes from the local database."""
    if limit < 1 or limit > 100:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 100")
    if offset < 0:
        raise HTTPException(status_code=400, detail="offset must be >= 0")

    items, total = crud.list_dishes(db, query=q, limit=limit, offset=offset)
    payload = [dish_to_response(item) for item in items]
    return response_schemas.DishListResponse(items=payload, total=total)
