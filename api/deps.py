from fastapi import Depends, Header, HTTPException
from sqlalchemy.orm import Session

from db import crud
from db.database import get_db
from db.models import User
from services.auth import AuthError, decode_access_token

def get_current_user(
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
) -> User:
    """Resolve the current user from a bearer token."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing authorization header")

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=401, detail="Invalid authorization header")

    try:
        claims = decode_access_token(token)
    except AuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    user_id = claims.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid access token")

    try:
        user = crud.get_user_by_id(db, int(user_id))
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=401, detail="Invalid access token") from exc
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


def get_optional_current_user(
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
) -> User | None:
    """Resolve current user optionally."""
    if not authorization:
        return None
    try:
        return get_current_user(authorization, db)
    except HTTPException:
        return None
