from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.deps import get_current_user
from db import crud
from db.database import get_db
from db.models import User
from schemas import request as request_schemas
from schemas import response as response_schemas
from services.auth import create_access_token, get_password_hash, verify_password
from utils.formatters import user_to_response

router = APIRouter()

@router.post("/guest", response_model=response_schemas.AuthResponse)
def guest_login(
    db: Session = Depends(get_db),
) -> response_schemas.AuthResponse:
    """Authenticate by creating a temporary guest user and return an app access token."""
    user = crud.create_guest_user(db)
    token = create_access_token(
        user_id=user.id,
        email=user.email,
        name=user.name,
        picture_url=user.picture_url,
    )
    return response_schemas.AuthResponse(
        access_token=token,
        user=user_to_response(user),
    )


@router.post("/logout")
def logout_user(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict[str, str]:
    """Logout user. If the user is a guest, delete their account and data."""
    if current_user.is_guest:
        crud.delete_guest_user(db, current_user.id)
        return {"status": "success", "message": "Guest user deleted"}
    return {"status": "success", "message": "Logged out"}


@router.post("/register", response_model=response_schemas.AuthResponse)
def register_user(
    payload: request_schemas.RegisterRequest,
    db: Session = Depends(get_db),
) -> response_schemas.AuthResponse:
    """Register a new user with email and password."""
    existing_user = crud.get_user_by_email(db, payload.email)
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed_password = get_password_hash(payload.password)
    user = crud.create_user_with_password(
        db,
        email=payload.email,
        hashed_password=hashed_password,
        name=payload.name,
    )
    token = create_access_token(
        user_id=user.id,
        email=user.email,
        name=user.name,
        picture_url=user.picture_url,
    )
    return response_schemas.AuthResponse(
        access_token=token,
        user=user_to_response(user),
    )


@router.post("/login", response_model=response_schemas.AuthResponse)
def login_user(
    payload: request_schemas.LoginRequest,
    db: Session = Depends(get_db),
) -> response_schemas.AuthResponse:
    """Authenticate with email and password."""
    user = crud.get_user_by_email(db, payload.email)
    if not user or not user.hashed_password:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_access_token(
        user_id=user.id,
        email=user.email,
        name=user.name,
        picture_url=user.picture_url,
    )
    return response_schemas.AuthResponse(
        access_token=token,
        user=user_to_response(user),
    )


@router.put("/profile", response_model=response_schemas.ProfileUpdateResponse)
def update_profile(
    payload: request_schemas.UserProfileUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> response_schemas.ProfileUpdateResponse:
    """Update the current user's profile information."""
    updated = crud.update_user_profile(db, current_user.id, allergies=payload.allergies)
    if not updated:
        raise HTTPException(status_code=404, detail="User not found")
    return response_schemas.ProfileUpdateResponse(user=user_to_response(updated))


@router.post("/add_allergy", response_model=response_schemas.ProfileUpdateResponse)
def add_allergy(
    payload: request_schemas.AddAllergyRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> response_schemas.ProfileUpdateResponse:
    """Append text ingredients to the current user's allergies profile."""
    actual_text = payload.text or payload.allergies or payload.description or ""
    if not actual_text.strip():
        raise HTTPException(status_code=400, detail="Missing text, allergies, or description field")
    updated = crud.add_user_allergy(db, current_user.id, new_allergies=actual_text)
    if not updated:
        raise HTTPException(status_code=404, detail="User not found")
    return response_schemas.ProfileUpdateResponse(user=user_to_response(updated))


@router.get("/me", response_model=response_schemas.UserResponse)
def get_me(current_user: User = Depends(get_current_user)) -> response_schemas.UserResponse:
    """Return the current authenticated user."""
    return user_to_response(current_user)
