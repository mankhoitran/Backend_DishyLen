"""CRUD operations for dish entities."""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from db.models import Dish, HistoryEntry, User


def get_dish_by_name(db: Session, dish_name: str) -> Dish | None:
    """Fetch a dish by normalized name."""

    return db.query(Dish).filter(Dish.name.ilike(dish_name.strip())).first()


def upsert_dish(
    db: Session,
    *,
    name: str,
    spicy_level: str,
    macros: dict[str, Any],
    summary: str,
) -> Dish:
    """Insert or update a dish row by name."""

    existing = get_dish_by_name(db, name)
    if existing:
        existing.spicy_level = spicy_level or existing.spicy_level
        existing.macros = macros or existing.macros
        existing.summary = summary or existing.summary
        db.add(existing)
        db.commit()
        db.refresh(existing)
        return existing

    dish = Dish(
        name=name.strip(),
        spicy_level=spicy_level or "unknown",
        macros=macros or {},
        summary=summary or "",
    )
    db.add(dish)
    db.commit()
    db.refresh(dish)
    return dish


def list_dishes(
    db: Session,
    *,
    query: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> tuple[list[Dish], int]:
    """List dishes with optional name filtering."""

    base_query = db.query(Dish)
    if query:
        base_query = base_query.filter(Dish.name.ilike(f"%{query.strip()}%"))

    total = base_query.count()
    items = (
        base_query.order_by(Dish.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return items, total


def get_user_by_id(db: Session, user_id: int) -> User | None:
    """Fetch a user by primary key."""

    return db.query(User).filter(User.id == user_id).first()


def get_user_by_google_sub(db: Session, google_sub: str) -> User | None:
    """Fetch a user by Google subject."""

    return db.query(User).filter(User.google_sub == google_sub.strip()).first()


def get_user_by_email(db: Session, email: str) -> User | None:
    """Fetch a user by email."""

    return db.query(User).filter(User.email.ilike(email.strip())).first()


def upsert_user(
    db: Session,
    *,
    google_sub: str,
    email: str,
    name: str | None,
    picture_url: str | None,
) -> User:
    """Insert or update a user by Google subject or email."""

    existing = get_user_by_google_sub(db, google_sub) or get_user_by_email(db, email)
    if existing:
        existing.google_sub = google_sub or existing.google_sub
        existing.email = email or existing.email
        if name:
            existing.name = name
        if picture_url:
            existing.picture_url = picture_url
        db.add(existing)
        db.commit()
        db.refresh(existing)
        return existing

    user = User(
        google_sub=google_sub.strip(),
        email=email.strip(),
        name=name or "",
        picture_url=picture_url or "",
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def update_user_profile(db: Session, user_id: int, allergies: str | None) -> User | None:
    """Update a user's personalized profile details."""
    user = get_user_by_id(db, user_id)
    if user:
        user.allergies = allergies or ""
        db.add(user)
        db.commit()
        db.refresh(user)
    return user


def add_user_allergy(db: Session, user_id: int, new_allergies: str) -> User | None:
    """Append new text to a user's existing allergies profile."""
    user = get_user_by_id(db, user_id)
    if not user:
        return None
    
    if not new_allergies.strip():
        return user

    current_items = [a.strip() for a in user.allergies.split(',')] if user.allergies else []
    current_lower = {a.lower() for a in current_items}
    
    new_items = [a.strip() for a in new_allergies.split(',')]
    
    for item in new_items:
        if item and item.lower() not in current_lower:
            current_items.append(item)
            current_lower.add(item.lower())
            
    user.allergies = ", ".join(current_items)
        
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def create_history_entry(
    db: Session,
    *,
    user_id: int,
    entry_type: str,
    title: str,
    payload: dict[str, Any],
) -> HistoryEntry:
    """Persist one user activity history entry."""

    entry = HistoryEntry(
        user_id=user_id,
        type=entry_type.strip(),
        title=title.strip(),
        payload=payload or {},
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry


def list_history_entries(
    db: Session,
    *,
    user_id: int,
    entry_type: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> tuple[list[HistoryEntry], int]:
    """List history entries for a user."""

    base_query = db.query(HistoryEntry).filter(HistoryEntry.user_id == user_id)
    if entry_type:
        base_query = base_query.filter(HistoryEntry.type == entry_type.strip())

    total = base_query.count()
    items = (
        base_query.order_by(HistoryEntry.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return items, total
