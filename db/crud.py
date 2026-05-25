"""CRUD operations for dish entities."""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from db.models import Dish


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
