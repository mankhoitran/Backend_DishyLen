"""Database engine/session setup."""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from config import get_settings


settings = get_settings()

engine = create_engine(
    settings.sqlite_db_url,
    connect_args={"check_same_thread": False} if settings.sqlite_db_url.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db() -> Session:
    """FastAPI dependency yielding a database session."""

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
