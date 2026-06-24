"""FastAPI demo app for the vLLM-backed agent and summarization tests."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from configs.configs import get_settings
from db.database import Base, engine
from routers import auth, dishes, history, vllm
from schemas import request as request_schemas
from schemas import response as response_schemas
from services.schema_logger import collect_pydantic_models, log_schema_snapshot
from utils.formatters import sanitize_bytes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

settings = get_settings()

app = FastAPI(title=f"{settings.app_name} (vLLM demo)", version=settings.app_version)

origins = [origin.strip() for origin in settings.cors_origins.split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path(__file__).resolve().parent / settings.uploads_dir
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    errors = exc.errors()
    sanitized_errors = sanitize_bytes(errors)
    return JSONResponse(
        status_code=422,
        content={"detail": jsonable_encoder(sanitized_errors)},
    )


@app.on_event("startup")
def on_startup() -> None:
    """Initialize required resources."""
    Base.metadata.create_all(bind=engine)
    try:
        current_module = sys.modules[__name__]
        models = collect_pydantic_models(request_schemas, response_schemas, current_module)
        log_schema_snapshot(Base, models)
    except Exception as exc:
        logging.getLogger(__name__).warning("Failed to log schema snapshot: %s", exc)


@app.get("/health")
def health_check() -> dict[str, str]:
    """Health endpoint."""
    return {"status": "ok"}


app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(history.router, prefix="/history", tags=["history"])
app.include_router(dishes.router, prefix="/dishes", tags=["dishes"])
app.include_router(vllm.router, prefix="/vllm", tags=["vllm"])
