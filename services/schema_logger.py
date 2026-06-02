"""Helpers for logging database and API schema snapshots."""

from __future__ import annotations

import json
import logging
from types import ModuleType
from typing import Any, Iterable

from pydantic import BaseModel
from sqlalchemy import Column
from sqlalchemy.orm import DeclarativeMeta

from services.logging_utils import get_schema_logger

logger = logging.getLogger(__name__)


def collect_pydantic_models(*modules: ModuleType) -> list[type[BaseModel]]:
    """Collect Pydantic models defined in the provided modules."""

    models: dict[str, type[BaseModel]] = {}
    for module in modules:
        for value in vars(module).values():
            if not isinstance(value, type):
                continue
            if not issubclass(value, BaseModel):
                continue
            key = f"{value.__module__}.{value.__name__}"
            models[key] = value
    return [models[key] for key in sorted(models)]


def log_schema_snapshot(db_base: DeclarativeMeta, models: Iterable[type[BaseModel]]) -> None:
    """Log database and API schema snapshots to the schema logger."""

    snapshot = {
        "db_schema": _build_db_schema(db_base),
        "api_schema": _build_api_schema(models),
    }

    schema_logger = get_schema_logger()
    try:
        schema_logger.info(json.dumps(snapshot, ensure_ascii=True))
    except Exception:
        logger.exception("Failed to write schema snapshot")


def _build_db_schema(db_base: DeclarativeMeta) -> dict[str, Any]:
    tables: list[dict[str, Any]] = []
    for table in db_base.metadata.sorted_tables:
        tables.append(
            {
                "name": table.name,
                "columns": [_serialize_column(column) for column in table.columns],
            }
        )
    return {"tables": tables}


def _serialize_column(column: Column[Any]) -> dict[str, Any]:
    default_value = _format_default(column)
    server_default = None
    if column.server_default is not None:
        try:
            server_default = str(column.server_default.arg)
        except Exception:
            server_default = str(column.server_default)

    payload: dict[str, Any] = {
        "name": column.name,
        "type": str(column.type),
        "nullable": column.nullable,
        "primary_key": column.primary_key,
    }
    if default_value is not None:
        payload["default"] = default_value
    if server_default is not None:
        payload["server_default"] = server_default
    return payload


def _format_default(column: Column[Any]) -> Any | None:
    if column.default is None:
        return None

    default = column.default
    try:
        if getattr(default, "is_scalar", False):
            return default.arg
        if getattr(default, "is_callable", False):
            arg = default.arg
            return getattr(arg, "__name__", "callable")
        return str(default.arg)
    except Exception:
        return str(default)


def _build_api_schema(models: Iterable[type[BaseModel]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for model in models:
        key = f"{model.__module__}.{model.__name__}"
        try:
            payload[key] = model.model_json_schema()
        except Exception:
            payload[key] = {"error": "failed_to_generate_schema"}
    return payload
