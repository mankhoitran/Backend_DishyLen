"""Logging helpers for capturing LLM responses."""

from __future__ import annotations

import logging
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
LLM_LOG_PATH = LOG_DIR / "llm_responses.log"


def get_llm_response_logger() -> logging.Logger:
    """Return a file logger for LLM responses."""

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("llm.responses")
    if not any(
        isinstance(handler, logging.FileHandler)
        and getattr(handler, "baseFilename", "") == str(LLM_LOG_PATH)
        for handler in logger.handlers
    ):
        handler = logging.FileHandler(LLM_LOG_PATH)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger
