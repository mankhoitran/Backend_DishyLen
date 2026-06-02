"""Load and format prompt templates from disk."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=64)
def _read_prompt(name: str) -> str:
    path = PROMPT_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")


def load_prompt(name: str) -> str:
    """Return raw prompt content."""

    return _read_prompt(name)


def format_prompt(name: str, **kwargs: object) -> str:
    """Format a prompt template with named parameters."""

    template = _read_prompt(name)
    try:
        return template.format(**kwargs)
    except KeyError as exc:
        missing = exc.args[0]
        raise KeyError(f"Missing prompt parameter: {missing}") from exc
