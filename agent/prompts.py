"""Prompt templates for the food agent."""

from __future__ import annotations

from prompt.loader import format_prompt, load_prompt


def system_prompt() -> str:
    """Core system prompt for the orchestration agent."""

    return load_prompt("agent_system.txt").strip()


def tool_selection_prompt(user_query: str, dish_name: str, target_language: str | None = None) -> str:
    """Prompt that asks model to decide and call functions."""

    translation_note = ""
    if target_language:
        translation_note = (
            f"If summary needs localization, call translate with target_language='{target_language}' "
            "after collecting the summary."
        )

    return format_prompt(
        "agent_tool_selection.txt",
        user_query=user_query,
        dish_name=dish_name,
        translation_note=translation_note,
    ).strip()


def final_answer_prompt(collected_context: dict) -> str:
    """Prompt that normalizes final response format."""

    return format_prompt(
        "agent_final_answer.txt",
        collected_context=collected_context,
    ).strip()
