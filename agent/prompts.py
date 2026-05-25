"""Prompt templates for the food agent."""

from __future__ import annotations


def system_prompt() -> str:
    """Core system prompt for the orchestration agent."""

    return (
        "You are a production food information agent. "
        "You MUST use tools to get facts. "
        "Primary objective: return dish name, spicy level, macros, summary, and one best image URL. "
        "If data exists in database, prefer it. Otherwise search and enrich, then persist. "
        "When enough data is collected, return strict JSON with keys: "
        "dish, spicy_level, macros, summary, image_url, source. "
        "source must be database or search."
    )


def tool_selection_prompt(user_query: str, dish_name: str, target_language: str | None = None) -> str:
    """Prompt that asks model to decide and call functions."""

    translation_note = ""
    if target_language:
        translation_note = (
            f"If summary needs localization, call translate with target_language='{target_language}' "
            "after collecting the summary."
        )

    return (
        f"User query: {user_query}\n"
        f"Extracted dish name: {dish_name}\n"
        "Use tools in this order when needed:\n"
        "1) get_processed_dish\n"
        "2) If missing, search_dish\n"
        "3) Fill gaps with get_spicy_level/get_dish_macro/get_dish_summary/get_dish_image_url\n"
        f"{translation_note}\n"
        "When done, output strict JSON only."
    )


def final_answer_prompt(collected_context: dict) -> str:
    """Prompt that normalizes final response format."""

    return (
        "Normalize the following context into strict JSON keys: "
        "dish, spicy_level, macros, summary, image_url, source. "
        f"Context: {collected_context}"
    )
