"""Centralised LLM prompt constants for the DishyLen backend.

All prompt strings (system prompts, user prompt templates, instruction
snippets) live here grouped by service so that they can be found,
reviewed, and updated in a single location.

Usage::

    from prompt.prompts import OCR, SEARCH, DUCKDUCKGO, TRANSLATION

    system = OCR.SYSTEM
    user   = OCR.PROMPT_TEMPLATE.format(ocr_text="…")
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# OCR service prompts
# ---------------------------------------------------------------------------

class OCR:
    """Prompts used by services/ocr_service.py."""

    SYSTEM = "You are a strict OCR post-processor."

    PROMPT_TEMPLATE = (
        "You are an OCR post-processor for menu images. Use the OCR text provided below as the only source.\n"
        "Return ONLY a strict JSON object with exactly one key: items. "
        "No extra keys, no markdown, no commentary.\n"
        "Step 1: Extract ONLY dish names and food/drink item names. Completely ignore prices, currency symbols, "
        "numbers, and headers like 'Menu' or 'Beverages'. Preserve the original reading order (top-to-bottom, "
        "left-to-right).\n"
        "Step 2: Correct the extracted items. Fix spelling errors, Vietnamese diacritics where applicable, and "
        "obvious OCR errors (e.g., 'Bow!' -> 'Bowl', 'eed Tea' -> 'Iced Tea'). Do NOT add or remove items.\n"
        "Each item must be a separate array element. Never combine multiple items into one string.\n"
        "Step 3: If User allergies are provided, infer if any item likely contains them. If there is a high likelihood of a conflict, append an allergy warning directly to the item name string (e.g., 'Pad Thai [Allergy Warning: Peanuts]').\n"
        "Step 4: If no valid food or drink items can be found in the OCR text, return an empty array for items.\n"
        "User allergies: {user_allergies}\n"
        'Output schema example: {{"items": ["Dish A", "Dish B [Allergy Warning: Peanuts]"]}} or {{"items": []}}\n'
        "OCR text:\n{ocr_text}"
    )


# ---------------------------------------------------------------------------
# Gemini search service prompts  (services/search.py)
# ---------------------------------------------------------------------------

class SEARCH:
    """Prompts used by services/search.py (Gemini-backed SearchService)."""

    SEARCH_DISH = (
        "You are a culinary data extractor. Use Google Search tool to find reliable dish information. "
        "Return ONLY raw JSON, do NOT wrap in markdown codeblocks.\n"
        "Keys must be exactly: dish, spicy_level, calories, protein, carbs, fats, summary, image_url.\n"
        "calories (kcal), protein (g), carbs (g), and fats (g) must be numerical values at the top level. NEVER return null or 0 (unless it is water). If exact nutritional values are unknown, you MUST estimate a realistic number based on a standard serving size.\n"
        "image_url must be one best direct image URL that clearly represents the dish, or empty string if unavailable.\n"
        "Dish to research: {dish_name}"
    )

    GET_SPICY_LEVEL = (
        "Return ONLY JSON with keys dish and spicy_level. "
        "spicy_level must be one of: not_spicy, mild, medium, hot, very_hot, unknown. "
        "Dish: {dish_name}"
    )

    GET_DISH_MACRO = (
        "Return ONLY raw JSON, do NOT wrap in markdown codeblocks.\n"
        "Keys must be exactly: dish, calories, protein, carbs, fats.\n"
        "calories (kcal), protein (g), carbs (g), and fats (g) must be numerical values at the top level. NEVER return null or 0 (unless it is water). If exact nutritional values are unknown, you MUST estimate a realistic number based on a standard serving size.\n"
        "Dish: {dish_name}"
    )

    GET_DISH_SUMMARY = (
        "Return ONLY JSON with keys dish and summary. "
        "summary should be factual and under 60 words. "
        "Dish: {dish_name}"
    )

    GET_DISH_IMAGE_URL = (
        "Use Google Search tool and return ONLY JSON with keys dish and image_url. "
        "Select one best direct image URL that visually represents the dish, "
        "prefer stable sources and avoid logo/thumbnail sprites. "
        "If no reliable image can be found, return an empty string for image_url. "
        "Dish: {dish_name}"
    )

    TRANSLATE = (
        "Return ONLY JSON with keys target_language and translated_text. "
        "target_language: {target_language}. text: {text}"
    )


# ---------------------------------------------------------------------------
# DuckDuckGo search service prompts  (services/duckduckgo_search.py)
# ---------------------------------------------------------------------------

class DUCKDUCKGO:
    """Prompts used by services/duckduckgo_search.py (vLLM-backed DuckDuckGoSearchService)."""

    EXTRACTION_SYSTEM = (
        "You are a culinary data extractor. Use the provided sources when possible. "
        "Return strict JSON only."
    )

    SEARCH_DISH = (
        "Return ONLY raw JSON, do NOT wrap in markdown codeblocks.\n"
        "Keys must be: dish, description, summary, calories, protein, carbs, fats, ingredients, allergens, spicy_level, image_url.\n"
        "description should be factual, 2-3 sentences, under 90 words. summary should be one sentence under 30 words. "
        "Avoid mentioning sources or search. If sources are empty, description and summary should be 'No sources found.'.\n"
        "calories (kcal), protein (g), carbs (g), and fats (g) must be numerical values at the top level. NEVER return null or 0 (unless it is water). If exact nutritional values are unknown, you MUST estimate a realistic number based on a standard serving size.\n"
        "ingredients must be a JSON array of strings. If not explicitly listed, you MUST infer the most common ingredients used to make that dish.\n"
        "allergens must be a JSON array of strings containing standard known allergens for the dish (e.g. ['Peanuts', 'Dairy']). If there are none, return an empty array [].\n"
        "spicy_level must be one of: not_spicy, mild, medium, hot, very_hot, unknown.\n"
        "image_url should be empty if no reliable image appears in sources."
    )

    GET_SPICY_LEVEL = (
        "Return ONLY JSON with keys dish and spicy_level. "
        "spicy_level must be one of: not_spicy, mild, medium, hot, very_hot, unknown."
    )

    GET_DISH_MACRO = (
        "Return ONLY raw JSON, do NOT wrap in markdown codeblocks.\n"
        "Keys must be exactly: dish, calories, protein, carbs, fats.\n"
        "calories (kcal), protein (g), carbs (g), and fats (g) must be numerical values at the top level. NEVER return null or 0 (unless it is water). If exact nutritional values are unknown, you MUST estimate a realistic number based on a standard serving size."
    )

    GET_DISH_SUMMARY = (
        "Return ONLY JSON with keys dish and summary. "
        "summary should be factual, 2-3 sentences, and under 80 words. "
        "Avoid mentioning sources or search. "
        "If sources are empty, summary should be 'No sources found.'."
    )

    GET_DISH_INGREDIENTS = (
        "Return ONLY raw JSON, do NOT wrap in markdown codeblocks.\n"
        "Keys must be exactly: dish, ingredients.\n"
        "ingredients must always return as a JSON array of strings. If there are none, return an empty array []. "
        "If sources are empty, ingredients should be an empty array []."
    )

    TRANSLATE_SYSTEM = "You are a translation engine."

    TRANSLATE = (
        "Return ONLY JSON with keys target_language and translated_text. "
        "target_language: {target_language}."
    )


# ---------------------------------------------------------------------------
# Translation prompts shared across services  (app.py, duckduckgo_search.py)
# ---------------------------------------------------------------------------

class TRANSLATION:
    """Generic translation prompts reused in multiple places."""

    SYSTEM = "You are a translation engine."

    USER = (
        "Return ONLY JSON with keys target_language and translated_text. "
        "target_language: {target_language}.\n"
        "text: {text}"
    )


# ---------------------------------------------------------------------------
# Summary prompts
# ---------------------------------------------------------------------------

class SUMMARY:
    """Prompts for summarizing arbitrary food text."""

    SYSTEM = "You are a culinary data extractor and summarizer. Return ONLY strict JSON."

    USER = (
        "Extract structured information from the text below.\n"
        "Return ONLY raw JSON, do NOT wrap in markdown codeblocks.\n"
        "Keys must be exactly: description, summary, calories, protein, carbs, fats, ingredients, allergens.\n"
        "description should be a factual 2-3 sentence overview.\n"
        "summary should be one short sentence (under {max_words} words).\n"
        "calories (kcal), protein (g), carbs (g), and fats (g) must be numerical values at the top level. NEVER return null or 0 (unless it is water). If exact nutritional values are unknown, you MUST estimate a realistic number based on a standard serving size.\n"
        "ingredients must be a JSON array of strings. If not explicitly listed, you MUST infer the most common ingredients used to make that dish.\n"
        "allergens must be a JSON array of strings containing standard known allergens for the dish (e.g. ['Peanuts', 'Dairy']). If there are none, return an empty array [].\n"
        "Text to summarize:\n{text}"
    )

class ALLERGY:
    """Prompts for checking allergy conflicts."""

    SYSTEM = "You are a health and dietary safety assistant."

    CHECK = (
        "Analyze the given dish and its ingredients against the User's allergies.\n"
        "Return ONLY raw JSON, do NOT wrap in markdown codeblocks.\n"
        "Keys must be exactly: allergyWarning (boolean) and allergens (array of strings).\n"
        "If there is any match or high risk of cross-contamination, allergyWarning MUST be true.\n"
        "If allergyWarning is true, the allergens array MUST NOT be empty. It must explicitly list the matching allergens (e.g., ['Peanuts', 'Dairy']).\n"
        "If there is no match, allergyWarning should be false, and allergens should be a list of all identified allergens in the dish, or [] if none exist.\n"
        "User allergies: {user_allergies}\n"
        "Dish: {dish_name}\n"
        "Ingredients: {ingredients}"
    )
