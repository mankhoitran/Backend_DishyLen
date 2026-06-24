import pytest
from services.ocr_service import _split_candidates, _normalize_line, _is_heading

def test_split_candidates():
    assert _split_candidates("Item 1 | Item 2") == ["Item 1", "Item 2"]
    assert _split_candidates("Item A / Item B") == ["Item A", "Item B"]
    assert _split_candidates("Item X  Item Y") == ["Item X", "Item Y"]
    assert _split_candidates("Single Item") == ["Single Item"]

def test_normalize_line():
    assert _normalize_line("Dish Name $10.99") == "Dish Name"
    assert _normalize_line("- Item") == " Item"
    assert _normalize_line("• Item") == " Item"
    assert _normalize_line("Too   Many    Spaces") == "Too Many Spaces"

def test_is_heading():
    assert _is_heading("Appetizers:") is True
    assert _is_heading("Main Menu") is True
    assert _is_heading("Just a dish name") is False
    assert _is_heading("Specials") is True
