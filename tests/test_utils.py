import pytest
from app import _sanitize_bytes, _extract_text, _to_number, _to_str_list, _dedupe_list, _short_summary

def test_sanitize_bytes():
    assert _sanitize_bytes(b"hello") == "hello"
    assert _sanitize_bytes({"key": b"value"}) == {"key": "value"}
    assert _sanitize_bytes([b"item"]) == ["item"]
    assert _sanitize_bytes((b"tuple",)) == ("tuple",)
    assert _sanitize_bytes(123) == 123
    assert _sanitize_bytes(b"\xff") == "<binary_data>" # Invalid UTF-8

def test_extract_text():
    assert _extract_text("simple text", ("key",)) == "simple text"
    assert _extract_text({"key": "value"}, ("key",)) == "value"
    assert _extract_text({"other": "value"}, ("key",)) == "value" # Fallback to first value
    assert _extract_text(["item1", "item2"], ("key",)) == "item1, item2"
    assert _extract_text('{"key": "json_value"}', ("key",)) == "json_value"
    assert _extract_text(None, ("key",)) == ""

def test_to_number():
    assert _to_number(123) == 123.0
    assert _to_number("123.45") == 123.45
    assert _to_number({"value": 42}) == 42.0
    assert _to_number("10-20") == 15.0 # Average
    assert _to_number("unknown") == 0.0
    assert _to_number("") == 0.0
    assert _to_number(None) == 0.0

def test_to_str_list():
    assert _to_str_list(["a", "b", "c", "a"]) == ["a", "b", "c"]
    assert _to_str_list('["x", "y"]') == ["x", "y"]
    assert _to_str_list("apple, banana, apple") == ["apple", "banana"]
    assert _to_str_list(None) == []

def test_dedupe_list():
    assert _dedupe_list(["a", "A", "b"]) == ["a", "b"] # Case-insensitive deduping
    assert _dedupe_list(["apple", "banana", "Apple"]) == ["apple", "banana"]

def test_short_summary():
    text = "word " * 40
    summary = _short_summary(text, 30)
    assert len(summary.split()) <= 30
    assert _short_summary("", 10) == ""
