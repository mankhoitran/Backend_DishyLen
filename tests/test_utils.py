import pytest
from utils.formatters import sanitize_bytes, extract_text, to_number, to_str_list, dedupe_list, short_summary

def test_sanitize_bytes():
    assert sanitize_bytes(b"hello") == "hello"
    assert sanitize_bytes({"key": b"value"}) == {"key": "value"}
    assert sanitize_bytes([b"item"]) == ["item"]
    assert sanitize_bytes((b"tuple",)) == ("tuple",)
    assert sanitize_bytes(123) == 123
    assert sanitize_bytes(b"\xff") == "<binary_data>" # Invalid UTF-8

def test_extract_text():
    assert extract_text("simple text", ("key",)) == "simple text"
    assert extract_text({"key": "value"}, ("key",)) == "value"
    assert extract_text({"other": "value"}, ("key",)) == "value" # Fallback to first value
    assert extract_text(["item1", "item2"], ("key",)) == "item1, item2"
    assert extract_text('{"key": "json_value"}', ("key",)) == "json_value"
    assert extract_text(None, ("key",)) == ""

def test_to_number():
    assert to_number(123) == 123.0
    assert to_number("123.45") == 123.45
    assert to_number({"value": 42}) == 42.0
    assert to_number("10-20") == 15.0 # Average
    assert to_number("unknown") == 0.0
    assert to_number("") == 0.0
    assert to_number(None) == 0.0

def test_to_str_list():
    assert to_str_list(["a", "b", "c", "a"]) == ["a", "b", "c"]
    assert to_str_list('["x", "y"]') == ["x", "y"]
    assert to_str_list("apple, banana, apple") == ["apple", "banana"]
    assert to_str_list(None) == []

def test_dedupe_list():
    assert dedupe_list(["a", "A", "b"]) == ["a", "b"] # Case-insensitive deduping
    assert dedupe_list(["apple", "banana", "Apple"]) == ["apple", "banana"]

def test_short_summary():
    text = "word " * 40
    summary = short_summary(text, 30)
    assert len(summary.split()) <= 30
    assert short_summary("", 10) == ""
