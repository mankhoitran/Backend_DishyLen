import pytest
from unittest.mock import MagicMock
from services.summary_service import SummaryService

def test_summarize_text_success():
    mock_vllm = MagicMock()
    # Mock the two calls: 1. summary, 2. allergy
    mock_vllm.generate_json.side_effect = [
        {
            "description": "A classic dish",
            "summary": "Classic dish",
            "calories": 100,
            "protein": 10,
            "carbs": 5,
            "fats": 2,
            "ingredients": ["egg", "milk"],
            "allergens": []
        },
        {
            "allergyWarning": True,
            "allergens": ["egg", "milk"]
        }
    ]
    
    service = SummaryService(mock_vllm)
    result = service.summarize_text("Some recipe text", max_words=30, user_allergies="egg")
    
    assert result["allergyWarning"] is True
    assert result["allergens"] == ["egg", "milk"]
    assert result["calories"] == 100
    assert result["description"] == "A classic dish"

def test_summarize_text_no_allergies():
    mock_vllm = MagicMock()
    mock_vllm.generate_json.return_value = {
        "description": "A classic dish",
        "summary": "Classic dish",
        "calories": 100,
        "protein": 10,
        "carbs": 5,
        "fats": 2,
        "ingredients": ["egg", "milk"],
        "allergens": []
    }
    
    service = SummaryService(mock_vllm)
    result = service.summarize_text("Some recipe text", max_words=30, user_allergies=None)
    
    assert "allergyWarning" not in result or result["allergyWarning"] is False
    # Only 1 call to generate_json since user_allergies is None
    mock_vllm.generate_json.assert_called_once()
