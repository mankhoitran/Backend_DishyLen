import pytest
from unittest.mock import MagicMock
from agent.search import DuckDuckGoSearchService

def test_check_allergy_conflict():
    mock_vllm = MagicMock()
    mock_vllm.generate_json.return_value = {
        "allergyWarning": True,
        "allergens": ["peanut"]
    }
    
    service = DuckDuckGoSearchService(mock_vllm)
    result = service.check_allergy("Pad Thai", ["noodles", "peanut"], "peanut")
    
    assert result["allergyWarning"] is True
    assert result["allergens"] == ["peanut"]
    mock_vllm.generate_json.assert_called_once()

def test_check_allergy_no_conflict():
    mock_vllm = MagicMock()
    mock_vllm.generate_json.return_value = {
        "allergyWarning": False,
        "allergens": []
    }
    
    service = DuckDuckGoSearchService(mock_vllm)
    result = service.check_allergy("Salad", ["lettuce"], "peanut")
    
    assert result["allergyWarning"] is False
    assert result["allergens"] == []

def test_check_allergy_empty_user_allergies():
    mock_vllm = MagicMock()
    service = DuckDuckGoSearchService(mock_vllm)
    
    result = service.check_allergy("Pad Thai", ["noodles"], None)
    assert result["allergyWarning"] is False
    assert result["allergens"] == []
    mock_vllm.generate_json.assert_not_called()
