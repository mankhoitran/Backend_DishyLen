import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_get_me_unauthorized(client):
    response = client.get("/auth/me")
    assert response.status_code == 401

def test_get_me_authorized(auth_client, test_user):
    response = auth_client.get("/auth/me")
    assert response.status_code == 200
    data = response.json()
    assert data["email"] == test_user.email
    assert data["allergies"] == "peanuts"

def test_update_profile(auth_client):
    response = auth_client.put("/auth/profile", json={"allergies": "dairy, gluten"})
    assert response.status_code == 200
    assert response.json()["user"]["allergies"] == "dairy, gluten"

def test_add_allergy(auth_client):
    # test_user initially has "peanuts"
    response = auth_client.post("/auth/add_allergy", json={"text": "shellfish"})
    assert response.status_code == 200
    assert response.json()["user"]["allergies"] == "peanuts, shellfish"

@patch("routers.vllm.TranslationService")
def test_translate_text(mock_translation_service_cls, auth_client):
    mock_instance = mock_translation_service_cls.return_value
    mock_instance.translate_text.return_value = {"translated_text": "Xin chao"}
    
    response = auth_client.post("/vllm/translate", json={"text": "Hello", "target_language": "vi"})
    assert response.status_code == 200
    assert response.json()["translated_text"] == "Xin chao"
