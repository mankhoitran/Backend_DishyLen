import pytest
from db import crud
from db.models import User

def test_create_user(db_session):
    user = crud.upsert_user(
        db_session,
        google_sub="sub123",
        email="newuser@example.com",
        name="New User",
        picture_url="http://example.com/pic.jpg",
    )
    assert user.id is not None
    assert user.email == "newuser@example.com"
    assert user.allergies == ""

def test_get_user_by_email(db_session, test_user):
    user = crud.get_user_by_email(db_session, test_user.email)
    assert user is not None
    assert user.id == test_user.id
    assert user.google_sub == test_user.google_sub

def test_update_user_profile(db_session, test_user):
    updated = crud.update_user_profile(db_session, test_user.id, "dairy, gluten")
    assert updated is not None
    assert updated.allergies == "dairy, gluten"

def test_add_user_allergy(db_session, test_user):
    # test_user initially has "peanuts"
    updated = crud.add_user_allergy(db_session, test_user.id, "shellfish")
    assert updated is not None
    assert updated.allergies == "peanuts, shellfish"

    # Add empty shouldn't change
    updated2 = crud.add_user_allergy(db_session, test_user.id, "   ")
    assert updated2.allergies == "peanuts, shellfish"

    # Add duplicate shouldn't change
    updated3 = crud.add_user_allergy(db_session, test_user.id, "Peanuts, squid")
    assert updated3.allergies == "peanuts, shellfish, squid"
