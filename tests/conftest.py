import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

from app import app
from api.deps import get_current_user
from db.database import Base, get_db
from db.models import User

# In-memory SQLite for tests
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

from sqlalchemy.pool import StaticPool

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, 
    connect_args={"check_same_thread": False},
    poolclass=StaticPool
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture(scope="function")
def db_session():
    """Create a fresh database for each test."""
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)

@pytest.fixture(scope="function")
def client(db_session):
    """Test client with db override."""
    def override_get_db():
        yield db_session
            
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.pop(get_db, None)

@pytest.fixture(scope="function")
def test_user(db_session):
    """A dummy user for testing."""
    user = User(
        google_sub="test_google_sub_123",
        email="testuser@example.com",
        name="Test User",
        picture_url="http://example.com/pic.jpg",
        allergies="peanuts"
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user

@pytest.fixture(scope="function")
def auth_client(client, test_user):
    """Test client that is authenticated as test_user."""
    def override_current_user():
        return test_user

    app.dependency_overrides[get_current_user] = override_current_user
    yield client
    app.dependency_overrides.pop(get_current_user, None)
