import sys
from fastapi.testclient import TestClient
from app import app
from db.database import SessionLocal
from db.models import User

client = TestClient(app)

def main():
    print("Testing guest login endpoint...")
    response = client.post("/auth/guest")
    if response.status_code != 200:
        print(f"Failed to create guest: {response.text}")
        sys.exit(1)
    
    data = response.json()
    token = data.get("access_token")
    user = data.get("user")
    print(f"Guest User Created: {user['email']} (is_guest: {user.get('is_guest', True)})")
    
    # Check DB
    db = SessionLocal()
    db_user = db.query(User).filter(User.id == user['id']).first()
    print(f"DB User is_guest: {db_user.is_guest}")
    db.close()

    print("Testing logout endpoint...")
    logout_res = client.post("/auth/logout", headers={"Authorization": f"Bearer {token}"})
    if logout_res.status_code != 200:
        print(f"Failed to logout: {logout_res.text}")
        sys.exit(1)
    
    print(f"Logout Response: {logout_res.json()}")

    # Verify user is deleted
    db = SessionLocal()
    db_user_after = db.query(User).filter(User.id == user['id']).first()
    if db_user_after is None:
        print("Success! Guest user has been deleted from the database.")
    else:
        print("Failed! Guest user still exists in the database.")
        sys.exit(1)
    db.close()

if __name__ == "__main__":
    main()
