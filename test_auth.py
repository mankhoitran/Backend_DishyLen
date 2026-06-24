import requests

url = "http://127.0.0.1:8000"

def test():
    try:
        # Register
        res = requests.post(f"{url}/auth/register", json={
            "email": "test@example.com",
            "password": "password123",
            "name": "Test User"
        })
        print("Register:", res.status_code, res.text)
        
        # Login
        res = requests.post(f"{url}/auth/login", json={
            "email": "test@example.com",
            "password": "password123"
        })
        print("Login:", res.status_code, res.text)
    except Exception as e:
        print("Error:", e)

test()
