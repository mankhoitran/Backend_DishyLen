"""Authentication helpers for Google login and JWT access tokens."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import jwt

from configs.configs import get_settings


settings = get_settings()


class AuthError(ValueError):
    """Raised when authentication fails."""

try:
    import bcrypt
except ImportError:
    bcrypt = None

def verify_password(plain_password: str, hashed_password: str) -> bool:
    if not bcrypt:
        raise AuthError("Password hashing library not installed")
    
    # DEBUG: Print the exact password being received
    print("DEBUG verify_password plain_password:", repr(plain_password), "Length:", len(plain_password))
    print("DEBUG verify_password hashed_password:", repr(hashed_password))

    # bcrypt absolutely does not support > 72 bytes.
    # If the string is longer, it MUST be truncated or pre-hashed.
    plain_password_bytes = plain_password[:72].encode('utf-8')
    hashed_password_bytes = hashed_password.encode('utf-8')
    return bcrypt.checkpw(plain_password_bytes, hashed_password_bytes)
 
def get_password_hash(password: str) -> str:
    if not bcrypt:
        raise AuthError("Password hashing library not installed")
    
    password_bytes = password[:72].encode('utf-8')
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password_bytes, salt).decode('utf-8')





def create_access_token(*, user_id: int, email: str, name: str, picture_url: str) -> str:
    """Create a signed JWT for the authenticated user."""

    if not settings.jwt_secret or settings.jwt_secret == "change-me":
        raise AuthError("JWT_SECRET is not configured")

    now = datetime.now(tz=timezone.utc)
    payload = {
        "sub": str(user_id),
        "email": email,
        "name": name,
        "picture": picture_url,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=settings.jwt_exp_minutes)).timestamp()),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> dict[str, Any]:
    """Decode a JWT access token."""

    try:
        return jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except jwt.PyJWTError as exc:
        raise AuthError("Invalid access token") from exc
