"""Authentication helpers for Google login and JWT access tokens."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import jwt
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token as google_id_token

from configs.configs import get_settings


settings = get_settings()


class AuthError(ValueError):
    """Raised when authentication fails."""


def _audiences() -> list[str]:
    return [aud.strip() for aud in settings.google_client_id.split(",") if aud.strip()]


def verify_google_id_token(id_token: str) -> dict[str, Any]:
    """Validate a Google ID token and return its claims."""

    audiences = _audiences()
    if not audiences:
        raise AuthError("GOOGLE_CLIENT_ID is not configured")

    request = google_requests.Request()
    payload = google_id_token.verify_oauth2_token(
        id_token,
        request,
        audiences[0] if len(audiences) == 1 else audiences,
    )

    issuer = payload.get("iss")
    if issuer not in ("accounts.google.com", "https://accounts.google.com"):
        raise AuthError("Invalid token issuer")

    if not payload.get("email"):
        raise AuthError("Token is missing email")

    if not payload.get("sub"):
        raise AuthError("Token is missing subject")

    if payload.get("email_verified") is False:
        raise AuthError("Email is not verified")

    return payload


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
