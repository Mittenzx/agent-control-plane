"""
Password hashing and JWT helpers for the control plane.

Password hashing uses PBKDF2-HMAC-SHA256 from the standard library (no external
crypto dependency). JWTs are created and verified with PyJWT (HS256).
"""

import base64
import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import jwt

# OWASP-recommended iteration count for PBKDF2-HMAC-SHA256 (2023+).
PBKDF2_ITERATIONS = 120_000
_SCHEME = "pbkdf2_sha256"
ALGORITHM = "HS256"


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text.encode("ascii"))


def hash_password(password: str) -> str:
    """Hash a plaintext password into a self-describing PBKDF2 string."""
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return f"{_SCHEME}${PBKDF2_ITERATIONS}${_b64(salt)}${_b64(dk)}"


def verify_password(password: str, stored: str) -> bool:
    """Constant-time comparison of a password against a stored PBKDF2 hash."""
    try:
        scheme, iterations_s, salt_b, hash_b = stored.split("$")
    except ValueError:
        return False
    if scheme != _SCHEME:
        return False
    try:
        salt = _unb64(salt_b)
        expected = _unb64(hash_b)
        iterations = int(iterations_s)
    except (ValueError, TypeError):
        return False
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return hmac.compare_digest(dk, expected)


def create_access_token(
    subject: str,
    secret_key: str,
    expires_minutes: int = 60,
    extra_claims: Optional[Dict[str, Any]] = None,
) -> str:
    """Create a signed HS256 JWT access token for ``subject``."""
    now = datetime.now(timezone.utc)
    payload: Dict[str, Any] = {
        "sub": subject,
        "iat": now,
        "exp": now + timedelta(minutes=expires_minutes),
        "type": "access",
    }
    if extra_claims:
        payload.update(extra_claims)
    return jwt.encode(payload, secret_key, algorithm=ALGORITHM)


def decode_token(token: str, secret_key: str) -> Dict[str, Any]:
    """Decode and validate a JWT. Raises ``jwt.PyJWTError`` on failure."""
    return jwt.decode(token, secret_key, algorithms=[ALGORITHM])


def generate_secret_key() -> str:
    """Generate a fresh, URL-safe secret key for signing JWTs."""
    return secrets.token_urlsafe(48)
