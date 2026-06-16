"""rtsp_warden.auth

Multi-user auth foundation for rtsp-warden.

Provides:
  - bcrypt password hashing
  - cookie-based sessions (HttpOnly, SameSite=Lax)
  - bearer API tokens (with prefix for display)
  - WSGI-style current user resolution

Designed to be import-light (only depends on rtsp_warden.db).
"""

from __future__ import annotations

import hashlib
import logging
import secrets
import string
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from http.cookies import SimpleCookie

import bcrypt

from .db.engine import get_session
from .db.models import ApiToken, Session, User

__all__ = [
    "SESSION_COOKIE_NAME",
    "SESSION_TTL_SECONDS",
    "API_TOKEN_PREFIX",
    "API_TOKEN_TTL_SECONDS",
    "BEARER_HEADER_PREFIX",
    "AuthSession",
    "AuthToken",
    "CurrentUser",
    "hash_password",
    "verify_password",
    "generate_session_token",
    "generate_api_token",
    "generate_admin_password",
    "generate_app_secret",
    "create_session",
    "lookup_session",
    "delete_session",
    "cleanup_expired_sessions",
    "create_api_token",
    "lookup_api_token",
    "revoke_api_token",
    "list_api_tokens",
    "get_current_user",
]

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SESSION_COOKIE_NAME = "warden_session"
SESSION_TTL_SECONDS = 7 * 24 * 60 * 60  # 7 days
API_TOKEN_PREFIX = "wdt_"
API_TOKEN_TTL_SECONDS = 365 * 24 * 60 * 60  # 1 year default
BEARER_HEADER_PREFIX = "Warden-Bearer "

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AuthSession:
    """Authenticated session resolved from a session cookie."""

    token: str
    user_id: int
    username: str
    role: str
    expires_at: datetime


@dataclass(slots=True)
class AuthToken:
    """API token — the raw value is only available at creation time."""

    raw: str
    prefix: str
    name: str
    user_id: int
    expires_at: datetime | None


@dataclass(slots=True)
class CurrentUser:
    """Resolved identity from a WSGI request."""

    user_id: int
    username: str
    role: str
    auth_method: str  # "session" | "bearer"


# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------


def hash_password(password: str) -> str:
    """Hash a plaintext password with bcrypt."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a plaintext password against a bcrypt hash."""
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Secret generation
# ---------------------------------------------------------------------------


def generate_session_token() -> str:
    """Generate a 64-char hex session token (32 random bytes)."""
    return secrets.token_hex(32)


def generate_api_token() -> str:
    """Generate an API token with the 'wdt_' prefix and ~40 urlsafe chars."""
    return API_TOKEN_PREFIX + secrets.token_urlsafe(30)


def generate_admin_password() -> str:
    """Generate a 16-char human-typeable admin password (alphanumeric only)."""
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(16))


def generate_app_secret() -> str:
    """Generate a 64-char hex secret for CSRF / signing (32 random bytes)."""
    return secrets.token_hex(32)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _hash_api_token(raw: str) -> str:
    """SHA-256 hash of the raw API token for storage."""
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _touch_api_token(raw: str) -> None:
    """Update last_used_at on the matching API token row. Best-effort."""
    token_hash = _hash_api_token(raw)
    try:
        with get_session() as session:
            row = session.query(ApiToken).filter(ApiToken.token_hash == token_hash).first()
            if row is not None:
                row.last_used_at = datetime.now(timezone.utc)
                session.commit()
    except Exception:
        log.debug("Failed to touch API token last_used_at", exc_info=True)


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------


def create_session(user: User) -> AuthSession:
    """Create a new browser session for the given user."""
    token = generate_session_token()
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=SESSION_TTL_SECONDS)

    with get_session() as session:
        row = Session(
            user_id=user.id,
            token=token,
            expires_at=expires_at,
        )
        session.add(row)
        session.commit()

    return AuthSession(
        token=token,
        user_id=user.id,
        username=user.username,
        role=user.role,
        expires_at=expires_at,
    )


def lookup_session(token: str) -> AuthSession | None:
    """Look up a session by token. Returns None if not found or expired."""
    with get_session() as session:
        row = (
            session.query(Session)
            .filter(Session.token == token)
            .filter(Session.expires_at > datetime.now(timezone.utc))
            .first()
        )
        if row is None:
            return None

        return AuthSession(
            token=row.token,
            user_id=row.user_id,
            username=row.user.username,
            role=row.user.role,
            expires_at=row.expires_at,
        )


def delete_session(token: str) -> bool:
    """Delete a session by token. Returns True if a session was deleted."""
    with get_session() as session:
        count = session.query(Session).filter(Session.token == token).delete()
        session.commit()
        return count > 0


def cleanup_expired_sessions() -> int:
    """Delete all expired sessions. Returns the number deleted."""
    with get_session() as session:
        count = (
            session.query(Session).filter(Session.expires_at < datetime.now(timezone.utc)).delete()
        )
        session.commit()
        return count


# ---------------------------------------------------------------------------
# API token management
# ---------------------------------------------------------------------------


def create_api_token(user: User, name: str, ttl_seconds: int | None = None) -> AuthToken:
    """Create a new API token for the given user.

    The raw token is only available in the returned AuthToken —
    it cannot be retrieved later.
    """
    raw = generate_api_token()
    prefix = raw[:8]  # includes "wdt_" + 4 chars
    token_hash = _hash_api_token(raw)

    expires_at: datetime | None = None
    effective_ttl = ttl_seconds if ttl_seconds is not None else API_TOKEN_TTL_SECONDS
    if effective_ttl > 0:
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=effective_ttl)

    with get_session() as session:
        tok = ApiToken(
            user_id=user.id,
            name=name,
            token_hash=token_hash,
            token_prefix=prefix,
            expires_at=expires_at,
        )
        session.add(tok)
        session.commit()

    return AuthToken(
        raw=raw,
        prefix=prefix,
        name=name,
        user_id=user.id,
        expires_at=expires_at,
    )


def lookup_api_token(raw_token: str) -> User | None:
    """Look up the user for a raw API token. Returns None if invalid or expired."""
    token_hash = _hash_api_token(raw_token)

    with get_session() as session:
        row = (
            session.query(ApiToken)
            .filter(ApiToken.token_hash == token_hash)
            .filter(
                (ApiToken.expires_at.is_(None)) | (ApiToken.expires_at > datetime.now(timezone.utc))
            )
            .first()
        )
        if row is None:
            return None

        user = row.user
        session.expunge(user)
        return user


def revoke_api_token(token_prefix: str) -> bool:
    """Revoke an API token by its prefix. Returns True if a token was revoked."""
    with get_session() as session:
        count = session.query(ApiToken).filter(ApiToken.token_prefix == token_prefix).delete()
        session.commit()
        return count > 0


def list_api_tokens(user_id: int) -> list[dict]:
    """List API tokens for a user (for display — no hashes included)."""
    with get_session() as session:
        rows = session.query(ApiToken).filter(ApiToken.user_id == user_id).all()
        return [
            {
                "id": row.id,
                "name": row.name,
                "prefix": row.token_prefix,
                "expires_at": row.expires_at.isoformat() if row.expires_at else None,
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "last_used_at": row.last_used_at.isoformat() if row.last_used_at else None,
            }
            for row in rows
        ]


# ---------------------------------------------------------------------------
# WSGI-style request resolution
# ---------------------------------------------------------------------------


def get_current_user(environ: dict) -> CurrentUser | None:
    """Resolve the current user from a WSGI environ dict.

    Checks (in order):
      1. ``warden_session`` cookie (HTTP_COOKIE header) — session lookup
      2. ``Authorization: Warden-Bearer <raw>`` header — API token lookup

    Returns None if no valid auth found.
    """
    # 1. Try session cookie
    cookie_header = environ.get("HTTP_COOKIE", "")
    if cookie_header:
        try:
            cookies = SimpleCookie(cookie_header)
            if SESSION_COOKIE_NAME in cookies:
                token = cookies[SESSION_COOKIE_NAME].value
                auth_session = lookup_session(token)
                if auth_session is not None:
                    return CurrentUser(
                        user_id=auth_session.user_id,
                        username=auth_session.username,
                        role=auth_session.role,
                        auth_method="session",
                    )
        except Exception:
            log.debug("Failed to parse session cookie", exc_info=True)

    # 2. Try bearer token
    auth_header = environ.get("HTTP_AUTHORIZATION", "")
    if auth_header.startswith(BEARER_HEADER_PREFIX):
        raw = auth_header[len(BEARER_HEADER_PREFIX) :]
        user = lookup_api_token(raw)
        if user is not None and user.is_active:
            _touch_api_token(raw)
            return CurrentUser(
                user_id=user.id,
                username=user.username,
                role=user.role,
                auth_method="bearer",
            )

    return None
