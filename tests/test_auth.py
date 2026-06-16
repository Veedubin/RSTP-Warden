"""Tests for auth.py — password hashing, sessions, tokens, get_current_user."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from rtsp_warden.auth import (
    BEARER_HEADER_PREFIX,
    SESSION_COOKIE_NAME,
    create_api_token,
    create_session,
    generate_admin_password,
    generate_api_token,
    generate_app_secret,
    generate_session_token,
    get_current_user,
    hash_password,
    lookup_api_token,
    lookup_session,
    revoke_api_token,
    verify_password,
)
from rtsp_warden.db.engine import get_session
from rtsp_warden.db.models import Session as SessionModel
from rtsp_warden.db.models import User

# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------


def test_hash_verify_password_roundtrip() -> None:
    """verify_password(plain, hash_password(plain)) is True; wrong password is False."""
    plain = "my_secret_pass_123!"
    hashed = hash_password(plain)
    assert verify_password(plain, hashed) is True
    assert verify_password("wrong_password", hashed) is False


# ---------------------------------------------------------------------------
# Secret generation
# ---------------------------------------------------------------------------


def test_generate_session_token_length() -> None:
    """generate_session_token() returns 64 hex chars."""
    token = generate_session_token()
    assert len(token) == 64
    # All hex chars
    int(token, 16)


def test_generate_api_token_has_wdt_prefix() -> None:
    """generate_api_token() starts with 'wdt_'."""
    token = generate_api_token()
    assert token.startswith("wdt_")


def test_generate_admin_password_alphanumeric_16() -> None:
    """generate_admin_password() returns 16 chars, all alphanumeric."""
    pw = generate_admin_password()
    assert len(pw) == 16
    assert pw.isalnum()


def test_generate_app_secret_length() -> None:
    """generate_app_secret() returns 64 hex chars."""
    secret = generate_app_secret()
    assert len(secret) == 64
    int(secret, 16)


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------


def test_create_and_lookup_session(admin_user: User) -> None:
    """Roundtrip via create_session(user) → lookup_session(token)."""
    auth_session = create_session(admin_user)
    assert auth_session.token is not None

    looked_up = lookup_session(auth_session.token)
    assert looked_up is not None
    assert looked_up.user_id == admin_user.id
    assert looked_up.username == admin_user.username
    assert looked_up.role == "admin"


def test_session_expires(admin_user: User) -> None:
    """Manually create expired session, lookup_session returns None."""
    token = generate_session_token()
    expired_at = datetime.now(timezone.utc) - timedelta(hours=1)

    with get_session() as session:
        row = SessionModel(
            user_id=admin_user.id,
            token=token,
            expires_at=expired_at,
        )
        session.add(row)
        session.commit()

    looked_up = lookup_session(token)
    assert looked_up is None


# ---------------------------------------------------------------------------
# API tokens
# ---------------------------------------------------------------------------


def test_create_api_token_returns_raw_only_once(admin_user: User) -> None:
    """tok.raw is set, hash is stored."""
    tok = create_api_token(admin_user, name="test-token")
    assert tok.raw is not None
    assert tok.raw.startswith("wdt_")
    assert tok.prefix == tok.raw[:8]
    assert tok.name == "test-token"
    assert tok.user_id == admin_user.id


def test_lookup_api_token_with_raw(admin_user: User) -> None:
    """Roundtrip via create_api_token → lookup_api_token(raw)."""
    tok = create_api_token(admin_user, name="lookup-test")
    user = lookup_api_token(tok.raw)
    assert user is not None
    assert user.id == admin_user.id
    assert user.username == admin_user.username


def test_revoke_api_token(admin_user: User) -> None:
    """revoke_api_token(prefix) removes the token."""
    tok = create_api_token(admin_user, name="revoke-test")
    assert lookup_api_token(tok.raw) is not None

    revoked = revoke_api_token(tok.prefix)
    assert revoked is True

    assert lookup_api_token(tok.raw) is None


# ---------------------------------------------------------------------------
# get_current_user
# ---------------------------------------------------------------------------


def test_get_current_user_via_session_cookie(admin_user: User, make_environ: callable) -> None:
    """get_current_user via session cookie returns CurrentUser(auth_method='session')."""
    auth_session = create_session(admin_user)
    environ = make_environ(cookie=f"{SESSION_COOKIE_NAME}={auth_session.token}")
    cu = get_current_user(environ)
    assert cu is not None
    assert cu.user_id == admin_user.id
    assert cu.username == admin_user.username
    assert cu.auth_method == "session"


def test_get_current_user_via_bearer(admin_user: User, make_environ: callable) -> None:
    """get_current_user via bearer returns CurrentUser(auth_method='bearer')."""
    tok = create_api_token(admin_user, name="bearer-test")
    environ = make_environ(bearer=f"{BEARER_HEADER_PREFIX}{tok.raw}")
    cu = get_current_user(environ)
    assert cu is not None
    assert cu.user_id == admin_user.id
    assert cu.username == admin_user.username
    assert cu.auth_method == "bearer"


def test_get_current_user_no_auth_returns_none(make_environ: callable) -> None:
    """get_current_user({}) returns None."""
    cu = get_current_user(make_environ())
    assert cu is None


def test_get_current_user_invalid_session_returns_none(make_environ: callable) -> None:
    """Bogus cookie returns None."""
    environ = make_environ(cookie=f"{SESSION_COOKIE_NAME}=bogus_token_here")
    cu = get_current_user(environ)
    assert cu is None


def test_get_current_user_invalid_bearer_returns_none(
    admin_user: User, make_environ: callable
) -> None:
    """Bogus token returns None."""
    environ = make_environ(bearer=f"{BEARER_HEADER_PREFIX}some_invalid_token")
    cu = get_current_user(environ)
    assert cu is None


def test_get_current_user_wrong_bearer_prefix_returns_none(make_environ: callable) -> None:
    """Authorization: Bearer foo returns None (only Warden-Bearer accepted)."""
    environ = make_environ(bearer="Bearer some_token")
    cu = get_current_user(environ)
    assert cu is None


def test_get_current_user_inactive_user_returns_none(
    clean_db: None, make_environ: callable
) -> None:
    """is_active=False user returns None for valid token."""
    from rtsp_warden.db.engine import get_session
    from rtsp_warden.db.models import User

    # Create an inactive user
    pw_hash = hash_password("testpass")
    with get_session() as session:
        user = User(
            username="inactive_user",
            password_hash=pw_hash,
            role="viewer",
            is_active=False,
        )
        session.add(user)
        session.commit()
        session.refresh(user)
        user_id = user.id

    # Create a token for the inactive user
    import hashlib

    from rtsp_warden.db.models import ApiToken

    raw = generate_api_token()
    token_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    prefix = raw[:8]

    with get_session() as session:
        tok = ApiToken(
            user_id=user_id,
            name="inactive-test",
            token_hash=token_hash,
            token_prefix=prefix,
        )
        session.add(tok)
        session.commit()

    environ = make_environ(bearer=f"{BEARER_HEADER_PREFIX}{raw}")
    cu = get_current_user(environ)
    assert cu is None
