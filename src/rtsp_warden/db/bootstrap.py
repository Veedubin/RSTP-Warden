"""First-run bootstrap used by ``serve``: schema, then an admin user if none exists."""

from __future__ import annotations

import logging
import os
import secrets
from collections.abc import Mapping

from ..auth import hash_password
from .schema import create_admin_user, ensure_schema, list_users

log = logging.getLogger(__name__)


def _generate_password() -> str:
    return secrets.token_urlsafe(12)


def ensure_admin_user(env: Mapping[str, str] | None = None) -> tuple[str, str] | None:
    """Create the first admin user when the users table is empty.

    Username and password come from WARDEN_ADMIN_USERNAME / WARDEN_ADMIN_PASSWORD
    when set; otherwise ``admin`` with a generated password. Returns the
    credentials that were created, or None when users already existed.
    """
    source = os.environ if env is None else env
    if list_users():
        return None
    username = source.get("WARDEN_ADMIN_USERNAME") or "admin"
    password = source.get("WARDEN_ADMIN_PASSWORD") or _generate_password()
    create_admin_user(username, hash_password(password))
    return username, password


def bootstrap_database() -> tuple[str, str] | None:
    """Ensure the schema exists and an admin user exists. Logs created credentials once."""
    ensure_schema()
    created = ensure_admin_user()
    if created is not None:
        username, password = created
        log.warning(
            "No users existed; created admin %r with password %r. "
            "Log in and change it, or set WARDEN_ADMIN_PASSWORD before first start.",
            username,
            password,
        )
    return created
