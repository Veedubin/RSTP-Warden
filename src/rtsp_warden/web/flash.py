"""One-shot flash messages carried across a redirect in a short-lived cookie.

A route calls ``set_flash(response, message, level)`` (``web/routes/_common.py``) on the
response it returns, usually a 303 redirect. ``ContextMiddleware`` (``web/context.py``)
decodes the cookie on the next full-page GET into ``request.state.flash``,
``partials/flash.html`` renders it, and the middleware deletes the cookie on that
response, so the message is shown exactly once.

The cookie value is base64url (no padding) of the JSON object ``{"m": message, "l": level}``.
Plain JSON is not a valid RFC 6265 cookie value (quotes, commas, spaces) and response
headers must be latin-1, so the encoded form keeps any message text safe.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Literal

FLASH_COOKIE = "warden_flash"
FLASH_MAX_AGE_SECONDS = 60
MAX_FLASH_CHARS = 500

FlashLevel = Literal["info", "success", "error"]
FLASH_LEVELS: tuple[str, ...] = ("info", "success", "error")


@dataclass(frozen=True, slots=True)
class Flash:
    """A decoded flash message. ``level`` is always one of FLASH_LEVELS."""

    message: str
    level: str


def _normalise_level(level: object) -> str:
    """Return ``level`` when it is a known level, else ``"info"``."""
    return level if isinstance(level, str) and level in FLASH_LEVELS else "info"


def encode_flash(message: str, level: str = "info") -> str:
    """Return the cookie value for a flash message (message cut to MAX_FLASH_CHARS)."""
    payload = json.dumps({"m": message[:MAX_FLASH_CHARS], "l": _normalise_level(level)})
    return base64.urlsafe_b64encode(payload.encode("utf-8")).decode("ascii").rstrip("=")


def decode_flash(raw: str | None) -> Flash | None:
    """Parse a cookie value made by encode_flash; None when absent or malformed."""
    if not raw:
        return None
    try:
        data = json.loads(base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4)))
    except ValueError:  # binascii.Error, UnicodeDecodeError and JSONDecodeError
        return None
    if not isinstance(data, dict):
        return None
    message = data.get("m")
    if not isinstance(message, str) or not message:
        return None
    return Flash(message=message[:MAX_FLASH_CHARS], level=_normalise_level(data.get("l")))
