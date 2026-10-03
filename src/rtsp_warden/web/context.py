"""Request context middleware: current user, CSRF token and flash message.

On every request, this middleware:
  1. Resolves ``current_user`` via the auth bridge and stores it on
     ``request.state.current_user``.
  2. Propagates the CSRF token (set by CSRFMiddleware) onto
     ``request.state.csrf_token`` for template access.
  3. On full-page GETs, decodes a pending ``warden_flash`` cookie onto
     ``request.state.flash`` (None otherwise) and deletes the cookie once an
     HTML page was rendered with it, so the message shows exactly once.

Templates can then use ``{{ request.state.current_user }}``,
``{{ request.state.csrf_token }}`` and ``{{ request.state.flash }}`` without
per-route plumbing.
"""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from .auth_bridge import get_current_user_from_request
from .flash import FLASH_COOKIE, decode_flash


def _shows_flash(request: Request) -> bool:
    """Only a full-page GET shows a flash; htmx fragments and form posts leave it pending."""
    return request.method == "GET" and request.headers.get("hx-request") != "true"


def _is_html_page(response: Response) -> bool:
    """True for a 200 HTML response (a redirect or an error does not consume the flash)."""
    content_type = response.headers.get("content-type", "")
    return response.status_code == 200 and content_type.startswith("text/html")


def _sets_cookie(response: Response, name: str) -> bool:
    """True when the response already sets the cookie ``name`` (a newer flash)."""
    prefix = f"{name}="
    return any(value.startswith(prefix) for value in response.headers.getlist("set-cookie"))


class ContextMiddleware(BaseHTTPMiddleware):
    """Starlette middleware that populates request.state with auth context."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Resolve the current user (may be None for unauthenticated requests)
        current_user = get_current_user_from_request(request)
        request.state.current_user = current_user

        # Propagate CSRF token if not already set by CSRFMiddleware
        if not hasattr(request.state, "csrf_token"):
            request.state.csrf_token = request.cookies.get("warden_csrf", "")

        raw_flash = request.cookies.get(FLASH_COOKIE) if _shows_flash(request) else None
        request.state.flash = decode_flash(raw_flash)

        response: Response = await call_next(request)

        if (
            raw_flash is not None
            and _is_html_page(response)
            and not _sets_cookie(response, FLASH_COOKIE)
        ):
            response.delete_cookie(FLASH_COOKIE, path="/")
        return response
