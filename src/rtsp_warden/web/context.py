"""Request context middleware: resolves the current user and CSRF token.

On every request, this middleware:
  1. Resolves ``current_user`` via the auth bridge and stores it on
     ``request.state.current_user``.
  2. Propagates the CSRF token (set by CSRFMiddleware) onto
     ``request.state.csrf_token`` for template access.

Templates can then use ``{{ request.state.current_user }}`` and
``{{ request.state.csrf_token }}`` without per-route plumbing.
"""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from .auth_bridge import get_current_user_from_request


class ContextMiddleware(BaseHTTPMiddleware):
    """Starlette middleware that populates request.state with auth context."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Resolve the current user (may be None for unauthenticated requests)
        current_user = get_current_user_from_request(request)
        request.state.current_user = current_user

        # Propagate CSRF token if not already set by CSRFMiddleware
        if not hasattr(request.state, "csrf_token"):
            request.state.csrf_token = request.cookies.get("warden_csrf", "")

        response: Response = await call_next(request)
        return response
