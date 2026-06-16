"""CSRF protection middleware using the double-submit cookie pattern.

Sets a ``warden_csrf`` cookie on every response (if absent). For mutating
HTTP methods (POST, PUT, DELETE, PATCH), validates that either:

1. The ``X-CSRF-Token`` header matches the cookie, OR
2. The ``csrf_token`` query parameter matches the cookie.

For form submissions, HTML forms should include a hidden ``csrf_token``
field and use htmx's ``hx-headers`` attribute to copy it into the
``X-CSRF-Token`` header. This is the standard htmx CSRF pattern.

GET, HEAD, and OPTIONS requests are exempt from validation.
"""

from __future__ import annotations

import secrets

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

CSRF_COOKIE_NAME = "warden_csrf"
CSRF_HEADER_NAME = "x-csrf-token"
CSRF_FORM_FIELD = "csrf_token"
CSRF_TOKEN_LENGTH = 32  # 32 bytes -> 64 hex chars

MUTATING_METHODS = {"POST", "PUT", "DELETE", "PATCH"}


class CSRFMiddleware(BaseHTTPMiddleware):
    """Starlette middleware enforcing CSRF protection via double-submit cookie.

    On every response, ensures the ``warden_csrf`` cookie is set.
    On mutating requests, validates the ``X-CSRF-Token`` header (or
    ``csrf_token`` query parameter) against the cookie value.
    Returns 403 if missing or mismatched.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        csrf_cookie = request.cookies.get(CSRF_COOKIE_NAME)

        # Generate a new token if none exists
        if not csrf_cookie:
            csrf_cookie = secrets.token_hex(CSRF_TOKEN_LENGTH)

        # Validate on mutating requests (header or query param)
        if request.method in MUTATING_METHODS:
            provided_token = request.headers.get(CSRF_HEADER_NAME)
            if not provided_token:
                provided_token = request.query_params.get(CSRF_FORM_FIELD)

            if not provided_token or provided_token != csrf_cookie:
                return JSONResponse(
                    status_code=403,
                    content={"detail": "CSRF token missing or invalid"},
                )

        # Store token on request.state for downstream use
        request.state.csrf_token = csrf_cookie

        # Process the request
        response = await call_next(request)

        # Set the CSRF cookie on every response
        response.set_cookie(
            key=CSRF_COOKIE_NAME,
            value=csrf_cookie,
            httponly=False,
            samesite="lax",
            path="/",
        )

        return response


def check_csrf_form(request: Request, form_token: str) -> bool:
    """Validate a form-submitted CSRF token against the cookie.

    Call this in route handlers after reading ``csrf_token`` from form data.
    Returns True if the token matches the cookie, False otherwise.
    """
    cookie_token = request.cookies.get(CSRF_COOKIE_NAME, "")
    return cookie_token == form_token
