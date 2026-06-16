"""Auth route handlers for the rtsp-warden web UI.

Provides login, logout, and session management endpoints.
All forms are protected by CSRF tokens.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.responses import Response
from starlette.templating import Jinja2Templates

from ... import auth
from ...db.schema import get_user_by_username
from ..auth_bridge import get_current_user_from_request
from ..csrf import CSRF_COOKIE_NAME, check_csrf_form
from ..paths import TEMPLATES_DIR
from ..rate_limit import get_login_limiter

router = APIRouter()

# Module-level templates instance (avoids re-creation per request).
_templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _is_auth_enabled() -> bool:
    """Check if auth is enabled via the WARDEN_AUTH_ENABLED env var."""
    return os.getenv("WARDEN_AUTH_ENABLED", "true").lower() not in ("false", "0", "no")


def _is_https() -> bool:
    """Check if HTTPS is expected (for secure cookie flag)."""
    return os.getenv("WARDEN_HTTPS", "false").lower() in ("true", "1", "yes")


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request) -> HTMLResponse:
    """Render the login form.

    If auth is disabled, redirect to the dashboard.
    If the user is already authenticated, redirect to the next URL or /.
    """
    if not _is_auth_enabled():
        return RedirectResponse(url="/", status_code=303)

    # Check if already authenticated
    current_user = get_current_user_from_request(request)
    if current_user is not None:
        next_url = request.query_params.get("next", "/")
        return RedirectResponse(url=next_url, status_code=303)

    csrf_token = getattr(request.state, "csrf_token", "")
    next_url = request.query_params.get("next", "/")

    return _templates.TemplateResponse(
        request,
        "login.html",
        {
            "request": request,
            "next": next_url,
            "csrf_token": csrf_token,
            "error": None,
        },
    )


@router.post("/login")
async def login_submit(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    csrf_token: str = Form(""),
    next: str = Form("/"),
) -> Response:
    """Process the login form submission.

    Validates CSRF, rate-limits failed attempts, and creates a session
    on success.
    """
    if not _is_auth_enabled():
        return RedirectResponse(url="/", status_code=303)

    # CSRF validation
    if not check_csrf_form(request, csrf_token):
        return RedirectResponse(url="/login", status_code=303)

    # Rate limiting
    client_ip = request.client.host if request.client else "unknown"
    limiter = get_login_limiter()
    if not limiter.check(client_ip):
        return Response(
            status_code=429,
            headers={"Retry-After": "60"},
            content="Too many login attempts. Try again later.",
        )

    # Authenticate
    user = get_user_by_username(username)
    if user is None or not auth.verify_password(password, user.password_hash):
        # Failed login -- re-render form with error
        csrf_cookie = request.cookies.get(CSRF_COOKIE_NAME, "")

        return _templates.TemplateResponse(
            request,
            "login.html",
            {
                "request": request,
                "next": next,
                "csrf_token": csrf_cookie,
                "error": "Invalid username or password",
            },
        )

    # Success -- create session
    auth_session = auth.create_session(user)
    limiter.reset(client_ip)

    # Calculate cookie max-age from session expiry
    now = datetime.now(timezone.utc)
    max_age = int((auth_session.expires_at - now).total_seconds())

    response = RedirectResponse(url=next, status_code=303)
    response.set_cookie(
        key=auth.SESSION_COOKIE_NAME,
        value=auth_session.token,
        httponly=True,
        secure=_is_https(),
        samesite="lax",
        max_age=max_age,
    )
    return response


@router.post("/logout")
async def logout_submit(
    request: Request,
    csrf_token: str = Form(""),
) -> RedirectResponse:
    """Destroy the current session and redirect to login."""
    # CSRF validation
    if not check_csrf_form(request, csrf_token):
        return RedirectResponse(url="/login", status_code=303)

    # Delete session if present
    session_token = request.cookies.get(auth.SESSION_COOKIE_NAME)
    if session_token:
        auth.delete_session(session_token)

    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie(key=auth.SESSION_COOKIE_NAME, path="/")
    return response
