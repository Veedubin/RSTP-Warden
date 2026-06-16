"""API token management route handlers for the rtsp-warden web UI.

Provides per-user token listing, creation (with one-time raw display),
and revocation. All routes require authentication (require_user).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from starlette.templating import Jinja2Templates

from ... import auth
from ...db import get_user_by_id
from ..auth_depends import CurrentUser, require_user
from ..csrf import check_csrf_form
from ..paths import TEMPLATES_DIR

router = APIRouter(prefix="/api-tokens")

_templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@router.get("", response_class=HTMLResponse)
async def tokens_list(request: Request, user: CurrentUser = Depends(require_user)) -> HTMLResponse:
    """Render the API tokens list page for the current user."""
    tokens = auth.list_api_tokens(user.user_id)
    return _templates.TemplateResponse(
        request,
        "tokens/list.html",
        {
            "request": request,
            "tokens": tokens,
            "new_token_raw": None,
            "error": None,
        },
    )


@router.post("", response_class=HTMLResponse)
async def create_token(
    request: Request,
    name: str = Form(""),
    expires_in_days: str = Form("365"),
    csrf_token: str = Form(""),
    user: CurrentUser = Depends(require_user),
) -> HTMLResponse:
    """Create a new API token for the current user and display the raw value once."""
    if not check_csrf_form(request, csrf_token):
        raise HTTPException(status_code=403, detail="CSRF token missing or invalid")

    errors: list[str] = []
    name = name.strip()
    if not name:
        errors.append("Token name is required.")

    # Parse expiry
    ttl_seconds: int | None = None
    try:
        days = int(expires_in_days)
        if days <= 0:
            errors.append("Expiry must be a positive number of days.")
        else:
            ttl_seconds = days * 86400
    except (ValueError, TypeError):
        errors.append("Expiry must be a valid number of days.")

    if errors:
        tokens = auth.list_api_tokens(user.user_id)
        return _templates.TemplateResponse(
            request,
            "tokens/list.html",
            {
                "request": request,
                "tokens": tokens,
                "new_token_raw": None,
                "error": " ".join(errors),
            },
        )

    # Look up the User ORM object for create_api_token
    db_user = get_user_by_id(user.user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")

    token_obj = auth.create_api_token(db_user, name=name, ttl_seconds=ttl_seconds)

    # Refresh token list (includes the new one minus the raw value)
    tokens = auth.list_api_tokens(user.user_id)

    return _templates.TemplateResponse(
        request,
        "tokens/list.html",
        {
            "request": request,
            "tokens": tokens,
            "new_token_raw": token_obj.raw,
            "error": None,
        },
    )


@router.post("/{token_id}/revoke")
async def revoke_token(
    request: Request,
    token_id: int,
    csrf_token: str = Form(""),
    user: CurrentUser = Depends(require_user),
) -> HTMLResponse:
    """Revoke an API token by its database ID.

    Only the token owner can revoke their own tokens.
    """
    if not check_csrf_form(request, csrf_token):
        raise HTTPException(status_code=403, detail="CSRF token missing or invalid")

    # Find the token to verify ownership
    tokens = auth.list_api_tokens(user.user_id)
    target = None
    for t in tokens:
        if t["id"] == token_id:
            target = t
            break

    if target is None:
        raise HTTPException(status_code=404, detail="Token not found")

    auth.revoke_api_token(target["prefix"])

    tokens = auth.list_api_tokens(user.user_id)
    return _templates.TemplateResponse(
        request,
        "tokens/list.html",
        {
            "request": request,
            "tokens": tokens,
            "new_token_raw": None,
            "error": None,
        },
    )
