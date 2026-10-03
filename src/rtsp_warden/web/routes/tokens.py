"""API token management route handlers for the rtsp-warden web UI.

Provides per-user token listing, creation (with one-time raw display),
and revocation. All routes require authentication (require_user).

The create and revoke forms on ``tokens/list.html`` are htmx forms that swap
``#tokens-panel``: an htmx request gets only ``partials/tokens_table.html``
back (200, errors shown inside it). Without htmx the routes render the whole
page (200) as before; an unknown token is still a 404 there.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse

from ... import auth
from ...db import get_user_by_id
from ..auth_depends import CurrentUser, require_user
from ..csrf import check_csrf_form
from ._common import is_htmx, templates

router = APIRouter(prefix="/api-tokens")


def _render_tokens(
    request: Request,
    user: CurrentUser,
    *,
    new_token_raw: str | None = None,
    notice: str | None = None,
    error: str | None = None,
) -> HTMLResponse:
    """Render the token panel for an htmx request, else the whole tokens page."""
    template = "partials/tokens_table.html" if is_htmx(request) else "tokens/list.html"
    return templates.TemplateResponse(
        request,
        template,
        {
            "request": request,
            "tokens": auth.list_api_tokens(user.user_id),
            "new_token_raw": new_token_raw,
            "notice": notice,
            "error": error,
        },
    )


@router.get("", response_class=HTMLResponse)
async def tokens_list(request: Request, user: CurrentUser = Depends(require_user)) -> HTMLResponse:
    """Render the API tokens list page for the current user."""
    return templates.TemplateResponse(
        request,
        "tokens/list.html",
        {
            "request": request,
            "tokens": auth.list_api_tokens(user.user_id),
            "new_token_raw": None,
            "notice": None,
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
        return _render_tokens(request, user, error=" ".join(errors))

    # Look up the User ORM object for create_api_token
    db_user = get_user_by_id(user.user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")

    token_obj = auth.create_api_token(db_user, name=name, ttl_seconds=ttl_seconds)
    return _render_tokens(request, user, new_token_raw=token_obj.raw)


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
    target = None
    for t in auth.list_api_tokens(user.user_id):
        if t["id"] == token_id:
            target = t
            break

    if target is None:
        if is_htmx(request):
            return _render_tokens(request, user, error="Token not found.")
        raise HTTPException(status_code=404, detail="Token not found")

    auth.revoke_api_token(target["prefix"])
    return _render_tokens(request, user, notice=f"Token {target['name']} revoked.")
