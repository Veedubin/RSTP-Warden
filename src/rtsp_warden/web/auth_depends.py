"""FastAPI dependency injection helpers for authentication.

Provides ``require_user`` and ``require_admin`` as ``Depends()`` targets
for route-level access control.
"""

from __future__ import annotations

from fastapi import Depends, HTTPException, Request, status

from ..auth import CurrentUser
from .auth_bridge import get_current_user_from_request


class LoginRequired(Exception):
    """Raised for browser requests with no session; handled by a redirect to /login."""

    def __init__(self, next_url: str) -> None:
        super().__init__(next_url)
        self.next_url = next_url


async def get_current_user(request: Request) -> CurrentUser | None:
    """Resolve the current user from the request, or return None."""
    return get_current_user_from_request(request)


def _wants_html(request: Request) -> bool:
    accept = request.headers.get("accept", "")
    return "text/html" in accept


async def require_user(
    request: Request,
    user: CurrentUser | None = Depends(get_current_user),
) -> CurrentUser:
    """Require an authenticated user.

    Browser page loads are redirected to /login. htmx partial requests get a
    401 with an ``HX-Redirect`` header so htmx navigates the whole page.
    Everything else gets a plain 401.
    """
    if user is None:
        if request.headers.get("hx-request") == "true":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Not authenticated",
                headers={"HX-Redirect": "/login"},
            )
        if _wants_html(request):
            raise LoginRequired(next_url=request.url.path)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": 'Bearer realm="warden"'},
        )
    return user


async def require_admin(user: CurrentUser = Depends(require_user)) -> CurrentUser:
    """Dependency that requires an authenticated admin user.

    Raises 403 if the user is authenticated but not an admin.
    """
    if user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required",
        )
    return user
