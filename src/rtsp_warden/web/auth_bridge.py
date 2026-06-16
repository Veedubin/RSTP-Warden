"""Auth bridge: adapts WSGI-style auth module to FastAPI/ASGI requests.

The auth module uses a WSGI environ dict for request resolution.
This bridge translates FastAPI Request objects into that format.
"""

from __future__ import annotations

from fastapi import Request

from ..auth import CurrentUser, get_current_user


def build_environ_from_fastapi(request: Request) -> dict[str, str | None]:
    """Build a WSGI-style environ dict from a FastAPI Request.

    Maps the ``warden_session`` cookie and ``Authorization`` header into
    the keys that :func:`rtsp_warden.auth.get_current_user` expects:
    ``HTTP_COOKIE`` and ``HTTP_AUTHORIZATION``.
    """
    # Build the raw Cookie header value from FastAPI's parsed cookies
    cookie_parts: list[str] = []
    for name, value in request.cookies.items():
        cookie_parts.append(f"{name}={value}")
    http_cookie = "; ".join(cookie_parts) if cookie_parts else None

    http_authorization = request.headers.get("authorization")

    environ: dict[str, str | None] = {}
    if http_cookie is not None:
        environ["HTTP_COOKIE"] = http_cookie
    if http_authorization is not None:
        environ["HTTP_AUTHORIZATION"] = http_authorization

    return environ


def get_current_user_from_request(request: Request) -> CurrentUser | None:
    """Resolve the current user from a FastAPI Request.

    Bridges the WSGI-oriented :func:`rtsp_warden.auth.get_current_user`
    into the ASGI world by converting the request to an environ dict.
    """
    environ = build_environ_from_fastapi(request)
    return get_current_user(environ)
