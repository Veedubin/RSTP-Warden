"""ntfy action: one POST per event, the thumbnail as the request body when there is one."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

from .base import (
    HTTP_TIMEOUT_S,
    ActionPayload,
    ActionResult,
    exception_text,
    http_error_text,
    notification_body,
    notification_title,
    read_attachment,
    send_test,
)

if TYPE_CHECKING:
    from ..config import NtfyActionSpec


class NtfyAction:
    """Publish to ``{url}/{topic}`` (or ``{url}`` when no topic is set).

    Title, message and filename travel as query parameters, which httpx
    percent-encodes as UTF-8, so non-ASCII camera and zone names work. The token
    is the only config value sent as a header (``Authorization: Bearer``).
    """

    def __init__(
        self,
        spec: NtfyActionSpec,
        client_factory: Callable[[], httpx.Client] = httpx.Client,
    ) -> None:
        self.name: str = spec.name
        self.type: str = "ntfy"
        base = spec.url.rstrip("/")
        self._endpoint = f"{base}/{spec.topic}" if spec.topic else base
        self._token = spec.token
        self._priority = spec.priority
        self._client_factory = client_factory

    def send(self, payload: ActionPayload, attachment: Path | None = None) -> ActionResult:
        message = notification_body(payload)
        params: dict[str, str] = {"title": notification_title(payload), "message": message}
        if self._priority is not None:
            params["priority"] = str(self._priority)
        jpeg = read_attachment(attachment)
        if jpeg is not None and attachment is not None:
            params["filename"] = attachment.name
            content = jpeg
        else:
            content = message.encode("utf-8")
        headers: dict[str, str] = {}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        try:
            with self._client_factory() as client:
                response = client.post(
                    self._endpoint,
                    params=params,
                    content=content,
                    headers=headers,
                    timeout=HTTP_TIMEOUT_S,
                )
        except Exception as exc:
            return ActionResult(ok=False, error=exception_text(exc))
        if response.is_success:
            return ActionResult(ok=True)
        return ActionResult(ok=False, error=http_error_text(response))

    def test(self) -> ActionResult:
        return send_test(self)
