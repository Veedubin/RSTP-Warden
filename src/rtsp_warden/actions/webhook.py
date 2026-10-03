"""Webhook action: the payload dict as a JSON body."""

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
    send_test,
)

if TYPE_CHECKING:
    from ..config import WebhookActionSpec


class WebhookAction:
    """``POST`` (or ``PUT``) ``payload.as_dict()`` as JSON to ``url`` with ``headers``."""

    def __init__(
        self,
        spec: WebhookActionSpec,
        client_factory: Callable[[], httpx.Client] = httpx.Client,
    ) -> None:
        self.name: str = spec.name
        self.type: str = "webhook"
        self._url = spec.url
        self._method = spec.method
        self._headers = dict(spec.headers)
        self._client_factory = client_factory

    def send(self, payload: ActionPayload, attachment: Path | None = None) -> ActionResult:
        """Send the JSON payload. ``attachment`` is ignored: the payload has ``thumbnail_url``."""
        try:
            with self._client_factory() as client:
                response = client.request(
                    self._method,
                    self._url,
                    json=payload.as_dict(),
                    headers=self._headers,
                    timeout=HTTP_TIMEOUT_S,
                )
        except Exception as exc:
            return ActionResult(ok=False, error=exception_text(exc))
        if response.is_success:
            return ActionResult(ok=True)
        return ActionResult(ok=False, error=http_error_text(response))

    def test(self) -> ActionResult:
        return send_test(self)
