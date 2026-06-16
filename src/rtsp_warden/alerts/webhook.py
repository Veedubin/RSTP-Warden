"""Generic webhook notifier -- POST JSON to any HTTP endpoint."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import httpx

from .base import NotificationResult


def _json_default(obj: Any) -> Any:
    """JSON serializer for non-standard types (datetime -> ISO string)."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _serialize_event(event: dict[str, Any]) -> str:
    """Serialize an event dict to JSON, handling non-standard types."""
    return json.dumps(event, default=_json_default)


class WebhookNotifier:
    """Send notifications by POSTing JSON to a configurable HTTP endpoint."""

    def __init__(
        self,
        name: str,
        url: str,
        method: str = "POST",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.name = name
        self.type = "webhook"
        self._url = url
        self._method = method.upper()
        self._headers = headers or {}
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Lazy-initialize the httpx async client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=10.0)
        return self._client

    async def send(self, event: dict[str, Any]) -> NotificationResult:
        """POST the event dict as JSON to the webhook endpoint."""
        now = datetime.now(tz=timezone.utc)
        try:
            client = await self._get_client()
            body = _serialize_event(event)
            resp = await client.request(
                method=self._method,
                url=self._url,
                content=body,
                headers={
                    "Content-Type": "application/json",
                    **self._headers,
                },
            )
            resp.raise_for_status()
            return NotificationResult(
                notifier_name=self.name,
                success=True,
                sent_at=now,
                http_status=resp.status_code,
            )
        except httpx.HTTPStatusError as exc:
            return NotificationResult(
                notifier_name=self.name,
                success=False,
                error=str(exc),
                sent_at=now,
                http_status=exc.response.status_code,
            )
        except Exception as exc:
            return NotificationResult(
                notifier_name=self.name,
                success=False,
                error=str(exc),
                sent_at=now,
            )

    async def test(self) -> NotificationResult:
        """Send a test payload to the webhook endpoint."""
        now = datetime.now(tz=timezone.utc)
        payload = {"test": True, "notifier": self.name}
        try:
            client = await self._get_client()
            resp = await client.request(
                method=self._method,
                url=self._url,
                json=payload,
                headers=self._headers,
            )
            resp.raise_for_status()
            return NotificationResult(
                notifier_name=self.name,
                success=True,
                sent_at=now,
                http_status=resp.status_code,
            )
        except httpx.HTTPStatusError as exc:
            return NotificationResult(
                notifier_name=self.name,
                success=False,
                error=str(exc),
                sent_at=now,
                http_status=exc.response.status_code,
            )
        except Exception as exc:
            return NotificationResult(
                notifier_name=self.name,
                success=False,
                error=str(exc),
                sent_at=now,
            )

    async def close(self) -> None:
        """Close the httpx client."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
