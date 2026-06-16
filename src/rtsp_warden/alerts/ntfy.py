"""Ntfy.sh notifier -- simple HTTP POST to https://ntfy.sh/<topic> or custom URL."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx

from .base import NotificationResult


class NtfyNotifier:
    """Send notifications via ntfy.sh HTTP API."""

    def __init__(
        self,
        name: str,
        url: str,
        topic: str | None = None,
        token: str | None = None,
        priority: int | None = None,
    ) -> None:
        self.name = name
        self.type = "ntfy"
        self._url = url.rstrip("/")
        self._topic = topic
        self._token = token
        self._priority = priority
        self._client: httpx.AsyncClient | None = None

    @property
    def _send_url(self) -> str:
        """Build the full ntfy URL by appending /<topic> if topic is set."""
        if self._topic:
            return f"{self._url}/{self._topic}"
        return self._url

    async def _get_client(self) -> httpx.AsyncClient:
        """Lazy-initialize the httpx async client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=10.0)
        return self._client

    async def send(self, event: dict[str, Any]) -> NotificationResult:
        """POST a notification for the given event to ntfy."""
        now = datetime.now(tz=timezone.utc)
        message = event.get("message", f"Event from {event.get('camera_name', 'unknown')}")
        event_type = event.get("event_type", "alert")
        severity = event.get("severity", "info")

        headers: dict[str, str] = {
            "Title": f"[{severity}] {event.get('camera_name', 'unknown')}: {event_type}",
            "Tags": event_type,
        }
        if self._priority is not None:
            headers["Priority"] = str(self._priority)
        if self._token is not None:
            headers["Authorization"] = f"Bearer {self._token}"

        try:
            client = await self._get_client()
            resp = await client.post(self._send_url, content=message, headers=headers)
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
        """Send a test notification."""
        now = datetime.now(tz=timezone.utc)
        headers: dict[str, str] = {
            "Title": "rtsp-warden test",
            "Tags": "test",
        }
        if self._priority is not None:
            headers["Priority"] = str(self._priority)
        if self._token is not None:
            headers["Authorization"] = f"Bearer {self._token}"

        try:
            client = await self._get_client()
            resp = await client.post(
                self._send_url, content="This is a test from rtsp-warden", headers=headers
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
