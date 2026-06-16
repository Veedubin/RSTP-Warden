"""Apprise notifier -- send notifications via 90+ services using the Apprise library.

Supports email (mailto://), Telegram (tgram://), Discord (discord://),
Slack (slack://), and many more. See https://github.com/caronc/apprise/wiki
for the full URL catalog.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import apprise

from .base import NotificationResult

_DEFAULT_TITLE_TEMPLATE = "[{severity}] {camera_name}: {event_type}"


class AppriseNotifier:
    """Send notifications via the Apprise library (email + 90+ services).

    Args:
        spec: AppriseSpec configuration from AppConfig.alerts.notifiers.
    """

    def __init__(self, spec: Any) -> None:
        from ..config import AppriseSpec

        if not isinstance(spec, AppriseSpec):
            raise TypeError(f"Expected AppriseSpec, got {type(spec).__name__}")
        self._spec = spec
        self.name: str = spec.name
        self.type: str = "apprise"
        self._urls: list[str] = spec.urls
        self._title_template: str = spec.title_template or _DEFAULT_TITLE_TEMPLATE

    async def send(self, event: dict[str, Any]) -> NotificationResult:
        """Send a notification for the given event via Apprise.

        Apprise.notify() is synchronous, so it is run in a thread to avoid
        blocking the event loop.
        """
        now = datetime.now(tz=timezone.utc)
        title = self._title_template.format(
            camera_name=event.get("camera_name", "unknown"),
            event_type=event.get("event_type", "alert"),
            severity=event.get("severity", "info"),
        )
        body = event.get("message", f"Event from {event.get('camera_name', 'unknown')}")

        try:
            result = await asyncio.to_thread(self._notify_sync, title=title, body=body)
            return result
        except Exception as exc:
            return NotificationResult(
                notifier_name=self.name,
                success=False,
                error=str(exc),
                sent_at=now,
            )

    async def test(self) -> NotificationResult:
        """Send a test notification via Apprise."""
        now = datetime.now(tz=timezone.utc)
        try:
            result = await asyncio.to_thread(
                self._notify_sync,
                title="rtsp-warden test",
                body="This is a test from rtsp-warden",
            )
            return result
        except Exception as exc:
            return NotificationResult(
                notifier_name=self.name,
                success=False,
                error=str(exc),
                sent_at=now,
            )

    async def close(self) -> None:
        """No-op: Apprise is stateless (no persistent connection to close)."""

    def _notify_sync(self, title: str, body: str) -> NotificationResult:
        """Synchronous Apprise notify call (run via asyncio.to_thread)."""
        now = datetime.now(tz=timezone.utc)
        apobj = apprise.Apprise()
        for url in self._urls:
            if not apobj.add(url):
                return NotificationResult(
                    notifier_name=self.name,
                    success=False,
                    error=f"Failed to add Apprise URL: {url}",
                    sent_at=now,
                )
        ok = apobj.notify(title=title, body=body)
        if ok:
            return NotificationResult(
                notifier_name=self.name,
                success=True,
                sent_at=now,
            )
        return NotificationResult(
            notifier_name=self.name,
            success=False,
            error="Apprise notify returned False",
            sent_at=now,
        )
