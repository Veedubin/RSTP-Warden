"""AlertManager -- dispatches events to notifiers with debouncing and severity filter."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from ..config import AlertsConfig
from .base import NotificationResult
from .factory import build_notifier

logger = logging.getLogger(__name__)


class AlertManager:
    """Dispatch events to configured notifiers with debouncing and severity filtering.

    Features:
      - Skips disabled notifiers
      - Filters by severity (event severity must be in notifier.severities)
      - Debounces: skips if same (camera, event_type) was sent to the same
        notifier within min_interval_seconds
      - Dispatches to all eligible notifiers in parallel via asyncio.gather
      - Catches per-notifier exceptions so one failure doesn't break others
    """

    def __init__(
        self,
        cfg: AlertsConfig,
        on_send: Callable[[NotificationResult], None] | None = None,
    ) -> None:
        self._cfg = cfg
        self._on_send = on_send
        self._notifiers: list[Any] = []  # built from cfg.notifiers
        self._spec_map: dict[str, Any] = {}  # name -> NotifierSpec for severity/interval lookup
        # Per-notifier debounce tracking: {(notifier_name, camera, event_type): last_sent_at}
        self._last_sent: dict[tuple[str, str, str], datetime] = {}

    async def start(self) -> None:
        """Initialize notifiers from config."""
        self._notifiers = []
        self._spec_map = {}
        for spec in self._cfg.notifiers:
            notifier = build_notifier(spec)
            self._notifiers.append(notifier)
            self._spec_map[spec.name] = spec
        logger.info("AlertManager started with %d notifier(s)", len(self._notifiers))

    async def stop(self) -> None:
        """Close httpx clients on all notifiers."""
        for notifier in self._notifiers:
            if hasattr(notifier, "close"):
                try:
                    await notifier.close()
                except Exception:
                    logger.debug("error closing notifier %s", notifier.name, exc_info=True)
        logger.info("AlertManager stopped")

    async def dispatch_event(self, event: dict[str, Any]) -> list[NotificationResult]:
        """Dispatch an event to all eligible notifiers.

        Args:
            event: Event dict with keys: camera_name, event_type, severity, message, etc.

        Returns:
            List of NotificationResult for each notifier that was attempted.
        """
        if not self._cfg.enabled:
            return []

        camera = event.get("camera_name", "")
        event_type = event.get("event_type", "")
        severity = event.get("severity", "info")
        results: list[NotificationResult] = []

        async def _dispatch_one(notifier: Any) -> NotificationResult:
            spec = self._spec_map.get(notifier.name)
            if spec is None:
                return NotificationResult(
                    notifier_name=notifier.name,
                    success=False,
                    error="no spec found",
                )

            # Skip disabled notifiers
            if not spec.enabled:
                return NotificationResult(
                    notifier_name=notifier.name,
                    success=False,
                    error="disabled",
                )

            # Severity filter
            if severity not in spec.severities:
                return NotificationResult(
                    notifier_name=notifier.name,
                    success=False,
                    error=f"severity {severity!r} not in {spec.severities}",
                )

            # Debounce check
            key = (notifier.name, camera, event_type)
            last = self._last_sent.get(key)
            now = datetime.now(tz=timezone.utc)
            if last is not None:
                elapsed = (now - last).total_seconds()
                if elapsed < spec.min_interval_seconds:
                    return NotificationResult(
                        notifier_name=notifier.name,
                        success=False,
                        error=f"debounced ({elapsed:.0f}s < {spec.min_interval_seconds}s)",
                    )

            # Send
            try:
                result = await notifier.send(event)
                if result.success:
                    self._last_sent[key] = now
            except Exception as exc:
                result = NotificationResult(
                    notifier_name=notifier.name,
                    success=False,
                    error=str(exc),
                )

            if self._on_send is not None:
                try:
                    self._on_send(result)
                except Exception:
                    logger.debug("on_send callback error", exc_info=True)

            return result

        # Run all notifiers in parallel
        coros = [_dispatch_one(n) for n in self._notifiers]
        if coros:
            results = await asyncio.gather(*coros)

        return list(results)

    async def test_notifier(self, name: str) -> NotificationResult:
        """Send a test notification to the named notifier.

        Args:
            name: The notifier name to test.

        Returns:
            NotificationResult from the test send.

        Raises:
            KeyError: If no notifier with the given name exists.
        """
        for notifier in self._notifiers:
            if notifier.name == name:
                return await notifier.test()
        raise KeyError(f"no notifier named {name!r}")

    def status(self) -> dict[str, Any]:
        """Return a status dict suitable for /status.json."""
        return {
            "enabled": self._cfg.enabled,
            "notifier_count": len(self._notifiers),
            "notifiers": [
                {
                    "name": n.name,
                    "type": n.type,
                    "enabled": self._spec_map.get(n.name, None) is not None
                    and self._spec_map[n.name].enabled,
                }
                for n in self._notifiers
            ],
        }
