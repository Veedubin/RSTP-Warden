"""Base notifier Protocol + errors."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol


@dataclass(slots=True)
class NotificationResult:
    """Result of a single notifier send attempt."""

    notifier_name: str
    success: bool
    error: str | None = None
    sent_at: datetime | None = None
    http_status: int | None = None


class NotifierSendError(Exception):
    """Raised when a notifier fails to send a notification."""


class Notifier(Protocol):
    """Protocol that all notifier backends must implement."""

    name: str
    type: str  # "ntfy" | "webhook"

    async def send(self, event: dict[str, Any]) -> NotificationResult: ...
    async def test(self) -> NotificationResult: ...
