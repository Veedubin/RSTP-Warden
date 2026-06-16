"""Tests for AppriseNotifier: protocol compliance, URL validation, send/test,
error handling, title template substitution, and factory integration."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from rtsp_warden.alerts import AppriseNotifier, build_notifier
from rtsp_warden.config import AppriseSpec, NtifySpec, WebhookSpec

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(coro: Any) -> Any:
    """Run an async coroutine synchronously."""
    return asyncio.run(coro)


def _make_event(
    camera_name: str = "front-door",
    event_type: str = "motion",
    severity: str = "info",
    message: str = "motion detected",
) -> dict[str, Any]:
    return {
        "id": 1,
        "camera_name": camera_name,
        "event_type": event_type,
        "severity": severity,
        "message": message,
        "metadata_json": "{}",
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
    }


def _apprise_spec(
    name: str = "test-apprise",
    urls: list[str] | None = None,
    title_template: str | None = None,
    min_severity: str = "info",
    min_interval_seconds: float = 60.0,
    enabled: bool = True,
) -> AppriseSpec:
    """Create an AppriseSpec for testing."""
    return AppriseSpec(
        name=name,
        type="apprise",
        urls=urls or ["mailto://user:pass@smtp.example.com:587"],
        title_template=title_template,
        min_severity=min_severity,
        min_interval_seconds=min_interval_seconds,
        enabled=enabled,
    )


# ---------------------------------------------------------------------------
# Protocol compliance tests
# ---------------------------------------------------------------------------


class TestAppriseProtocolCompliance:
    """Verify AppriseNotifier satisfies the Notifier Protocol (duck typing)."""

    def test_has_name_attribute(self) -> None:
        """AppriseNotifier has a .name attribute."""
        spec = _apprise_spec(name="my-email")
        notifier = AppriseNotifier(spec=spec)
        assert notifier.name == "my-email"

    def test_has_type_attribute(self) -> None:
        """AppriseNotifier has .type == 'apprise'."""
        spec = _apprise_spec()
        notifier = AppriseNotifier(spec=spec)
        assert notifier.type == "apprise"

    def test_has_send_method(self) -> None:
        """AppriseNotifier has an async send() method."""
        spec = _apprise_spec()
        notifier = AppriseNotifier(spec=spec)
        assert hasattr(notifier, "send")
        assert asyncio.iscoroutinefunction(notifier.send)

    def test_has_test_method(self) -> None:
        """AppriseNotifier has an async test() method."""
        spec = _apprise_spec()
        notifier = AppriseNotifier(spec=spec)
        assert hasattr(notifier, "test")
        assert asyncio.iscoroutinefunction(notifier.test)

    def test_has_close_method(self) -> None:
        """AppriseNotifier has a close() method (no-op for Apprise)."""
        spec = _apprise_spec()
        notifier = AppriseNotifier(spec=spec)
        assert hasattr(notifier, "close")
        assert asyncio.iscoroutinefunction(notifier.close)

    def test_close_is_noop(self) -> None:
        """AppriseNotifier.close() does not raise and returns None."""
        spec = _apprise_spec()
        notifier = AppriseNotifier(spec=spec)
        result = _run(notifier.close())
        assert result is None


# ---------------------------------------------------------------------------
# URL validation tests
# ---------------------------------------------------------------------------


class TestAppriseURLValidation:
    """Verify AppriseSpec URL validation raises on invalid input."""

    def test_empty_urls_list_raises(self) -> None:
        """AppriseSpec with empty urls list raises ValidationError."""
        with pytest.raises(ValueError):
            AppriseSpec(name="bad", type="apprise", urls=[])

    def test_blank_urls_raises(self) -> None:
        """AppriseSpec with only blank strings raises ValidationError."""
        with pytest.raises(ValueError):
            AppriseSpec(name="bad", type="apprise", urls=["  ", ""])

    def test_invalid_min_severity_raises(self) -> None:
        """AppriseSpec with invalid min_severity raises ValidationError."""
        with pytest.raises(ValueError):
            AppriseSpec(
                name="bad",
                type="apprise",
                urls=["mailto://u:p@smtp.example.com"],
                min_severity="critical",
            )

    def test_valid_urls_accepted(self) -> None:
        """AppriseSpec with valid URLs is accepted."""
        spec = AppriseSpec(
            name="ok",
            type="apprise",
            urls=["mailto://u:p@smtp.example.com", "tgram://bottoken/chatid"],
        )
        assert len(spec.urls) == 2


# ---------------------------------------------------------------------------
# Send / test with mocked Apprise
# ---------------------------------------------------------------------------


class TestAppriseSendSuccess:
    """Verify AppriseNotifier.send() works when Apprise succeeds."""

    @patch("apprise.Apprise")
    def test_send_success(self, mock_apprise_cls: MagicMock) -> None:
        """When Apprise.notify() returns True, NotificationResult.success is True."""
        mock_instance = MagicMock()
        mock_instance.add.return_value = True
        mock_instance.notify.return_value = True
        mock_apprise_cls.return_value = mock_instance

        spec = _apprise_spec()
        notifier = AppriseNotifier(spec=spec)
        result = _run(notifier.send(_make_event()))

        assert result.success is True
        assert result.notifier_name == "test-apprise"
        assert result.error is None
        mock_instance.add.assert_called_once_with("mailto://user:pass@smtp.example.com:587")
        mock_instance.notify.assert_called_once()

    @patch("apprise.Apprise")
    def test_send_failure_returns_false(self, mock_apprise_cls: MagicMock) -> None:
        """When Apprise.notify() returns False, NotificationResult.success is False."""
        mock_instance = MagicMock()
        mock_instance.add.return_value = True
        mock_instance.notify.return_value = False
        mock_apprise_cls.return_value = mock_instance

        spec = _apprise_spec()
        notifier = AppriseNotifier(spec=spec)
        result = _run(notifier.send(_make_event()))

        assert result.success is False
        assert "Apprise notify returned False" in (result.error or "")

    @patch("apprise.Apprise")
    def test_send_url_add_failure(self, mock_apprise_cls: MagicMock) -> None:
        """When Apprise.add() returns False, the result indicates failure."""
        mock_instance = MagicMock()
        mock_instance.add.return_value = False
        mock_apprise_cls.return_value = mock_instance

        spec = _apprise_spec()
        notifier = AppriseNotifier(spec=spec)
        result = _run(notifier.send(_make_event()))

        assert result.success is False
        assert "Failed to add Apprise URL" in (result.error or "")

    @patch("apprise.Apprise")
    def test_send_exception_caught(self, mock_apprise_cls: MagicMock) -> None:
        """When Apprise raises an exception, it is caught and returned as error."""
        mock_instance = MagicMock()
        mock_instance.add.return_value = True
        mock_instance.notify.side_effect = RuntimeError("connection failed")
        mock_apprise_cls.return_value = mock_instance

        spec = _apprise_spec()
        notifier = AppriseNotifier(spec=spec)
        result = _run(notifier.send(_make_event()))

        assert result.success is False
        assert "connection failed" in (result.error or "")


class TestAppriseTestMethod:
    """Verify AppriseNotifier.test() sends a test notification."""

    @patch("apprise.Apprise")
    def test_test_success(self, mock_apprise_cls: MagicMock) -> None:
        """test() sends a test message and returns success."""
        mock_instance = MagicMock()
        mock_instance.add.return_value = True
        mock_instance.notify.return_value = True
        mock_apprise_cls.return_value = mock_instance

        spec = _apprise_spec()
        notifier = AppriseNotifier(spec=spec)
        result = _run(notifier.test())

        assert result.success is True
        assert result.notifier_name == "test-apprise"
        call_args = mock_instance.notify.call_args
        assert call_args.kwargs["title"] == "rtsp-warden test"


# ---------------------------------------------------------------------------
# Title template substitution tests
# ---------------------------------------------------------------------------


class TestAppriseTitleTemplate:
    """Verify title template formatting works with and without template."""

    @patch("apprise.Apprise")
    def test_default_template(self, mock_apprise_cls: MagicMock) -> None:
        """Default template formats [severity] camera_name: event_type."""
        mock_instance = MagicMock()
        mock_instance.add.return_value = True
        mock_instance.notify.return_value = True
        mock_apprise_cls.return_value = mock_instance

        spec = _apprise_spec(title_template=None)
        notifier = AppriseNotifier(spec=spec)
        result = _run(
            notifier.send(_make_event(camera_name="yard", event_type="person", severity="warn"))
        )

        assert result.success is True
        call_args = mock_instance.notify.call_args
        title = call_args.kwargs["title"]
        assert title == "[warn] yard: person"

    @patch("apprise.Apprise")
    def test_custom_template(self, mock_apprise_cls: MagicMock) -> None:
        """Custom title_template is used for notification titles."""
        mock_instance = MagicMock()
        mock_instance.add.return_value = True
        mock_instance.notify.return_value = True
        mock_apprise_cls.return_value = mock_instance

        spec = _apprise_spec(title_template="ALERT: {severity} on {camera_name}")
        notifier = AppriseNotifier(spec=spec)
        result = _run(notifier.send(_make_event(camera_name="garage", severity="error")))

        assert result.success is True
        call_args = mock_instance.notify.call_args
        title = call_args.kwargs["title"]
        assert title == "ALERT: error on garage"


# ---------------------------------------------------------------------------
# Factory integration tests
# ---------------------------------------------------------------------------


class TestAppriseFactory:
    """Verify build_notifier returns AppriseNotifier for AppriseSpec."""

    def test_factory_builds_apprise(self) -> None:
        """build_notifier(AppriseSpec) returns an AppriseNotifier instance."""
        spec = AppriseSpec(
            name="gmail-alerts",
            type="apprise",
            urls=["mailto://user:pass@smtp.gmail.com:587"],
            min_severity="warn",
        )
        notifier = build_notifier(spec)
        assert isinstance(notifier, AppriseNotifier)
        assert notifier.name == "gmail-alerts"
        assert notifier.type == "apprise"

    def test_factory_still_builds_ntfy(self) -> None:
        """build_notifier still works for NtifySpec after the refactor."""
        spec = NtifySpec(
            name="phone",
            type="ntfy",
            url="https://ntfy.sh",
            topic="alerts",
        )
        notifier = build_notifier(spec)
        assert notifier.name == "phone"
        assert notifier.type == "ntfy"

    def test_factory_still_builds_webhook(self) -> None:
        """build_notifier still works for WebhookSpec after the refactor."""
        spec = WebhookSpec(
            name="slack",
            type="webhook",
            url="https://hooks.slack.com/services/abc",
        )
        notifier = build_notifier(spec)
        assert notifier.name == "slack"
        assert notifier.type == "webhook"


# ---------------------------------------------------------------------------
# AppriseSpec severities property tests
# ---------------------------------------------------------------------------


class TestAppriseSpecSeverities:
    """Verify the severities property derives from min_severity."""

    def test_info_severity(self) -> None:
        """min_severity='info' yields all three levels."""
        spec = AppriseSpec(
            name="test", type="apprise", urls=["mailto://x@y.com"], min_severity="info"
        )
        assert spec.severities == ["info", "warn", "error"]

    def test_warn_severity(self) -> None:
        """min_severity='warn' yields warn and error."""
        spec = AppriseSpec(
            name="test", type="apprise", urls=["mailto://x@y.com"], min_severity="warn"
        )
        assert spec.severities == ["warn", "error"]

    def test_error_severity(self) -> None:
        """min_severity='error' yields error only."""
        spec = AppriseSpec(
            name="test", type="apprise", urls=["mailto://x@y.com"], min_severity="error"
        )
        assert spec.severities == ["error"]

    def test_default_severity_is_info(self) -> None:
        """Default min_severity is 'info'."""
        spec = AppriseSpec(name="test", type="apprise", urls=["mailto://x@y.com"])
        assert spec.severities == ["info", "warn", "error"]


# ---------------------------------------------------------------------------
# Multiple URLs test
# ---------------------------------------------------------------------------


class TestAppriseMultipleURLs:
    """Verify AppriseNotifier handles multiple URLs."""

    @patch("apprise.Apprise")
    def test_sends_to_multiple_urls(self, mock_apprise_cls: MagicMock) -> None:
        """AppriseNotifier adds all URLs to the Apprise object."""
        mock_instance = MagicMock()
        mock_instance.add.return_value = True
        mock_instance.notify.return_value = True
        mock_apprise_cls.return_value = mock_instance

        spec = AppriseSpec(
            name="multi",
            type="apprise",
            urls=["mailto://u:p@smtp.example.com", "tgram://bottoken/chatid"],
        )
        notifier = AppriseNotifier(spec=spec)
        result = _run(notifier.send(_make_event()))

        assert result.success is True
        assert mock_instance.add.call_count == 2
        mock_instance.add.assert_any_call("mailto://u:p@smtp.example.com")
        mock_instance.add.assert_any_call("tgram://bottoken/chatid")
