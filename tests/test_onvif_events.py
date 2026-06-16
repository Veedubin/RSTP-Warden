"""Tests for ONVIF event subscription (OnvifEventSubscriber, OnvifEvent, OnvifEventType)."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

import httpx
import pytest

from rtsp_warden.config import OnvifConfig, OnvifEventConfig
from rtsp_warden.onvif.discovery import OnvifError
from rtsp_warden.onvif.events import (
    OnvifEvent,
    OnvifEventSubscriber,
    OnvifEventType,
    _build_pull_messages_envelope,
    _build_subscribe_envelope,
    _build_unsubscribe_envelope,
    _check_soap_fault,
    _parse_event_capability,
    _parse_pull_messages,
    _parse_subscription_ref,
    get_active_subscribers,
    get_subscription_states,
    register_subscriber,
    unregister_subscriber,
)
from rtsp_warden.onvif.ptz import OnvifClient

# ---------------------------------------------------------------------------
# Sample SOAP responses
# ---------------------------------------------------------------------------

CAPABILITIES_WITH_EVENTS = """<?xml version="1.0" encoding="utf-8"?>
<soap:Envelope xmlns:soap="http://www.w3.org/2003/05/soap-envelope"
               xmlns:tds="http://www.onvif.org/ver10/device/wsdl">
  <soap:Body>
    <tds:GetCapabilitiesResponse>
      <tds:Capabilities>
        <tt:Events xmlns:tt="http://www.onvif.org/ver10/schema">
          <tt:XAddr>http://192.168.1.100:80/onvif/event_service</tt:XAddr>
        </tt:Events>
      </tds:Capabilities>
    </tds:GetCapabilitiesResponse>
  </soap:Body>
</soap:Envelope>"""

SUBSCRIBE_RESPONSE = """<?xml version="1.0" encoding="utf-8"?>
<soap:Envelope xmlns:soap="http://www.w3.org/2003/05/soap-envelope"
               xmlns:wsa="http://schemas.xmlsoap.org/ws/2004/08/addressing">
  <soap:Body>
    <tev:CreatePullPointSubscriptionResponse xmlns:tev="http://www.onvif.org/ver10/events/wsdl">
      <tev:SubscriptionReference>
        <wsa:Address>http://192.168.1.100:80/onvif/subscriptions?id=42</wsa:Address>
      </tev:SubscriptionReference>
    </tev:CreatePullPointSubscriptionResponse>
  </soap:Body>
</soap:Envelope>"""

PULL_MESSAGES_WITH_MOTION = """<?xml version="1.0" encoding="utf-8"?>
<soap:Envelope xmlns:soap="http://www.w3.org/2003/05/soap-envelope"
               xmlns:tev="http://www.onvif.org/ver10/events/wsdl"
               xmlns:tt="http://www.onvif.org/ver10/schema">
  <soap:Body>
    <tev:PullMessagesResponse>
      <tev:CurrentTime>2026-06-16T12:00:00Z</tev:CurrentTime>
      <tev:NotificationMessage>
        <tt:Topic>tns1:VideoSource/MotionAlarm</tt:Topic>
        <tt:UtcTime>2026-06-16T12:00:00Z</tt:UtcTime>
        <tt:Message>
          <tt:Source>
            <tt:SimpleItem Name="VideoSourceToken" Value="VideoSource_1"/>
          </tt:Source>
          <tt:Data>
            <tt:SimpleItem Name="State" Value="true"/>
          </tt:Data>
        </tt:Message>
      </tev:NotificationMessage>
    </tev:PullMessagesResponse>
  </soap:Body>
</soap:Envelope>"""

PULL_MESSAGES_WITH_TAMPER = """<?xml version="1.0" encoding="utf-8"?>
<soap:Envelope xmlns:soap="http://www.w3.org/2003/05/soap-envelope"
               xmlns:tt="http://www.onvif.org/ver10/schema">
  <soap:Body>
    <tev:PullMessagesResponse xmlns:tev="http://www.onvif.org/ver10/events/wsdl">
      <tev:NotificationMessage>
        <tt:Topic>tns1:VideoSource/ImagingAlarm/Tampering</tt:Topic>
        <tt:UtcTime>2026-06-16T12:05:00Z</tt:UtcTime>
        <tt:Message>
          <tt:Data>
            <tt:SimpleItem Name="State" Value="true"/>
          </tt:Data>
        </tt:Message>
      </tev:NotificationMessage>
    </tev:PullMessagesResponse>
  </soap:Body>
</soap:Envelope>"""

PULL_MESSAGES_EMPTY = """<?xml version="1.0" encoding="utf-8"?>
<soap:Envelope xmlns:soap="http://www.w3.org/2003/05/soap-envelope">
  <soap:Body>
    <tev:PullMessagesResponse xmlns:tev="http://www.onvif.org/ver10/events/wsdl"/>
  </soap:Body>
</soap:Envelope>"""

UNSUBSCRIBE_RESPONSE = """<?xml version="1.0" encoding="utf-8"?>
<soap:Envelope xmlns:soap="http://www.w3.org/2003/05/soap-envelope">
  <soap:Body>
    <wsnt:UnsubscribeResponse xmlns:wsnt="http://docs.oasis-open.org/wsn/b-2"/>
  </soap:Body>
</soap:Envelope>"""

FAULT_RESPONSE = """<?xml version="1.0" encoding="utf-8"?>
<soap:Envelope xmlns:soap="http://www.w3.org/2003/05/soap-envelope">
  <soap:Body>
    <soap:Fault>
      <soap:Code>soap:Sender</soap:Code>
      <soap:Reason>
        <soap:Text xml:lang="en">ActionNotSupported</soap:Text>
      </soap:Reason>
    </soap:Fault>
  </soap:Body>
</soap:Envelope>"""

AUTH_FAILURE_RESPONSE = "Unauthorized"


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_sequenced_transport(
    responses: list[tuple[int, str]],
) -> httpx.MockTransport:
    """Create a MockTransport that returns responses in sequence."""

    idx = [0]

    def handler(request: httpx.Request) -> httpx.Response:
        if idx[0] >= len(responses):
            return httpx.Response(200, text=UNSUBSCRIBE_RESPONSE)
        status, body = responses[idx[0]]
        idx[0] += 1
        return httpx.Response(status, text=body)

    return httpx.MockTransport(handler)


def _make_subscriber(
    transport: httpx.MockTransport | None = None,
    callback: object = None,
    poll_interval: float = 10.0,
) -> OnvifEventSubscriber:
    """Create an OnvifEventSubscriber with a mock transport for testing."""
    client = OnvifClient(
        device_xaddr="http://192.168.1.100/onvif/device_service",
        username="admin",
        password="pass",
    )
    sub = OnvifEventSubscriber(
        client=client,
        camera_name="test_cam",
        topics=["tns1:VideoSource/MotionAlarm"],
        callback=callback,
        poll_interval_seconds=poll_interval,
    )
    if transport is not None:
        sub._transport = transport
    return sub


# ---------------------------------------------------------------------------
# OnvifEventType and OnvifEvent tests
# ---------------------------------------------------------------------------


class TestOnvifEventType:
    """Tests for OnvifEventType enum values."""

    def test_motion_value(self) -> None:
        assert OnvifEventType.MOTION == "motion"

    def test_tamper_value(self) -> None:
        assert OnvifEventType.TAMPER == "tamper"

    def test_analytic_value(self) -> None:
        assert OnvifEventType.ANALYTIC == "analytic"

    def test_unknown_value(self) -> None:
        assert OnvifEventType.UNKNOWN == "unknown"


class TestOnvifEventFromTopic:
    """Tests for OnvifEvent.from_topic() topic classification."""

    def test_motion_alarm_topic(self) -> None:
        """MotionAlarm topic maps to MOTION."""
        event = OnvifEvent.from_topic("test_cam", "tns1:VideoSource/MotionAlarm")
        assert event.event_type == OnvifEventType.MOTION
        assert event.camera_name == "test_cam"

    def test_cell_motion_topic(self) -> None:
        """CellMotionDetector topic maps to MOTION."""
        event = OnvifEvent.from_topic("test_cam", "tns1:RuleEngine/CellMotionDetector/Motion")
        assert event.event_type == OnvifEventType.MOTION

    def test_tampering_topic(self) -> None:
        """Tampering topic maps to TAMPER."""
        event = OnvifEvent.from_topic("test_cam", "tns1:VideoSource/ImagingAlarm/Tampering")
        assert event.event_type == OnvifEventType.TAMPER

    def test_imaging_alarm_topic(self) -> None:
        """ImagingAlarm topic maps to TAMPER."""
        event = OnvifEvent.from_topic("test_cam", "tns1:VideoSource/ImagingAlarm")
        assert event.event_type == OnvifEventType.TAMPER

    def test_rule_engine_topic(self) -> None:
        """RuleEngine topic maps to ANALYTIC."""
        event = OnvifEvent.from_topic("test_cam", "tns1:RuleEngine/LineCross")
        assert event.event_type == OnvifEventType.ANALYTIC

    def test_unknown_topic(self) -> None:
        """Unknown topic maps to UNKNOWN."""
        event = OnvifEvent.from_topic("test_cam", "tns1:SomeUnknownTopic")
        assert event.event_type == OnvifEventType.UNKNOWN

    def test_case_insensitive_matching(self) -> None:
        """Topic matching is case-insensitive."""
        event = OnvifEvent.from_topic("test_cam", "tns1:VideoSource/MOTIONALARM")
        assert event.event_type == OnvifEventType.MOTION

    def test_default_timestamp(self) -> None:
        """When no timestamp provided, defaults to now UTC."""
        before = datetime.now(tz=timezone.utc)
        event = OnvifEvent.from_topic("test_cam", "tns1:VideoSource/MotionAlarm")
        after = datetime.now(tz=timezone.utc)
        assert before <= event.timestamp <= after

    def test_custom_timestamp(self) -> None:
        """Custom timestamp is preserved."""
        ts = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        event = OnvifEvent.from_topic("test_cam", "tns1:VideoSource/MotionAlarm", timestamp=ts)
        assert event.timestamp == ts

    def test_raw_message_preserved(self) -> None:
        """raw_message dict is preserved in the event."""
        data = {"State": "true", "VideoSourceToken": "src1"}
        event = OnvifEvent.from_topic("test_cam", "tns1:VideoSource/MotionAlarm", raw_message=data)
        assert event.raw_message == data


class TestOnvifEventToAlertEvent:
    """Tests for OnvifEvent.to_alert_event() conversion."""

    def test_motion_to_alert_event(self) -> None:
        """Motion event converts to info severity alert event dict."""
        event = OnvifEvent.from_topic("test_cam", "tns1:VideoSource/MotionAlarm")
        alert = event.to_alert_event()
        assert alert["camera_name"] == "test_cam"
        assert alert["event_type"] == "motion"
        assert alert["severity"] == "info"
        assert "MotionAlarm" in alert["message"]

    def test_tamper_to_alert_event(self) -> None:
        """Tamper event converts to warn severity alert event dict."""
        event = OnvifEvent.from_topic("test_cam", "tns1:VideoSource/ImagingAlarm/Tampering")
        alert = event.to_alert_event()
        assert alert["severity"] == "warn"

    def test_unknown_to_alert_event(self) -> None:
        """Unknown event type converts to info severity."""
        event = OnvifEvent.from_topic("test_cam", "tns1:SomeTopic")
        alert = event.to_alert_event()
        assert alert["severity"] == "info"
        assert alert["event_type"] == "unknown"


# ---------------------------------------------------------------------------
# SOAP envelope builder tests
# ---------------------------------------------------------------------------


class TestSoapEnvelopeBuilders:
    """Tests for SOAP envelope construction."""

    def test_subscribe_envelope_contains_action(self) -> None:
        """Subscribe envelope includes CreatePullPointSubscription action."""
        envelope = _build_subscribe_envelope(timeout_seconds=60)
        assert "CreatePullPointSubscription" in envelope
        assert "PT60S" in envelope

    def test_subscribe_envelope_custom_timeout(self) -> None:
        """Subscribe envelope uses the specified timeout value."""
        envelope = _build_subscribe_envelope(timeout_seconds=120)
        assert "PT120S" in envelope

    def test_pull_messages_envelope(self) -> None:
        """PullMessages envelope includes timeout and message limit."""
        envelope = _build_pull_messages_envelope(
            subscription_ref="http://example.com/sub/1",
            timeout_seconds=10,
            message_limit=5,
        )
        assert "PullMessages" in envelope
        assert "PT10S" in envelope
        assert "5" in envelope  # message limit

    def test_unsubscribe_envelope(self) -> None:
        """Unsubscribe envelope includes Unsubscribe action."""
        envelope = _build_unsubscribe_envelope()
        assert "Unsubscribe" in envelope


# ---------------------------------------------------------------------------
# XML response parser tests
# ---------------------------------------------------------------------------


class TestParseSubscriptionRef:
    """Tests for _parse_subscription_ref."""

    def test_parse_subscription_ref_success(self) -> None:
        """Extract subscription address from valid response."""
        ref = _parse_subscription_ref(SUBSCRIBE_RESPONSE)
        assert ref == "http://192.168.1.100:80/onvif/subscriptions?id=42"

    def test_parse_subscription_ref_invalid_xml(self) -> None:
        """Invalid XML returns None."""
        ref = _parse_subscription_ref("not xml")
        assert ref is None

    def test_parse_subscription_ref_missing_address(self) -> None:
        """Response without SubscriptionReference returns None."""
        no_ref_xml = """<?xml version="1.0" encoding="utf-8"?>
        <soap:Envelope xmlns:soap="http://www.w3.org/2003/05/soap-envelope">
          <soap:Body>
            <tev:CreatePullPointSubscriptionResponse xmlns:tev="http://www.onvif.org/ver10/events/wsdl"/>
          </soap:Body>
        </soap:Envelope>"""
        ref = _parse_subscription_ref(no_ref_xml)
        assert ref is None


class TestParsePullMessages:
    """Tests for _parse_pull_messages."""

    def test_parse_motion_message(self) -> None:
        """Parse a PullMessages response with a motion notification."""
        msgs = _parse_pull_messages(PULL_MESSAGES_WITH_MOTION)
        assert len(msgs) == 1
        assert "MotionAlarm" in msgs[0]["topic"]
        assert "State" in msgs[0]["data"]
        assert msgs[0]["data"]["State"] == "true"

    def test_parse_tamper_message(self) -> None:
        """Parse a PullMessages response with a tampering notification."""
        msgs = _parse_pull_messages(PULL_MESSAGES_WITH_TAMPER)
        assert len(msgs) == 1
        assert "Tampering" in msgs[0]["topic"]

    def test_parse_empty_messages(self) -> None:
        """Parse an empty PullMessages response."""
        msgs = _parse_pull_messages(PULL_MESSAGES_EMPTY)
        assert msgs == []

    def test_parse_invalid_xml(self) -> None:
        """Invalid XML returns empty list."""
        msgs = _parse_pull_messages("not xml")
        assert msgs == []


class TestParseEventCapability:
    """Tests for _parse_event_capability."""

    def test_parse_event_xaddr(self) -> None:
        """Extract Events.XAddr from GetCapabilities response."""
        xaddr = _parse_event_capability(CAPABILITIES_WITH_EVENTS)
        assert xaddr == "http://192.168.1.100:80/onvif/event_service"

    def test_parse_event_xaddr_invalid_xml(self) -> None:
        """Invalid XML returns None."""
        xaddr = _parse_event_capability("not xml")
        assert xaddr is None

    def test_parse_event_xaddr_no_events(self) -> None:
        """Response without Events capability returns None."""
        no_events = """<?xml version="1.0" encoding="utf-8"?>
        <soap:Envelope xmlns:soap="http://www.w3.org/2003/05/soap-envelope">
          <soap:Body>
            <tds:GetCapabilitiesResponse xmlns:tds="http://www.onvif.org/ver10/device/wsdl">
              <tds:Capabilities>
                <tt:PTZ xmlns:tt="http://www.onvif.org/ver10/schema">
                  <tt:XAddr>http://192.168.1.100/onvif/ptz_service</tt:XAddr>
                </tt:PTZ>
              </tds:Capabilities>
            </tds:GetCapabilitiesResponse>
          </soap:Body>
        </soap:Envelope>"""
        xaddr = _parse_event_capability(no_events)
        assert xaddr is None


class TestCheckSoapFault:
    """Tests for _check_soap_fault."""

    def test_fault_raises_onvif_error(self) -> None:
        """SOAP fault raises OnvifError with reason."""
        with pytest.raises(OnvifError, match="SOAP fault"):
            _check_soap_fault(FAULT_RESPONSE, "TestOp")

    def test_success_no_raise(self) -> None:
        """Successful response does not raise."""
        _check_soap_fault(SUBSCRIBE_RESPONSE, "TestOp")

    def test_invalid_xml_no_raise(self) -> None:
        """Invalid XML does not raise (assumed success)."""
        _check_soap_fault("not xml", "TestOp")


# ---------------------------------------------------------------------------
# OnvifEventSubscriber integration tests (with httpx MockTransport)
# ---------------------------------------------------------------------------


class TestOnvifEventSubscriber:
    """Tests for OnvifEventSubscriber using MockTransport."""

    async def test_subscribe_success(self) -> None:
        """subscribe() sends CreatePullPointSubscription and returns ref."""
        transport = _make_sequenced_transport(
            [
                (200, CAPABILITIES_WITH_EVENTS),
                (200, SUBSCRIBE_RESPONSE),
            ]
        )
        sub = _make_subscriber(transport=transport)

        ref = await sub.subscribe()
        assert ref == "http://192.168.1.100:80/onvif/subscriptions?id=42"
        assert sub.subscription_ref == ref

    async def test_subscribe_auth_failure(self) -> None:
        """401 response raises OnvifError."""
        transport = _make_sequenced_transport([(401, AUTH_FAILURE_RESPONSE)])
        sub = _make_subscriber(transport=transport)

        with pytest.raises(OnvifError, match="auth failed"):
            await sub.subscribe()

    async def test_subscribe_http_error(self) -> None:
        """500 response raises OnvifError."""
        transport = _make_sequenced_transport([(500, "Internal Server Error")])
        sub = _make_subscriber(transport=transport)

        with pytest.raises(OnvifError, match="HTTP error"):
            await sub.subscribe()

    async def test_subscribe_no_ref_in_response(self) -> None:
        """Response without subscription ref raises OnvifError."""
        no_ref_response = """<?xml version="1.0" encoding="utf-8"?>
        <soap:Envelope xmlns:soap="http://www.w3.org/2003/05/soap-envelope">
          <soap:Body>
            <tev:CreatePullPointSubscriptionResponse xmlns:tev="http://www.onvif.org/ver10/events/wsdl"/>
          </soap:Body>
        </soap:Envelope>"""
        transport = _make_sequenced_transport(
            [
                (200, CAPABILITIES_WITH_EVENTS),
                (200, no_ref_response),
            ]
        )
        sub = _make_subscriber(transport=transport)

        with pytest.raises(OnvifError, match="No subscription reference"):
            await sub.subscribe()

    async def test_subscribe_soap_fault(self) -> None:
        """SOAP fault in subscribe response raises OnvifError."""
        transport = _make_sequenced_transport(
            [
                (200, CAPABILITIES_WITH_EVENTS),
                (200, FAULT_RESPONSE),
            ]
        )
        sub = _make_subscriber(transport=transport)

        with pytest.raises(OnvifError, match="SOAP fault"):
            await sub.subscribe()

    async def test_pull_messages_returns_events(self) -> None:
        """pull_messages() parses notifications into OnvifEvent list."""
        transport = _make_sequenced_transport(
            [
                (200, CAPABILITIES_WITH_EVENTS),
                (200, SUBSCRIBE_RESPONSE),
                (200, PULL_MESSAGES_WITH_MOTION),
            ]
        )
        sub = _make_subscriber(transport=transport)

        await sub.subscribe()
        events = await sub.pull_messages()

        assert len(events) == 1
        assert events[0].event_type == OnvifEventType.MOTION
        assert events[0].camera_name == "test_cam"
        assert "MotionAlarm" in events[0].raw_topic
        assert events[0].raw_message.get("State") == "true"

    async def test_pull_messages_empty(self) -> None:
        """pull_messages() returns empty list when no messages."""
        transport = _make_sequenced_transport(
            [
                (200, CAPABILITIES_WITH_EVENTS),
                (200, SUBSCRIBE_RESPONSE),
                (200, PULL_MESSAGES_EMPTY),
            ]
        )
        sub = _make_subscriber(transport=transport)

        await sub.subscribe()
        events = await sub.pull_messages()
        assert events == []

    async def test_pull_messages_not_subscribed(self) -> None:
        """pull_messages() returns empty list when not subscribed."""
        transport = _make_sequenced_transport([])
        sub = _make_subscriber(transport=transport)

        events = await sub.pull_messages()
        assert events == []

    async def test_unsubscribe(self) -> None:
        """unsubscribe() terminates the subscription."""
        transport = _make_sequenced_transport(
            [
                (200, CAPABILITIES_WITH_EVENTS),
                (200, SUBSCRIBE_RESPONSE),
                (200, UNSUBSCRIBE_RESPONSE),
            ]
        )
        sub = _make_subscriber(transport=transport)

        await sub.subscribe()
        assert sub.subscription_ref is not None

        await sub.unsubscribe()
        assert sub.subscription_ref is None

    async def test_poll_once_fires_callback(self) -> None:
        """_poll_once() fires callback for each received event."""
        callback_events: list[OnvifEvent] = []

        async def on_event(event: OnvifEvent) -> None:
            callback_events.append(event)

        transport = _make_sequenced_transport(
            [
                (200, CAPABILITIES_WITH_EVENTS),
                (200, SUBSCRIBE_RESPONSE),
                (200, PULL_MESSAGES_WITH_MOTION),
            ]
        )
        sub = _make_subscriber(transport=transport, callback=on_event)

        await sub.subscribe()
        await sub._poll_once()

        assert len(callback_events) == 1
        assert callback_events[0].event_type == OnvifEventType.MOTION
        assert sub.last_event_time is not None

    async def test_poll_once_no_callback(self) -> None:
        """_poll_once() with no callback still returns events."""
        transport = _make_sequenced_transport(
            [
                (200, CAPABILITIES_WITH_EVENTS),
                (200, SUBSCRIBE_RESPONSE),
                (200, PULL_MESSAGES_WITH_MOTION),
            ]
        )
        sub = _make_subscriber(transport=transport, callback=None)

        await sub.subscribe()
        events = await sub._poll_once()
        assert len(events) == 1

    async def test_start_creates_subscription(self) -> None:
        """start() creates subscription and starts poll task."""
        transport = _make_sequenced_transport(
            [
                (200, CAPABILITIES_WITH_EVENTS),
                (200, SUBSCRIBE_RESPONSE),
                (200, PULL_MESSAGES_EMPTY),  # first poll
            ]
        )
        sub = _make_subscriber(transport=transport, poll_interval=100.0)

        await sub.start()
        assert sub.is_running is True
        assert sub.subscription_ref is not None

        # Clean up
        await sub.stop()
        assert sub.is_running is False

    async def test_stop_cancels_polling(self) -> None:
        """stop() cancels polling and unsubscribes."""
        transport = _make_sequenced_transport(
            [
                (200, CAPABILITIES_WITH_EVENTS),
                (200, SUBSCRIBE_RESPONSE),
                (200, PULL_MESSAGES_EMPTY),
                (200, UNSUBSCRIBE_RESPONSE),
            ]
        )
        sub = _make_subscriber(transport=transport, poll_interval=100.0)

        await sub.start()
        assert sub.is_running is True

        await sub.stop()
        assert sub.is_running is False
        assert sub.subscription_ref is None

    async def test_graceful_shutdown_on_error(self) -> None:
        """stop() handles unsubscribe errors gracefully."""
        transport = _make_sequenced_transport(
            [
                (200, CAPABILITIES_WITH_EVENTS),
                (200, SUBSCRIBE_RESPONSE),
                (500, "Error"),  # unsubscribe fails
            ]
        )
        sub = _make_subscriber(transport=transport, poll_interval=100.0)

        await sub.start()
        # Should not raise even though unsubscribe fails
        await sub.stop()
        assert sub.is_running is False

    async def test_callback_exception_handled(self) -> None:
        """_poll_once() handles callback exceptions without crashing."""

        async def bad_callback(event: OnvifEvent) -> None:
            raise RuntimeError("callback failed")

        transport = _make_sequenced_transport(
            [
                (200, CAPABILITIES_WITH_EVENTS),
                (200, SUBSCRIBE_RESPONSE),
                (200, PULL_MESSAGES_WITH_MOTION),
            ]
        )
        sub = _make_subscriber(transport=transport, callback=bad_callback)

        await sub.subscribe()
        # Should not raise
        events = await sub._poll_once()
        assert len(events) == 1

    async def test_tamper_event_parsing(self) -> None:
        """PullMessages with tamper topic parses to TAMPER event type."""
        transport = _make_sequenced_transport(
            [
                (200, CAPABILITIES_WITH_EVENTS),
                (200, SUBSCRIBE_RESPONSE),
                (200, PULL_MESSAGES_WITH_TAMPER),
            ]
        )
        sub = _make_subscriber(transport=transport)

        await sub.subscribe()
        events = await sub.pull_messages()
        assert len(events) == 1
        assert events[0].event_type == OnvifEventType.TAMPER


# ---------------------------------------------------------------------------
# Config validation tests
# ---------------------------------------------------------------------------


class TestOnvifEventConfig:
    """Tests for OnvifEventConfig pydantic model."""

    def test_default_values(self) -> None:
        """Default config has type=all, min_interval=30."""
        cfg = OnvifEventConfig()
        assert cfg.type == "all"
        assert cfg.min_interval_seconds == 30.0

    def test_motion_type(self) -> None:
        """Can create config with motion type."""
        cfg = OnvifEventConfig(type="motion")
        assert cfg.type == "motion"

    def test_tamper_type(self) -> None:
        """Can create config with tamper type."""
        cfg = OnvifEventConfig(type="tamper")
        assert cfg.type == "tamper"

    def test_invalid_type(self) -> None:
        """Invalid event type raises ValidationError."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            OnvifEventConfig(type="invalid")

    def test_min_interval_zero_raises(self) -> None:
        """min_interval_seconds=0 raises ValidationError."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            OnvifEventConfig(min_interval_seconds=0)

    def test_min_interval_negative_raises(self) -> None:
        """Negative min_interval_seconds raises ValidationError."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            OnvifEventConfig(min_interval_seconds=-5.0)


class TestOnvifConfigEventsFields:
    """Tests for events fields in OnvifConfig."""

    def test_events_enabled_default_false(self) -> None:
        """events_enabled defaults to False."""
        cfg = OnvifConfig()
        assert cfg.events_enabled is False

    def test_events_poll_interval_default(self) -> None:
        """events_poll_interval_seconds defaults to 10."""
        cfg = OnvifConfig()
        assert cfg.events_poll_interval_seconds == 10

    def test_events_poll_interval_zero_raises(self) -> None:
        """events_poll_interval_seconds=0 raises ValidationError."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            OnvifConfig(events_poll_interval_seconds=0)

    def test_events_poll_interval_negative_raises(self) -> None:
        """Negative events_poll_interval_seconds raises ValidationError."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            OnvifConfig(events_poll_interval_seconds=-1)

    def test_events_enabled_true(self) -> None:
        """Can enable events in config."""
        cfg = OnvifConfig(events_enabled=True)
        assert cfg.events_enabled is True


# ---------------------------------------------------------------------------
# Subscriber registry tests
# ---------------------------------------------------------------------------


class TestSubscriberRegistry:
    """Tests for the module-level subscriber registry."""

    def setup_method(self) -> None:
        """Clear the registry before each test."""
        # Remove any leftovers from prior tests
        for name in list(get_active_subscribers().keys()):
            unregister_subscriber(name)

    def test_register_and_lookup(self) -> None:
        """Registered subscriber is found by camera name."""
        client = OnvifClient(device_xaddr="http://192.168.1.100/onvif/device_service")
        sub = OnvifEventSubscriber(client=client, camera_name="cam1")
        register_subscriber("cam1", sub)

        active = get_active_subscribers()
        assert "cam1" in active
        assert active["cam1"] is sub

    def test_unregister_removes(self) -> None:
        """Unregistered subscriber is removed from registry."""
        client = OnvifClient(device_xaddr="http://192.168.1.100/onvif/device_service")
        sub = OnvifEventSubscriber(client=client, camera_name="cam1")
        register_subscriber("cam1", sub)
        unregister_subscriber("cam1")

        assert "cam1" not in get_active_subscribers()

    def test_unregister_missing_no_error(self) -> None:
        """Unregistering a non-existent name does not raise."""
        unregister_subscriber("nonexistent")

    def test_get_subscription_states(self) -> None:
        """get_subscription_states returns state dicts."""
        client = OnvifClient(device_xaddr="http://192.168.1.100/onvif/device_service")
        sub = OnvifEventSubscriber(client=client, camera_name="cam1")
        register_subscriber("cam1", sub)

        states = get_subscription_states()
        assert len(states) >= 1
        found = [s for s in states if s["camera_name"] == "cam1"]
        assert len(found) == 1
        assert found[0]["is_running"] is False
        assert found[0]["subscription_ref"] is None
        assert found[0]["last_event_time"] is None

    def teardown_method(self) -> None:
        """Clear the registry after each test."""
        for name in list(get_active_subscribers().keys()):
            unregister_subscriber(name)


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Tests for error handling in OnvifEventSubscriber."""

    async def test_timeout_raises_onvif_error(self) -> None:
        """HTTP timeout raises OnvifError."""

        def timeout_handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("Connection timed out")

        transport = httpx.MockTransport(timeout_handler)
        sub = _make_subscriber(transport=transport)

        with pytest.raises(OnvifError, match="timed out"):
            await sub.subscribe()

    async def test_request_error_raises_onvif_error(self) -> None:
        """Network error raises OnvifError."""

        def error_handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("Connection refused")

        transport = httpx.MockTransport(error_handler)
        sub = _make_subscriber(transport=transport)

        with pytest.raises(OnvifError, match="request failed"):
            await sub.subscribe()

    async def test_malformed_xml_in_pull_messages(self) -> None:
        """Malformed XML in PullMessages returns empty list gracefully."""
        transport = _make_sequenced_transport(
            [
                (200, CAPABILITIES_WITH_EVENTS),
                (200, SUBSCRIBE_RESPONSE),
                (200, "not valid xml at all"),
            ]
        )
        sub = _make_subscriber(transport=transport)

        await sub.subscribe()
        events = await sub.pull_messages()
        assert events == []


# ---------------------------------------------------------------------------
# Poll loop tests
# ---------------------------------------------------------------------------


class TestPollLoop:
    """Tests for the polling loop behavior."""

    async def test_poll_loop_retries_on_onvif_error(self) -> None:
        """Poll loop continues after OnvifError."""
        call_count = [0]

        async def mock_poll_once(self: OnvifEventSubscriber) -> list[OnvifEvent]:
            call_count[0] += 1
            if call_count[0] == 1:
                raise OnvifError("temporary error")
            # After the error, cancel ourselves to stop the loop
            self._running = False
            return []

        transport = _make_sequenced_transport(
            [
                (200, CAPABILITIES_WITH_EVENTS),
                (200, SUBSCRIBE_RESPONSE),
            ]
        )
        sub = _make_subscriber(transport=transport, poll_interval=0.01)

        await sub.subscribe()
        sub._running = True

        with patch.object(OnvifEventSubscriber, "_poll_once", mock_poll_once):
            await sub._poll_loop()

        # Should have been called twice (once error, once success)
        assert call_count[0] == 2
