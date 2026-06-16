"""Tests for ONVIF discovery and PTZ control."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from rtsp_warden.onvif.discovery import (
    DiscoveredCamera,
    OnvifDiscovery,
    OnvifError,
    _parse_probe_response,
)
from rtsp_warden.onvif.ptz import (
    OnvifPTZ,
    _build_velocity_xml,
    _check_soap_fault,
    _parse_capabilities_ptz,
    _parse_profile_token,
)

# ---------------------------------------------------------------------------
# Discovery tests
# ---------------------------------------------------------------------------


# A realistic WS-Discovery ProbeMatches response
SAMPLE_PROBE_MATCH = """<?xml version="1.0" encoding="utf-8"?>
<soap:Envelope xmlns:soap="http://www.w3.org/2003/05/soap-envelope"
               xmlns:wsd="http://schemas.xmlsoap.org/ws/2005/04/discovery"
               xmlns:dn="http://www.onvif.org/ver10/network/wsdl">
  <soap:Body>
    <wsd:ProbeMatches>
      <wsd:ProbeMatch>
        <wsd:XAddrs>http://192.168.1.100:80/onvif/device_service</wsd:XAddrs>
        <wsd:Types>dn:NetworkVideoTransmitter</wsd:Types>
        <wsd:Scopes>
          onvif://www.onvif.org/Manufacturer/Hikvision
          onvif://www.onvif.org/Model/DS-2CD2142FWD
          onvif://www.onvif.org/name/FrontDoor
        </wsd:Scopes>
      </wsd:ProbeMatch>
    </wsd:ProbeMatches>
  </soap:Body>
</soap:Envelope>"""


def test_discovery_parses_response() -> None:
    """Feed a handcrafted WS-Discovery response, verify DiscoveredCamera extraction."""
    result = _parse_probe_response(SAMPLE_PROBE_MATCH.encode())
    assert result is not None
    assert result.xaddr == "http://192.168.1.100:80/onvif/device_service"
    assert result.address == "192.168.1.100"
    assert "NetworkVideoTransmitter" in " ".join(result.types)
    assert result.name == "FrontDoor"
    assert result.manufacturer == "Hikvision"
    assert result.model == "DS-2CD2142FWD"


def test_discovery_dedupes_by_address() -> None:
    """Two responses from same IP should yield one DiscoveredCamera."""
    cameras: dict[str, DiscoveredCamera] = {}
    for _ in range(2):
        camera = _parse_probe_response(SAMPLE_PROBE_MATCH.encode())
        if camera and camera.address not in cameras:
            cameras[camera.address] = camera

    assert len(cameras) == 1
    assert cameras["192.168.1.100"].xaddr == "http://192.168.1.100:80/onvif/device_service"


def test_discovery_filters_non_nvt_types() -> None:
    """Response with types that do not include NetworkVideoTransmitter is filtered."""
    non_nvt_response = """<?xml version="1.0" encoding="utf-8"?>
<soap:Envelope xmlns:soap="http://www.w3.org/2003/05/soap-envelope"
               xmlns:wsd="http://schemas.xmlsoap.org/ws/2005/04/discovery">
  <soap:Body>
    <wsd:ProbeMatches>
      <wsd:ProbeMatch>
        <wsd:XAddrs>http://192.168.1.200:80/device_service</wsd:XAddrs>
        <wsd:Types>dn:SomeOtherDevice</wsd:Types>
      </wsd:ProbeMatch>
    </wsd:ProbeMatches>
  </soap:Body>
</soap:Envelope>"""
    result = _parse_probe_response(non_nvt_response.encode())
    assert result is None


def test_discovery_handles_no_responses() -> None:
    """Returns empty list on timeout when no devices respond."""
    discovery = OnvifDiscovery(timeout_seconds=0)

    from unittest.mock import patch

    with patch("rtsp_warden.onvif.discovery.socket.socket") as mock_socket_cls:
        mock_sock = mock_socket_cls.return_value
        mock_sock.recvfrom.side_effect = OSError("no data")
        with patch("rtsp_warden.onvif.discovery.select.select", return_value=([], [], [])):
            cameras = discovery.discover()

    assert cameras == []


def test_discovery_extracts_manufacturer_and_model() -> None:
    """Parses manufacturer and model from ONVIF scopes correctly."""
    result = _parse_probe_response(SAMPLE_PROBE_MATCH.encode())
    assert result is not None
    assert result.manufacturer == "Hikvision"
    assert result.model == "DS-2CD2142FWD"
    assert result.name == "FrontDoor"


# ---------------------------------------------------------------------------
# PTZ tests
# ---------------------------------------------------------------------------

CAPABILITIES_RESPONSE = """<?xml version="1.0" encoding="utf-8"?>
<soap:Envelope xmlns:soap="http://www.w3.org/2003/05/soap-envelope"
               xmlns:tds="http://www.onvif.org/ver10/device/wsdl">
  <soap:Body>
    <tds:GetCapabilitiesResponse>
      <tds:Capabilities>
        <tt:PTZ xmlns:tt="http://www.onvif.org/ver10/schema">
          <tt:XAddr>http://192.168.1.100:80/onvif/ptz_service</tt:XAddr>
        </tt:PTZ>
      </tds:Capabilities>
    </tds:GetCapabilitiesResponse>
  </soap:Body>
</soap:Envelope>"""

PROFILES_RESPONSE = """<?xml version="1.0" encoding="utf-8"?>
<soap:Envelope xmlns:soap="http://www.w3.org/2003/05/soap-envelope"
               xmlns:tptz="http://www.onvif.org/ver20/ptz/wsdl">
  <soap:Body>
    <tptz:GetProfilesResponse>
      <tptz:Profiles token="profile1">
        <tptz:Name>MainProfile</tptz:Name>
      </tptz:Profiles>
    </tptz:GetProfilesResponse>
  </soap:Body>
</soap:Envelope>"""

SUCCESS_RESPONSE = """<?xml version="1.0" encoding="utf-8"?>
<soap:Envelope xmlns:soap="http://www.w3.org/2003/05/soap-envelope">
  <soap:Body>
    <tptz:ContinuousMoveResponse xmlns:tptz="http://www.onvif.org/ver20/ptz/wsdl"/>
  </soap:Body>
</soap:Envelope>"""

STOP_SUCCESS_RESPONSE = """<?xml version="1.0" encoding="utf-8"?>
<soap:Envelope xmlns:soap="http://www.w3.org/2003/05/soap-envelope">
  <soap:Body>
    <tptz:StopResponse xmlns:tptz="http://www.onvif.org/ver20/ptz/wsdl"/>
  </soap:Body>
</soap:Envelope>"""

FAULT_RESPONSE = """<?xml version="1.0" encoding="utf-8"?>
<soap:Envelope xmlns:soap="http://www.w3.org/2003/05/soap-envelope">
  <soap:Body>
    <soap:Fault>
      <soap:Code>soap:Sender</soap:Code>
      <soap:Reason>
        <soap:Text xml:lang="en">Invalid profile token</soap:Text>
      </soap:Reason>
    </soap:Fault>
  </soap:Body>
</soap:Envelope>"""


def _make_sequenced_transport(
    responses: list[tuple[int, str]],
) -> httpx.MockTransport:
    """Create a MockTransport that returns responses in sequence."""

    idx = [0]

    def handler(request: httpx.Request) -> httpx.Response:
        if idx[0] >= len(responses):
            return httpx.Response(200, text=SUCCESS_RESPONSE)
        status, body = responses[idx[0]]
        idx[0] += 1
        return httpx.Response(status, text=body)

    return httpx.MockTransport(handler)


def _make_capturing_transport(
    responses: list[tuple[int, str]],
    captured: list[httpx.Request],
) -> httpx.MockTransport:
    """Create a MockTransport that captures requests and returns responses in sequence."""

    idx = [0]

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        if idx[0] >= len(responses):
            return httpx.Response(200, text=SUCCESS_RESPONSE)
        status, body = responses[idx[0]]
        idx[0] += 1
        return httpx.Response(status, text=body)

    return httpx.MockTransport(handler)


def test_ptz_continuous_move_builds_correct_envelope() -> None:
    """Verify continuous_move sends correct pan/tilt/zoom velocity."""
    captured: list[httpx.Request] = []
    transport = _make_capturing_transport(
        [
            (200, CAPABILITIES_RESPONSE),
            (200, PROFILES_RESPONSE),
            (200, SUCCESS_RESPONSE),
        ],
        captured,
    )

    ptz = OnvifPTZ(
        device_xaddr="http://192.168.1.100/onvif/device_service",
        timeout_seconds=5,
        transport=transport,
    )

    asyncio.run(ptz.continuous_move(pan=-1.0, tilt=0.5, zoom=0.0))

    # The third request should be ContinuousMove
    move_req = captured[2]
    body = move_req.content.decode()
    assert "ContinuousMove" in body
    assert "-1.0000" in body  # pan
    assert "0.5000" in body  # tilt
    assert "0.0000" in body  # zoom


def test_ptz_stop_builds_correct_envelope() -> None:
    """Verify Stop request includes PanTilt and Zoom flags."""
    captured: list[httpx.Request] = []
    transport = _make_capturing_transport(
        [
            (200, CAPABILITIES_RESPONSE),
            (200, PROFILES_RESPONSE),
            (200, STOP_SUCCESS_RESPONSE),
        ],
        captured,
    )

    ptz = OnvifPTZ(
        device_xaddr="http://192.168.1.100/onvif/device_service",
        timeout_seconds=5,
        transport=transport,
    )

    asyncio.run(ptz.stop())

    stop_req = captured[2]
    body = stop_req.content.decode()
    assert "Stop" in body
    assert "PanTilt" in body
    assert "Zoom" in body


def test_ptz_handles_auth_failure() -> None:
    """A 401 response should raise OnvifError with auth message."""
    transport = _make_sequenced_transport([(401, "Unauthorized")])

    ptz = OnvifPTZ(
        device_xaddr="http://192.168.1.100/onvif/device_service",
        timeout_seconds=5,
        transport=transport,
    )

    with pytest.raises(OnvifError, match="auth failed|Authentication failed"):
        asyncio.run(ptz._get_ptz_xaddr())


def test_ptz_handles_timeout() -> None:
    """A slow response should raise OnvifError with timeout message."""

    def timeout_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("Connection timed out")

    transport = httpx.MockTransport(timeout_handler)

    ptz = OnvifPTZ(
        device_xaddr="http://192.168.1.100/onvif/device_service",
        timeout_seconds=1,
        transport=transport,
    )

    with pytest.raises(OnvifError, match="timed out"):
        asyncio.run(ptz._get_ptz_xaddr())


def test_ptz_get_capabilities_extracts_ptz_xaddr() -> None:
    """Given a GetCapabilities response, extract the PTZ service URL."""
    result = _parse_capabilities_ptz(CAPABILITIES_RESPONSE)
    assert result == "http://192.168.1.100:80/onvif/ptz_service"


def test_parse_profile_token() -> None:
    """Extract profile token from GetProfiles response."""
    result = _parse_profile_token(PROFILES_RESPONSE)
    assert result == "profile1"


def test_check_soap_fault_raises() -> None:
    """A SOAP Fault response should raise OnvifError."""
    with pytest.raises(OnvifError, match="SOAP fault"):
        _check_soap_fault(FAULT_RESPONSE, "ContinuousMove")


def test_check_soap_fault_success() -> None:
    """A successful SOAP response should not raise."""
    _check_soap_fault(SUCCESS_RESPONSE, "ContinuousMove")  # Should not raise


def test_parse_probe_response_invalid_xml() -> None:
    """Invalid XML should return None gracefully."""
    result = _parse_probe_response(b"not xml at all")
    assert result is None


def test_ptz_continuous_move_clamps_values() -> None:
    """Values outside [-1, 1] should be clamped."""
    captured: list[httpx.Request] = []
    transport = _make_capturing_transport(
        [
            (200, CAPABILITIES_RESPONSE),
            (200, PROFILES_RESPONSE),
            (200, SUCCESS_RESPONSE),
        ],
        captured,
    )

    ptz = OnvifPTZ(
        device_xaddr="http://192.168.1.100/onvif/device_service",
        timeout_seconds=5,
        transport=transport,
    )

    asyncio.run(ptz.continuous_move(pan=5.0, tilt=-3.0, zoom=2.0))

    move_req = captured[2]
    body = move_req.content.decode()
    # Clamped: pan=1.0, tilt=-1.0, zoom=1.0
    assert "1.0000" in body
    assert "-1.0000" in body


def test_build_velocity_xml_values() -> None:
    """Verify _build_velocity_xml generates correct XML fragments."""
    xml = _build_velocity_xml(pan=0.5, tilt=-0.3, zoom=0.0)
    assert 'x="0.5000"' in xml
    assert 'y="-0.3000"' in xml
    assert 'x="0.0000"' in xml  # zoom x
