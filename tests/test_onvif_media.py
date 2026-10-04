"""Tests for ONVIF stream-URI discovery (onvif/media.py).

The fake camera behaves like the owner's port-forwarded Foscam: ONVIF on 888, WS-UsernameToken
auth, and its own LAN address (192.0.2.29) in every XAddr and stream URI.
"""

from __future__ import annotations

import base64
import hashlib
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import httpx
import pytest

from rtsp_warden.onvif.discovery import OnvifError
from rtsp_warden.onvif.media import (
    DEFAULT_PORTS,
    StreamUris,
    discover_stream_uris,
    rewrite_host,
)
from rtsp_warden.onvif.soap import (
    DEVICE_NS,
    MEDIA_NS,
    PASSWORD_DIGEST_TYPE,
    SCHEMA_NS,
    SOAP_NS,
    WSSE_NS,
    WSU_NS,
    OnvifAuthError,
    local_tag,
)

LAN = "192.0.2.29"  # the camera's own address, reported in every XAddr and URI
HOST = "192.0.2.10"  # the address the camera is reached on (port forward)
USER = "admin"
PASSWORD = "p@ss/w#rd?%"  # review focus 1: URL-special characters
LOCAL_NOW = datetime(2026, 10, 2, 12, 0, 0, tzinfo=timezone.utc)
STREAMS = {"prof0": f"rtsp://{LAN}:554/videoMain", "prof1": f"rtsp://{LAN}:554/videoSub"}


def _clock() -> datetime:
    return LOCAL_NOW


# --- canned camera replies -----------------------------------------------------------------


def _reply(body: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<env:Envelope xmlns:env="{SOAP_NS}" xmlns:tds="{DEVICE_NS}" xmlns:trt="{MEDIA_NS}"'
        f' xmlns:tt="{SCHEMA_NS}"><env:Body>{body}</env:Body></env:Envelope>'
    )


def _fault(code: str, reason: str) -> str:
    return _reply(
        "<env:Fault><env:Code><env:Value>env:Sender</env:Value>"
        f"<env:Subcode><env:Value>ter:{code}</env:Value></env:Subcode></env:Code>"
        f"<env:Reason><env:Text>{reason}</env:Text></env:Reason></env:Fault>"
    )


def _time_reply(when: datetime) -> str:
    return _reply(
        "<tds:GetSystemDateAndTimeResponse><tds:SystemDateAndTime><tt:UTCDateTime>"
        f"<tt:Time><tt:Hour>{when.hour}</tt:Hour><tt:Minute>{when.minute}</tt:Minute>"
        f"<tt:Second>{when.second}</tt:Second></tt:Time>"
        f"<tt:Date><tt:Year>{when.year}</tt:Year><tt:Month>{when.month}</tt:Month>"
        f"<tt:Day>{when.day}</tt:Day></tt:Date>"
        "</tt:UTCDateTime></tds:SystemDateAndTime></tds:GetSystemDateAndTimeResponse>"
    )


def _caps_reply(media_xaddr: str | None) -> str:
    media = f"<tt:Media><tt:XAddr>{media_xaddr}</tt:XAddr></tt:Media>" if media_xaddr else ""
    return _reply(
        "<tds:GetCapabilitiesResponse><tds:Capabilities>"
        f"<tt:Device><tt:XAddr>http://{LAN}:888/onvif/device_service</tt:XAddr></tt:Device>"
        f"{media}</tds:Capabilities></tds:GetCapabilitiesResponse>"
    )


def _profiles_reply(tokens: list[str]) -> str:
    profiles = "".join(
        f'<trt:Profiles token="{token}" fixed="true"><tt:Name>{token}</tt:Name></trt:Profiles>'
        for token in tokens
    )
    return _reply(f"<trt:GetProfilesResponse>{profiles}</trt:GetProfilesResponse>")


def _uri_reply(uri: str) -> str:
    return _reply(
        "<trt:GetStreamUriResponse><trt:MediaUri>"
        f"<tt:Uri>{uri}</tt:Uri><tt:InvalidAfterConnect>false</tt:InvalidAfterConnect>"
        "</trt:MediaUri></trt:GetStreamUriResponse>"
    )


# --- request inspection --------------------------------------------------------------------


def _port(request: httpx.Request) -> int:
    return request.url.port or 80  # httpx drops the default port from the URL


def _operation(request: httpx.Request) -> str:
    body = ET.fromstring(request.content).find(f"{{{SOAP_NS}}}Body")
    assert body is not None
    return local_tag(body[0].tag)


def _username_token(request: httpx.Request) -> ET.Element | None:
    return ET.fromstring(request.content).find(f".//{{{WSSE_NS}}}UsernameToken")


def _created(request: httpx.Request) -> str | None:
    token = _username_token(request)
    return None if token is None else token.findtext(f"{{{WSU_NS}}}Created")


def _digest_ok(request: httpx.Request, camera_time: datetime) -> bool:
    """What a camera checks: user, PasswordDigest recipe, and Created within 5 s of its clock."""
    token = _username_token(request)
    if token is None:
        return False
    password = token.find(f"{{{WSSE_NS}}}Password")
    if password is None or password.get("Type") != PASSWORD_DIGEST_TYPE:
        return False
    nonce = base64.b64decode(token.findtext(f"{{{WSSE_NS}}}Nonce") or "")
    created = token.findtext(f"{{{WSU_NS}}}Created") or ""
    expected = base64.b64encode(
        hashlib.sha1(nonce + created.encode("utf-8") + PASSWORD.encode("utf-8")).digest()
    ).decode("ascii")
    when = datetime.strptime(created, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    return (
        token.findtext(f"{{{WSSE_NS}}}Username") == USER
        and password.text == expected
        and abs((when - camera_time).total_seconds()) <= 5
    )


class FakeCamera:
    """httpx.MockTransport handler for a port-forwarded ONVIF camera.

    ``closed`` is what every port other than ``onvif_port`` does ("refuse", "timeout" or
    "http404"); ``other_ports`` overrides it per port. ``auth`` is "wsse" (checks the
    UsernameToken), "digest" (HTTP Digest only), "reject", "http401" or "none".
    """

    def __init__(
        self,
        *,
        onvif_port: int | None = 888,
        closed: str = "refuse",
        other_ports: dict[int, str] | None = None,
        auth: str = "wsse",
        camera_time: datetime = LOCAL_NOW,
        time_needs_auth: bool = False,
        media_xaddr: str | None = f"http://{LAN}:8999/onvif/media_service",
        streams: dict[str, str] | None = None,
        fail_tokens: frozenset[str] = frozenset(),
    ) -> None:
        self.onvif_port = onvif_port
        self.closed = closed
        self.other_ports = other_ports or {}
        self.auth = auth
        self.camera_time = camera_time
        self.time_needs_auth = time_needs_auth
        self.media_xaddr = media_xaddr
        self.streams = dict(STREAMS) if streams is None else streams
        self.fail_tokens = fail_tokens
        self.requests: list[httpx.Request] = []

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)

    def ports(self) -> list[int]:
        return [_port(request) for request in self.requests]

    def signed(self) -> list[httpx.Request]:
        """Requests that reached the ONVIF port, except the unauthenticated clock read."""
        return [
            request
            for request in self.requests
            if _port(request) == self.onvif_port and _operation(request) != "GetSystemDateAndTime"
        ]

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        port = _port(request)
        if port != self.onvif_port:
            behaviour = self.other_ports.get(port, self.closed)
            if behaviour == "http404":
                return httpx.Response(404, text="<html><body>Not Found</body></html>")
            if behaviour == "timeout":
                raise httpx.ConnectTimeout("timed out", request=request)
            raise httpx.ConnectError("connection refused", request=request)
        assert request.headers["content-type"].startswith("application/soap+xml")
        operation = _operation(request)
        if operation == "GetSystemDateAndTime":
            if self.time_needs_auth:
                return httpx.Response(400, text=_fault("NotAuthorized", "Sender not Authorized"))
            return httpx.Response(200, text=_time_reply(self.camera_time))
        denied = self._deny(request)
        if denied is not None:
            return denied
        if operation == "GetCapabilities":
            return httpx.Response(200, text=_caps_reply(self.media_xaddr))
        if operation == "GetProfiles":
            return httpx.Response(200, text=_profiles_reply(list(self.streams)))
        if operation == "GetStreamUri":
            body = request.content.decode("utf-8")
            assert "<tt:Stream>RTP-Unicast</tt:Stream>" in body
            assert "<tt:Protocol>RTSP</tt:Protocol>" in body
            token = ET.fromstring(request.content).findtext(f".//{{{MEDIA_NS}}}ProfileToken")
            if token in self.fail_tokens or token not in self.streams:
                return httpx.Response(500, text=_fault("NoProfile", "Profile has no stream"))
            return httpx.Response(200, text=_uri_reply(self.streams[token]))
        return httpx.Response(400, text=_fault("ActionNotSupported", "Unknown action"))

    def _deny(self, request: httpx.Request) -> httpx.Response | None:
        if self.auth == "none":
            return None
        if self.auth == "http401":
            return httpx.Response(401, text="Unauthorized")
        if self.auth == "digest":
            if request.headers.get("authorization", "").startswith(f'Digest username="{USER}"'):
                return None
            challenge = 'Digest realm="cam", nonce="n1", qop="auth"'
            return httpx.Response(401, headers={"WWW-Authenticate": challenge})
        if self.auth == "wsse" and _digest_ok(request, self.camera_time):
            return None
        return httpx.Response(400, text=_fault("NotAuthorized", "Sender not Authorized"))


async def _discover(camera: FakeCamera, host: str = HOST, **kwargs: object) -> StreamUris:
    return await discover_stream_uris(
        host, USER, PASSWORD, transport=camera.transport, clock=_clock, **kwargs
    )


# --- rewrite_host --------------------------------------------------------------------------


def test_rewrite_host_keeps_scheme_port_path_query_and_drops_userinfo() -> None:
    uri = f"rtsp://admin:secret@{LAN}:554/videoMain?channel=1"

    assert rewrite_host(uri, HOST) == f"rtsp://{HOST}:554/videoMain?channel=1"


def test_rewrite_host_without_a_port_adds_none() -> None:
    assert rewrite_host(f"rtsp://{LAN}/live/ch0", HOST) == f"rtsp://{HOST}/live/ch0"


@pytest.mark.parametrize("host", ["2001:db8::5", "[2001:db8::5]", " 2001:db8::5 "])
def test_rewrite_host_brackets_ipv6_literals(host: str) -> None:
    assert rewrite_host(f"rtsp://{LAN}:554/videoMain", host) == "rtsp://[2001:db8::5]:554/videoMain"
    assert rewrite_host("rtsp://[fe80::1]:554/x", "192.0.2.10") == "rtsp://192.0.2.10:554/x"


@pytest.mark.parametrize(
    "host", ["", "   ", "cam/one", "u:p@cam", "192.0.2.10:888", "cam one", "[::1", "fe80::1%eth0"]
)
def test_rewrite_host_rejects_anything_but_a_bare_host(host: str) -> None:
    with pytest.raises(ValueError) as info:
        rewrite_host(f"rtsp://{LAN}:554/videoMain", host)

    assert "p@cam" not in str(info.value)


def test_rewrite_host_rejects_a_uri_without_host() -> None:
    with pytest.raises(ValueError, match="no scheme or host"):
        rewrite_host("videoMain", HOST)


# --- discover_stream_uris ------------------------------------------------------------------


def test_default_ports_are_the_common_onvif_ports() -> None:
    assert DEFAULT_PORTS == (80, 8080, 888, 2020)


async def test_skips_dead_and_non_onvif_ports_and_rewrites_lan_addresses() -> None:
    camera = FakeCamera(other_ports={80: "http404", 8080: "refuse"})

    found = await _discover(camera)

    assert found == StreamUris(
        host=HOST,
        port=888,
        main=f"rtsp://{HOST}:554/videoMain",
        sub=f"rtsp://{HOST}:554/videoSub",
        profiles=["prof0", "prof1"],
    )
    assert camera.ports()[:2] == [80, 8080]
    assert 2020 not in camera.ports()
    urls = {_operation(r): str(r.url) for r in camera.requests if _port(r) == 888}
    assert urls["GetSystemDateAndTime"] == f"http://{HOST}:888/onvif/device_service"
    assert urls["GetCapabilities"] == f"http://{HOST}:888/onvif/device_service"
    # The Media XAddr said LAN:8999; its path is kept, host and port are the ones that answered.
    assert urls["GetProfiles"] == f"http://{HOST}:888/onvif/media_service"
    assert urls["GetStreamUri"] == f"http://{HOST}:888/onvif/media_service"


async def test_signed_requests_carry_a_valid_digest_and_never_the_password() -> None:
    # Review focus 1: the password with / # ? @ % never leaves the process in clear text.
    camera = FakeCamera()

    await _discover(camera)

    signed = camera.signed()
    operations = [_operation(r) for r in signed]
    assert operations == ["GetCapabilities", "GetProfiles", "GetStreamUri", "GetStreamUri"]
    assert all(_digest_ok(r, LOCAL_NOW) for r in signed)
    nonces = set()
    for request in signed:
        token = _username_token(request)
        assert token is not None
        nonces.add(token.findtext(f"{{{WSSE_NS}}}Nonce"))
    assert len(nonces) == 4  # a fresh nonce for every request
    clock_reads = [r for r in camera.requests if _port(r) == 888 and r not in signed]
    assert len(clock_reads) == 1
    assert _username_token(clock_reads[0]) is None
    for request in camera.requests:
        assert PASSWORD not in request.content.decode("utf-8")
        assert PASSWORD not in str(request.url)
        assert "authorization" not in request.headers


async def test_created_follows_the_camera_clock() -> None:
    camera_time = datetime(2030, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    camera = FakeCamera(camera_time=camera_time)

    found = await _discover(camera)

    assert found.port == 888
    assert {_created(r) for r in camera.signed()} == {"2030-01-01T00:00:00Z"}


async def test_camera_that_hides_its_clock_gets_local_time() -> None:
    camera = FakeCamera(time_needs_auth=True)

    found = await _discover(camera)

    assert found.main == f"rtsp://{HOST}:554/videoMain"
    assert {_created(r) for r in camera.signed()} == {"2026-10-02T12:00:00Z"}


async def test_single_profile_leaves_sub_empty() -> None:
    camera = FakeCamera(streams={"prof0": f"rtsp://{LAN}:554/videoMain"})

    found = await _discover(camera)

    assert found.main == f"rtsp://{HOST}:554/videoMain"
    assert found.sub is None
    assert found.profiles == ["prof0"]


async def test_sub_stream_failure_keeps_main() -> None:
    camera = FakeCamera(fail_tokens=frozenset({"prof1"}))

    found = await _discover(camera)

    assert found.main == f"rtsp://{HOST}:554/videoMain"
    assert found.sub is None
    assert found.profiles == ["prof0", "prof1"]


async def test_same_uri_for_both_profiles_leaves_sub_empty() -> None:
    same = f"rtsp://{LAN}:554/live"
    camera = FakeCamera(streams={"prof0": same, "prof1": same})

    found = await _discover(camera)

    assert found.main == f"rtsp://{HOST}:554/live"
    assert found.sub is None


async def test_missing_media_xaddr_falls_back_to_the_device_service() -> None:
    camera = FakeCamera(media_xaddr=None)

    found = await _discover(camera)

    assert found.main == f"rtsp://{HOST}:554/videoMain"
    urls = {_operation(r): str(r.url) for r in camera.signed()}
    assert urls["GetProfiles"] == f"http://{HOST}:888/onvif/device_service"


async def test_auth_fault_stops_without_trying_other_ports() -> None:
    camera = FakeCamera(onvif_port=80, auth="reject")

    with pytest.raises(OnvifAuthError) as info:
        await _discover(camera)

    assert str(info.value).startswith("authentication failed on port 80")
    assert set(camera.ports()) == {80}
    assert PASSWORD not in str(info.value)


async def test_http_401_without_a_digest_challenge_is_an_auth_failure() -> None:
    camera = FakeCamera(auth="http401")

    with pytest.raises(OnvifAuthError, match=r"^authentication failed on port 888"):
        await _discover(camera)

    assert 2020 not in camera.ports()


async def test_http_digest_firmware_gets_digest_credentials() -> None:
    camera = FakeCamera(auth="digest")

    found = await _discover(camera)

    assert found.main == f"rtsp://{HOST}:554/videoMain"
    challenged = [r for r in camera.requests if "authorization" in r.headers]
    assert challenged
    assert all(r.headers["authorization"].startswith('Digest username="admin"') for r in challenged)


async def test_dead_host_costs_one_attempt_per_port_at_the_timeout() -> None:
    # Review focus 4: four ports x 3 s is the ceiling, and the error names every port tried.
    camera = FakeCamera(onvif_port=None, closed="timeout")

    with pytest.raises(OnvifError) as info:
        await _discover(camera)

    assert not isinstance(info.value, OnvifAuthError)
    assert camera.ports() == [80, 8080, 888, 2020]
    for request in camera.requests:
        assert request.extensions["timeout"] == {
            "connect": 3.0,
            "read": 3.0,
            "write": 3.0,
            "pool": 3.0,
        }
    assert str(info.value) == (
        "no ONVIF media service found on 192.0.2.10 (tried ports 80, 8080, 888, 2020: "
        "80 timed out; 8080 timed out; 888 timed out; 2020 timed out)"
    )


async def test_no_profiles_moves_on_and_reports_every_port() -> None:
    camera = FakeCamera(streams={})

    with pytest.raises(OnvifError) as info:
        await _discover(camera)

    message = str(info.value)
    assert "tried ports 80, 8080, 888, 2020" in message
    assert "80 connection failed" in message
    assert "888 camera reported no media profiles" in message
    assert "2020 connection failed" in message


async def test_custom_ports_and_timeout_are_honoured() -> None:
    camera = FakeCamera(onvif_port=2020)

    found = await _discover(camera, ports=(2020, 80), timeout_s=1.5)

    assert found.port == 2020
    assert set(camera.ports()) == {2020}
    assert camera.requests[0].extensions["timeout"]["connect"] == 1.5


async def test_ipv6_host_literal_survives() -> None:
    camera = FakeCamera(onvif_port=80)

    found = await _discover(camera, host="[2001:db8::5]")

    assert found.host == "2001:db8::5"
    assert found.port == 80
    assert found.main == "rtsp://[2001:db8::5]:554/videoMain"
    assert str(camera.requests[0].url) == "http://[2001:db8::5]/onvif/device_service"


@pytest.mark.parametrize(
    ("host", "ports"), [("u:p@cam", DEFAULT_PORTS), (HOST, ()), (HOST, (0,)), (HOST, (65536,))]
)
async def test_bad_host_or_ports_raise_onvif_error_before_any_request(
    host: str, ports: tuple[int, ...]
) -> None:
    camera = FakeCamera()

    with pytest.raises(OnvifError) as info:
        await _discover(camera, host=host, ports=ports)

    assert camera.requests == []
    assert "p@cam" not in str(info.value)


async def test_without_a_username_no_credentials_are_sent() -> None:
    camera = FakeCamera(auth="none")

    found = await discover_stream_uris(HOST, "", "", transport=camera.transport, clock=_clock)

    assert found.main == f"rtsp://{HOST}:554/videoMain"
    assert all(_username_token(r) is None for r in camera.signed())
    assert all("authorization" not in r.headers for r in camera.requests)
