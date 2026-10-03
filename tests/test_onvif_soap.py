"""Tests for the shared ONVIF SOAP helpers (onvif/soap.py)."""

from __future__ import annotations

import base64
import hashlib
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

import pytest

from rtsp_warden.onvif.discovery import OnvifError
from rtsp_warden.onvif.soap import (
    BASE64_ENCODING_TYPE,
    PASSWORD_DIGEST_TYPE,
    SOAP_NS,
    WSSE_NS,
    WSU_NS,
    OnvifAuthError,
    check_soap_fault,
    find_local,
    local_tag,
    soap_envelope,
    wsse_header,
)

NOON = datetime(2026, 10, 2, 12, 0, 0, tzinfo=timezone.utc)
NONCE = bytes(range(16))


def _token(header_xml: str) -> ET.Element:
    """Parse a wsse_header() result inside a real envelope and return the UsernameToken."""
    root = ET.fromstring(soap_envelope("<tds:GetCapabilities/>", header_xml=header_xml))
    token = root.find(f".//{{{WSSE_NS}}}UsernameToken")
    assert token is not None
    return token


def _fault(code: str, reason: str) -> str:
    return (
        f'<env:Envelope xmlns:env="{SOAP_NS}"><env:Body><env:Fault>'
        "<env:Code><env:Value>env:Sender</env:Value>"
        f"<env:Subcode><env:Value>{code}</env:Value></env:Subcode></env:Code>"
        f'<env:Reason><env:Text xml:lang="en">{reason}</env:Text></env:Reason>'
        "</env:Fault></env:Body></env:Envelope>"
    )


# --- wsse_header ---------------------------------------------------------------------------


def test_wsse_header_known_digest_for_fixed_nonce_and_time() -> None:
    token = _token(wsse_header("admin", "secret", now=NOON, nonce=NONCE))

    assert token.findtext(f"{{{WSSE_NS}}}Username") == "admin"
    password = token.find(f"{{{WSSE_NS}}}Password")
    assert password is not None
    assert password.get("Type") == PASSWORD_DIGEST_TYPE
    # Base64(SHA1(nonce + "2026-10-02T12:00:00Z" + "secret")), computed once and pinned.
    assert password.text == "FkBdXnxtUgWwX6lXsb1tIMQCjHU="
    nonce = token.find(f"{{{WSSE_NS}}}Nonce")
    assert nonce is not None
    assert nonce.get("EncodingType") == BASE64_ENCODING_TYPE
    assert nonce.text == "AAECAwQFBgcICQoLDA0ODw=="
    assert token.findtext(f"{{{WSU_NS}}}Created") == "2026-10-02T12:00:00Z"


def test_wsse_header_digest_matches_the_ws_security_recipe() -> None:
    # Review focus 1: a password with URL-special characters only ever enters the SHA1 input.
    password = "p@ss/w#rd?%"
    header = wsse_header("admin", password, now=NOON, nonce=NONCE)

    expected = base64.b64encode(
        hashlib.sha1(NONCE + b"2026-10-02T12:00:00Z" + password.encode("utf-8")).digest()
    ).decode("ascii")
    assert _token(header).findtext(f"{{{WSSE_NS}}}Password") == expected
    assert password not in header


def test_wsse_header_escapes_the_username() -> None:
    header = wsse_header("a<b&c", "pw", now=NOON, nonce=NONCE)

    assert "<wsse:Username>a&lt;b&amp;c</wsse:Username>" in header
    assert _token(header).findtext(f"{{{WSSE_NS}}}Username") == "a<b&c"


def test_wsse_header_writes_created_in_utc() -> None:
    plus_two = datetime(2026, 10, 2, 14, 0, 0, tzinfo=timezone(timedelta(hours=2)))
    naive = datetime(2026, 10, 2, 12, 0, 0)

    aware = _token(wsse_header("u", "p", now=plus_two, nonce=NONCE))
    assert aware.findtext(f"{{{WSU_NS}}}Created") == "2026-10-02T12:00:00Z"
    taken_as_utc = _token(wsse_header("u", "p", now=naive, nonce=NONCE))
    assert taken_as_utc.findtext(f"{{{WSU_NS}}}Created") == "2026-10-02T12:00:00Z"


def test_wsse_header_uses_a_fresh_nonce_per_call() -> None:
    first = _token(wsse_header("u", "p", now=NOON)).findtext(f"{{{WSSE_NS}}}Nonce")
    second = _token(wsse_header("u", "p", now=NOON)).findtext(f"{{{WSSE_NS}}}Nonce")

    assert first and second
    assert len(base64.b64decode(first)) == 16
    assert first != second


# --- soap_envelope -------------------------------------------------------------------------


def test_soap_envelope_wraps_header_and_body() -> None:
    root = ET.fromstring(soap_envelope("<tds:GetCapabilities/>", header_xml="<tt:Marker/>"))

    assert root.tag == f"{{{SOAP_NS}}}Envelope"
    header, body = list(root)
    assert header.tag == f"{{{SOAP_NS}}}Header"
    assert local_tag(header[0].tag) == "Marker"
    assert body.tag == f"{{{SOAP_NS}}}Body"
    assert body[0].tag == "{http://www.onvif.org/ver10/device/wsdl}GetCapabilities"


def test_soap_envelope_without_header_has_no_header_element() -> None:
    root = ET.fromstring(soap_envelope("<trt:GetProfiles/>"))

    assert [local_tag(child.tag) for child in root] == ["Body"]
    assert root[0][0].tag == "{http://www.onvif.org/ver10/media/wsdl}GetProfiles"


# --- check_soap_fault ----------------------------------------------------------------------


def test_check_soap_fault_ignores_normal_replies_and_non_xml() -> None:
    check_soap_fault(soap_envelope("<tds:GetCapabilitiesResponse/>"))
    check_soap_fault("<html><body>Not Found</body>")  # not XML: not a fault
    check_soap_fault("")


def test_check_soap_fault_raises_reason_text() -> None:
    with pytest.raises(OnvifError) as info:
        check_soap_fault(_fault("ter:ActionNotSupported", "Optional Action Not Implemented"))

    assert not isinstance(info.value, OnvifAuthError)
    assert str(info.value) == "Optional Action Not Implemented"


@pytest.mark.parametrize(
    ("code", "reason"),
    [
        ("ter:NotAuthorized", "Sender not Authorized"),
        ("wsse:FailedAuthentication", "The security token could not be authenticated"),
        (
            "ter:OperationProhibited",
            "The action requested requires authorization and the sender is not authorized",
        ),
    ],
)
def test_check_soap_fault_auth_faults_raise_onvif_auth_error(code: str, reason: str) -> None:
    with pytest.raises(OnvifAuthError) as info:
        check_soap_fault(_fault(code, reason))

    assert str(info.value) == reason


def test_check_soap_fault_reads_soap11_faultstring() -> None:
    soap11 = (
        '<s:Envelope xmlns:s="http://schemas.xmlsoap.org/soap/envelope/"><s:Body><s:Fault>'
        "<faultcode>s:Server</faultcode><faultstring>Internal error</faultstring>"
        "</s:Fault></s:Body></s:Envelope>"
    )
    with pytest.raises(OnvifError, match="^Internal error$"):
        check_soap_fault(soap11)


def test_check_soap_fault_without_reason() -> None:
    bare = f'<env:Envelope xmlns:env="{SOAP_NS}"><env:Body><env:Fault/></env:Body></env:Envelope>'
    with pytest.raises(OnvifError, match=r"no reason given"):
        check_soap_fault(bare)


# --- find_local / local_tag ----------------------------------------------------------------


def test_find_local_matches_by_local_name_in_any_namespace() -> None:
    root = ET.fromstring(
        '<a:Root xmlns:a="urn:a" xmlns:b="urn:b"><b:MediaUri><a:Uri>rtsp://x/1</a:Uri>'
        "</b:MediaUri><Uri>rtsp://x/2</Uri></a:Root>"
    )

    found = find_local(root, "Uri")
    assert found is not None and found.text == "rtsp://x/1"
    assert find_local(root, "Root") is root
    assert find_local(root, "Missing") is None
    assert local_tag("{urn:a}Uri") == "Uri"
    assert local_tag("Uri") == "Uri"
