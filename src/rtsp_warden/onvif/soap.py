"""Shared SOAP 1.2 helpers for ONVIF: envelopes, WS-UsernameToken, faults, XML lookups.

New ONVIF code (``media.py``) builds on these. ``ptz.py`` and ``events.py`` still carry
their own copies and keep HTTP Digest auth; moving them over is deferred (decision R7).
"""

from __future__ import annotations

import base64
import hashlib
import os
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from xml.sax.saxutils import escape

from .discovery import OnvifError

SOAP_NS = "http://www.w3.org/2003/05/soap-envelope"
WSA_NS = "http://schemas.xmlsoap.org/ws/2004/08/addressing"
DEVICE_NS = "http://www.onvif.org/ver10/device/wsdl"
MEDIA_NS = "http://www.onvif.org/ver10/media/wsdl"
PTZ_NS = "http://www.onvif.org/ver20/ptz/wsdl"
SCHEMA_NS = "http://www.onvif.org/ver10/schema"

_WSS = "http://docs.oasis-open.org/wss/2004/01/"
WSSE_NS = _WSS + "oasis-200401-wss-wssecurity-secext-1.0.xsd"
WSU_NS = _WSS + "oasis-200401-wss-wssecurity-utility-1.0.xsd"
PASSWORD_DIGEST_TYPE = _WSS + "oasis-200401-wss-username-token-profile-1.0#PasswordDigest"
BASE64_ENCODING_TYPE = _WSS + "oasis-200401-wss-soap-message-security-1.0#Base64Binary"

# Prefixes declared on every envelope so bodies can use them without their own xmlns.
ENVELOPE_PREFIXES: dict[str, str] = {
    "soap": SOAP_NS,
    "wsa": WSA_NS,
    "tds": DEVICE_NS,
    "trt": MEDIA_NS,
    "tptz": PTZ_NS,
    "tt": SCHEMA_NS,
}

# Fault codes (local part, lower-cased) and reason phrases that mean "bad credentials".
_AUTH_FAULT_CODES = frozenset(
    {"notauthorized", "failedauthentication", "invalidsecurity", "invalidsecuritytoken"}
)
_AUTH_FAULT_PHRASES = ("not authorized", "not authorised", "unauthorized", "authentication")


class OnvifAuthError(OnvifError):
    """The camera rejected the credentials (SOAP NotAuthorized fault or HTTP 401)."""


def local_tag(tag: str) -> str:
    """Return the local part of an ElementTree tag: ``{namespace}Name`` -> ``Name``."""
    return tag.rsplit("}", 1)[-1]


def find_local(elem: ET.Element, local_name: str) -> ET.Element | None:
    """Return the first element in ``elem.iter()`` (elem itself included) with that local name."""
    for child in elem.iter():
        if local_tag(child.tag) == local_name:
            return child
    return None


def soap_envelope(body_xml: str, *, header_xml: str = "") -> str:
    """Wrap ``body_xml`` (and an optional ``header_xml``) in a SOAP 1.2 envelope.

    The envelope declares the ``soap``, ``wsa``, ``tds``, ``trt``, ``tptz`` and ``tt``
    prefixes. No ``<soap:Header>`` is written when ``header_xml`` is empty.
    """
    xmlns = " ".join(f'xmlns:{prefix}="{uri}"' for prefix, uri in ENVELOPE_PREFIXES.items())
    header = f"<soap:Header>{header_xml}</soap:Header>" if header_xml else ""
    return (
        '<?xml version="1.0" encoding="utf-8"?>'
        f"<soap:Envelope {xmlns}>{header}<soap:Body>{body_xml}</soap:Body></soap:Envelope>"
    )


def _created(now: datetime | None) -> str:
    moment = datetime.now(timezone.utc) if now is None else now
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def wsse_header(
    username: str,
    password: str,
    *,
    now: datetime | None = None,
    nonce: bytes | None = None,
) -> str:
    """Build a ``<wsse:Security>`` UsernameToken header with a PasswordDigest.

    ``Digest = Base64(SHA1(nonce + created + password))``. ``now`` defaults to the current
    UTC time (a naive value is taken as UTC) and is written as ``YYYY-MM-DDTHH:MM:SSZ``;
    ``nonce`` defaults to 16 random bytes. The password itself never appears in the result.
    """
    raw_nonce = os.urandom(16) if nonce is None else nonce
    created = _created(now)
    digest = hashlib.sha1(raw_nonce + created.encode("utf-8") + password.encode("utf-8")).digest()
    return (
        f'<wsse:Security xmlns:wsse="{WSSE_NS}" xmlns:wsu="{WSU_NS}">'
        "<wsse:UsernameToken>"
        f"<wsse:Username>{escape(username)}</wsse:Username>"
        f'<wsse:Password Type="{PASSWORD_DIGEST_TYPE}">{_b64(digest)}</wsse:Password>'
        f'<wsse:Nonce EncodingType="{BASE64_ENCODING_TYPE}">{_b64(raw_nonce)}</wsse:Nonce>'
        f"<wsu:Created>{created}</wsu:Created>"
        "</wsse:UsernameToken>"
        "</wsse:Security>"
    )


def check_soap_fault(xml_text: str) -> None:
    """Raise when ``xml_text`` carries a SOAP Fault; return quietly otherwise.

    Call it BEFORE ``raise_for_status``: ONVIF devices answer auth and action errors with
    HTTP 400/500 plus a fault body. Raises ``OnvifAuthError`` for NotAuthorized-style
    faults and ``OnvifError`` for every other fault; the message is the fault's reason
    text. Text that is not XML is not a fault.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return
    fault = find_local(root, "Fault")
    if fault is None:
        return
    reason = ""
    for name in ("Text", "faultstring"):
        elem = find_local(fault, name)
        if elem is not None and elem.text and elem.text.strip():
            reason = elem.text.strip()
            break
    codes = {
        (elem.text or "").strip().rsplit(":", 1)[-1].lower()
        for elem in fault.iter()
        if local_tag(elem.tag) in ("Value", "faultcode")
    }
    message = reason or "SOAP fault (no reason given)"
    lowered = reason.lower()
    if codes & _AUTH_FAULT_CODES or any(phrase in lowered for phrase in _AUTH_FAULT_PHRASES):
        raise OnvifAuthError(message)
    raise OnvifError(message)
