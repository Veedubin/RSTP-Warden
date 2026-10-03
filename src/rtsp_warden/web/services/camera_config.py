"""Camera config services for the add/edit/delete camera routes.

Everything here works on the RAW YAML (``${VAR}`` references unexpanded), never on the
loaded ``AppConfig``, whose strings hold expanded secrets. Credentials typed into the
add-camera form are written to config.yaml only as ``${CAM_<SLUG>_USER}`` /
``${CAM_<SLUG>_PASS}`` references; the percent-encoded values go to the ``.env`` next to
config.yaml (see ``web/env_file.py``).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import SplitResult, quote, unquote, urlsplit, urlunsplit

import yaml
from pydantic import ValidationError

from ... import ports
from ...config import _ENV_REF, AppConfig, CameraConfig, expand_env
from ..config_lock import CONFIG_RMW_LOCK, _locked_write_yaml

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$")
_RESERVED_NAMES = frozenset({"new"})
_DEFAULT_RTSP_PORT = 554


def env_slug(name: str) -> str:
    """Upper-case ``name`` and turn ``-`` into ``_`` (``front-door`` -> ``FRONT_DOOR``)."""
    return name.upper().replace("-", "_")


def env_var_names(slug: str) -> tuple[str, str]:
    """Return the ``(user, password)`` variable names for a camera slug."""
    return f"CAM_{slug}_USER", f"CAM_{slug}_PASS"


def credential_env_values(slug: str, username: str, password: str) -> dict[str, str]:
    """Return the .env values for a camera: percent-encoded, ready to sit in a URL."""
    user_var, pass_var = env_var_names(slug)
    return {user_var: quote(username, safe=""), pass_var: quote(password, safe="")}


def validate_name(name: str, existing: Iterable[str]) -> str:
    """Return the stripped camera name, or raise ValueError with a message for the form.

    Rules: ``NAME_RE``; not ``new`` (it would shadow ``/cameras/new``); not equal to an
    existing name ignoring case; and no two cameras may share an ``env_slug`` (their
    credential variables would collide, e.g. ``front-door`` and ``front_door``).
    """
    cleaned = name.strip()
    if not cleaned:
        raise ValueError("Enter a camera name.")
    if not NAME_RE.match(cleaned):
        raise ValueError(
            "Camera names are 1-32 characters: letters, digits, '-' or '_', "
            "starting with a letter or digit."
        )
    if cleaned.lower() in _RESERVED_NAMES:
        raise ValueError(f"{cleaned!r} is reserved; choose another camera name.")
    slug = env_slug(cleaned)
    for other in existing:
        other_clean = other.strip()
        if other_clean.lower() == cleaned.lower():
            raise ValueError(f"A camera named {other_clean!r} already exists.")
        if env_slug(other_clean) == slug:
            raise ValueError(
                f"{cleaned!r} would share the credential variables CAM_{slug}_USER and "
                f"CAM_{slug}_PASS with camera {other_clean!r}; choose another name."
            )
    return cleaned


def allocate_proxy_port(cfg: AppConfig, *, host: str = "0.0.0.0", start: int = 9001) -> int:
    """Return the lowest port >= ``start`` that no camera uses and that is free to bind.

    Ports listed in any camera's ``proxy.port`` are skipped without probing (a stopped
    camera still owns its port); the others are checked with ``ports.port_is_free``.
    """
    used = {int(cam.proxy.port) for cam in cfg.cameras}
    for port in range(start, 65536):
        if port in used:
            continue
        if ports.port_is_free(host, port):
            return port
    raise ValueError(f"No free proxy port at or above {start}.")


def _clean_host(host: str) -> str:
    """Return ``host`` ready for a URL netloc (IPv6 literals get brackets)."""
    cleaned = host.strip()
    if not cleaned or any(ch in cleaned for ch in "/@?#% \t"):
        raise ValueError("Enter the camera's host name or IP address (no scheme, path or user).")
    if ":" in cleaned and not cleaned.startswith("["):
        return f"[{cleaned}]"
    return cleaned


def build_rtsp_url(host: str, port: int, path: str, slug: str) -> str:
    """Return ``rtsp://${CAM_<slug>_USER}:${CAM_<slug>_PASS}@host:port/path``.

    ``path`` may be given with or without its leading ``/``.
    """
    user_var, pass_var = env_var_names(slug)
    clean_path = "/" + path.strip().lstrip("/")
    return f"rtsp://${{{user_var}}}:${{{pass_var}}}@{_clean_host(host)}:{int(port)}{clean_path}"


def _split_rtsp_url(url: str) -> SplitResult:
    """Parse an rtsp(s) URL; error messages never repeat the URL (it may hold a password)."""
    parts = urlsplit(url.strip())
    if parts.scheme not in ("rtsp", "rtsps"):
        raise ValueError("RTSP URLs must start with rtsp:// or rtsps://.")
    try:
        _ = parts.port  # ValueError for a password with a raw '/', '#' or '?' in it
    except ValueError:
        raise ValueError(
            "The RTSP URL could not be parsed. Leave the user name and password out of the "
            "URL and type them in the User name and Password fields."
        ) from None
    if not parts.hostname:
        raise ValueError("The RTSP URL has no host.")
    return parts


def _literal_credentials(parts: SplitResult) -> tuple[str, str] | None:
    """Return the decoded ``(user, password)`` typed into a URL, or None.

    None when the URL has no userinfo or its userinfo uses ``${VAR}`` references.
    """
    userinfo, sep, _hostport = parts.netloc.rpartition("@")
    if not sep or not userinfo or _ENV_REF.search(userinfo):
        return None
    user, _, password = userinfo.partition(":")
    return unquote(user), unquote(password)


def _rewrite_userinfo(url: str, slug: str | None) -> str:
    """Drop literal userinfo from ``url``; with a slug, put the camera's env refs there.

    A URL whose userinfo already uses ``${VAR}`` references is returned unchanged.
    """
    parts = _split_rtsp_url(url)
    userinfo, _, hostport = parts.netloc.rpartition("@")
    if userinfo and _ENV_REF.search(userinfo):
        return url.strip()
    netloc = hostport
    if slug is not None:
        user_var, pass_var = env_var_names(slug)
        netloc = f"${{{user_var}}}:${{{pass_var}}}@{hostport}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def url_with_env_credentials(url: str, slug: str) -> str:
    """Return ``url`` with its userinfo replaced by ``${CAM_<slug>_USER}:${CAM_<slug>_PASS}``.

    Use it for URLs that carry no credentials (ONVIF ``GetStreamUri`` results, typed
    URLs). A URL that already uses ``${VAR}`` references is returned unchanged.
    """
    return _rewrite_userinfo(url, slug)


@dataclass(slots=True)
class NewCameraInput:
    """The add-camera form, already stripped; ``name`` already passed ``validate_name``."""

    name: str
    host: str
    username: str
    password: str
    main_url: str | None = None
    sub_url: str | None = None
    onvif_port: int | None = None
    record_enabled: bool = True


def raw_camera_entry(inp: NewCameraInput, proxy_port: int) -> tuple[dict[str, Any], dict[str, str]]:
    """Build the raw ``cameras[]`` YAML dict and the .env values for a new camera.

    Credentials come from the form; when the user name is blank, credentials typed into
    ``main_url`` (then ``sub_url``) are taken instead and removed from the URL. With
    credentials, every URL gets ``${CAM_<SLUG>_USER}:${CAM_<SLUG>_PASS}@`` and the second
    return value holds the percent-encoded values; without, it is empty. ``main_url``
    defaults to ``build_rtsp_url(host, 554, "", slug)`` (no userinfo without credentials).
    The proxy is MJPEG on ``proxy_port`` reading ``sub`` when a sub URL is given.
    """
    slug = env_slug(inp.name)
    main_given = (inp.main_url or "").strip()
    sub_given = (inp.sub_url or "").strip()

    username, password = inp.username, inp.password
    if not username:
        for given in (main_given, sub_given):
            found = _literal_credentials(_split_rtsp_url(given)) if given else None
            if found is not None:
                username, password = found
                break
    cred_slug = slug if username else None

    if main_given:
        main_url = _rewrite_userinfo(main_given, cred_slug)
    elif cred_slug is not None:
        main_url = build_rtsp_url(inp.host, _DEFAULT_RTSP_PORT, "", slug)
    else:
        main_url = f"rtsp://{_clean_host(inp.host)}:{_DEFAULT_RTSP_PORT}/"

    entry: dict[str, Any] = {"name": inp.name, "main_url": main_url}
    if sub_given:
        entry["sub_url"] = _rewrite_userinfo(sub_given, cred_slug)
    if inp.onvif_port is not None:
        entry["onvif_port"] = int(inp.onvif_port)
    entry["record"] = {"enabled": bool(inp.record_enabled)}
    entry["proxy"] = {
        "enabled": True,
        "mode": "mjpeg",
        "stream": "sub" if sub_given else "main",
        "port": int(proxy_port),
    }
    env_values = credential_env_values(slug, username, password) if cred_slug else {}
    return entry, env_values


# --- raw config.yaml edits ------------------------------------------------------------

# _locked_write_yaml's flock covers only the write. Every read-modify-write of config.yaml
# in this process goes through update_raw_config, which holds this lock from the read to
# the write, so two writers (add camera, zone save, preset save) cannot lose each other's
# change. Writers outside this module (zones, retention, detectors) must use it too.
_RMW_LOCK = CONFIG_RMW_LOCK
_NESTED_KEYS = frozenset({"record", "proxy"})


def _read_raw(config_path: Path) -> dict[str, Any]:
    """Load config.yaml without expanding ``${VAR}``; ``cameras`` is always a list."""
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{config_path} does not hold a YAML mapping")
    cameras = data.get("cameras")
    if cameras is None:
        data["cameras"] = []
    elif not isinstance(cameras, list):
        raise ValueError(f"{config_path}: 'cameras' is not a list")
    return data


def _entry_name(entry: object) -> str:
    return str(entry.get("name", "")).strip() if isinstance(entry, dict) else ""


def raw_camera_names(config_path: Path) -> list[str]:
    """Names of the cameras in config.yaml as it is on disk (it may differ from cfg).

    Raises yaml.YAMLError for a file that is not valid YAML, ValueError for one that is
    not a config mapping, and OSError when it cannot be read.
    """
    return [name for name in (_entry_name(e) for e in _read_raw(config_path)["cameras"]) if name]


def referenced_env_vars(config_path: Path) -> set[str]:
    """Every ``${VAR}`` name config.yaml mentions (comments too, to stay on the safe side)."""
    return set(_ENV_REF.findall(config_path.read_text(encoding="utf-8")))


def _find_entry(data: dict[str, Any], name: str) -> dict[str, Any]:
    for entry in data["cameras"]:
        if _entry_name(entry) == name:
            return entry
    raise KeyError(name)


def update_raw_config(config_path: Path, mutate: Callable[[dict[str, Any]], None]) -> None:
    """Read config.yaml raw, call ``mutate(data)``, write the result, all under one lock.

    ``mutate`` edits ``data`` in place; ``data["cameras"]`` is always a list and ``${VAR}``
    references are unexpanded. When ``mutate`` raises, nothing is written and the
    exception propagates. File errors (``OSError``) propagate too.
    """
    with _RMW_LOCK:
        data = _read_raw(config_path)
        mutate(data)
        _locked_write_yaml(config_path, data)


def append_camera(config_path: Path, entry: dict[str, Any]) -> None:
    """Append ``entry`` to ``cameras`` in config.yaml (raw YAML, written under the lock).

    Raises ValueError when a camera with the same name (ignoring case and ``-``/``_``,
    i.e. the same ``env_slug``) is already in the file.
    """
    name = str(entry["name"]).strip()

    def _append(data: dict[str, Any]) -> None:
        for existing in data["cameras"]:
            other = _entry_name(existing)
            if other and env_slug(other) == env_slug(name):
                raise ValueError(f"A camera named {other!r} is already in {config_path.name}.")
        data["cameras"].append(dict(entry))

    update_raw_config(config_path, _append)


def patch_camera(
    config_path: Path,
    name: str,
    patch: Mapping[str, object],
    *,
    remove_keys: Iterable[str] = (),
) -> None:
    """Merge ``patch`` into the raw entry of camera ``name`` and drop ``remove_keys``.

    Top-level keys are replaced; for ``record`` and ``proxy`` a mapping is merged one
    level deep, so keys the form does not show (``output_dir``, ``port``) survive. Keys
    not named in ``patch`` keep their raw text, ``${VAR}`` references included.
    Raises KeyError when the camera is not in the file and ValueError on an attempt to
    change or remove ``name``.
    """
    remove = tuple(remove_keys)
    if "name" in remove or ("name" in patch and str(patch["name"]).strip() != name):
        raise ValueError("A camera cannot be renamed; delete it and add it again.")

    def _patch(data: dict[str, Any]) -> None:
        entry = _find_entry(data, name)
        for key, value in patch.items():
            current = entry.get(key)
            if key in _NESTED_KEYS and isinstance(current, dict) and isinstance(value, Mapping):
                current.update(value)
            elif isinstance(value, Mapping):
                entry[key] = dict(value)
            else:
                entry[key] = value
        for key in remove:
            entry.pop(key, None)

    update_raw_config(config_path, _patch)


def remove_camera(config_path: Path, name: str) -> None:
    """Remove camera ``name`` from config.yaml; KeyError when it is not there.

    Removing the last camera leaves ``cameras: []`` (AppConfig requires the key).
    """

    def _remove(data: dict[str, Any]) -> None:
        kept = [entry for entry in data["cameras"] if _entry_name(entry) != name]
        if len(kept) == len(data["cameras"]):
            raise KeyError(name)
        data["cameras"] = kept

    update_raw_config(config_path, _remove)


def _validation_message(exc: ValidationError) -> str:
    """Field paths and messages only: pydantic's str() repeats the input (secrets)."""
    details = []
    for err in exc.errors():
        loc = ".".join(str(part) for part in err.get("loc", ())) or "camera"
        details.append(f"{loc}: {err.get('msg', 'invalid value')}")
    return "Invalid camera settings: " + "; ".join(details)


def validate_entry(entry: Mapping[str, Any], env: Mapping[str, str]) -> CameraConfig:
    """Expand ``${VAR}`` in ``entry`` from ``env`` and validate it as a CameraConfig.

    The SystemExit that ``expand_env`` raises for a missing variable and pydantic's
    ValidationError both become ValueError; neither message contains a value.
    """
    try:
        expanded = expand_env(dict(entry), env)
    except SystemExit as exc:
        raise ValueError(str(exc)) from None
    try:
        return CameraConfig.model_validate(expanded)
    except ValidationError as exc:
        raise ValueError(_validation_message(exc)) from None
