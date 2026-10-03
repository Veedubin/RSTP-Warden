"""Edit helpers for one camera's raw ``config.yaml`` entry (the camera detail page).

The edit form is filled from the RAW YAML entry, so ``${VAR}`` references and
passwords never reach the browser, and a save patches only the keys that changed.
New credentials are stored percent-encoded in the ``.env`` next to ``config.yaml``
as ``CAM_<SLUG>_USER`` / ``CAM_<SLUG>_PASS`` and both URLs then reference them;
nothing writes a plaintext password into ``config.yaml``.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

import yaml

from ...config import CameraConfig
from ...probe import validate_rtsp_url
from .camera_config import credential_env_values, env_slug, env_var_names

# Userinfo typed into a URL field may carry a password only as a ${VAR} reference.
_TYPED_USERINFO_RE = re.compile(r"^[^:]*(?::\$\{[A-Za-z_][A-Za-z0-9_]*\})?$")


class CameraEditError(ValueError):
    """Invalid edit-form input; ``field_name`` is the form field the message belongs to."""

    def __init__(self, field_name: str, message: str) -> None:
        super().__init__(message)
        self.field_name = field_name


@dataclass(slots=True)
class EditCameraInput:
    """The edit form as submitted (``record_enabled`` is the checkbox state)."""

    main_url: str
    sub_url: str
    username: str
    password: str
    onvif_port: str
    record_enabled: bool


@dataclass(slots=True)
class CameraEditPlan:
    """What one save changes in config.yaml and .env, and whether the ingest restarts."""

    patch: dict[str, object] = field(default_factory=dict)
    remove_keys: list[str] = field(default_factory=list)
    env_values: dict[str, str] = field(default_factory=dict)
    restart: bool = False

    @property
    def is_empty(self) -> bool:
        """True when the submitted form matches the saved entry."""
        return not (self.patch or self.remove_keys or self.env_values)


def split_userinfo(url: str) -> tuple[str, str]:
    """Split ``scheme://userinfo@rest`` into ``(userinfo, url without userinfo)``.

    Works on raw YAML text such as ``rtsp://${CAM_X_USER}:${CAM_X_PASS}@h:554/p``.
    Returns ``("", url)`` when the URL has no userinfo.
    """
    sep = url.find("://")
    if sep < 0:
        return "", url
    start = sep + 3
    end = len(url)
    for ch in "/?#":
        i = url.find(ch, start)
        if i != -1:
            end = min(end, i)
    authority = url[start:end]
    at = authority.rfind("@")
    if at < 0:
        return "", url
    return authority[:at], url[:start] + authority[at + 1 :] + url[end:]


def join_userinfo(userinfo: str, url: str) -> str:
    """Insert ``userinfo@`` after the scheme of a URL without userinfo ("" changes nothing)."""
    sep = url.find("://")
    if not userinfo or sep < 0:
        return url
    return f"{url[: sep + 3]}{userinfo}@{url[sep + 3 :]}"


def env_userinfo(name: str) -> str:
    """Return ``${CAM_<SLUG>_USER}:${CAM_<SLUG>_PASS}`` for camera ``name``."""
    user_var, pass_var = env_var_names(env_slug(name))
    return f"${{{user_var}}}:${{{pass_var}}}"


def read_camera_entry(config_path: Path, name: str) -> dict[str, Any]:
    """Return a deep copy of the raw (unexpanded) ``cameras[]`` entry called ``name``.

    Raises OSError when the file cannot be read, ``yaml.YAMLError`` when it does not
    parse, and KeyError when no entry has that name.
    """
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    cameras = data.get("cameras") if isinstance(data, dict) else None
    for entry in cameras if isinstance(cameras, list) else []:
        if isinstance(entry, dict) and str(entry.get("name", "")).strip() == name:
            return copy.deepcopy(entry)
    raise KeyError(name)


def current_credentials(cam: CameraConfig) -> tuple[str, str]:
    """Return the decoded ``(user name, password)`` of the camera's expanded main URL."""
    parts = urlsplit(cam.main_url)
    return unquote(parts.username or ""), unquote(parts.password or "")


def edit_form_values(raw: Mapping[str, Any], cam: CameraConfig) -> dict[str, object]:
    """Initial edit-form values: URLs from the raw entry without userinfo, never a password."""
    main_url = str(raw.get("main_url") or "")
    sub_url = str(raw.get("sub_url") or "")
    onvif_port = raw.get("onvif_port")
    return {
        "name": cam.name,
        "username": current_credentials(cam)[0],
        "main_url": split_userinfo(main_url)[1],
        "sub_url": split_userinfo(sub_url)[1] if sub_url else "",
        "onvif_port": "" if onvif_port is None else str(onvif_port),
        "record_enabled": cam.record.enabled,
        "password_entered": False,
        "proxy_port": cam.proxy.port,
    }


def _clean_url(
    value: str, saved: str, field_name: str, label: str, *, required: bool
) -> tuple[str, str]:
    """Check one submitted URL; return ``(typed userinfo, URL without userinfo)``.

    ``saved`` is what the form showed (the raw URL without its userinfo). A field sent
    back unchanged is kept as it is, even if today's checks would reject it (for
    example a whole-URL ``${VAR}`` reference), so other fields can still be edited.
    """
    text = value.strip()
    if not text:
        if required:
            raise CameraEditError(field_name, f"{label} is required.")
        return "", ""
    if text == saved:
        return "", text
    userinfo, bare = split_userinfo(text)
    if userinfo and not _TYPED_USERINFO_RE.match(userinfo):
        raise CameraEditError(
            field_name,
            f"{label}: put the user name and password in their own fields, not in the URL.",
        )
    if urlsplit(bare).scheme not in ("rtsp", "rtsps"):
        raise CameraEditError(field_name, f"{label} must start with rtsp:// or rtsps://.")
    try:
        validate_rtsp_url(bare)
    except ValueError as exc:
        raise CameraEditError(field_name, f"{label}: {exc}") from None
    return userinfo, bare


def _parse_onvif_port(value: str) -> int | None:
    text = value.strip()
    if not text:
        return None
    if not text.isdigit() or not 1 <= int(text) <= 65535:
        raise CameraEditError("onvif_port", "ONVIF port must be a whole number from 1 to 65535.")
    return int(text)


def plan_camera_edit(
    raw: Mapping[str, Any], cam: CameraConfig, inp: EditCameraInput
) -> CameraEditPlan:
    """Compare the submitted form with the raw entry and return only what changed.

    Rules (decisions R3 and R10): a blank password and a blank or unchanged user name
    keep the current credentials; changed credentials go to ``.env`` and both URLs then
    reference ``${CAM_<SLUG>_USER}:${CAM_<SLUG>_PASS}``; otherwise each URL keeps the
    userinfo text it had (a new sub URL borrows the main URL's); a blank sub URL removes
    ``sub_url``; a blank ONVIF port removes ``onvif_port``. URL, credential and
    recording changes need an ingest restart; an ONVIF port change does not.
    Raises CameraEditError (a ValueError) naming the field when the input is invalid.
    """
    plan = CameraEditPlan()
    raw_main = str(raw.get("main_url") or "")
    raw_sub = str(raw.get("sub_url") or "")
    main_ui, main_saved = split_userinfo(raw_main)
    sub_ui, sub_saved = split_userinfo(raw_sub) if raw_sub else ("", "")

    typed_main_ui, main_bare = _clean_url(
        inp.main_url, main_saved, "main_url", "Main stream URL", required=True
    )
    typed_sub_ui, sub_bare = _clean_url(
        inp.sub_url, sub_saved, "sub_url", "Sub stream URL", required=False
    )
    onvif_port = _parse_onvif_port(inp.onvif_port)

    cur_user, cur_pass = current_credentials(cam)
    new_user = inp.username.strip() or cur_user
    new_pass = inp.password or cur_pass
    if (new_user, new_pass) != (cur_user, cur_pass):
        plan.env_values = credential_env_values(env_slug(cam.name), new_user, new_pass)
        main_ui = sub_ui = env_userinfo(cam.name)
        plan.restart = True
    else:
        main_ui = typed_main_ui or main_ui
        sub_ui = typed_sub_ui or (sub_ui if raw_sub else main_ui)

    new_main = join_userinfo(main_ui, main_bare)
    if new_main != raw_main:
        plan.patch["main_url"] = new_main
        plan.restart = True
    if sub_bare:
        new_sub = join_userinfo(sub_ui, sub_bare)
        if new_sub != raw_sub:
            plan.patch["sub_url"] = new_sub
            plan.restart = True
    elif "sub_url" in raw:
        plan.remove_keys.append("sub_url")
        plan.restart = plan.restart or bool(raw_sub)

    if inp.record_enabled != cam.record.enabled:
        plan.patch["record"] = {"enabled": inp.record_enabled}
        plan.restart = True

    if onvif_port is None:
        if "onvif_port" in raw:
            plan.remove_keys.append("onvif_port")
    elif onvif_port != raw.get("onvif_port"):
        plan.patch["onvif_port"] = onvif_port
    return plan


def apply_plan(raw: Mapping[str, Any], plan: CameraEditPlan) -> dict[str, Any]:
    """Return the raw entry as it will read after the save (``record`` merged one level)."""
    entry = copy.deepcopy(dict(raw))
    for key, value in plan.patch.items():
        current = entry.get(key)
        if isinstance(value, dict) and isinstance(current, dict):
            entry[key] = {**current, **value}
        else:
            entry[key] = value
    for key in plan.remove_keys:
        entry.pop(key, None)
    return entry


def apply_to_running_config(cam: CameraConfig, updated: CameraConfig) -> None:
    """Copy the editable fields of a validated entry onto the shared in-memory camera.

    ``app.state.cfg`` and the runtime hold the same ``CameraConfig`` object and
    ``AppRuntime.request_restart_camera`` rebuilds the camera from it, so the fields
    are assigned in place instead of replacing the object.
    """
    cam.main_url = updated.main_url
    cam.sub_url = updated.sub_url
    cam.record.enabled = updated.record.enabled
    cam.proxy.stream = updated.proxy.stream
    cam.onvif_port = updated.onvif_port
