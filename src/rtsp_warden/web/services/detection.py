"""Detection settings for the camera page (RW-3 Task 15).

Builds what the Detection panel shows (detector rows merged with the runner's live
status, the label groups of the classes page), patches ``config.yaml`` in place, and
fires the synthetic test event of the rules panel.

Every config write reads the raw YAML (not env-expanded), changes one key, and writes
it back through ``_locked_write_yaml``, so ``${VAR}`` references and keys this code
does not know survive. The read, the change and the write run as one step under an
in-process lock (``update_config_yaml``), so two saves never drop each other's change.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import yaml

from ...actions.base import placeholder_jpeg
from ...actions.rules import RuleDecision, RuleEngine
from ...db import schema
from ...detectors.builtin.onnx import CUDA_PROVIDER
from ...detectors.event_builder import EventBuilder, EventInfo
from ...detectors.model_registry import (
    DEFAULT_MODEL,
    ModelError,
    camera_model_labels,
    load_descriptor,
    load_labels,
)
from ...detectors.registry import LEGACY_DETECTOR_TYPES
from ...status_model import camera_detection_summary
from ..config_lock import _locked_write_yaml

if TYPE_CHECKING:
    from ...config import AppConfig, CameraConfig, DetectorSpec

log = logging.getLogger(__name__)

#: Label groups of the classes page, in display order. Labels in none of them go to "other".
CLASS_CATEGORIES: dict[str, tuple[str, ...]] = {
    "person": ("person",),
    "pet": ("cat", "dog"),
    "vehicle": ("bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"),
    "critter": ("bird", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"),
}
_CATEGORY_OF: dict[str, str] = {
    label: category for category, labels in CLASS_CATEGORIES.items() for label in labels
}

#: Spec fields shown in the detector table's "Settings" column, per detector type.
_SUMMARY_FIELDS: dict[str, tuple[str, ...]] = {
    "motion": ("min_area", "sensitivity"),
    "person": ("min_confidence", "scale_factor", "min_neighbors"),
    "vehicle": ("min_confidence",),
    "dnn": ("min_confidence",),
    "onnx": ("min_confidence",),
    "custom": ("import_path",),
}

TEST_EVENT_LABEL = "person"
TEST_EVENT_CONFIDENCE = 0.99


# --- display -----------------------------------------------------------------------------


def class_groups_for(labels: Sequence[str]) -> list[tuple[str, list[str]]]:
    """Group labels for the classes page.

    Known categories come first in CLASS_CATEGORIES order, then "other" for every
    label in no category (a custom model's labels land there). Labels keep the order
    they are given in; empty groups are dropped.
    """
    groups: dict[str, list[str]] = {category: [] for category in (*CLASS_CATEGORIES, "other")}
    for label in labels:
        groups[_CATEGORY_OF.get(label, "other")].append(label)
    return [(category, items) for category, items in groups.items() if items]


def label_choices(cam: CameraConfig, models_dir: Path) -> list[str]:
    """Labels the classes page offers for this camera.

    The union of the labels of the camera's ``onnx`` models (enabled or not), in
    first-use order. A camera without an ``onnx`` detector, or whose model files
    cannot be read, gets the default model's labels (the 80 COCO names); an empty
    list when even those cannot be read.
    """
    try:
        per_model = camera_model_labels(cam, models_dir)
    except ModelError as exc:
        log.warning("camera %s: cannot read model labels (%s); offering defaults", cam.name, exc)
        per_model = {}
    if not per_model:
        try:
            return load_labels(load_descriptor(DEFAULT_MODEL, models_dir))
        except ModelError as exc:
            log.warning("cannot read the %s labels: %s", DEFAULT_MODEL, exc)
            return []
    seen: set[str] = set()
    out: list[str] = []
    for labels in per_model.values():
        for label in labels:
            if label not in seen:
                seen.add(label)
                out.append(label)
    return out


def detector_summary(spec: DetectorSpec) -> str:
    """``"min_area=500, sensitivity=0.5"`` style summary of a spec, or ``"(defaults)"``."""
    parts = [
        f"{name}={getattr(spec, name)}"
        for name in _SUMMARY_FIELDS.get(spec.type, ())
        if getattr(spec, name, None) is not None
    ]
    return ", ".join(parts) if parts else "(defaults)"


def runtime_detection_status(runtime: Any, camera: str) -> dict[str, Any] | None:
    """The runtime's detection status for one camera, or None.

    Reads ``runtime.detection_status(camera)`` (Task 12). A runtime without that
    method (test fakes, no runtime at all), a call that raises, or a result that is
    not a dict all give None.
    """
    getter = getattr(runtime, "detection_status", None) if runtime is not None else None
    if not callable(getter):
        return None
    try:
        status = getter(camera)
    except Exception:
        log.warning("detection status for camera %s failed", camera, exc_info=True)
        return None
    return status if isinstance(status, dict) else None


def _text(item: Mapping[str, Any] | None, key: str) -> str | None:
    value = item.get(key) if item else None
    return str(value) if value else None


def _count(item: Mapping[str, Any] | None, key: str) -> int:
    value = item.get(key) if item else None
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def detector_rows(cfg: AppConfig, runtime: Any, camera: str) -> list[dict[str, Any]]:
    """One display row per detector spec of the camera, keyed by its config index.

    Config values come from the in-memory ``CameraConfig``; live values (provider,
    fallback warning, load error, frame counters) come from the runner's per-detector
    status, matched on ``"index"`` (the spec's position in ``cameras[].detectors``);
    ``error`` is the model load error (``"error"``) or the setup exception
    (``"setup_error"``).
    A disabled spec, or one the runner does not report, has ``running`` False.
    Every value is a plain str / int / float / bool / None.
    """
    cam = next((c for c in cfg.cameras if c.name == camera), None)
    if cam is None:
        return []
    status = runtime_detection_status(runtime, camera) or {}
    live_by_index: dict[int, Mapping[str, Any]] = {}
    for item in status.get("detectors") or ():
        if isinstance(item, Mapping) and isinstance(item.get("index"), int):
            live_by_index[int(item["index"])] = item

    rows: list[dict[str, Any]] = []
    for index, spec in enumerate(cam.detectors):
        live = live_by_index.get(index) if spec.enabled else None
        provider = _text(live, "provider")
        fps = float(cam.effective_fps(spec))
        rows.append(
            {
                "index": index,
                "type": spec.type,
                "enabled": spec.enabled,
                "deprecated": spec.type in LEGACY_DETECTOR_TYPES,
                "model": (spec.model or DEFAULT_MODEL) if spec.type == "onnx" else None,
                "device": spec.device if spec.type == "onnx" else None,
                "fps": fps,
                "fps_label": f"{fps:g}" if spec.fps is not None else f"{fps:g} (camera)",
                "spec_fps": float(spec.fps) if spec.fps is not None else None,
                "motion_events": (
                    cam.motion_events_enabled(spec) if spec.type == "motion" else None
                ),
                "summary_str": detector_summary(spec),
                "has_roi": spec.roi is not None,
                "has_masks": bool(spec.masks),
                "running": live is not None,
                "provider": provider,
                "provider_label": provider.removesuffix("ExecutionProvider") if provider else None,
                "fallback_warning": _text(live, "fallback_warning"),
                "error": _text(live, "error") or _text(live, "setup_error"),
                "processed": _count(live, "processed"),
                "skipped": _count(live, "skipped"),
                "errors": _count(live, "errors"),
            }
        )
    return rows


# --- config.yaml write-back --------------------------------------------------------------


def write_failed_message(config_path: Path, exc: OSError) -> str:
    """User-facing text for a config.yaml write that failed (read-only mount, permissions)."""
    reason = exc.strerror or str(exc) or type(exc).__name__
    return f"Could not save {config_path}: {reason}. The change is active until the next restart."


# One read-modify-write of config.yaml at a time in this process. _locked_write_yaml
# locks only the write, so two saves that each read the file first (a detector toggle
# and a detection-settings save, say) would both write their own stale copy and lose
# the other's change. RW-2's write-backs can share this helper after the merge.
_CONFIG_RMW_LOCK = threading.Lock()


def update_config_yaml(config_path: Path, mutate: Callable[[dict[str, Any]], bool]) -> bool:
    """Read the raw config.yaml, let *mutate* change it, write it back when it returns True.

    The read, the change and the locked atomic write (``_locked_write_yaml``) run under
    one in-process lock. Returns what *mutate* returned; nothing is written when it is
    False. Raises OSError when the file cannot be read or written.
    """
    with _CONFIG_RMW_LOCK:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not mutate(data):
            return False
        _locked_write_yaml(config_path, data)
        return True


def _persist_camera_field(
    config_path: Path, camera_name: str, field_name: str, value: object
) -> None:
    """Persist one camera-level field for one camera to config.yaml.

    Args:
        config_path: Path to config.yaml.
        camera_name: The camera whose field changed.
        field_name: Field name on CameraConfig to persist.
        value: Value to write for the field.
    """

    def mutate(data: dict[str, Any]) -> bool:
        for cam_dict in data.get("cameras") or []:
            if isinstance(cam_dict, dict) and cam_dict.get("name") == camera_name:
                cam_dict[field_name] = value
                return True
        return False  # not in the file: nothing to write

    update_config_yaml(config_path, mutate)


def _persist_camera_retention(config_path: Path, cfg: AppConfig) -> None:
    """Write every camera's in-memory ``retention`` block back to config.yaml."""

    def mutate(data: dict[str, Any]) -> bool:
        changed = False
        for cam_dict in data.get("cameras") or []:
            if not isinstance(cam_dict, dict) or cam_dict.get("name") is None:
                continue
            for cam_cfg in cfg.cameras:
                if cam_cfg.name == cam_dict["name"]:
                    if cam_cfg.retention is not None:
                        cam_dict["retention"] = cam_cfg.retention.model_dump(exclude_none=True)
                        changed = True
                    elif "retention" in cam_dict:
                        del cam_dict["retention"]
                        changed = True
                    break
        return changed

    update_config_yaml(config_path, mutate)


def _persist_detector_entry(
    config_path: Path,
    camera: str,
    index: int,
    patch: Mapping[str, object],
    *,
    expected_type: str | None = None,
) -> bool:
    """Set ``patch``'s keys on the raw ``detectors[index]`` entry of one camera.

    Only that entry changes: other keys of the entry, other detectors, other cameras,
    unknown keys and ``${VAR}`` text stay as they are in the file. Returns False and
    writes nothing when the camera is not in the file, the index is out of range, the
    entry is not a mapping, or its ``type`` differs from ``expected_type`` (the file
    was edited since it was loaded). Raises OSError when the file cannot be written.
    """

    def mutate(data: dict[str, Any]) -> bool:
        for cam_dict in data.get("cameras") or []:
            if not isinstance(cam_dict, dict) or str(cam_dict.get("name", "")).strip() != camera:
                continue
            detectors = cam_dict.get("detectors")
            if not isinstance(detectors, list) or not 0 <= index < len(detectors):
                return False
            entry = detectors[index]
            if not isinstance(entry, dict):
                return False
            if expected_type is not None and entry.get("type") != expected_type:
                return False
            entry.update(patch)
            return True
        return False

    return update_config_yaml(config_path, mutate)


# --- test event (spec 8.5, ruling R13) ---------------------------------------------------


@dataclass(slots=True)
class FiredTestEvent:
    """Result of ``fire_test_event``: the event row, the rule decision, the queued actions."""

    event_id: int
    thumbnail_path: str | None
    decision: RuleDecision
    queued: list[str] = field(default_factory=list)  # action names handed to the ActionQueue
    note: str = ""


def placeholder_frame() -> np.ndarray:
    """The 320x180 test picture of ``actions.base.placeholder_jpeg`` as a BGR frame."""
    jpeg = placeholder_jpeg("rtsp-warden test event")
    return cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)


def fire_test_event(
    cam: CameraConfig,
    *,
    dispatch: Callable[..., RuleDecision | None] | None,
    now: datetime | None = None,
    db: Any = schema,
) -> FiredTestEvent:
    """Create a synthetic ``person`` event and push it through the camera's rules (R13).

    Inserts a real ``events`` row (``event_type="test"``, ``ended_at`` equal to
    ``created_at``) with a placeholder thumbnail at
    ``<record.output_dir>/<camera>/thumbnails/<id>.jpg``, then calls
    ``dispatch(info, bypass_cooldown=True, allow_clip=False)``, which is
    ``AppRuntime.dispatch_event`` (Task 12): the camera's real ``RuleEngine`` with the
    cooldown bypassed (no stamp recorded) and the matched actions queued once each on
    the real ``ActionQueue``; no clip. Without a runtime (``dispatch`` None, or it
    returns no decision, or it raises) the rules are evaluated here with a fresh
    ``RuleEngine`` so the answer still says what would match, nothing is sent, and
    ``note`` says why.
    """
    started = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    stored = started.replace(tzinfo=None)  # the database keeps naive UTC
    event_id = db.insert_event(
        camera_name=cam.name,
        event_type="test",
        label=TEST_EVENT_LABEL,
        confidence=TEST_EVENT_CONFIDENCE,
        zone="",
        track_id=None,
        message=f"Test event fired from the web UI on {cam.name}",
        created_at=stored,
        metadata={"test": True},
    )
    rel: str | None = EventBuilder.thumbnail_rel_path(cam.name, event_id)
    try:
        builder = EventBuilder(
            camera=cam.name, output_dir=Path(cam.record.output_dir), area_masks=()
        )
        builder.write_thumbnail(rel, placeholder_frame(), None)
    except Exception as exc:
        log.warning("test event %s on %s: thumbnail not written: %s", event_id, cam.name, exc)
        rel = None
    db.close_event(event_id, stored, thumbnail_path=rel)

    info = EventInfo(
        id=event_id,
        camera=cam.name,
        label=TEST_EVENT_LABEL,
        confidence=TEST_EVENT_CONFIDENCE,
        zone="",
        started_at=started,
        ended_at=started,
        thumbnail_path=rel,
        clip_path=None,
        track_id=None,
        event_type="test",
    )
    decision: Any = None
    if dispatch is not None:
        try:
            decision = dispatch(info, bypass_cooldown=True, allow_clip=False)
        except Exception as exc:
            log.warning("test event %s on %s: rules failed: %s", event_id, cam.name, exc)
            decision = None
    if isinstance(decision, RuleDecision):
        queued: list[str] = []
        for match in decision.matched:
            for name in match.actions:
                if name not in queued:
                    queued.append(name)
        return FiredTestEvent(event_id, rel, decision, queued=queued)

    local = RuleEngine(cam.name, cam.rules).evaluate(info, bypass_cooldown=True)
    note = ""
    if any(match.actions for match in local.matched):
        note = "Actions were not sent: the detection runtime is not available."
    return FiredTestEvent(event_id, rel, local, note=note)


# --- status badge (Task 17) --------------------------------------------------------------


def camera_badge(runtime: Any, name: str) -> dict[str, Any] | None:
    """Detection badge for camera *name*, or None when it runs no detection.

    ``text``: ``detection error`` (a detector could not load its model), ``CPU fallback``
    (``device: cuda`` but the session runs on CPU), ``GPU``, ``model loading`` (an ``onnx``
    detector has no session yet) or ``CPU``. ``level`` is ``error`` | ``warn`` | ``ok`` (the
    CSS modifier), ``title`` the warnings (else the provider) for the tooltip, ``dropped`` and
    ``processed`` the runner's frame counters since it last started (a hot reload resets them).
    """
    det = camera_detection_summary(runtime, name)
    if det is None:
        return None
    rows = det["detectors"]
    providers = {d["provider"] for d in rows if d["provider"]}
    if any(d["error"] for d in rows):
        text, level = "detection error", "error"
    elif det["fallback_warning"]:
        text, level = "CPU fallback", "warn"
    elif CUDA_PROVIDER in providers:
        text, level = "GPU", "ok"
    elif not providers and any(d["type"] == "onnx" for d in rows):
        text, level = "model loading", "ok"
    else:
        text, level = "CPU", "ok"
    return {
        "text": text,
        "level": level,
        "title": "; ".join(det["warnings"]) or det["provider"] or "OpenCV on CPU",
        "dropped": det["dropped"],
        "processed": det["processed"],
    }
