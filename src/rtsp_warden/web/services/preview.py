"""Same-origin MJPEG and snapshot helpers backed by the in-process FrameHub.

Detection boxes (spec 9.3, 9.4): `/cameras/{name}/live-boxes.mjpeg` passes every hub frame
through `box_annotator`, which draws the camera runner's current `LiveBoxes` on it. Boxes are
in tap-frame pixels (`LiveBoxes.frame_w` x `frame_h`) and are scaled to the decoded hub frame,
whose size follows `proxy.scale_width`. A frame passes through untouched, with no decode and
no encode, when there is no runner, no box, or the boxes are older than the age limit.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

if TYPE_CHECKING:
    from ...config import CameraConfig
    from ...detectors.event_builder import LiveBoxes

log = logging.getLogger(__name__)

BOUNDARY = "frame"
MJPEG_CONTENT_TYPE = f"multipart/x-mixed-replace; boundary={BOUNDARY}"

DEFAULT_BOX_MAX_AGE_S = 1.5
BOX_COLOR_BGR = (0, 255, 0)
TEXT_OUTLINE_BGR = (0, 0, 0)
JPEG_QUALITY = 80
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _now() -> float:
    """Wall-clock seconds. A module function so tests can replace it with a fixed clock."""
    return time.time()


def mjpeg_frames(
    hub: Any,
    stop_after: int | None = None,
    *,
    annotate: Callable[[bytes], bytes] | None = None,
) -> Iterator[bytes]:
    """Yield multipart MJPEG parts as new frames arrive on *hub*.

    stop_after bounds the number of frames for tests; None streams forever. annotate, when
    given, maps each JPEG to the bytes that are sent (Content-Length follows the result).
    """
    last_id = 0
    sent = 0
    while stop_after is None or sent < stop_after:
        jpeg, fid, _ts = hub.wait_for_new(last_id, timeout=2.0)
        if fid == last_id or not jpeg:
            continue
        last_id = fid
        sent += 1
        if annotate is not None:
            jpeg = annotate(jpeg)
        header = (
            f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpeg)}\r\n\r\n"
        ).encode("ascii")
        yield header + jpeg + b"\r\n"


def find_hub(runtime: Any, camera_name: str) -> Any | None:
    """Return the FrameHub of the named camera, or None when there is no runtime or hub."""
    for cam_rt in getattr(runtime, "cameras", None) or []:
        if cam_rt.camera.name == camera_name:
            return getattr(cam_rt, "hub", None)
    return None


def find_runner(runtime: Any, camera_name: str) -> Any | None:
    """Return the camera's current DetectorRunner (named ``detector_<camera>``), or None.

    Runners are replaced on every detector rebuild, so callers resolve this per frame.
    """
    wanted = f"detector_{camera_name}"
    for runner in getattr(runtime, "detector_runners", None) or []:
        if getattr(runner, "name", None) == wanted:
            return runner
    return None


def live_boxes_for(runtime: Any, camera_name: str) -> LiveBoxes | None:
    """The camera runner's latest LiveBoxes, or None when there is no runner or no boxes."""
    runner = find_runner(runtime, camera_name)
    get_boxes = getattr(runner, "live_boxes", None)
    if not callable(get_boxes):
        return None
    return get_boxes()


def boxes_max_age(cam: CameraConfig | None) -> float:
    """Seconds after which boxes count as stale and are no longer drawn.

    At least DEFAULT_BOX_MAX_AGE_S, and at least two periods of the slowest enabled
    non-motion detector, so boxes from a 0.5 fps detector do not blink off between runs.
    """
    if cam is None:
        return DEFAULT_BOX_MAX_AGE_S
    rates = [cam.effective_fps(s) for s in cam.detectors if s.enabled and s.type != "motion"]
    if not rates:
        return DEFAULT_BOX_MAX_AGE_S
    return max(DEFAULT_BOX_MAX_AGE_S, 2.0 / min(rates))


def _draw_boxes(img: np.ndarray, boxes: LiveBoxes) -> None:
    """Draw *boxes* on *img* in place, scaled from the tap frame to img's size."""
    h, w = img.shape[:2]
    sx = w / boxes.frame_w
    sy = h / boxes.frame_h
    thickness = max(2, round(w / 480))
    font_scale = max(0.5, w / 1280)
    for box in boxes.boxes:
        x, y, bw, bh = box.bbox
        x1 = min(max(int(round(x * sx)), 0), w - 1)
        y1 = min(max(int(round(y * sy)), 0), h - 1)
        x2 = min(max(int(round((x + bw) * sx)), 0), w - 1)
        y2 = min(max(int(round((y + bh) * sy)), 0), h - 1)
        cv2.rectangle(img, (x1, y1), (x2, y2), BOX_COLOR_BGR, thickness)
        text = f"{box.label} {round(box.confidence * 100)}%"
        (_tw, th), _baseline = cv2.getTextSize(text, _FONT, font_scale, thickness)
        ty = y1 - 4 if y1 - 4 - th >= 0 else min(y1 + th + 4, h - 1)
        cv2.putText(
            img, text, (x1, ty), _FONT, font_scale, TEXT_OUTLINE_BGR, thickness + 2, cv2.LINE_AA
        )
        cv2.putText(img, text, (x1, ty), _FONT, font_scale, BOX_COLOR_BGR, thickness, cv2.LINE_AA)


def annotate_jpeg(
    jpeg: bytes,
    boxes: LiveBoxes | None,
    *,
    max_age_s: float = DEFAULT_BOX_MAX_AGE_S,
    now: float | None = None,
) -> bytes:
    """Return *jpeg* with *boxes* drawn on it, or *jpeg* itself when there is nothing to draw.

    Nothing to draw: no boxes, an empty box tuple, a non-positive tap size, boxes older than
    max_age_s (compared with *now*, default the wall clock), or a JPEG cv2 cannot decode.
    """
    if boxes is None or not boxes.boxes:
        return jpeg
    if boxes.frame_w <= 0 or boxes.frame_h <= 0:
        return jpeg
    current = _now() if now is None else now
    if current - boxes.ts_unix > max_age_s:
        return jpeg
    img = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jpeg
    _draw_boxes(img, boxes)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        return jpeg
    return buf.tobytes()


def box_annotator(
    get_boxes: Callable[[], LiveBoxes | None],
    *,
    max_age_s: float = DEFAULT_BOX_MAX_AGE_S,
) -> Callable[[bytes], bytes]:
    """Build the ``annotate`` callable for mjpeg_frames.

    get_boxes is called once per frame (the runner may have been rebuilt since the last one).
    The callable never raises: on any error it logs at debug level and returns the frame as
    it came, so one bad frame cannot end a viewer's stream.
    """

    def annotate(jpeg: bytes) -> bytes:
        try:
            return annotate_jpeg(jpeg, get_boxes(), max_age_s=max_age_s)
        except Exception:
            log.debug("live box annotation failed; sending the frame unannotated", exc_info=True)
            return jpeg

    return annotate
