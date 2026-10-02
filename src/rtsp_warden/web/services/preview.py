"""Same-origin MJPEG and snapshot helpers backed by the in-process FrameHub."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

BOUNDARY = "frame"
MJPEG_CONTENT_TYPE = f"multipart/x-mixed-replace; boundary={BOUNDARY}"


def mjpeg_frames(hub: Any, stop_after: int | None = None) -> Iterator[bytes]:
    """Yield multipart MJPEG parts as new frames arrive on *hub*.

    stop_after bounds the number of frames for tests; None streams forever.
    """
    last_id = 0
    sent = 0
    while stop_after is None or sent < stop_after:
        jpeg, fid, _ts = hub.wait_for_new(last_id, timeout=2.0)
        if fid == last_id or not jpeg:
            continue
        last_id = fid
        sent += 1
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
