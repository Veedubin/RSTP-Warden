"""Live preview with detection boxes (spec 9.3, 9.4; RW-3 Task 16).

Offline: frames are synthetic black JPEGs, the runtime and runner are SimpleNamespace fakes,
and the clock is passed in (``now=``) or replaced (``preview._now``).
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from rtsp_warden.config import AppConfig, CameraConfig
from rtsp_warden.detectors.event_builder import LiveBox, LiveBoxes
from rtsp_warden.proxy.mjpeg import FrameHub
from rtsp_warden.web.app import create_app
from rtsp_warden.web.config import WebSettings
from rtsp_warden.web.routes import detection as detection_routes
from rtsp_warden.web.services import preview
from rtsp_warden.web.services.preview import (
    DEFAULT_BOX_MAX_AGE_S,
    annotate_jpeg,
    box_annotator,
    boxes_max_age,
    find_runner,
    live_boxes_for,
    mjpeg_frames,
)

GREEN_MIN = 200  # box pixels are (0, 255, 0) before JPEG; measured 241..252 after
DARK_MAX = 40  # untouched black pixels decode to 0


def _jpeg(width: int, height: int) -> bytes:
    ok, buf = cv2.imencode(".jpg", np.zeros((height, width, 3), dtype=np.uint8))
    assert ok
    return buf.tobytes()


def _decode(jpeg: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert img is not None
    return img


def _boxes(
    ts_unix: float = 100.0, frame_w: int = 320, frame_h: int = 180, *items: LiveBox
) -> LiveBoxes:
    if not items:
        items = (LiveBox(label="person", bbox=(100, 50, 80, 60), confidence=0.9),)
    return LiveBoxes(boxes=tuple(items), frame_w=frame_w, frame_h=frame_h, ts_unix=ts_unix)


def _first_part_jpeg(body: bytes) -> bytes:
    head, rest = body.split(b"\r\n\r\n", 1)
    assert head.startswith(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ")
    length = int(head.rsplit(b": ", 1)[1])
    assert rest[length:] == b"\r\n"
    return rest[:length]


# --- annotate_jpeg -------------------------------------------------------------------------


def test_annotate_jpeg_draws_fresh_boxes() -> None:
    src = _jpeg(320, 180)
    out = annotate_jpeg(src, _boxes(ts_unix=100.0), now=100.5)
    assert out != src
    img = _decode(out)
    assert img.shape == (180, 320, 3)
    top_edge = img[48:53, 130:150]  # box top edge is y=50, x 100..180
    assert top_edge[:, :, 1].max() > GREEN_MIN
    assert top_edge[:, :, 2].max() < 100
    left_edge = img[70:90, 98:103]  # box left edge is x=100
    assert left_edge[:, :, 1].max() > GREEN_MIN
    assert img[75:86, 130:151].max() < DARK_MAX  # only the outline is drawn


def test_annotate_jpeg_scales_boxes_to_a_larger_hub_frame() -> None:
    out = annotate_jpeg(_jpeg(640, 360), _boxes(ts_unix=100.0), now=100.5)
    img = _decode(out)
    assert img.shape == (360, 640, 3)
    # tap box (100, 50, 80, 60) on 320x180 -> (200, 100)-(360, 220) on 640x360
    assert img[98:103, 270:290, 1].max() > GREEN_MIN  # top edge
    assert img[218:223, 270:290, 1].max() > GREEN_MIN  # bottom edge
    assert img[150:170, 358:363, 1].max() > GREEN_MIN  # right edge
    assert img[48:53, 130:150].max() < DARK_MAX  # nothing at the unscaled position
    assert img[150:171, 270:291].max() < DARK_MAX


def test_annotate_jpeg_scales_each_axis_separately() -> None:
    out = annotate_jpeg(_jpeg(640, 480), _boxes(ts_unix=100.0), now=100.5)
    img = _decode(out)
    # sx = 2.0, sy = 480 / 180: top edge y = round(50 * 2.667) = 133, bottom y = 293
    assert img[131:136, 270:290, 1].max() > GREEN_MIN
    assert img[291:296, 270:290, 1].max() > GREEN_MIN
    assert img[98:103, 270:290].max() < DARK_MAX  # where a width-only scale would put it


@pytest.mark.parametrize(("now", "drawn"), [(101.5, True), (101.6, False), (250.0, False)])
def test_annotate_jpeg_age_limit(now: float, drawn: bool) -> None:
    src = _jpeg(320, 180)
    out = annotate_jpeg(src, _boxes(ts_unix=100.0), max_age_s=1.5, now=now)
    assert (out != src) is drawn
    if not drawn:
        assert out is src  # passed through: no decode, no re-encode


@pytest.mark.parametrize(
    "boxes",
    [
        None,
        LiveBoxes(boxes=(), frame_w=320, frame_h=180, ts_unix=100.0),
        LiveBoxes(
            boxes=(LiveBox(label="person", bbox=(1, 1, 5, 5), confidence=0.5),),
            frame_w=0,
            frame_h=180,
            ts_unix=100.0,
        ),
    ],
)
def test_annotate_jpeg_passes_through_when_there_is_nothing_to_draw(boxes) -> None:
    src = _jpeg(320, 180)
    assert annotate_jpeg(src, boxes, now=100.0) is src


def test_annotate_jpeg_passes_through_an_undecodable_frame() -> None:
    src = b"\xff\xd8\xff\xd9"
    assert annotate_jpeg(src, _boxes(ts_unix=100.0), now=100.0) is src


def test_annotate_jpeg_reads_the_module_clock_when_now_is_missing(monkeypatch) -> None:
    src = _jpeg(320, 180)
    monkeypatch.setattr(preview, "_now", lambda: 100.2)
    assert annotate_jpeg(src, _boxes(ts_unix=100.0)) != src
    monkeypatch.setattr(preview, "_now", lambda: 105.0)
    assert annotate_jpeg(src, _boxes(ts_unix=100.0)) is src


def test_annotate_jpeg_clamps_boxes_that_leave_the_frame() -> None:
    boxes = _boxes(100.0, 320, 180, LiveBox(label="car", bbox=(300, 170, 80, 60), confidence=0.7))
    img = _decode(annotate_jpeg(_jpeg(320, 180), boxes, now=100.0))
    assert img.shape == (180, 320, 3)
    assert img[168:173, 302:318, 1].max() > GREEN_MIN  # top edge inside the frame


# --- mjpeg_frames, box_annotator, runner lookup ---------------------------------------------


def test_mjpeg_frames_sends_the_annotated_bytes_with_their_length() -> None:
    hub = FrameHub()
    hub.update(b"\xff\xd8\xff\xd9")
    parts = list(mjpeg_frames(hub, stop_after=1, annotate=lambda jpeg: b"XY" + jpeg))
    assert parts[0].startswith(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: 6\r\n\r\n")
    assert parts[0].endswith(b"XY\xff\xd8\xff\xd9\r\n")


def test_box_annotator_resolves_boxes_on_every_frame(monkeypatch) -> None:
    monkeypatch.setattr(preview, "_now", lambda: 100.0)
    calls: list[int] = []
    current: dict[str, LiveBoxes | None] = {"boxes": None}

    def get_boxes() -> LiveBoxes | None:
        calls.append(1)
        return current["boxes"]

    annotate = box_annotator(get_boxes)
    src = _jpeg(320, 180)
    assert annotate(src) is src
    current["boxes"] = _boxes(ts_unix=100.0)
    assert annotate(src) != src
    assert len(calls) == 2


def test_box_annotator_never_raises(monkeypatch) -> None:
    def broken() -> LiveBoxes | None:
        raise RuntimeError("runner gone")

    debug_calls: list[str] = []
    monkeypatch.setattr(preview.log, "debug", lambda msg, *a, **k: debug_calls.append(msg))
    src = _jpeg(320, 180)
    assert box_annotator(broken)(src) is src
    assert len(debug_calls) == 1


def test_find_runner_and_live_boxes_for() -> None:
    boxes = _boxes(ts_unix=100.0)
    runner = SimpleNamespace(name="detector_cam", live_boxes=lambda: boxes)
    other = SimpleNamespace(name="detector_other", live_boxes=lambda: None)
    runtime = SimpleNamespace(detector_runners=[other, runner])
    assert find_runner(runtime, "cam") is runner
    assert find_runner(runtime, "missing") is None
    assert find_runner(None, "cam") is None
    assert find_runner(SimpleNamespace(cameras=[]), "cam") is None  # no detector_runners attr
    assert live_boxes_for(runtime, "cam") is boxes
    assert live_boxes_for(runtime, "other") is None
    assert live_boxes_for(None, "cam") is None
    no_method = SimpleNamespace(detector_runners=[SimpleNamespace(name="detector_cam")])
    assert live_boxes_for(no_method, "cam") is None


def test_boxes_max_age() -> None:
    assert boxes_max_age(None) == DEFAULT_BOX_MAX_AGE_S
    motion_only = CameraConfig(name="m", main_url="rtsp://u:p@h/m", detectors=[{"type": "motion"}])
    assert boxes_max_age(motion_only) == DEFAULT_BOX_MAX_AGE_S
    slow = CameraConfig(
        name="s",
        main_url="rtsp://u:p@h/m",
        detect_fps=5.0,
        detectors=[{"type": "motion"}, {"type": "onnx", "fps": 0.5}],
    )
    assert boxes_max_age(slow) == 4.0  # two periods of a 0.5 fps detector
    fast = CameraConfig(
        name="f", main_url="rtsp://u:p@h/m", detect_fps=5.0, detectors=[{"type": "onnx"}]
    )
    assert boxes_max_age(fast) == DEFAULT_BOX_MAX_AGE_S  # 2 / 5 fps is below the floor
    disabled = CameraConfig(
        name="d",
        main_url="rtsp://u:p@h/m",
        detectors=[{"type": "onnx", "fps": 0.5, "enabled": False}],
    )
    assert boxes_max_age(disabled) == DEFAULT_BOX_MAX_AGE_S


# --- the route and the detail page -----------------------------------------------------------


def _login(client: TestClient) -> None:
    client.get("/login")
    token = client.cookies.get("warden_csrf", "")
    client.post(
        "/login", data={"username": "admin", "password": "testpass123", "csrf_token": token}
    )


def _client(runtime: object, cam: CameraConfig) -> TestClient:
    cfg = AppConfig(cameras=[cam])
    app = create_app(WebSettings(), cfg=cfg, runtime_provider=lambda: runtime, runtime=runtime)
    client = TestClient(app)
    _login(client)
    return client


def _runtime(cam: CameraConfig, hub: object, runners: list | None = None) -> SimpleNamespace:
    cam_rt = SimpleNamespace(
        camera=cam, hub=hub, proxy=None, recorder=SimpleNamespace(processes=lambda: [])
    )
    runtime = SimpleNamespace(cameras=[cam_rt])
    if runners is not None:
        runtime.detector_runners = runners
    return runtime


@pytest.fixture
def cam() -> CameraConfig:
    return CameraConfig(name="cam", main_url="rtsp://u:p@h/m")


@pytest.fixture
def frame() -> bytes:
    return _jpeg(320, 180)


@pytest.fixture
def hub(frame: bytes) -> FrameHub:
    h = FrameHub()
    h.update(frame)
    return h


@pytest.fixture
def one_frame_stream(monkeypatch) -> None:
    """The route streams forever; make it send one frame so TestClient can read the body."""
    real = preview.mjpeg_frames

    def bounded(hub, annotate=None):
        return real(hub, stop_after=1, annotate=annotate)

    monkeypatch.setattr(detection_routes, "mjpeg_frames", bounded)
    monkeypatch.setattr(preview, "_now", lambda: 100.5)


def test_live_boxes_route_draws_the_runner_boxes(db_with_user, cam, hub, one_frame_stream):
    runner = SimpleNamespace(name="detector_cam", live_boxes=lambda: _boxes(ts_unix=100.0))
    client = _client(_runtime(cam, hub, [runner]), cam)
    r = client.get("/cameras/cam/live-boxes.mjpeg")
    assert r.status_code == 200
    assert r.headers["content-type"] == "multipart/x-mixed-replace; boundary=frame"
    assert r.headers["cache-control"] == "no-store"
    img = _decode(_first_part_jpeg(r.content))
    assert img[48:53, 130:150, 1].max() > GREEN_MIN


def test_live_boxes_route_passes_frames_through_without_a_runner(
    db_with_user, cam, hub, frame, one_frame_stream
):
    client = _client(_runtime(cam, hub), cam)
    r = client.get("/cameras/cam/live-boxes.mjpeg")
    assert r.status_code == 200
    assert _first_part_jpeg(r.content) == frame


def test_live_boxes_route_looks_the_runner_up_per_frame(db_with_user, cam, frame, one_frame_stream):
    """A detector rebuild replaces the runner; an open stream must follow the new one."""
    stale_runner = SimpleNamespace(name="detector_cam", live_boxes=lambda: None)
    new_runner = SimpleNamespace(name="detector_cam", live_boxes=lambda: _boxes(ts_unix=100.0))
    holder: dict[str, SimpleNamespace] = {}

    class RebuildingHub:
        """Swaps the runner (as rebuild_camera_detectors does) while the frame is in flight."""

        def wait_for_new(self, last_id: int, timeout: float = 2.0):
            holder["runtime"].detector_runners = [new_runner]
            return frame, last_id + 1, 0.0

    holder["runtime"] = _runtime(cam, RebuildingHub(), [stale_runner])
    client = _client(holder["runtime"], cam)
    r = client.get("/cameras/cam/live-boxes.mjpeg")
    assert r.status_code == 200
    img = _decode(_first_part_jpeg(r.content))
    assert img[48:53, 130:150, 1].max() > GREEN_MIN


def test_live_boxes_route_503_without_a_hub(db_with_user, cam):
    client = _client(SimpleNamespace(cameras=[]), cam)
    assert client.get("/cameras/cam/live-boxes.mjpeg").status_code == 503


def test_live_boxes_route_requires_login(db_with_user, cam, hub):
    runtime = _runtime(cam, hub)
    app = create_app(
        WebSettings(),
        cfg=AppConfig(cameras=[cam]),
        runtime_provider=lambda: runtime,
        runtime=runtime,
    )
    r = TestClient(app).get("/cameras/cam/live-boxes.mjpeg", follow_redirects=False)
    assert r.status_code == 401


def test_detail_page_has_the_show_boxes_toggle(db_with_user, cam, hub):
    client = _client(_runtime(cam, hub), cam)
    html = client.get("/cameras/cam").text
    assert "Show boxes" in html
    assert 'role="switch"' in html
    assert 'data-boxes-src="/cameras/cam/live-boxes.mjpeg"' in html
    assert 'data-plain-src="/cameras/cam/live.mjpeg"' in html
    # The img's own src is the plain stream: boxes stay off until the switch is turned on.
    assert re.search(r'<img\b[^>]*\ssrc="/cameras/cam/live\.mjpeg"', html)
    assert "warden.preview.boxes" in html
