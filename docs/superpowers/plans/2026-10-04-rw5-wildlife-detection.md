# RW-5: Wildlife Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tell a cat, a fox and a raccoon apart on the Foscam, day and night, with a second YOLOX model trained on camera-trap data, plus the runtime pieces that make two models and a day/night signal work in the existing pipeline.

**Architecture:** The package gains a frame-based night flag (`detectors/daylight.py`), `when` and `classes` fields on detector specs with their runner, config, status and web surfaces, and a `night` badge on events. A separate uv project in `tools/wildlife/` downloads ENA24, Open Images and the raccoon sets, merges them into one COCO json with the 18-label `wildlife.txt`, fine-tunes YOLOX-S with grayscale augmentation, exports a raw-head ONNX file and writes the `model.yaml` the registry already understands. The model is dropped into `data/models/wildlife-yolox-s/` as a user model.

**Tech Stack:** Python 3.11+, pydantic 2, FastAPI + htmx, OpenCV, NumPy, onnxruntime; training: PyTorch cu128, YOLOX (Apache-2.0, commit `6ddff4824372906469a7fae2dc3206c7aa4bbaee`), pycocotools, onnx.

**Spec:** `docs/superpowers/specs/2026-10-04-wildlife-detection-design.md`

## Deviations found while executing (2026-10-04)

- **YOLOX is a pinned source checkout, not a uv dependency.** Its `setup.py` imports torch at build
  time (`AssertionError: torch is required for pre-compiling ops`), so `uv sync` cannot build the git
  dependency. `tools/wildlife/setup.sh` runs `uv sync` and fetches commit
  `6ddff4824372906469a7fae2dc3206c7aa4bbaee` into `tools/wildlife/.yolox/`; every script puts that
  directory on `sys.path`. The YOLOX runtime helpers (`loguru`, `tabulate`, `thop`, `ninja`, `psutil`,
  `tensorboard`) are the project's dependencies instead.
- **No `train.py` of our own.** With the checkout present, `train.sh` runs YOLOX's `tools/train.py -f exp.py`
  directly (Task 10's file list shrinks by one; `exp.py` is loaded through YOLOX's `get_exp`).
- **The public ENA24 zip has no human images** (8789 of the 9676 listed), so `person` had zero boxes.
  Open Images `Person` and `Car` (capped at 1500 images each) were added as hard negatives, and the
  Open Images validation and test splits add fox, raccoon and skunk images (the train split alone has
  only 422 fox and 285 raccoon images without groups and drawings).

## Global Constraints

- `uv` only: `uv run pytest`, `uv run ruff check src/ tests/`, `uv run ruff format --check src/ tests/`; the gate is "no new ruff errors" over the 4 baseline E501 in `cli.py`, `proxy/mjpeg.py`, `recorder.py`.
- Every test stays offline: no ffmpeg, cameras, network, model weights or torch in `tests/`.
- Every `config.yaml` read-modify-write goes through `web/services/detection.update_config_yaml` or `camera_config.update_raw_config`; a route patches only the key it owns and `${VAR}` text survives.
- Labels file `wildlife.txt` order is fixed by the spec (18 labels, `cat` first, `vehicle` last); COCO category id = label index + 1.
- Model tensor contract unchanged: input `images` float32 `[1,3,640,640]` BGR 0–255, output `output` `[1,8400,23]` raw YOLOX rows; `postprocess: yolox`.
- MIT-clean: no Ultralytics or YOLO-World code or weights; data licenses as in the spec section 2.7.
- Never commit anything under `tools/wildlife/data/`, `tools/wildlife/YOLOX_outputs/`, `data/`, or the real camera password.
- Version bump to 1.4.0 touches `pyproject.toml`, `src/rtsp_warden/__init__.py`, `tests/test_admin.py`.
- Commit after each task; push only when the owner says "push".

## Review Focus

1. A camera whose first frames arrive while the ingest is still black (ffmpeg warm-up) must not lock the state to `night` for long: the flag follows the frames, and three colour frames flip it back. Test in Task 1 (`test_dark_then_colour_flips_after_three_frames`).
2. A `when: night` slot that never ran must still show on the Detection panel as a configured detector with its skip count, not as "not running" with an error. Test in Task 3 (`test_when_skipped_slot_is_reported_as_running`).
3. Two `onnx` slots with overlapping `classes` must still both be accepted; the pipeline warns rather than refuses, because a user may want it. Test in Task 4 (`test_overlapping_classes_load_with_a_warning`).
4. An `onnx` spec `classes` that intersects to nothing with the camera `detect_classes` must build a detector that reports nothing, not raise at startup. Test in Task 4 (`test_empty_intersection_builds_a_silent_detector`).
5. Event rows written before this release have no `night` key; the events page must render them with no badge and no error. Test in Task 6 (`test_old_rows_without_night_render_without_badge`).

---

### Task 1: `detectors/daylight.py` — channel spread and the DayNight state

**Files:**
- Create: `src/rtsp_warden/detectors/daylight.py`
- Test: `tests/test_daylight.py`

**Interfaces:**
- Produces: `channel_spread(frame_bgr: np.ndarray) -> float`; `NIGHT_SPREAD = 4.0`; `SWITCH_FRAMES = 3`; `class DayNight` with `update(frame_bgr, ts_unix) -> bool`, fields `night: bool | None`, `since_ts: float | None`, `switches: int`, `last_spread: float`; `allows(when: str, night: bool | None) -> bool`.

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for detectors/daylight.py: grayscale (IR) detection with hysteresis."""

from __future__ import annotations

import numpy as np
import pytest

from rtsp_warden.detectors.daylight import (
    NIGHT_SPREAD,
    SWITCH_FRAMES,
    DayNight,
    allows,
    channel_spread,
)

H, W = 90, 160


def grey(level: int = 120) -> np.ndarray:
    return np.full((H, W, 3), level, dtype=np.uint8)


def colour() -> np.ndarray:
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    frame[..., 0] = 40  # B
    frame[..., 1] = 120  # G
    frame[..., 2] = 200  # R
    return frame


def dark() -> np.ndarray:
    return np.zeros((H, W, 3), dtype=np.uint8)


def test_spread_is_zero_for_grey_and_large_for_colour() -> None:
    assert channel_spread(grey()) == 0.0
    assert channel_spread(colour()) == pytest.approx(160.0)
    assert channel_spread(dark()) == 0.0


def test_spread_tolerates_bad_input() -> None:
    assert channel_spread(np.zeros((H, W), dtype=np.uint8)) == 0.0
    assert channel_spread(np.zeros((0, 0, 3), dtype=np.uint8)) == 0.0


def test_first_frame_sets_the_state_at_once() -> None:
    dn = DayNight()
    assert dn.night is None
    assert dn.update(grey(), 10.0) is True
    assert dn.night is True and dn.since_ts == 10.0 and dn.switches == 0
    assert dn.last_spread == 0.0


def test_two_frames_do_not_flip_three_do() -> None:
    dn = DayNight()
    dn.update(colour(), 1.0)
    assert dn.update(grey(), 2.0) is False
    assert dn.update(grey(), 3.0) is False
    assert dn.update(grey(), 4.0) is True
    assert dn.since_ts == 4.0 and dn.switches == 1


def test_a_single_odd_frame_resets_the_pending_count() -> None:
    dn = DayNight()
    dn.update(colour(), 1.0)
    dn.update(grey(), 2.0)
    dn.update(grey(), 3.0)
    dn.update(colour(), 4.0)  # back on the current side: pending resets
    dn.update(grey(), 5.0)
    assert dn.update(grey(), 6.0) is False
    assert dn.update(grey(), 7.0) is True


def test_dark_then_colour_flips_after_three_frames() -> None:
    """(review focus) Black warm-up frames count as night; daylight takes over quickly."""
    dn = DayNight()
    for ts in (1.0, 2.0, 3.0):
        assert dn.update(dark(), ts) is True
    assert dn.update(colour(), 4.0) is True
    assert dn.update(colour(), 5.0) is True
    assert dn.update(colour(), 6.0) is False
    assert dn.switches == 1


def test_threshold_and_switch_frames_are_configurable() -> None:
    dn = DayNight(threshold=200.0, switch_frames=1)
    dn.update(grey(), 1.0)
    assert dn.update(colour(), 2.0) is True  # 160 < 200: still "night"
    dn2 = DayNight(threshold=NIGHT_SPREAD, switch_frames=1)
    dn2.update(grey(), 1.0)
    assert dn2.update(colour(), 2.0) is False
    assert SWITCH_FRAMES == 3


def test_allows() -> None:
    assert allows("always", None) and allows("always", True) and allows("always", False)
    assert allows("night", True) and not allows("night", False) and not allows("night", None)
    assert allows("day", False) and allows("day", None) and not allows("day", True)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_daylight.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'rtsp_warden.detectors.daylight'`

- [ ] **Step 3: Write the implementation**

```python
"""Day / night (IR) detection from the decoded tap frame.

The Foscam, like most IR cameras, flips its IR-cut filter at night and the frame becomes
grayscale: red, green and blue are equal within JPEG chroma noise. ``channel_spread``
measures that (mean of ``max(B,G,R) - min(B,G,R)`` over a subsampled frame; about 0-2 for
an IR frame, tens for daylight). ``DayNight`` turns the measure into a state with
hysteresis so dusk does not flap it. A very dark colour frame also measures low and counts
as night, which is the intended meaning: "IR, or too dark for colour".

Pure NumPy; no detector or runtime imports, so the training tool can copy the formula.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Mean channel spread below which a frame counts as grayscale (night).
NIGHT_SPREAD = 4.0
#: Consecutive frames on the other side before the state flips.
SWITCH_FRAMES = 3
_SUBSAMPLE = 4


def channel_spread(frame_bgr: np.ndarray) -> float:
    """Mean of ``max(B,G,R) - min(B,G,R)`` over every 4th pixel; 0.0 for unusable input."""
    if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[2] < 3 or frame_bgr.size == 0:
        return 0.0
    small = frame_bgr[::_SUBSAMPLE, ::_SUBSAMPLE, :3]
    if small.size == 0:
        return 0.0
    spread = small.max(axis=2).astype(np.int16) - small.min(axis=2).astype(np.int16)
    return float(spread.mean())


@dataclass
class DayNight:
    """Night state of one camera, updated once per decoded frame.

    The first frame sets ``night`` at once. Afterwards the state flips only after
    ``switch_frames`` consecutive frames measured on the other side of ``threshold``;
    a frame back on the current side resets that count. ``since_ts`` is the frame time
    of the last change, ``switches`` counts changes (not the first frame).
    """

    threshold: float = NIGHT_SPREAD
    switch_frames: int = SWITCH_FRAMES
    night: bool | None = None
    since_ts: float | None = None
    switches: int = 0
    last_spread: float = 0.0
    _pending: int = field(default=0, init=False, repr=False)

    def update(self, frame_bgr: np.ndarray, ts_unix: float) -> bool:
        spread = channel_spread(frame_bgr)
        self.last_spread = spread
        observed = spread < self.threshold
        if self.night is None:
            self.night = observed
            self.since_ts = float(ts_unix)
            self._pending = 0
            return observed
        if observed == self.night:
            self._pending = 0
            return self.night
        self._pending += 1
        if self._pending >= max(1, int(self.switch_frames)):
            self.night = observed
            self.since_ts = float(ts_unix)
            self.switches += 1
            self._pending = 0
        return self.night


def allows(when: str, night: bool | None) -> bool:
    """Whether a detector with ``when`` runs in the given state (``None`` counts as day)."""
    if when == "night":
        return night is True
    if when == "day":
        return not night
    return True


__all__ = ["NIGHT_SPREAD", "SWITCH_FRAMES", "DayNight", "allows", "channel_spread"]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_daylight.py -q`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/rtsp_warden/detectors/daylight.py tests/test_daylight.py
git commit -m "feat(detection): day/night state from the tap frame's channel spread"
```

---

### Task 2: Runner night flag, event metadata, status surfaces

**Files:**
- Modify: `src/rtsp_warden/detectors/runner.py` (fields, `__post_init__`, `_process_job`, `status`)
- Modify: `src/rtsp_warden/detectors/event_builder.py` (`__init__`, `_open_track`, `_open_motion`)
- Modify: `src/rtsp_warden/status_model.py` (`DetectionStatus`, `summarize_detection`)
- Modify: `src/rtsp_warden/web/templates/partials/detection_panel.html` (status line)
- Test: `tests/test_runner_tracking.py`, `tests/test_event_builder.py`, `tests/test_status_detection_surfaces.py`, `tests/test_detection_panel.py`

**Interfaces:**
- Consumes: `DayNight`, `channel_spread` from Task 1.
- Produces: `DetectorRunner.daynight: DayNight` (constructor field); runner status keys `night: bool | None`, `night_since: float | None`, `night_switches: int`; `EventBuilder.night: bool | None` attribute, metadata key `"night"` on object and motion events; `DetectionStatus["night" | "night_since" | "night_switches"]`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_runner_tracking.py` (reuse `ScriptedDetector`, `_job`, `W`, `H`, `PERSON_BOX`; add a colour JPEG helper):

```python
def _colour_jpeg() -> bytes:
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    frame[..., 2] = 200
    ok, buf = cv2.imencode(".jpg", frame)
    assert ok
    return buf.tobytes()


def _job_bytes(ts: float, jpeg: bytes) -> _FrameJob:
    return _FrameJob(camera="yard", stream="main", jpeg_bytes=jpeg, ts_unix=ts)


def test_runner_tracks_night_state_and_reports_it() -> None:
    det = ScriptedDetector(lambda ts: [])
    spec = DetectorSpec(type="onnx")
    slot = DetectorSlot(index=0, spec=spec, detector=det, fps=5.0, tracked=True,
                        motion_events=False, input_width=640)
    runner = DetectorRunner(name="d", slots=(slot,), worker_count=1, tap_fps=5.0,
                            tracker=Tracker(grace_seconds=1.0, min_frames=1))
    assert runner.status()["night"] is None
    runner._process_job(_job(1.0))  # the black test JPEG is grayscale
    st = runner.status()
    assert st["night"] is True and st["night_since"] == 1.0 and st["night_switches"] == 0
    colour = _colour_jpeg()
    for ts in (2.0, 3.0):
        runner._process_job(_job_bytes(ts, colour))
    assert runner.status()["night"] is True
    runner._process_job(_job_bytes(4.0, colour))
    st = runner.status()
    assert st["night"] is False and st["night_since"] == 4.0 and st["night_switches"] == 1


def test_runner_hands_the_night_flag_to_the_event_builder(tmp_path: Path) -> None:
    class Builder:
        night: bool | None = None
        seen: list[bool | None] = []

        def on_tracks(self, update: TrackerUpdate, shape: tuple[int, int]) -> None:
            self.seen.append(self.night)

        def on_motion(self, *a: Any) -> None:
            pass

        def close_all(self, ts: float) -> None:
            pass

    builder = Builder()
    det = ScriptedDetector(lambda ts: [Detection(kind="person", confidence=0.9, bbox=PERSON_BOX,
                                                  ts_unix=ts)])
    spec = DetectorSpec(type="onnx")
    slot = DetectorSlot(index=0, spec=spec, detector=det, fps=5.0, tracked=True,
                        motion_events=False, input_width=640)
    runner = DetectorRunner(name="d", slots=(slot,), worker_count=1, tap_fps=5.0,
                            tracker=Tracker(grace_seconds=1.0, min_frames=1),
                            event_builder=builder)  # type: ignore[arg-type]
    runner._process_job(_job(1.0))
    assert builder.seen == [True]


def test_masks_do_not_bias_the_night_measure() -> None:
    """A privacy mask blacks out pixels; the measure runs before masks are applied."""
    from rtsp_warden.detectors.roi import Mask

    det = ScriptedDetector(lambda ts: [])
    spec = DetectorSpec(type="onnx")
    slot = DetectorSlot(index=0, spec=spec, detector=det, fps=5.0, tracked=True,
                        motion_events=False, input_width=640)
    full = Mask(polygon=[(0, 0), (W, 0), (W, H), (0, H)], name="all")
    runner = DetectorRunner(name="d", slots=(slot,), worker_count=1, tap_fps=5.0,
                            masks=[full], tracker=Tracker(grace_seconds=1.0, min_frames=1))
    runner._process_job(_job_bytes(1.0, _colour_jpeg()))
    assert runner.status()["night"] is False
```

In `tests/test_event_builder.py` (reuse `FakeDb`, `_track`, `SHAPE`, `tmp_path` style of the existing tests; look at how the existing tests construct `EventBuilder(camera=..., output_dir=..., db=FakeDb())` and `TrackerUpdate(opened=[...], updated=[], closed=[])` and copy that construction):

```python
def test_object_event_metadata_carries_the_night_flag(tmp_path: Path) -> None:
    db = FakeDb()
    builder = EventBuilder(camera="yard", output_dir=tmp_path, db=db)
    assert builder.night is None
    builder.night = True
    builder.on_tracks(TrackerUpdate(opened=[_track(1)], updated=[], closed=[]), SHAPE)
    (_kind, _id, fields), = db.of("insert")
    assert fields["metadata"]["night"] is True


def test_motion_event_metadata_carries_the_night_flag(tmp_path: Path) -> None:
    db = FakeDb()
    builder = EventBuilder(camera="yard", output_dir=tmp_path, db=db)
    builder.night = False
    builder.on_motion(True, False, 10.0)
    (_kind, _id, fields), = db.of("insert")
    assert fields["metadata"] == {"night": False}


def test_metadata_night_is_null_before_the_first_frame(tmp_path: Path) -> None:
    db = FakeDb()
    builder = EventBuilder(camera="yard", output_dir=tmp_path, db=db)
    builder.on_tracks(TrackerUpdate(opened=[_track(1)], updated=[], closed=[]), SHAPE)
    (_kind, _id, fields), = db.of("insert")
    assert fields["metadata"]["night"] is None
```

In `tests/test_status_detection_surfaces.py` (find the existing test that feeds a raw dict to `summarize_detection` and add):

```python
def test_summary_carries_the_night_fields() -> None:
    raw = {"detectors": [], "frames_processed": 1, "night": True, "night_since": 12.5,
           "night_switches": 2}
    out = summarize_detection(raw)
    assert out is not None
    assert out["night"] is True and out["night_since"] == 12.5 and out["night_switches"] == 2


def test_summary_night_defaults_when_absent() -> None:
    out = summarize_detection({"detectors": [], "frames_processed": 1})
    assert out is not None
    assert out["night"] is None and out["night_since"] is None and out["night_switches"] == 0
```

In `tests/test_detection_panel.py`, extend `_live_status` to return `"night": True, "night_since": 1759536000.0, "night_switches": 1` for `yard`, and add:

```python
def test_panel_shows_night_mode(env: SimpleNamespace) -> None:
    r = env.client.get("/cameras/yard/detection")
    assert r.status_code == 200
    assert "night mode on" in r.text
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_runner_tracking.py tests/test_event_builder.py tests/test_status_detection_surfaces.py tests/test_detection_panel.py -q -k "night"`
Expected: FAIL (`KeyError: 'night'`, `AttributeError: ... has no attribute 'night'`, missing text).

- [ ] **Step 3: Implement**

`runner.py`:

```python
from .daylight import DayNight
# dataclass fields, after tap_fps:
    daynight: DayNight = field(default_factory=DayNight)
# in _process_job, replace the frame decode / masks lines with:
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is None:
            logger.debug("failed to decode JPEG for %s/%s", job.camera, job.stream)
            return
        night = self.daynight.update(frame, job.ts_unix)  # before masks: masked pixels are black
        if self.event_builder is not None:
            self.event_builder.night = night
        frame = apply_masks(frame, self.masks)
# in status(), after "stationary_suppressed":
            "night": self.daynight.night,
            "night_since": self.daynight.since_ts,
            "night_switches": int(self.daynight.switches),
```

`event_builder.py`: in `EventBuilder.__init__` add `self.night: bool | None = None` with the comment `# set by the runner once per frame; written into event metadata`; in `_open_track` metadata add `"night": self.night`; in `_open_motion` change `metadata={}` to `metadata={"night": self.night}`. Update the module docstring's metadata description.

`status_model.py`: `DetectionStatus` gains `night: bool | None`, `night_since: float | None`, `night_switches: int`; in `summarize_detection` add

```python
        "night": raw.get("night") if isinstance(raw.get("night"), bool) else None,
        "night_since": _as_float(raw.get("night_since")),
        "night_switches": _as_int(raw.get("night_switches")),
```

(check `_as_float` returns None for None; if it returns 0.0, write `raw.get("night_since") if isinstance(raw.get("night_since"), (int, float)) else None`).

`detection_panel.html`, inside the `{% if status %}` paragraph after the dropped-frames line:

```jinja
    {% if status.get('night') is not none %}
    &middot; night mode {{ 'on' if status.get('night') else 'off' }}
    {% endif %}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_runner_tracking.py tests/test_event_builder.py tests/test_status_detection_surfaces.py tests/test_detection_panel.py tests/test_detector_runner.py -q`
Expected: all pass (existing status tests that compare whole dicts need the three new keys added).

- [ ] **Step 5: Commit**

```bash
git add src/rtsp_warden/detectors/runner.py src/rtsp_warden/detectors/event_builder.py src/rtsp_warden/status_model.py src/rtsp_warden/web/templates/partials/detection_panel.html tests/
git commit -m "feat(detection): night flag on runner status, event metadata and the Detection panel"
```

---

### Task 3: `when: always | day | night` on detector specs

**Files:**
- Modify: `src/rtsp_warden/detectors/registry.py` (`DetectorSpec.when`, `DetectorSlot.when`, `_make_slot`)
- Modify: `src/rtsp_warden/detectors/runner.py` (`__post_init__`, `_process_job`, `_slot_status`)
- Modify: `src/rtsp_warden/status_model.py` (`DetectorStatus`, `summarize_detection`)
- Modify: `src/rtsp_warden/web/services/detection.py` (`detector_summary`, `detector_rows`)
- Test: `tests/test_config_detection.py`, `tests/test_runner_tracking.py`, `tests/test_detection_service.py`

**Interfaces:**
- Consumes: `allows(when, night)` from Task 1; runner `daynight` from Task 2.
- Produces: `DetectorWhen = Literal["always", "day", "night"]`; `DetectorSpec.when: DetectorWhen = "always"`; `DetectorSlot.when: str`; runner per-slot status `when: str`, `when_skipped: int`; `detector_rows` row key `when`.

- [ ] **Step 1: Write the failing tests**

`tests/test_config_detection.py` (use the file's existing camera-dict helpers):

```python
def test_detector_when_defaults_to_always_and_accepts_day_night() -> None:
    assert DetectorSpec(type="motion").when == "always"
    assert DetectorSpec(type="onnx", when="night").when == "night"
    assert DetectorSpec(type="motion", when="day").when == "day"
    with pytest.raises(ValidationError):
        DetectorSpec(type="onnx", when="dusk")
```

`tests/test_runner_tracking.py`:

```python
def _slot(index: int, det: ScriptedDetector, *, when: str = "always") -> DetectorSlot:
    spec = DetectorSpec(type="onnx", when=when)
    return DetectorSlot(index=index, spec=spec, detector=det, fps=5.0, tracked=True,
                        motion_events=False, input_width=640, when=when)


def test_when_night_slot_runs_only_at_night() -> None:
    always = ScriptedDetector(lambda ts: [])
    nightly = ScriptedDetector(lambda ts: [])
    runner = DetectorRunner(name="d", slots=(_slot(0, always), _slot(1, nightly, when="night")),
                            worker_count=1, tap_fps=5.0,
                            tracker=Tracker(grace_seconds=1.0, min_frames=1))
    runner._process_job(_job(1.0))  # black JPEG: night
    assert always.calls == [1.0] and nightly.calls == [1.0]
    colour = _colour_jpeg()
    for ts in (2.0, 3.0, 4.0):
        runner._process_job(_job_bytes(ts, colour))  # day from ts 4.0
    runner._process_job(_job_bytes(5.0, colour))
    assert nightly.calls == [1.0, 2.0, 3.0]
    assert always.calls == [1.0, 2.0, 3.0, 4.0, 5.0]
    rows = runner.status()["detectors"]
    assert rows[1]["when"] == "night" and rows[1]["when_skipped"] == 2
    assert rows[0]["when"] == "always" and rows[0]["when_skipped"] == 0


def test_when_skipped_slot_is_reported_as_running() -> None:
    """(review focus) A paused slot is configured and healthy; it just did not run."""
    day = ScriptedDetector(lambda ts: [])
    runner = DetectorRunner(name="d", slots=(_slot(0, day, when="day"),), worker_count=1,
                            tap_fps=5.0, tracker=Tracker(grace_seconds=1.0, min_frames=1))
    runner.setup()
    try:
        runner._process_job(_job(1.0))  # night: the day slot pauses
    finally:
        runner.teardown()
    row = runner.status()["detectors"][0]
    assert row["setup_error"] is None and row["processed"] == 0 and row["when_skipped"] == 1
```

`tests/test_detection_service.py` (uses `detector_summary` / `detector_rows`; follow its fixtures):

```python
def test_detector_summary_shows_when_unless_always() -> None:
    assert "when=" not in detector_summary(DetectorSpec(type="onnx"))
    assert "when=night" in detector_summary(DetectorSpec(type="onnx", when="night"))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_config_detection.py tests/test_runner_tracking.py tests/test_detection_service.py -q -k "when"`
Expected: FAIL (`ValidationError: extra fields not permitted` or `TypeError: unexpected keyword 'when'`).

- [ ] **Step 3: Implement**

`registry.py`:

```python
DetectorWhen = Literal["always", "day", "night"]
# DetectorSpec, after device:
    when: DetectorWhen = "always"  # run always, only by day, or only at night (IR frames)
# DetectorSlot, after input_width:
    when: str = "always"
# _make_slot: add
        when=spec.when,
```

`runner.py`:

```python
from .daylight import DayNight, allows
# __post_init__, after _slot_errors:
        self._slot_when_skipped: list[int] = [0] * count
# _process_job loop, before the _due check:
            if slot is not None and not allows(slot.when, night):
                self._slot_when_skipped[i] += 1
                continue
# _slot_status: add
            "when": str(slot.when),
            "when_skipped": int(self._slot_when_skipped[i]),
```

`status_model.py`: `DetectorStatus` gains `when: str` and `when_skipped: int`; `summarize_detection` adds `"when": _as_text(entry.get("when")) or "always"` and `"when_skipped": _as_int(entry.get("when_skipped"))`.

`web/services/detection.py`:

```python
def detector_summary(spec: DetectorSpec) -> str:
    parts = [
        f"{name}={getattr(spec, name)}"
        for name in _SUMMARY_FIELDS.get(spec.type, ())
        if getattr(spec, name, None) is not None
    ]
    if spec.when != "always":
        parts.append(f"when={spec.when}")
    return ", ".join(parts) if parts else "(defaults)"
# detector_rows: add
                "when": spec.when,
                "when_skipped": _count(live, "when_skipped"),
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_config_detection.py tests/test_runner_tracking.py tests/test_detection_service.py tests/test_status_detection_surfaces.py tests/test_detection_status.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/rtsp_warden/detectors/registry.py src/rtsp_warden/detectors/runner.py src/rtsp_warden/status_model.py src/rtsp_warden/web/services/detection.py tests/
git commit -m "feat(detection): per-detector when: always | day | night"
```

---

### Task 4: `classes` on `onnx` detector specs

**Files:**
- Modify: `src/rtsp_warden/detectors/registry.py` (`DetectorSpec.classes`, validator, `_build_onnx_detector`)
- Modify: `src/rtsp_warden/config.py` (`_validate_labels`)
- Test: `tests/test_config_labels.py`, `tests/test_onnx_registry.py`

**Interfaces:**
- Consumes: `effective_classes(camera_classes, detector_classes)` from `detectors/class_filter.py`; `unknown_labels_message(camera, where, unknown, per_model)`.
- Produces: `DetectorSpec.classes: list[str] | None = None`; the built `OnnxDetector.classes` is `effective_classes(cam.detect_classes, spec.classes)`.

- [ ] **Step 1: Write the failing tests**

`tests/test_config_labels.py` (uses `write_model`, `app`, `cam`, `default_models`):

```python
def test_onnx_classes_are_validated_against_that_model(default_models: Path) -> None:
    write_model(default_models, "yolox-s", ["person", "car", "cat"])
    write_model(default_models, "wild", ["cat", "fox", "raccoon"])
    ok = AppConfig.model_validate(app(cam(detectors=[
        {"type": "onnx", "model": "yolox-s", "classes": ["person"]},
        {"type": "onnx", "model": "wild", "classes": ["fox", "raccoon"]},
    ])))
    assert ok.cameras[0].detectors[1].classes == ["fox", "raccoon"]
    with pytest.raises(ValidationError, match=r"detectors\[0\] classes.*'fox'"):
        AppConfig.model_validate(app(cam(detectors=[
            {"type": "onnx", "model": "yolox-s", "classes": ["fox"]},
            {"type": "onnx", "model": "wild"},
        ])))


def test_classes_on_a_non_onnx_detector_is_rejected() -> None:
    with pytest.raises(ValidationError, match="classes is only valid for type: onnx"):
        DetectorSpec(type="motion", classes=["person"])
```

`tests/test_onnx_registry.py` (it builds detectors through `build_detectors_for_camera(cam, specs, models_dir=...)` with descriptors written to a temp models dir; follow its helper for writing a descriptor):

```python
def test_spec_classes_intersect_with_camera_detect_classes(models_dir: Path) -> None:
    _write_descriptor(models_dir, "wild", ["cat", "fox", "raccoon", "person"])
    cam = CameraConfig(name="yard", main_url=URL, detect_classes=["fox", "person", "car"],
                       detectors=[DetectorSpec(type="onnx", model="wild",
                                               classes=["cat", "fox", "raccoon"])])
    bundle = build_detectors_for_camera(cam, cam.detectors, models_dir=models_dir)
    assert bundle.detectors[0].classes == ["fox"]


def test_spec_classes_alone_filter_the_model(models_dir: Path) -> None:
    _write_descriptor(models_dir, "wild", ["cat", "fox", "raccoon", "person"])
    cam = CameraConfig(name="yard", main_url=URL,
                       detectors=[DetectorSpec(type="onnx", model="wild", classes=["raccoon"])])
    bundle = build_detectors_for_camera(cam, cam.detectors, models_dir=models_dir)
    assert bundle.detectors[0].classes == ["raccoon"]


def test_empty_intersection_builds_a_silent_detector(models_dir: Path) -> None:
    """(review focus) Nothing in common means "report nothing", never a startup error."""
    _write_descriptor(models_dir, "wild", ["cat", "fox"])
    cam = CameraConfig(name="yard", main_url=URL, detect_classes=["person"],
                       detectors=[DetectorSpec(type="onnx", model="wild", classes=["fox"])])
    bundle = build_detectors_for_camera(cam, cam.detectors, models_dir=models_dir)
    det = bundle.detectors[0]
    assert det.classes == []
    assert det.process(np.zeros((64, 64, 3), dtype=np.uint8), 1.0) == []


def test_overlapping_classes_load_with_a_warning(models_dir: Path, caplog) -> None:
    """(review focus) Two models allowed to report the same label are accepted, with a warning."""
    _write_descriptor(models_dir, "a", ["dog", "cat"])
    _write_descriptor(models_dir, "b", ["dog", "fox"])
    cam = CameraConfig(name="yard", main_url=URL, detectors=[
        DetectorSpec(type="onnx", model="a", classes=["dog"]),
        DetectorSpec(type="onnx", model="b", classes=["dog", "fox"]),
    ])
    with caplog.at_level("WARNING"):
        bundle = build_detectors_for_camera(cam, cam.detectors, models_dir=models_dir)
    assert len(bundle.detectors) == 2
    assert "both report 'dog'" in caplog.text
```

(If `test_onnx_registry.py` has no `models_dir` fixture or `_write_descriptor` helper, add them at the top of the file: a `tmp_path`-based fixture and a helper that writes `model.yaml` with `postprocess: yolox`, `input_size: [64, 64]`, `file: m.onnx`, `labels: labels.txt` plus the labels file, like `write_model` in `test_config_labels.py`.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_config_labels.py tests/test_onnx_registry.py -q -k "classes"`
Expected: FAIL (`extra fields not permitted: classes`).

- [ ] **Step 3: Implement**

`registry.py`:

```python
# DetectorSpec, after when:
    classes: list[str] | None = None  # onnx: labels this slot may report (None = all)

    @model_validator(mode="after")
    def _classes_only_for_onnx(self) -> DetectorSpec:
        if self.classes is not None and self.type != "onnx":
            raise ValueError("classes is only valid for type: onnx")
        return self

# _build_onnx_detector: replace the classes line with
    classes = effective_classes(camera_detect_classes, spec.classes)
```

In `build_detectors_for_camera`, after the slots are built, warn once per overlapping label:

```python
    seen: dict[str, int] = {}
    for slot in bundle.slots:
        if slot.spec.type != "onnx":
            continue
        reported = getattr(slot.detector, "classes", None)
        if reported is None:
            reported = getattr(slot.detector, "labels", None) or []
            try:
                reported = load_labels(slot.detector.descriptor)  # type: ignore[attr-defined]
            except Exception:
                reported = []
        for label in reported:
            if label in seen:
                logger.warning(
                    "camera %s: detectors %d and %d both report %r; one object may open two "
                    "events (set classes on one of them)",
                    cam.name, seen[label], slot.index, label,
                )
            else:
                seen[label] = slot.index
```

`config.py` `_validate_labels`, inside the camera loop after `universe` is computed:

```python
            for index, spec in enumerate(cam.detectors):
                if spec.type != "onnx" or spec.classes is None:
                    continue
                model_name = spec.model or DEFAULT_MODEL
                labels = per_model.get(model_name, [])
                unknown = [c for c in spec.classes if c not in labels]
                if unknown:
                    raise ValueError(
                        unknown_labels_message(
                            cam.name, f"detectors[{index}] classes", unknown,
                            {model_name: labels},
                        )
                    )
```

(`DEFAULT_MODEL` is already imported in `config.py` from `model_registry`; if not, import it.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_config_labels.py tests/test_onnx_registry.py tests/test_config_detection.py tests/test_detector_integration.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/rtsp_warden/detectors/registry.py src/rtsp_warden/config.py tests/
git commit -m "feat(detection): per-slot classes on onnx detectors, validated per model"
```

---

### Task 5: Detection panel — `when` select and `classes` field

**Files:**
- Modify: `src/rtsp_warden/web/routes/detection.py` (two routes after `set_detector_fps`)
- Modify: `src/rtsp_warden/web/services/detection.py` (`detector_rows`: `classes_text`; `CLASS_CATEGORIES`)
- Modify: `src/rtsp_warden/web/templates/partials/detector_list.html`
- Test: `tests/test_detection_panel.py`, `tests/test_detection_service.py`

**Interfaces:**
- Consumes: `_persist_detector_entry(config_path, camera, index, patch, expected_type=)`, `_try_rebuild_detectors`, `_detector_list_response`, `load_descriptor`, `load_labels`, `ModelError`.
- Produces: `POST /cameras/{name}/detectors/{index}/when` (form `when`), `POST /cameras/{name}/detectors/{index}/classes` (form `classes`, comma-separated, empty clears).

- [ ] **Step 1: Write the failing tests**

`tests/test_detection_panel.py` (the `env` fixture's `yard` camera has detectors 0,1 motion and 2,3 onnx; `yolox-s` labels are COCO, read from the package since `models_dir` is empty):

```python
def test_set_when_patches_only_that_key(env: SimpleNamespace) -> None:
    before = _raw_camera(env.path)
    r = _post(env, "/cameras/yard/detectors/2/when", {"when": "night"})
    assert r.status_code == 200
    assert env.cfg.cameras[0].detectors[2].when == "night"
    after = _raw_camera(env.path)
    assert after["detectors"][2] == {**before["detectors"][2], "when": "night"}
    assert after["detectors"][1]["note"] == "an unknown key that must survive"
    assert "${T15_USER}:${T15_PASS}" in env.path.read_text(encoding="utf-8")
    env.runtime.rebuild_camera_detectors.assert_called_once_with("yard")
    assert '<option value="night" selected' in r.text


def test_set_when_rejects_unknown_values_and_mismatched_type(env: SimpleNamespace) -> None:
    assert _post(env, "/cameras/yard/detectors/2/when", {"when": "dusk"}).status_code == 422
    raw = yaml.safe_load(env.path.read_text(encoding="utf-8"))
    del raw["cameras"][0]["detectors"][0]
    env.path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    assert _post(env, "/cameras/yard/detectors/1/when", {"when": "night"}).status_code == 409
    env.runtime.rebuild_camera_detectors.assert_not_called()


def test_set_classes_validates_against_the_models_labels(env: SimpleNamespace) -> None:
    r = _post(env, "/cameras/yard/detectors/2/classes", {"classes": "person, car ,truck"})
    assert r.status_code == 200
    assert env.cfg.cameras[0].detectors[2].classes == ["person", "car", "truck"]
    assert _raw_camera(env.path)["detectors"][2]["classes"] == ["person", "car", "truck"]
    assert 'value="person, car, truck"' in r.text
    r = _post(env, "/cameras/yard/detectors/2/classes", {"classes": "person, unicorn"})
    assert r.status_code == 422 and "unicorn" in r.text
    assert env.cfg.cameras[0].detectors[2].classes == ["person", "car", "truck"]
    r = _post(env, "/cameras/yard/detectors/2/classes", {"classes": ""})
    assert r.status_code == 200
    assert env.cfg.cameras[0].detectors[2].classes is None
    assert _raw_camera(env.path)["detectors"][2]["classes"] is None


def test_set_classes_is_only_for_onnx_rows(env: SimpleNamespace) -> None:
    assert _post(env, "/cameras/yard/detectors/0/classes", {"classes": "person"}).status_code == 422
    r = env.client.get("/cameras/yard/detectors")
    assert r.text.count('/classes"') == 2  # one form per onnx row
    assert r.text.count('/when"') == 4  # every row
```

`tests/test_detection_service.py`:

```python
def test_class_groups_put_wildlife_labels_with_the_critters() -> None:
    groups = dict(class_groups_for(["fox", "raccoon", "cat", "skunk", "person", "toaster"]))
    assert groups["pet"] == ["cat"]
    assert groups["critter"] == ["fox", "raccoon", "skunk"]
    assert groups["other"] == ["toaster"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_detection_panel.py tests/test_detection_service.py -q -k "when or classes or groups"`
Expected: FAIL with 404/405 on the new routes and a `KeyError`/wrong group.

- [ ] **Step 3: Implement**

`web/services/detection.py`: extend `CLASS_CATEGORIES["critter"]` with
`"fox", "raccoon", "skunk", "opossum", "squirrel", "rabbit", "coyote", "bobcat", "deer", "chipmunk", "woodchuck"`
and `CLASS_CATEGORIES["vehicle"]` with `"vehicle"`. In `detector_rows` add
`"classes": list(spec.classes) if spec.classes is not None else None` and
`"classes_text": ", ".join(spec.classes) if spec.classes else ""`.

`web/routes/detection.py`, after `set_detector_fps`:

```python
_WHEN_VALUES = ("always", "day", "night")


async def _patch_detector_key(
    request: Request, name: str, index: int, patch: dict[str, Any]
) -> str | None:
    """Patch one key of detectors[index] in config.yaml; 409 when the entry changed under us."""
    cfg = get_cfg(request)
    cam = _camera_or_404(cfg, name)
    spec = cam.detectors[index]
    config_path = get_config_path(request)
    if config_path is None:
        return None
    try:
        written = await run_in_threadpool(
            _persist_detector_entry, config_path, name, index, patch, expected_type=spec.type
        )
    except OSError as exc:
        log.warning("camera %s: detector #%d %s not saved: %s", name, index, list(patch), exc)
        return write_failed_message(config_path, exc)
    if not written:
        raise HTTPException(
            status_code=409,
            detail=(
                f"config.yaml no longer has a {spec.type} detector at index {index} "
                f"for camera {name!r}; restart rtsp-warden to load the edited file"
            ),
        )
    return None


@router.post("/{name}/detectors/{index}/when", response_model=None)
async def set_detector_when(
    request: Request, name: str, index: int, user: CurrentUser = Depends(require_admin)
) -> Response:
    """Set one detector's ``when`` (always | day | night); admin-only, hot reload."""
    cfg = get_cfg(request)
    cam = _camera_or_404(cfg, name)
    if not 0 <= index < len(cam.detectors):
        raise HTTPException(status_code=404, detail=f"Camera {name!r} has no detector #{index}")
    form = await request.form()
    when = str(form.get("when", "")).strip()
    if when not in _WHEN_VALUES:
        raise HTTPException(status_code=422, detail="when must be always, day or night")
    message = await _patch_detector_key(request, name, index, {"when": when})
    cam.detectors[index].when = when  # type: ignore[assignment]
    await run_in_threadpool(_try_rebuild_detectors, request, name)
    if is_htmx(request):
        return _detector_list_response(request, cfg, cam, user, message)
    return RedirectResponse(url=f"/cameras/{name}", status_code=303)


@router.post("/{name}/detectors/{index}/classes", response_model=None)
async def set_detector_classes(
    request: Request, name: str, index: int, user: CurrentUser = Depends(require_admin)
) -> Response:
    """Set the labels one onnx detector may report (comma-separated; empty = all)."""
    cfg = get_cfg(request)
    cam = _camera_or_404(cfg, name)
    if not 0 <= index < len(cam.detectors):
        raise HTTPException(status_code=404, detail=f"Camera {name!r} has no detector #{index}")
    spec = cam.detectors[index]
    if spec.type != "onnx":
        raise HTTPException(status_code=422, detail="classes applies to onnx detectors only")
    form = await request.form()
    wanted = [c.strip() for c in str(form.get("classes", "")).split(",") if c.strip()]
    classes: list[str] | None = wanted or None
    if classes is not None:
        model_name = spec.model or DEFAULT_MODEL
        try:
            labels = load_labels(load_descriptor(model_name, cfg.runtime.models_dir))
        except ModelError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None
        unknown = [c for c in classes if c not in labels]
        if unknown:
            raise HTTPException(
                status_code=422,
                detail=unknown_labels_message(name, "classes", unknown, {model_name: labels}),
            )
    message = await _patch_detector_key(request, name, index, {"classes": classes})
    spec.classes = classes
    await run_in_threadpool(_try_rebuild_detectors, request, name)
    if is_htmx(request):
        return _detector_list_response(request, cfg, cam, user, message)
    return RedirectResponse(url=f"/cameras/{name}", status_code=303)
```

Imports needed in the route module: `DEFAULT_MODEL, ModelError, load_descriptor, load_labels, unknown_labels_message` from `...detectors.model_registry`.

`detector_list.html`: add a `<th>When</th>` after `Fps` and, per row:

```jinja
        <td>
          {% if is_admin %}
          <form method="POST" action="/cameras/{{ camera_name }}/detectors/{{ d.index }}/when"
                hx-post="/cameras/{{ camera_name }}/detectors/{{ d.index }}/when"
                hx-target="#detector-list" hx-swap="innerHTML" hx-trigger="change">
            <input type="hidden" name="csrf_token"
                   value="{{ request.state.csrf_token if request.state.csrf_token else '' }}">
            <select name="when" aria-label="When detector {{ d.index }} runs">
              {% for value in ('always', 'day', 'night') %}
              <option value="{{ value }}"{% if d.when == value %} selected{% endif %}>{{ value }}</option>
              {% endfor %}
            </select>
          </form>
          {% else %}{{ d.when }}{% endif %}
        </td>
```

and in the Settings cell, for `d.type == 'onnx'` and `is_admin`:

```jinja
          {% if d.type == 'onnx' and is_admin %}
          <form class="detector-classes-form" method="POST"
                action="/cameras/{{ camera_name }}/detectors/{{ d.index }}/classes"
                hx-post="/cameras/{{ camera_name }}/detectors/{{ d.index }}/classes"
                hx-target="#detector-list" hx-swap="innerHTML">
            <input type="hidden" name="csrf_token"
                   value="{{ request.state.csrf_token if request.state.csrf_token else '' }}">
            <input type="text" name="classes" value="{{ d.classes_text }}"
                   placeholder="all labels"
                   aria-label="Labels detector {{ d.index }} may report (comma-separated)">
            <button type="submit" class="outline secondary">Set</button>
          </form>
          {% elif d.type == 'onnx' and d.classes_text %}<br><small>classes: {{ d.classes_text }}</small>{% endif %}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_detection_panel.py tests/test_detection_service.py tests/test_web_detectors.py tests/test_ui_shell.py tests/test_deploy_docs.py -q`
Expected: all pass (`test_deploy_docs.py` checks the README route table; add the two new routes there if it lists per-detector routes).

- [ ] **Step 5: Commit**

```bash
git add src/rtsp_warden/web/ tests/
git commit -m "feat(web): when and classes controls on the detector table"
```

---

### Task 6: `night` badge on events

**Files:**
- Modify: `src/rtsp_warden/web/services/events.py` (`event_to_dict`)
- Modify: `src/rtsp_warden/web/templates/partials/event_card.html`, `src/rtsp_warden/web/templates/events/detail.html`
- Test: `tests/test_events_ui.py`

**Interfaces:**
- Produces: `event_to_dict(...)["night"]: bool | None`.

- [ ] **Step 1: Write the failing tests**

```python
def test_event_dict_reads_the_night_flag(clean_db: None, tmp_path: Path, cfg: AppConfig) -> None:
    night_id = insert_event(camera_name="front", event_type="object", label="raccoon",
                            confidence=0.8, metadata={"night": True}, created_at=T0)
    day_id = insert_event(camera_name="front", event_type="object", label="cat",
                          confidence=0.8, metadata={"night": False}, created_at=T0)
    rows = {e["id"]: e for e in svc.list_events(cfg, limit=10)}
    assert rows[night_id]["night"] is True and rows[day_id]["night"] is False


def test_old_rows_without_night_render_without_badge(client: TestClient, rec_dir: Path) -> None:
    """(review focus) Rows from before 1.4.0 have no night key: no badge, no error."""
    _seed(rec_dir, label="person")
    r = client.get("/events")
    assert r.status_code == 200
    assert ">night<" not in r.text


def test_night_events_show_a_badge_on_the_grid_and_the_detail(
    client: TestClient, rec_dir: Path
) -> None:
    event_id = insert_event(camera_name="front", event_type="object", label="fox",
                            confidence=0.7, metadata={"night": True}, created_at=T0)
    r = client.get("/events")
    assert '<span class="event-badge">night</span>' in r.text
    r = client.get(f"/events/{event_id}")
    assert r.status_code == 200
    assert r.text.count('<span class="event-badge">night</span>') >= 1
```

(Check `insert_event`'s signature in `db/schema.py` for the `metadata` keyword and the `list_events` wrapper name in `web/services/events.py`; adjust the calls, not the assertions.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_events_ui.py -q -k night`
Expected: FAIL (`KeyError: 'night'`, missing badge).

- [ ] **Step 3: Implement**

`web/services/events.py`:

```python
def _night_flag(metadata_json: str | None) -> bool | None:
    """The ``night`` flag of an event's metadata, or None when absent or unreadable."""
    if not metadata_json:
        return None
    try:
        data = json.loads(metadata_json)
    except (TypeError, ValueError):
        return None
    value = data.get("night") if isinstance(data, dict) else None
    return value if isinstance(value, bool) else None
# event_to_dict: add
        "night": _night_flag(row.metadata_json),
```

`event_card.html`, after the test badge: `{% if evt.night %}<span class="event-badge">night</span>{% endif %}`.
`events/detail.html`: same after the test badge in the `<h1>`, and a table row
`<tr><th scope="row">Night</th><td>{{ 'yes' if event.night else ('no' if event.night is false else '-') }}</td></tr>` after the Zone row.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_events_ui.py tests/test_db_events.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/rtsp_warden/web/services/events.py src/rtsp_warden/web/templates/ tests/test_events_ui.py
git commit -m "feat(events): night badge from event metadata"
```

---

### Task 7: Training tool scaffold and the pure data functions

**Files:**
- Create: `tools/wildlife/pyproject.toml`, `tools/wildlife/README.md`, `tools/wildlife/wildlife.txt`, `tools/wildlife/wildlife_data.py`, `tools/wildlife/.gitignore`
- Modify: `.gitignore` (add `tools/wildlife/data/`, `tools/wildlife/YOLOX_outputs/`, `tools/wildlife/*.pth`)
- Test: `tests/test_wildlife_tool.py`

**Interfaces:**
- Produces (in `wildlife_data.py`, stdlib + numpy + cv2 only):
  - `LABELS: tuple[str, ...]` (18, spec order), `label_id(name) -> int` (0-based), `COCO_CATEGORIES: list[dict]` (`id = index + 1`)
  - `ENA24_MAP: dict[str, str]`, `map_ena24_category(name: str) -> str` (raises `UnknownCategory` listing the name)
  - `OPEN_IMAGES_MIDS: dict[str, str]` (`/m/01yrx` → `cat`, …), `filter_open_images_rows(rows: Iterable[dict[str, str]], cap_per_class: int) -> dict[str, list[dict]]` (image id → boxes; skips `IsGroupOf=1`, `IsDepiction=1`, caps *images* per class)
  - `voc_box_to_coco(xmin, ymin, xmax, ymax) -> list[float]` (`[x, y, w, h]`), `normalized_box_to_coco(xmin, xmax, ymin, ymax, width, height) -> list[float]`
  - `split_image_ids(ids: Sequence[str], val_fraction: float, seed: int) -> tuple[list[str], list[str]]`
  - `channel_spread(frame_bgr) -> float` and `is_gray(frame_bgr, threshold=4.0) -> bool` (same formula as `rtsp_warden.detectors.daylight`)
  - `grayscale_copy(img_bgr, rng) -> np.ndarray` (gray replicated to 3 channels, brightness 0.4–1.0, Gaussian noise sigma 0–6, clipped uint8)

- [ ] **Step 1: Write the failing tests**

```python
"""tools/wildlife/wildlife_data.py: pure data-prep functions, imported by file path (no torch)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from rtsp_warden.detectors.daylight import channel_spread as runtime_spread

TOOL = Path(__file__).resolve().parents[1] / "tools" / "wildlife"


def _load():
    spec = importlib.util.spec_from_file_location("wildlife_data", TOOL / "wildlife_data.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


wd = _load()


def test_labels_match_the_shipped_file_and_the_spec_order() -> None:
    lines = (TOOL / "wildlife.txt").read_text(encoding="utf-8").split()
    assert lines == list(wd.LABELS)
    assert wd.LABELS[0] == "cat" and wd.LABELS[-1] == "vehicle" and len(wd.LABELS) == 18
    assert wd.label_id("raccoon") == 3
    assert [c["id"] for c in wd.COCO_CATEGORIES] == list(range(1, 19))
    assert wd.COCO_CATEGORIES[2]["name"] == "fox"


@pytest.mark.parametrize(
    ("ena", "label"),
    [("Red Fox", "fox"), ("Grey Fox", "fox"), ("Northern Raccoon", "raccoon"),
     ("Domestic Cat", "cat"), ("White_Tailed_Deer", "deer"), ("Wild Turkey", "bird"),
     ("American Crow", "bird"), ("Chicken", "bird"), ("Eastern Fox Squirrel", "squirrel"),
     ("Eastern Gray Squirrel", "squirrel"), ("Human", "person"), ("Vehicle", "vehicle")],
)
def test_ena24_mapping(ena: str, label: str) -> None:
    assert wd.map_ena24_category(ena) == label


def test_ena24_unknown_category_is_an_error_naming_it() -> None:
    with pytest.raises(wd.UnknownCategory, match="Sasquatch"):
        wd.map_ena24_category("Sasquatch")


def test_open_images_filter_caps_images_and_skips_groups_and_drawings() -> None:
    def row(img: str, mid: str, **extra: str) -> dict[str, str]:
        base = {"ImageID": img, "LabelName": mid, "XMin": "0.1", "XMax": "0.5", "YMin": "0.2",
                "YMax": "0.6", "IsGroupOf": "0", "IsDepiction": "0"}
        return {**base, **extra}

    cat, fox, bus = "/m/01yrx", "/m/0306r", "/m/01bjv"
    rows = [row("a", cat), row("a", cat), row("b", cat), row("c", cat), row("d", fox),
            row("e", fox, IsGroupOf="1"), row("f", fox, IsDepiction="1"), row("g", bus)]
    out = wd.filter_open_images_rows(rows, cap_per_class=2)
    assert sorted(out) == ["a", "b", "d"]
    assert len(out["a"]) == 2 and out["a"][0]["label"] == "cat"
    assert out["d"][0]["label"] == "fox"


def test_box_conversions() -> None:
    assert wd.voc_box_to_coco(10, 20, 50, 80) == [10.0, 20.0, 40.0, 60.0]
    assert wd.normalized_box_to_coco(0.1, 0.5, 0.2, 0.6, 200, 100) == [20.0, 20.0, 80.0, 40.0]


def test_split_is_seeded_and_disjoint() -> None:
    ids = [f"img{i}" for i in range(100)]
    train, val = wd.split_image_ids(ids, 0.1, seed=7)
    assert len(val) == 10 and len(train) == 90 and not set(train) & set(val)
    assert wd.split_image_ids(ids, 0.1, seed=7) == (train, val)
    assert wd.split_image_ids(ids, 0.1, seed=8) != (train, val)


def test_channel_spread_matches_the_runtime_formula() -> None:
    rng = np.random.default_rng(1)
    for _ in range(5):
        frame = rng.integers(0, 256, size=(45, 80, 3), dtype=np.uint8)
        assert wd.channel_spread(frame) == pytest.approx(runtime_spread(frame))
    assert wd.is_gray(np.full((45, 80, 3), 90, dtype=np.uint8)) is True
    assert wd.is_gray(frame) is False


def test_grayscale_copy_is_grey_and_darker_or_equal() -> None:
    rng = np.random.default_rng(3)
    img = rng.integers(0, 256, size=(45, 80, 3), dtype=np.uint8)
    out = wd.grayscale_copy(img, rng)
    assert out.shape == img.shape and out.dtype == np.uint8
    assert wd.channel_spread(out) < 12.0  # grey plus a little noise
    assert out.mean() <= img.mean() + 6.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_wildlife_tool.py -q`
Expected: FAIL (`FileNotFoundError` on `wildlife_data.py`).

- [ ] **Step 3: Write the files**

`tools/wildlife/wildlife.txt`: the 18 labels of the spec, one per line.

`tools/wildlife/wildlife_data.py`:

```python
"""Pure data-prep helpers for the wildlife model (stdlib + NumPy + OpenCV only).

Imported by ``fetch.py``, ``prepare.py``, ``exp.py`` and by ``tests/test_wildlife_tool.py``
(by file path, from the main venv), so this module never imports torch or yolox.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
LABELS: tuple[str, ...] = tuple((HERE / "wildlife.txt").read_text(encoding="utf-8").split())
COCO_CATEGORIES: list[dict] = [
    {"id": i + 1, "name": name, "supercategory": "animal" if name not in ("person", "vehicle") else name}
    for i, name in enumerate(LABELS)
]


class UnknownCategory(ValueError):
    """A source category with no entry in the mapping table."""


def label_id(name: str) -> int:
    return LABELS.index(name)


ENA24_MAP: dict[str, str] = {
    "american black bear": "bear", "american crow": "bird", "bird": "bird", "bobcat": "bobcat",
    "chicken": "bird", "coyote": "coyote", "dog": "dog", "domestic cat": "cat",
    "eastern chipmunk": "chipmunk", "eastern cottontail": "rabbit",
    "eastern fox squirrel": "squirrel", "eastern gray squirrel": "squirrel", "grey fox": "fox",
    "horse": "horse", "human": "person", "northern raccoon": "raccoon", "red fox": "fox",
    "striped skunk": "skunk", "vehicle": "vehicle", "virginia opossum": "opossum",
    "white tailed deer": "deer", "wild turkey": "bird", "woodchuck": "woodchuck",
}


def map_ena24_category(name: str) -> str:
    key = name.strip().lower().replace("_", " ").replace("-", " ")
    try:
        return ENA24_MAP[key]
    except KeyError:
        raise UnknownCategory(f"ENA24 category {name!r} has no label mapping") from None


OPEN_IMAGES_MIDS: dict[str, str] = {
    "/m/01yrx": "cat", "/m/0306r": "fox", "/m/0dq75": "raccoon", "/m/0km7z": "skunk",
    "/m/071qp": "squirrel", "/m/06mf6": "rabbit", "/m/0bt9lr": "dog",
}


def filter_open_images_rows(
    rows: Iterable[dict[str, str]], cap_per_class: int
) -> dict[str, list[dict]]:
    """Boxes of wanted classes grouped by image id, at most ``cap_per_class`` images per label.

    Rows with ``IsGroupOf`` or ``IsDepiction`` set are skipped (crowds and drawings). The
    first ``cap_per_class`` images seen per label are kept; every wanted box of a kept image
    is kept. Boxes stay normalized (``xmin, xmax, ymin, ymax``) until the image size is known.
    """
    kept_images: dict[str, set[str]] = {label: set() for label in OPEN_IMAGES_MIDS.values()}
    out: dict[str, list[dict]] = {}
    pending: dict[str, list[dict]] = {}
    for row in rows:
        label = OPEN_IMAGES_MIDS.get(row.get("LabelName", ""))
        if label is None or row.get("IsGroupOf") == "1" or row.get("IsDepiction") == "1":
            continue
        image_id = row["ImageID"]
        box = {"label": label, "xmin": float(row["XMin"]), "xmax": float(row["XMax"]),
               "ymin": float(row["YMin"]), "ymax": float(row["YMax"])}
        if image_id in out:
            out[image_id].append(box)
            continue
        if len(kept_images[label]) >= cap_per_class:
            pending.setdefault(image_id, []).append(box)
            continue
        kept_images[label].add(image_id)
        out[image_id] = pending.pop(image_id, []) + [box]
    return out


def voc_box_to_coco(xmin: float, ymin: float, xmax: float, ymax: float) -> list[float]:
    return [float(xmin), float(ymin), float(xmax - xmin), float(ymax - ymin)]


def normalized_box_to_coco(
    xmin: float, xmax: float, ymin: float, ymax: float, width: int, height: int
) -> list[float]:
    return [xmin * width, ymin * height, (xmax - xmin) * width, (ymax - ymin) * height]


def split_image_ids(
    ids: Sequence[str], val_fraction: float, seed: int
) -> tuple[list[str], list[str]]:
    order = list(ids)
    random.Random(seed).shuffle(order)
    n_val = int(round(len(order) * val_fraction))
    return sorted(order[n_val:]), sorted(order[:n_val])


_SUBSAMPLE = 4


def channel_spread(frame_bgr: np.ndarray) -> float:
    """Same formula as rtsp_warden.detectors.daylight.channel_spread (pinned by a test)."""
    if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[2] < 3 or frame_bgr.size == 0:
        return 0.0
    small = frame_bgr[::_SUBSAMPLE, ::_SUBSAMPLE, :3]
    if small.size == 0:
        return 0.0
    spread = small.max(axis=2).astype(np.int16) - small.min(axis=2).astype(np.int16)
    return float(spread.mean())


def is_gray(frame_bgr: np.ndarray, threshold: float = 4.0) -> bool:
    return channel_spread(frame_bgr) < threshold


def grayscale_copy(img_bgr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """IR look-alike: gray x3 channels, brightness 0.4-1.0, Gaussian noise sigma 0-6."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray *= float(rng.uniform(0.4, 1.0))
    gray += rng.normal(0.0, float(rng.uniform(0.0, 6.0)), size=gray.shape).astype(np.float32)
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)
```

`tools/wildlife/pyproject.toml`:

```toml
[project]
name = "rtsp-warden-wildlife"
version = "0.1.0"
description = "Trains the wildlife-yolox-s model for rtsp-warden (not part of the package)."
requires-python = ">=3.11,<3.12"
dependencies = [
  "torch>=2.6",
  "torchvision>=0.21",
  "yolox @ git+https://github.com/Megvii-BaseDetection/YOLOX.git@6ddff4824372906469a7fae2dc3206c7aa4bbaee",
  "pycocotools>=2.0.8",
  "onnx>=1.17",
  "onnxruntime>=1.20",
  "opencv-python-headless>=4.10",
  "numpy>=1.26",
  "httpx>=0.28",
  "pyyaml>=6",
  "tqdm>=4.66",
]

[tool.uv]
package = false

[tool.uv.sources]
torch = [{ index = "pytorch-cu128" }]
torchvision = [{ index = "pytorch-cu128" }]

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true
```

If `uv sync` in `tools/wildlife` fails on YOLOX's pinned `onnx-simplifier==0.4.10`, use the fallback documented in the README: `git clone --depth 1 https://github.com/Megvii-BaseDetection/YOLOX.git .yolox && git -C .yolox checkout 6ddff48`, remove the `yolox @ git+...` line, and add `.yolox` to `PYTHONPATH` in `train.sh` plus `loguru`, `tabulate`, `thop`, `ninja`, `psutil`, `tensorboard` to the dependencies.

`tools/wildlife/.gitignore`: `data/`, `YOLOX_outputs/`, `.yolox/`, `*.pth`, `*.onnx`.
Root `.gitignore`: append `tools/wildlife/data/`, `tools/wildlife/YOLOX_outputs/`, `tools/wildlife/.yolox/`.

`tools/wildlife/README.md`: the commands in order (`uv sync`, `uv run python fetch.py`, `uv run python prepare.py`, `./train.sh`, `uv run python evaluate.py`, `uv run python export.py`, then from the repo root `uv run python tools/wildlife/verify.py tools/wildlife/out/wildlife-yolox-s`), the data licenses table from the spec, the Roboflow manual step, disk and time estimates (about 6 GB, 1–3 h on a 16 GB GPU), and the fallback above.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_wildlife_tool.py -q`
Expected: all pass. Also `uv run ruff check tools/ && uv run ruff format --check tools/` (informal; keep it clean).

- [ ] **Step 5: Commit**

```bash
git add tools/wildlife/pyproject.toml tools/wildlife/README.md tools/wildlife/wildlife.txt tools/wildlife/wildlife_data.py tools/wildlife/.gitignore .gitignore tests/test_wildlife_tool.py
git commit -m "feat(tools): wildlife training project scaffold with tested data helpers"
```

---

### Task 8: `fetch.py` — download the sources

**Files:**
- Create: `tools/wildlife/fetch.py`
- Test: `tests/test_wildlife_tool.py` (URL table only)

**Interfaces:**
- Consumes: `OPEN_IMAGES_MIDS`, `filter_open_images_rows` from Task 7.
- Produces: `data/raw/ena24/ena24.json`, `data/raw/ena24/images/*.jpg` (unzipped), `data/raw/openimages/boxes.json` (`{image_id: [box...]}` with normalized boxes) and `data/raw/openimages/images/<id>.jpg`, `data/raw/raccoon_dataset-master/{images,annotations}`. Constants `ENA24_IMAGES_URL`, `ENA24_JSON_URL`, `OI_BOXES_CSV_URL`, `OI_IMAGE_URL` (format string), `RACCOON_ZIP_URL`.

- [ ] **Step 1: Write the failing test**

```python
def test_fetch_urls_are_the_verified_mirrors() -> None:
    spec = importlib.util.spec_from_file_location("fetch", TOOL / "fetch.py")
    fetch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fetch)
    assert fetch.ENA24_IMAGES_URL == "https://storage.googleapis.com/public-datasets-lila/ena24/ena24.zip"
    assert fetch.ENA24_JSON_URL == "https://storage.googleapis.com/public-datasets-lila/ena24/ena24.json"
    assert fetch.OI_BOXES_CSV_URL == "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv"
    assert fetch.OI_IMAGE_URL.format(image_id="abc") == "https://open-images-dataset.s3.amazonaws.com/train/abc.jpg"
    assert fetch.RACCOON_ZIP_URL == "https://github.com/datitran/raccoon_dataset/archive/refs/heads/master.zip"
```

(`fetch.py` must import only stdlib, httpx, tqdm and `wildlife_data` at module level, and run nothing on import, so the main venv can import it: guard httpx/tqdm imports inside functions or `try`.)

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_wildlife_tool.py -q -k fetch`
Expected: FAIL (`FileNotFoundError`).

- [ ] **Step 3: Write `fetch.py`**

```python
"""Download the wildlife training sources into data/raw/ (resumable; skips existing files).

    uv run python fetch.py [--cap 1500] [--workers 16] [--only ena24|openimages|raccoon]
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from wildlife_data import filter_open_images_rows  # noqa: E402

DATA = HERE / "data"
RAW = DATA / "raw"
ENA24_IMAGES_URL = "https://storage.googleapis.com/public-datasets-lila/ena24/ena24.zip"
ENA24_JSON_URL = "https://storage.googleapis.com/public-datasets-lila/ena24/ena24.json"
OI_BOXES_CSV_URL = "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv"
OI_IMAGE_URL = "https://open-images-dataset.s3.amazonaws.com/train/{image_id}.jpg"
RACCOON_ZIP_URL = "https://github.com/datitran/raccoon_dataset/archive/refs/heads/master.zip"


def download(url: str, dest: Path, *, desc: str | None = None) -> Path:
    """Stream ``url`` to ``dest`` (via ``dest.part``), resuming with a Range header."""
    import httpx
    from tqdm import tqdm

    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    with httpx.stream("GET", url, headers=headers, follow_redirects=True, timeout=120) as r:
        if r.status_code == 416:  # already complete
            part.rename(dest)
            return dest
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0)) + have
        mode = "ab" if r.status_code == 206 else "wb"
        with part.open(mode) as fh, tqdm(total=total, initial=have, unit="B", unit_scale=True,
                                        desc=desc or dest.name) as bar:
            for chunk in r.iter_bytes(1 << 20):
                fh.write(chunk)
                bar.update(len(chunk))
    part.rename(dest)
    return dest


def fetch_ena24() -> None:
    out = RAW / "ena24"
    download(ENA24_JSON_URL, out / "ena24.json")
    zip_path = download(ENA24_IMAGES_URL, out / "ena24.zip", desc="ena24.zip (3.6 GB)")
    images = out / "images"
    if not images.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(out / "unzipped")
        # the zip holds one top-level folder of jpgs; normalise to images/
        top = next(p for p in (out / "unzipped").iterdir() if p.is_dir())
        top.rename(images)
        (out / "unzipped").rmdir()


def fetch_openimages(cap: int, workers: int) -> None:
    import httpx

    out = RAW / "openimages"
    boxes_path = out / "boxes.json"
    if not boxes_path.exists():
        csv_path = download(OI_BOXES_CSV_URL, out / "oidv6-train-annotations-bbox.csv",
                            desc="open images boxes csv (2.3 GB)")
        with csv_path.open(newline="", encoding="utf-8") as fh:
            boxes = filter_open_images_rows(csv.DictReader(fh), cap_per_class=cap)
        boxes_path.write_text(json.dumps(boxes), encoding="utf-8")
    boxes = json.loads(boxes_path.read_text(encoding="utf-8"))
    images = out / "images"
    images.mkdir(parents=True, exist_ok=True)
    todo = [i for i in boxes if not (images / f"{i}.jpg").exists()]
    print(f"open images: {len(boxes)} images, {len(todo)} to download")

    def one(image_id: str) -> None:
        with httpx.Client(timeout=60, follow_redirects=True) as client:
            r = client.get(OI_IMAGE_URL.format(image_id=image_id))
            if r.status_code == 200:
                (images / f"{image_id}.jpg").write_bytes(r.content)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in as_completed([pool.submit(one, i) for i in todo]):
            pass


def fetch_raccoon() -> None:
    out = RAW / "raccoon_dataset-master"
    if out.exists():
        return
    import httpx

    r = httpx.get(RACCOON_ZIP_URL, follow_redirects=True, timeout=120)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        zf.extractall(RAW)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cap", type=int, default=1500, help="Open Images: images per class")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--only", choices=["ena24", "openimages", "raccoon"])
    args = ap.parse_args()
    if args.only in (None, "raccoon"):
        fetch_raccoon()
    if args.only in (None, "ena24"):
        fetch_ena24()
    if args.only in (None, "openimages"):
        fetch_openimages(args.cap, args.workers)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test, then the download in the background**

Run: `uv run pytest tests/test_wildlife_tool.py -q` → all pass.
Then from `tools/wildlife/`: `uv sync` (first time; falls back as the README says if YOLOX fails to install), then `uv run python fetch.py` as a background job with its log in the scratchpad; check the ENA24 zip layout once unzipped (`ls data/raw/ena24/images | head`) and fix `fetch_ena24`'s folder normalisation if the zip layout differs.

- [ ] **Step 5: Commit**

```bash
git add tools/wildlife/fetch.py tests/test_wildlife_tool.py
git commit -m "feat(tools): fetch ENA24, Open Images and the raccoon set"
```

---

### Task 9: `prepare.py` — one COCO dataset with the wildlife labels

**Files:**
- Create: `tools/wildlife/prepare.py`
- Test: `tests/test_wildlife_tool.py` (merge logic through a pure function)

**Interfaces:**
- Consumes: Task 7 helpers; Task 8 outputs.
- Produces: `data/coco/annotations/train.json`, `data/coco/annotations/val.json` (COCO; categories `COCO_CATEGORIES`; images carry `source` and, on val, `is_gray`), `data/coco/train2017/<name>.jpg` and `data/coco/val2017/<name>.jpg` as symlinks. Pure function `build_records(ena24_json, oi_boxes, oi_sizes, raccoon_xmls, roboflow_json=None) -> list[Record]` where `Record = {"file": Path, "width": int, "height": int, "source": str, "boxes": [(label, [x, y, w, h]), ...]}`, and `to_coco(records, ids) -> dict`.

- [ ] **Step 1: Write the failing test**

```python
def test_build_records_and_to_coco_use_the_wildlife_categories(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location("prepare", TOOL / "prepare.py")
    prep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prep)
    ena = {
        "images": [{"id": "1", "file_name": "1.jpg", "width": 1920, "height": 1080}],
        "annotations": [{"image_id": "1", "category_id": 5, "bbox": [10, 20, 30, 40]}],
        "categories": [{"id": 5, "name": "Northern Raccoon"}],
    }
    oi_boxes = {"img": [{"label": "fox", "xmin": 0.0, "xmax": 0.5, "ymin": 0.0, "ymax": 0.5}]}
    oi_sizes = {"img": (200, 100)}
    records = prep.build_records(ena, tmp_path / "ena", oi_boxes, oi_sizes, tmp_path / "oi",
                                 raccoon_xmls=[])
    assert [r["source"] for r in records] == ["ena24", "openimages"]
    assert records[0]["boxes"] == [("raccoon", [10.0, 20.0, 30.0, 40.0])]
    assert records[1]["boxes"] == [("fox", [0.0, 0.0, 100.0, 50.0])]
    coco = prep.to_coco(records, {"ena24/1.jpg": True, "openimages/img.jpg": False})
    assert [c["name"] for c in coco["categories"]][:4] == ["cat", "dog", "fox", "raccoon"]
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    assert [cats[a["category_id"]] for a in coco["annotations"]] == ["raccoon", "fox"]
    assert coco["images"][0]["is_gray"] is True and coco["images"][1]["is_gray"] is False
    assert all(a["iscrowd"] == 0 and a["area"] > 0 for a in coco["annotations"])
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_wildlife_tool.py -q -k prepare` → FAIL.

- [ ] **Step 3: Write `prepare.py`**

```python
"""Merge the raw sources into data/coco/{annotations,train2017,val2017} for YOLOX.

    uv run python prepare.py [--val-fraction 0.1] [--seed 20261004]
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from wildlife_data import (  # noqa: E402
    COCO_CATEGORIES,
    LABELS,
    is_gray,
    map_ena24_category,
    normalized_box_to_coco,
    split_image_ids,
    voc_box_to_coco,
)

DATA = HERE / "data"
RAW = DATA / "raw"
COCO = DATA / "coco"


def build_records(
    ena_json: dict,
    ena_images: Path,
    oi_boxes: dict[str, list[dict]],
    oi_sizes: dict[str, tuple[int, int]],
    oi_images: Path,
    raccoon_xmls: list[Path],
    roboflow: tuple[dict, Path] | None = None,
) -> list[dict]:
    records: list[dict] = []
    cats = {c["id"]: c["name"] for c in ena_json["categories"]}
    by_image: dict[str, list] = {}
    for ann in ena_json["annotations"]:
        if "bbox" in ann:
            by_image.setdefault(str(ann["image_id"]), []).append(ann)
    for img in ena_json["images"]:
        anns = by_image.get(str(img["id"]))
        if not anns:
            continue
        boxes = [(map_ena24_category(cats[a["category_id"]]), [float(v) for v in a["bbox"]])
                 for a in anns]
        records.append({"file": ena_images / img["file_name"], "width": img["width"],
                        "height": img["height"], "source": "ena24", "boxes": boxes})
    for image_id, boxes in oi_boxes.items():
        size = oi_sizes.get(image_id)
        if size is None:
            continue
        w, h = size
        records.append({"file": oi_images / f"{image_id}.jpg", "width": w, "height": h,
                        "source": "openimages",
                        "boxes": [(b["label"], normalized_box_to_coco(b["xmin"], b["xmax"],
                                                                     b["ymin"], b["ymax"], w, h))
                                  for b in boxes]})
    for xml_path in raccoon_xmls:
        root = ET.parse(xml_path).getroot()
        size = root.find("size")
        w, h = int(size.find("width").text), int(size.find("height").text)
        boxes = []
        for obj in root.findall("object"):
            bb = obj.find("bndbox")
            boxes.append(("raccoon", voc_box_to_coco(*(float(bb.find(k).text)
                                                        for k in ("xmin", "ymin", "xmax", "ymax")))))
        records.append({"file": xml_path.parent.parent / "images" / root.find("filename").text,
                        "width": w, "height": h, "source": "raccoon", "boxes": boxes})
    if roboflow is not None:
        rf_json, rf_dir = roboflow
        rf_cats = {c["id"]: c["name"].strip().lower() for c in rf_json["categories"]}
        rf_by_image: dict[int, list] = {}
        for ann in rf_json["annotations"]:
            rf_by_image.setdefault(ann["image_id"], []).append(ann)
        for img in rf_json["images"]:
            boxes = [(rf_cats[a["category_id"]], [float(v) for v in a["bbox"]])
                     for a in rf_by_image.get(img["id"], []) if rf_cats[a["category_id"]] in LABELS]
            if boxes:
                records.append({"file": rf_dir / img["file_name"], "width": img["width"],
                                "height": img["height"], "source": "roboflow", "boxes": boxes})
    return records


def record_key(rec: dict) -> str:
    return f"{rec['source']}/{Path(rec['file']).name}"


def to_coco(records: list[dict], gray_flags: dict[str, bool]) -> dict:
    cat_id = {c["name"]: c["id"] for c in COCO_CATEGORIES}
    images, annotations = [], []
    for i, rec in enumerate(records, start=1):
        key = record_key(rec)
        entry = {"id": i, "file_name": key.replace("/", "__"), "width": rec["width"],
                 "height": rec["height"], "source": rec["source"]}
        if key in gray_flags:
            entry["is_gray"] = bool(gray_flags[key])
        images.append(entry)
        for label, (x, y, w, h) in rec["boxes"]:
            if w <= 0 or h <= 0:
                continue
            annotations.append({"id": len(annotations) + 1, "image_id": i,
                                "category_id": cat_id[label], "bbox": [x, y, w, h],
                                "area": w * h, "iscrowd": 0})
    return {"images": images, "annotations": annotations, "categories": COCO_CATEGORIES}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=20261004)
    args = ap.parse_args()

    ena_json = json.loads((RAW / "ena24" / "ena24.json").read_text(encoding="utf-8"))
    oi_boxes = json.loads((RAW / "openimages" / "boxes.json").read_text(encoding="utf-8"))
    oi_images = RAW / "openimages" / "images"
    oi_sizes: dict[str, tuple[int, int]] = {}
    for image_id in oi_boxes:
        img = cv2.imread(str(oi_images / f"{image_id}.jpg"))
        if img is not None:
            oi_sizes[image_id] = (img.shape[1], img.shape[0])
    raccoon_xmls = sorted((RAW / "raccoon_dataset-master" / "annotations").glob("*.xml"))
    roboflow = None
    rf_dir = RAW / "roboflow-cat-raccoons"
    if rf_dir.exists():
        merged = {"images": [], "annotations": [], "categories": None}
        for split in ("train", "valid", "test"):
            ann = rf_dir / split / "_annotations.coco.json"
            if ann.exists():
                part = json.loads(ann.read_text(encoding="utf-8"))
                offset = len(merged["images"])
                for img in part["images"]:
                    img["id"] += offset
                    img["file_name"] = f"{split}/{img['file_name']}"
                for a in part["annotations"]:
                    a["image_id"] += offset
                merged["images"] += part["images"]
                merged["annotations"] += part["annotations"]
                merged["categories"] = part["categories"]
        if merged["categories"]:
            roboflow = (merged, rf_dir)

    records = [r for r in build_records(ena_json, RAW / "ena24" / "images", oi_boxes, oi_sizes,
                                        oi_images, raccoon_xmls, roboflow)
               if Path(r["file"]).exists()]
    keys = [record_key(r) for r in records]
    train_keys, val_keys = split_image_ids(keys, args.val_fraction, args.seed)
    by_key = {record_key(r): r for r in records}
    gray = {}
    for key in val_keys:
        img = cv2.imread(str(by_key[key]["file"]))
        gray[key] = bool(img is not None and is_gray(img))

    for split, split_keys in (("train", train_keys), ("val", val_keys)):
        recs = [by_key[k] for k in split_keys]
        coco = to_coco(recs, gray)
        img_dir = COCO / ("train2017" if split == "train" else "val2017")
        img_dir.mkdir(parents=True, exist_ok=True)
        for entry, rec in zip(coco["images"], recs, strict=True):
            link = img_dir / entry["file_name"]
            if not link.exists():
                link.symlink_to(Path(rec["file"]).resolve())
        (COCO / "annotations").mkdir(parents=True, exist_ok=True)
        (COCO / "annotations" / f"{split}.json").write_text(json.dumps(coco), encoding="utf-8")
        counts = Counter(LABELS[a["category_id"] - 1] for a in coco["annotations"])
        print(f"{split}: {len(coco['images'])} images, {len(coco['annotations'])} boxes")
        for label in LABELS:
            print(f"  {counts.get(label, 0):6d}  {label}")
    print(f"val grayscale images: {sum(gray.values())} of {len(gray)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test, then prepare the data**

`uv run pytest tests/test_wildlife_tool.py -q` → pass. Then in `tools/wildlife/`: `uv run python prepare.py` once Task 8's downloads are complete; keep the printed per-class table for HANDOFF (raccoon and fox counts matter). Spot-check two symlinked images open with `cv2.imread`.

- [ ] **Step 5: Commit**

```bash
git add tools/wildlife/prepare.py tests/test_wildlife_tool.py
git commit -m "feat(tools): merge the wildlife sources into one COCO dataset"
```

---

### Task 10: `exp.py`, `train.py`, `train.sh`, `evaluate.py` — fine-tune and measure

**Files:**
- Create: `tools/wildlife/exp.py`, `tools/wildlife/train.py`, `tools/wildlife/train.sh`, `tools/wildlife/evaluate.py`
- Test: none in the main suite (torch); `exp.py` is exercised by the training run.

**Interfaces:**
- Consumes: `data/coco/` from Task 9; `grayscale_copy`, `is_gray`, `LABELS` from Task 7.
- Produces: `YOLOX_outputs/wildlife_yolox_s/best_ckpt.pth`; `evaluate.py` prints and writes `out/eval.json` with AP50 per class for `all`, `colour`, `gray`.

- [ ] **Step 1: Write `exp.py`**

```python
"""YOLOX-S experiment for the wildlife model: 18 classes, grayscale augmentation."""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import numpy as np
from yolox.data import COCODataset, TrainTransform
from yolox.exp import Exp as BaseExp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from wildlife_data import LABELS, grayscale_copy, is_gray  # noqa: E402


class GrayAugCOCODataset(COCODataset):
    """COCODataset whose colour images come back as an IR look-alike half of the time."""

    def __init__(self, *args, gray_prob: float = 0.5, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.gray_prob = gray_prob
        self._rng = np.random.default_rng()

    def load_resized_img(self, index):
        img = super().load_resized_img(index)
        if self.gray_prob > 0 and random.random() < self.gray_prob and not is_gray(img):
            img = grayscale_copy(img, self._rng)
        return img


class Exp(BaseExp):
    def __init__(self) -> None:
        super().__init__()
        self.depth = 0.33
        self.width = 0.50
        self.num_classes = len(LABELS)
        self.data_dir = str(HERE / "data" / "coco")
        self.train_ann = "train.json"
        self.val_ann = "val.json"
        self.input_size = (640, 640)
        self.test_size = (640, 640)
        self.max_epoch = int(os.environ.get("WILDLIFE_EPOCHS", "50"))
        self.no_aug_epochs = 5
        self.warmup_epochs = 2
        self.eval_interval = 5
        self.print_interval = 50
        self.data_num_workers = 6
        self.exp_name = "wildlife_yolox_s"

    def get_dataset(self, cache: bool = False, cache_type: str = "ram"):
        return GrayAugCOCODataset(
            data_dir=self.data_dir,
            json_file=self.train_ann,
            img_size=self.input_size,
            preproc=TrainTransform(max_labels=50, flip_prob=self.flip_prob,
                                   hsv_prob=self.hsv_prob),
            cache=cache,
            cache_type=cache_type,
        )
```

- [ ] **Step 2: Write `train.py` and `train.sh`**

`train.py` mirrors YOLOX's `tools/train.py` without depending on its location:

```python
"""Fine-tune the wildlife YOLOX-S (single GPU).

    uv run python train.py -b 32 --fp16 -c yolox_s.pth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from yolox.core import launch
from yolox.exp import check_exp_value
from yolox.utils import configure_module, configure_nccl, configure_omp, get_num_devices

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from exp import Exp  # noqa: E402


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("wildlife train")
    p.add_argument("-b", "--batch-size", type=int, default=32)
    p.add_argument("-d", "--devices", type=int, default=1)
    p.add_argument("-c", "--ckpt", default=str(HERE / "yolox_s.pth"))
    p.add_argument("--resume", action="store_true")
    p.add_argument("-e", "--start_epoch", type=int, default=None)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--cache", type=str, nargs="?", const="ram")
    p.add_argument("-o", "--occupy", action="store_true")
    p.add_argument("-l", "--logger", default="tensorboard")
    p.add_argument("--dist-backend", default="nccl")
    p.add_argument("--dist-url", default=None)
    p.add_argument("--num_machines", type=int, default=1)
    p.add_argument("--machine_rank", type=int, default=0)
    p.add_argument("opts", nargs=argparse.REMAINDER, default=[])
    return p


def main_worker(exp: Exp, args: argparse.Namespace) -> None:
    configure_nccl()
    configure_omp()
    torch.backends.cudnn.benchmark = True
    trainer = exp.get_trainer(args)
    trainer.train()


if __name__ == "__main__":
    configure_module()
    args = make_parser().parse_args()
    args.experiment_name = None
    args.name = None
    args.exp_file = str(HERE / "exp.py")
    exp = Exp()
    exp.merge(args.opts)
    check_exp_value(exp)
    args.experiment_name = exp.exp_name
    num_gpu = get_num_devices() if args.devices is None else args.devices
    launch(main_worker, num_gpu, args.num_machines, args.machine_rank,
           backend=args.dist_backend, dist_url=args.dist_url, args=(exp, args))
```

`train.sh`:

```bash
#!/usr/bin/env bash
# Fine-tune wildlife-yolox-s on the GPU. Run from tools/wildlife/ after prepare.py.
set -euo pipefail
cd "$(dirname "$0")"
[ -f yolox_s.pth ] || curl -L -o yolox_s.pth \
  https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth
exec uv run python train.py -b "${BATCH:-32}" --fp16 -c yolox_s.pth "$@"
```

If the trainer raises `CUDA out of memory`, rerun with `BATCH=16`.

- [ ] **Step 3: Write `evaluate.py`**

```python
"""AP50 per class on the validation set: all, colour-only and grayscale-only images.

    uv run python evaluate.py [--ckpt YOLOX_outputs/wildlife_yolox_s/best_ckpt.pth]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from yolox.data.data_augment import ValTransform
from yolox.utils import postprocess

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from exp import Exp  # noqa: E402
from wildlife_data import LABELS  # noqa: E402


def run(ckpt: Path, conf: float, nms: float) -> dict:
    exp = Exp()
    model = exp.get_model().cuda().eval()
    model.load_state_dict(torch.load(ckpt, map_location="cuda")["model"])
    coco = COCO(str(Path(exp.data_dir) / "annotations" / exp.val_ann))
    cat_ids = sorted(coco.getCatIds())
    transform = ValTransform(legacy=False)
    results = []
    for img_id in coco.getImgIds():
        info = coco.loadImgs(img_id)[0]
        img = cv2.imread(str(Path(exp.data_dir) / "val2017" / info["file_name"]))
        if img is None:
            continue
        ratio = min(exp.test_size[0] / img.shape[0], exp.test_size[1] / img.shape[1])
        tensor, _ = transform(img, None, exp.test_size)
        with torch.no_grad():
            out = postprocess(model(torch.from_numpy(tensor).unsqueeze(0).cuda()),
                              exp.num_classes, conf, nms, class_agnostic=False)[0]
        if out is None:
            continue
        out = out.cpu().numpy()
        for x1, y1, x2, y2, obj, cls, cls_id in out:
            x1, y1, x2, y2 = (v / ratio for v in (x1, y1, x2, y2))
            results.append({"image_id": img_id, "category_id": cat_ids[int(cls_id)],
                            "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                            "score": float(obj * cls)})
    dets = coco.loadRes(results) if results else None
    report: dict[str, dict[str, float]] = {}
    subsets = {
        "all": coco.getImgIds(),
        "colour": [i for i in coco.getImgIds() if not coco.imgs[i].get("is_gray")],
        "gray": [i for i in coco.getImgIds() if coco.imgs[i].get("is_gray")],
    }
    for name, img_ids in subsets.items():
        if dets is None or not img_ids:
            report[name] = {}
            continue
        ev = COCOeval(coco, dets, "bbox")
        ev.params.imgIds = img_ids
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        precision = ev.eval["precision"]  # [T, R, K, A, M]
        per_class = {}
        for k, cat_id in enumerate(ev.params.catIds):
            p = precision[0, :, k, 0, 2]  # IoU 0.5, all areas, maxDets 100
            p = p[p > -1]
            per_class[LABELS[cat_id - 1]] = float(p.mean()) if p.size else float("nan")
        report[name] = per_class
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path,
                    default=HERE / "YOLOX_outputs" / "wildlife_yolox_s" / "best_ckpt.pth")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--nms", type=float, default=0.65)
    args = ap.parse_args()
    report = run(args.ckpt, args.conf, args.nms)
    (HERE / "out").mkdir(exist_ok=True)
    (HERE / "out" / "eval.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"{'label':10s} {'all':>6s} {'colour':>7s} {'gray':>6s}")
    for label in LABELS:
        row = [report[s].get(label, float("nan")) for s in ("all", "colour", "gray")]
        print(f"{label:10s} {row[0]:6.3f} {row[1]:7.3f} {row[2]:6.3f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Train and evaluate**

From `tools/wildlife/`: `chmod +x train.sh && ./train.sh` as a background job (log to the scratchpad; expect 1–3 h; the compose stack keeps running on the same GPU, so watch `nvidia-smi` memory and drop `BATCH=16` if the two together exceed 16 GB). When it finishes: `uv run python evaluate.py`. Record the table. Soft target: `cat`, `fox`, `raccoon` AP50 >= 0.6 on `gray`. Below target: one more run with `WILDLIFE_EPOCHS=80` (resume with `--resume`), then report the numbers to the owner either way.

- [ ] **Step 5: Commit**

```bash
git add tools/wildlife/exp.py tools/wildlife/train.py tools/wildlife/train.sh tools/wildlife/evaluate.py
git commit -m "feat(tools): YOLOX-S wildlife experiment with grayscale augmentation and split evaluation"
```

---

### Task 11: `export.py` and `verify.py` — the model the registry loads

**Files:**
- Create: `tools/wildlife/export.py`, `tools/wildlife/verify.py`
- Test: `tests/test_wildlife_tool.py` (descriptor writer, pure)

**Interfaces:**
- Produces: `tools/wildlife/out/wildlife-yolox-s/{wildlife_yolox_s.onnx, wildlife.txt, model.yaml}`; `write_descriptor(out_dir: Path, onnx_name: str, sha256: str) -> Path` in `export.py` (pure; stdlib + yaml).

- [ ] **Step 1: Write the failing test**

```python
def test_export_writes_a_descriptor_the_registry_accepts(tmp_path: Path) -> None:
    from rtsp_warden.detectors.model_registry import load_descriptor, load_labels

    spec = importlib.util.spec_from_file_location("export", TOOL / "export.py")
    export = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(export)
    out = tmp_path / "wildlife-yolox-s"
    out.mkdir()
    (out / "wildlife_yolox_s.onnx").write_bytes(b"fake")
    export.write_descriptor(out, "wildlife_yolox_s.onnx", "a" * 64)
    desc = load_descriptor("wildlife-yolox-s", tmp_path)
    assert desc.file == "wildlife_yolox_s.onnx" and desc.input_size == (640, 640)
    assert desc.sha256 == "a" * 64 and desc.postprocess == "yolox"
    assert load_labels(desc) == list(wd.LABELS)
```

(`export.py` imports torch and yolox only inside `main()` so this import works from the main venv.)

- [ ] **Step 2: Run the test to verify it fails** → FAIL (`FileNotFoundError`).

- [ ] **Step 3: Write `export.py` and `verify.py`**

```python
"""Export the trained checkpoint to ONNX (raw YOLOX head) and write model.yaml.

    uv run python export.py [--ckpt ...] [--out out/wildlife-yolox-s]
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
MODEL_NAME = "wildlife-yolox-s"
ONNX_NAME = "wildlife_yolox_s.onnx"


def write_descriptor(out_dir: Path, onnx_name: str, sha256: str) -> Path:
    shutil.copyfile(HERE / "wildlife.txt", out_dir / "wildlife.txt")
    desc = {
        "name": MODEL_NAME,
        "file": onnx_name,
        "labels": "wildlife.txt",
        "input_size": [640, 640],
        "postprocess": "yolox",
        "sha256": sha256,
    }
    path = out_dir / "model.yaml"
    path.write_text(
        "# wildlife-yolox-s: YOLOX-S fine-tuned on ENA24 + Open Images + raccoon sets (RW-5).\n"
        "# Same tensor contract as yolox-s: images [1,3,640,640] BGR 0-255, raw head output.\n"
        + yaml.safe_dump(desc, sort_keys=False),
        encoding="utf-8",
    )
    return path


def main() -> None:
    import onnxruntime as ort
    import torch

    sys.path.insert(0, str(HERE))
    from exp import Exp

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path,
                    default=HERE / "YOLOX_outputs" / "wildlife_yolox_s" / "best_ckpt.pth")
    ap.add_argument("--out", type=Path, default=HERE / "out" / MODEL_NAME)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    exp = Exp()
    model = exp.get_model().eval()
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu")["model"])
    model.head.decode_in_inference = False  # raw rows: the runtime decodes (decode_yolox)
    dummy = torch.randn(1, 3, *exp.test_size)
    onnx_path = args.out / ONNX_NAME
    torch.onnx.export(model, dummy, str(onnx_path), input_names=["images"],
                      output_names=["output"], opset_version=13, dynamo=False)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"images": dummy.numpy()})[0]
    expected = (1, 8400, 5 + exp.num_classes)
    if tuple(out.shape) != expected:
        raise SystemExit(f"ONNX output shape {out.shape}, expected {expected}")
    sha = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
    write_descriptor(args.out, ONNX_NAME, sha)
    print(f"wrote {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB) sha256 {sha}")
    print(f"install: cp -r {args.out} <models_dir>/   (compose: data/models/)")


if __name__ == "__main__":
    main()
```

`verify.py` (run from the repo root with the main venv):

```python
"""Load the exported model through rtsp-warden's own registry and detector, run one image.

    uv run python tools/wildlife/verify.py tools/wildlife/out/wildlife-yolox-s [image.jpg]
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from rtsp_warden.detectors.builtin.onnx import OnnxDetector
from rtsp_warden.detectors.model_registry import load_descriptor


def main() -> None:
    model_dir = Path(sys.argv[1]).resolve()
    desc = load_descriptor(model_dir.name, model_dir.parent)
    det = OnnxDetector(descriptor=desc, models_dir=model_dir.parent, device="cpu",
                       min_confidence=0.3)
    det.setup()
    if det.error:
        raise SystemExit(det.error)
    if len(sys.argv) > 2:
        frame = cv2.imread(sys.argv[2])
    else:
        frame = np.full((720, 1280, 3), 114, dtype=np.uint8)
    found = det.process(frame, 1.0)
    print(f"provider {det.provider}, {len(det.labels)} labels, {len(found)} detections")
    for d in found:
        print(f"  {d.kind:10s} {d.confidence:.2f} {d.bbox}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Export, verify, install**

`uv run pytest tests/test_wildlife_tool.py -q` → pass. From `tools/wildlife/`: `uv run python export.py`. From the repo root: `uv run python tools/wildlife/verify.py tools/wildlife/out/wildlife-yolox-s tools/wildlife/data/coco/val2017/<a raccoon image>` and confirm a `raccoon` detection. Then `cp -r tools/wildlife/out/wildlife-yolox-s data/models/` and edit the live `config/config.yaml` to the spec's section 5.6 camera block (keep the Foscam's `main_url` and `vendor` as they are), and `docker compose -f docker-compose.yml -f docker-compose.gpu.yml restart warden`. Check the Detection panel shows two onnx rows with `CUDA` providers and the night-mode line.

- [ ] **Step 5: Commit**

```bash
git add tools/wildlife/export.py tools/wildlife/verify.py tests/test_wildlife_tool.py
git commit -m "feat(tools): export wildlife-yolox-s to ONNX with a registry descriptor; verify through the runtime"
```

---

### Task 12: Docs, version, gate, handoff

**Files:**
- Modify: `README.md` (Detection section: "Wildlife model" subsection, `when` and `classes` fields, night flag; route table rows for the two new routes if it lists per-detector routes), `CLAUDE.md` (RW-5 notes under the detector pipeline), `pyproject.toml`, `src/rtsp_warden/__init__.py`, `tests/test_admin.py` (1.4.0), `HANDOFF.md`, `TASKS.md`
- Test: `tests/test_deploy_docs.py` (README examples; mark the new README config block only if the built-in descriptor exists, else leave it unmarked)

- [ ] **Step 1: README "Wildlife model" subsection**

Under the detection section add: what the model is, the camera block from the spec section 5.6 (unmarked fenced yaml, since `wildlife-yolox-s` is a user model), where to put the model (`WARDEN_MODELS_DIR`; compose `data/models/`), the three training commands pointing to `tools/wildlife/README.md`, the `when` and `classes` fields, the `night` flag and badge, and the data licenses line. Add `when` and `classes` to the detector field table.

- [ ] **Step 2: Version bump**

`pyproject.toml` and `src/rtsp_warden/__init__.py` to `1.4.0`; update the assertion in `tests/test_admin.py`.

- [ ] **Step 3: Gate**

```bash
uv run ruff check src/ tests/          # exactly the 4 baseline E501
uv run ruff format --check src/ tests/
uv run pytest -q                       # full suite
```

Fix anything the gate reports; rerun until clean.

- [ ] **Step 4: HANDOFF and TASKS**

HANDOFF top block: what shipped, the eval table (AP50 per class, all / colour / gray), the exact model install path, what the owner should look for (two onnx rows on the Detection panel, night mode line, a `fox` / `raccoon` / `cat` event with the right label day and night), and the follow-ups (release asset + built-in descriptor; event relabel button). TASKS.md: RW-5 card status.

- [ ] **Step 5: Commit**

```bash
git add README.md CLAUDE.md pyproject.toml src/rtsp_warden/__init__.py tests/test_admin.py HANDOFF.md TASKS.md
git commit -m "docs: RW-5 wildlife detection shipped; version 1.4.0"
```

Push only when the owner says "push".
