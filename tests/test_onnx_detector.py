"""OnnxDetector: inference on a tiny graph, provider selection, model loading.

Every model here is a tiny constant graph from ``tests/helpers_onnx.py`` run by the
real onnxruntime CPU build. No weights, no network, no GPU.

Worked numbers (input 64x64, labels person/car, 84 anchor rows):
- row 19 = stride 8, cell (gx=3, gy=2): ``[0.5, 0.5, ln 2, ln 2, obj, person, car]``
  decodes to centre (28, 20), size 16x16, input box x1y1x2y2 (20, 12, 36, 28).
- row 69 = stride 16 (level starts at row 64), cell (gx=1, gy=1): ``[0, 0, 0, 0, ...]``
  decodes to centre (16, 16), size 16x16, input box (8, 8, 24, 24).
- A 320x180 frame letterboxes with ratio 0.2 (resized to 64x36), so frame boxes are
  input boxes times 5: row 19 -> xywh (100, 60, 80, 80); row 69 -> (40, 40, 80, 80).
"""

from __future__ import annotations

import math
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from rtsp_warden.detectors.builtin import onnx as onnx_mod
from rtsp_warden.detectors.builtin.onnx import CPU_PROVIDER, CUDA_PROVIDER, OnnxDetector
from rtsp_warden.detectors.model_registry import ModelDescriptor, load_descriptor
from tests.helpers_onnx import write_model_dir, yolox_output

LN2 = math.log(2.0)
INPUT = (64, 64)
PERSON_ROW = 19  # stride 8, gx=3, gy=2
NEXT_ROW = 20  # stride 8, gx=4, gy=2
CAR_ROW = 69  # stride 16, gx=1, gy=1
FRAME_W, FRAME_H = 320, 180


def _frame(bgr: tuple[int, int, int] = (10, 20, 30)) -> np.ndarray:
    frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    frame[:, :] = bgr
    return frame


def _two_objects() -> dict[int, list[float]]:
    return {
        PERSON_ROW: [0.5, 0.5, LN2, LN2, 0.9, 0.8, 0.1],  # person 0.72
        CAR_ROW: [0.0, 0.0, 0.0, 0.0, 0.95, 0.05, 0.9],  # car 0.855
    }


def _detector(tmp_path: Path, rows: dict[int, list[float]], **kwargs: Any) -> OnnxDetector:
    models_dir = tmp_path / "models"
    write_model_dir(models_dir, yolox_output(INPUT, 2, rows))
    return OnnxDetector(
        descriptor=load_descriptor("tiny", models_dir), models_dir=models_dir, **kwargs
    )


def _capture(monkeypatch: pytest.MonkeyPatch, level: str) -> list[str]:
    seen: list[str] = []

    def record(msg: str, *args: object, **_kwargs: object) -> None:
        seen.append(msg % args if args else msg)

    monkeypatch.setattr(onnx_mod.log, level, record)
    return seen


@pytest.fixture
def cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the real onnxruntime look like the CPU build, even on a GPU dev box."""
    import onnxruntime

    monkeypatch.setattr(onnxruntime, "get_available_providers", lambda: [CPU_PROVIDER])
    monkeypatch.setattr(onnx_mod, "_preload_done", True)
    monkeypatch.setattr(onnx_mod, "_both_onnxruntime_packages", lambda: False)


# ---------------------------------------------------------------------------
# OnnxDetector.process on a tiny constant graph (real onnxruntime, CPU)
# ---------------------------------------------------------------------------


def test_process_returns_xywh_in_frame_pixels(tmp_path: Path) -> None:
    det = _detector(tmp_path, _two_objects(), device="cpu")
    det.setup()

    found = det.process(_frame(), 1234.5)

    assert [d.kind for d in found] == ["car", "person"]  # best score first
    assert found[0].bbox == (40, 40, 80, 80)
    assert found[0].confidence == pytest.approx(0.855, abs=1e-5)
    assert found[1].bbox == (100, 60, 80, 80)
    assert found[1].confidence == pytest.approx(0.72, abs=1e-5)
    assert found[1].metadata == {"class_id": 0, "class_name": "person", "model": "tiny"}
    assert all(d.ts_unix == 1234.5 for d in found)
    assert all(isinstance(v, int) for d in found for v in d.bbox)
    assert all(type(d.confidence) is float for d in found)


def test_min_confidence_drops_low_scores(tmp_path: Path) -> None:
    det = _detector(tmp_path, _two_objects(), device="cpu", min_confidence=0.75)
    det.setup()

    assert [d.kind for d in det.process(_frame(), 1.0)] == ["car"]


def test_classes_keep_only_wanted_labels(tmp_path: Path) -> None:
    det = _detector(tmp_path, _two_objects(), device="cpu", classes=["person"])
    det.setup()

    assert [d.kind for d in det.process(_frame(), 1.0)] == ["person"]


def test_class_filter_applies_before_the_best_class_is_picked(tmp_path: Path) -> None:
    # The row's best class is car (0.9); the camera only wants person (0.6).
    rows = {PERSON_ROW: [0.5, 0.5, LN2, LN2, 1.0, 0.6, 0.9]}
    det = _detector(tmp_path, rows, device="cpu", classes=["person"])
    det.setup()

    found = det.process(_frame(), 1.0)

    assert [(d.kind, round(d.confidence, 3)) for d in found] == [("person", 0.6)]


def test_empty_classes_report_nothing(tmp_path: Path) -> None:
    det = _detector(tmp_path, _two_objects(), device="cpu", classes=[], min_confidence=0.0)
    det.setup()

    assert det.process(_frame(), 1.0) == []


def test_nms_drops_same_class_duplicate_and_keeps_other_class(tmp_path: Path) -> None:
    rows = {
        PERSON_ROW: [0.5, 0.5, LN2, LN2, 0.9, 0.8, 0.0],  # person 0.72
        NEXT_ROW: [-0.5, 0.5, LN2, LN2, 0.9, 0.7, 0.0],  # same box, person 0.63
        21: [-1.5, 0.5, LN2, LN2, 0.9, 0.0, 0.9],  # same box, car 0.81
    }
    det = _detector(tmp_path, rows, device="cpu")
    det.setup()

    found = det.process(_frame(), 1.0)

    assert [(d.kind, round(d.confidence, 2), d.bbox) for d in found] == [
        ("car", 0.81, (100, 60, 80, 80)),
        ("person", 0.72, (100, 60, 80, 80)),
    ]


def test_two_overlapping_people_stay_two_detections(tmp_path: Path) -> None:
    """(review focus 2, detector half) IoU 1/3 is below nms_iou 0.45: both survive."""
    rows = {
        PERSON_ROW: [0.5, 0.5, LN2, LN2, 0.9, 0.8, 0.0],  # input box (20, 12, 36, 28)
        NEXT_ROW: [0.5, 0.5, LN2, LN2, 0.9, 0.7, 0.0],  # input box (28, 12, 44, 28)
    }
    det = _detector(tmp_path, rows, device="cpu")
    det.setup()

    found = det.process(_frame(), 1.0)

    assert [(d.kind, d.bbox) for d in found] == [
        ("person", (100, 60, 80, 80)),
        ("person", (140, 60, 80, 80)),
    ]


def test_degenerate_frames_report_nothing(tmp_path: Path) -> None:
    det = _detector(tmp_path, _two_objects(), device="cpu")
    det.setup()

    assert det.process(np.zeros((4, 4, 3), dtype=np.uint8), 1.0) == []
    assert det.process(np.zeros((FRAME_H, FRAME_W), dtype=np.uint8), 1.0) == []


def test_input_width_comes_from_the_descriptor(tmp_path: Path) -> None:
    det = _detector(tmp_path, _two_objects())

    assert det.input_width == 64
    assert det.loaded is False  # building never loads the model


def test_teardown_releases_the_session(tmp_path: Path) -> None:
    det = _detector(tmp_path, _two_objects(), device="cpu")
    det.setup()
    det.teardown()

    assert det.loaded is False
    assert det.process(_frame(), 1.0) == []


# ---------------------------------------------------------------------------
# Provider selection
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("cpu_only")
def test_setup_logs_the_provider_actually_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    infos = _capture(monkeypatch, "info")
    det = _detector(tmp_path, _two_objects(), device="auto")

    det.setup()

    assert det.provider == CPU_PROVIDER
    assert det.fallback_warning is None
    assert det.error is None
    assert det.labels == ["person", "car"]
    assert any("provider: CPUExecutionProvider" in line for line in infos)


@pytest.mark.usefixtures("cpu_only")
def test_cuda_requested_without_cuda_falls_back_to_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    warnings = _capture(monkeypatch, "warning")
    det = _detector(tmp_path, _two_objects(), device="cuda")

    det.setup()

    assert det.provider == CPU_PROVIDER
    assert det.fallback_warning == "CUDA requested but unavailable; running on CPUExecutionProvider"
    assert any("CUDA requested but unavailable" in line for line in warnings)
    assert len(det.process(_frame(), 1.0)) == 2  # still detects, on CPU


class _FakeSession:
    def __init__(self, path: str, providers: list[str], used: list[str]) -> None:
        self.path = path
        self.requested = list(providers)
        self._used = used

    def get_inputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name="images")]

    def get_providers(self) -> list[str]:
        return list(self._used)


def _fake_ort(
    monkeypatch: pytest.MonkeyPatch,
    *,
    available: list[str],
    used: list[str],
    package_name: str = "onnxruntime-gpu",
) -> SimpleNamespace:
    calls = SimpleNamespace(preload=0, sessions=[])

    def preload_dlls() -> None:
        calls.preload += 1

    def inference_session(path: str, providers: list[str]) -> _FakeSession:
        session = _FakeSession(path, providers, used)
        calls.sessions.append(session)
        return session

    fake = SimpleNamespace(
        package_name=package_name,
        get_available_providers=lambda: list(available),
        preload_dlls=preload_dlls,
        InferenceSession=inference_session,
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake)
    monkeypatch.setattr(onnx_mod, "_preload_done", False)
    monkeypatch.setattr(onnx_mod, "_both_onnxruntime_packages", lambda: False)
    return calls


def test_auto_uses_cuda_and_preloads_the_gpu_libraries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _fake_ort(
        monkeypatch, available=[CUDA_PROVIDER, CPU_PROVIDER], used=[CUDA_PROVIDER, CPU_PROVIDER]
    )
    det = _detector(tmp_path, _two_objects(), device="auto")

    det.setup()
    det.setup()  # a second detector (or a hot reload) does not preload again

    assert calls.preload == 1
    assert calls.sessions[0].requested == [CUDA_PROVIDER, CPU_PROVIDER]
    assert det.provider == CUDA_PROVIDER
    assert det.fallback_warning is None


def test_cpu_device_never_asks_for_cuda(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_ort(monkeypatch, available=[CUDA_PROVIDER, CPU_PROVIDER], used=[CPU_PROVIDER])
    det = _detector(tmp_path, _two_objects(), device="cpu")

    det.setup()

    assert calls.preload == 0
    assert calls.sessions[0].requested == [CPU_PROVIDER]
    assert det.provider == CPU_PROVIDER
    assert det.fallback_warning is None


def test_cuda_listed_but_session_on_cpu_sets_the_fallback_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The GPU package lists CUDA even when libcublas cannot load; trust the session.
    _fake_ort(monkeypatch, available=[CUDA_PROVIDER, CPU_PROVIDER], used=[CPU_PROVIDER])
    det = _detector(tmp_path, _two_objects(), device="cuda")

    det.setup()

    assert det.provider == CPU_PROVIDER
    assert det.fallback_warning == "CUDA requested but unavailable; running on CPUExecutionProvider"


def test_cpu_package_never_preloads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_ort(
        monkeypatch, available=[CPU_PROVIDER], used=[CPU_PROVIDER], package_name="onnxruntime"
    )
    det = _detector(tmp_path, _two_objects(), device="auto")

    det.setup()

    assert calls.preload == 0
    assert calls.sessions[0].requested == [CPU_PROVIDER]


@pytest.mark.usefixtures("cpu_only")
def test_both_packages_installed_logs_the_fix_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    warnings = _capture(monkeypatch, "warning")
    monkeypatch.setattr(onnx_mod, "_both_onnxruntime_packages", lambda: True)
    det = _detector(tmp_path, _two_objects(), device="auto")

    det.setup()

    assert det.provider == CPU_PROVIDER
    assert any(onnx_mod.GPU_FIX_COMMAND in line for line in warnings)


# ---------------------------------------------------------------------------
# Model file problems never crash or block (review focus 1)
# ---------------------------------------------------------------------------


def _without_model_file(tmp_path: Path) -> tuple[OnnxDetector, Path]:
    """A detector whose descriptor exists but whose .onnx file is not downloaded yet."""
    det = _detector(tmp_path, _two_objects(), device="cpu")
    stash = tmp_path / "stash.onnx"
    (tmp_path / "models" / "tiny" / "tiny.onnx").rename(stash)
    return det, stash


def test_setup_never_downloads_the_first_frame_does(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(review focus 1) serve and hot reload never wait on a model download."""
    det, stash = _without_model_file(tmp_path)
    fetched: list[str] = []

    def fake_download(desc: ModelDescriptor, models_dir: Path, **_kwargs: object) -> Path:
        fetched.append(desc.name)
        target = models_dir / desc.name / desc.file
        shutil.copyfile(stash, target)
        return target

    monkeypatch.setattr(onnx_mod, "ensure_model_file", fake_download)

    det.setup()

    assert fetched == []
    assert det.loaded is False
    assert det.error is None

    found = det.process(_frame(), 10.0)  # runs on the detector worker thread in production

    assert fetched == ["tiny"]
    assert det.provider == CPU_PROVIDER
    assert [d.kind for d in found] == ["car", "person"]


def test_download_failure_sets_error_and_never_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(review focus 1) Offline home lab: no model file and the download fails."""
    errors = _capture(monkeypatch, "error")

    def offline(desc: ModelDescriptor, models_dir: Path, **_kwargs: object) -> Path:
        raise OSError("network is unreachable")

    monkeypatch.setattr(onnx_mod, "ensure_model_file", offline)
    det, _stash = _without_model_file(tmp_path)

    det.setup()
    found = det.process(_frame(), 1000.0)

    assert found == []
    assert det.loaded is False
    assert det.provider is None
    assert det.error == "model 'tiny' unavailable: network is unreachable"
    assert errors == ["onnx detector onnx: model 'tiny' unavailable: network is unreachable"]


def test_failed_download_retries_after_the_interval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(review focus 1) The detector recovers when the network comes back."""
    _capture(monkeypatch, "error")
    det, stash = _without_model_file(tmp_path)
    attempts: list[str] = []
    online = {"value": False}

    def flaky(desc: ModelDescriptor, models_dir: Path, **_kwargs: object) -> Path:
        attempts.append(desc.name)
        if not online["value"]:
            raise OSError("network is unreachable")
        target = models_dir / desc.name / desc.file
        shutil.copyfile(stash, target)
        return target

    monkeypatch.setattr(onnx_mod, "ensure_model_file", flaky)
    det.setup()

    assert det.process(_frame(), 1000.0) == []  # first attempt fails; retry_interval_s = 300
    assert det.process(_frame(), 1299.0) == []
    assert len(attempts) == 1

    online["value"] = True
    found = det.process(_frame(), 1300.0)

    assert len(attempts) == 2
    assert det.error is None
    assert [d.kind for d in found] == ["car", "person"]


def test_failed_setup_waits_one_interval_before_retrying(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _capture(monkeypatch, "error")
    attempts: list[str] = []

    def broken(desc: ModelDescriptor, models_dir: Path, **_kwargs: object) -> Path:
        attempts.append(desc.name)
        raise OSError("disk read error")

    monkeypatch.setattr(onnx_mod, "ensure_model_file", broken)
    det = _detector(tmp_path, _two_objects(), device="cpu", retry_interval_s=300.0)

    det.setup()  # the file is on disk, so setup() loads (and fails) right away
    det.process(_frame(), 1000.0)
    det.process(_frame(), 1299.0)

    assert len(attempts) == 1

    det.process(_frame(), 1300.0)

    assert len(attempts) == 2
    assert det.error == "model 'tiny' unavailable: disk read error"


def test_hash_mismatch_sets_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _capture(monkeypatch, "error")
    models_dir = tmp_path / "models"
    write_model_dir(models_dir, yolox_output(INPUT, 2, _two_objects()), sha256="0" * 64)
    det = OnnxDetector(descriptor=load_descriptor("tiny", models_dir), models_dir=models_dir)

    det.setup()

    assert det.loaded is False
    assert det.error is not None
    assert det.error.startswith("model 'tiny' unavailable: ")


def test_teardown_does_not_wait_for_a_stalled_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(review finding) A hot reload tears the old detector down while its worker is stuck
    in a download: teardown() returns at once, and a session that finishes loading after
    teardown() is dropped instead of being installed."""
    import threading
    import time

    det, stash = _without_model_file(tmp_path)
    started = threading.Event()
    release = threading.Event()

    def stalled(desc: ModelDescriptor, models_dir: Path, **_kwargs: object) -> Path:
        started.set()
        assert release.wait(timeout=10.0)
        target = models_dir / desc.name / desc.file
        shutil.copyfile(stash, target)
        return target

    monkeypatch.setattr(onnx_mod, "ensure_model_file", stalled)
    det.setup()
    results: list[list[Any]] = []
    worker = threading.Thread(target=lambda: results.append(det.process(_frame(), 1.0)))
    worker.start()
    try:
        assert started.wait(timeout=5.0)
        t0 = time.monotonic()
        det.teardown()
        assert time.monotonic() - t0 < 0.5
        assert det.process(_frame(), 2.0) == []  # torn down: no second load starts
    finally:
        release.set()
        worker.join(timeout=10.0)

    assert not worker.is_alive()
    assert results == [[]]
    assert det.loaded is False
