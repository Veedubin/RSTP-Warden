"""Model registry: built-in descriptors, user models, labels, models_dir, verified downloads."""

from __future__ import annotations

import ast
import hashlib
import io
import threading
import time
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml

from rtsp_warden.detectors import model_registry as mr
from rtsp_warden.detectors.builtin.model_utils import load_class_names
from rtsp_warden.detectors.model_registry import (
    DEFAULT_MODEL,
    SUPPORTED_POSTPROCESS,
    ModelDescriptor,
    ModelDescriptorError,
    ModelDownloadError,
    ModelError,
    ModelNotFound,
    ModelVerifyError,
    builtin_models_dir,
    default_models_dir,
    ensure_model_file,
    labels_path,
    list_models,
    load_descriptor,
    load_labels,
    model_file_path,
    unknown_labels_message,
)

YOLOX_S_SHA = "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063"
YOLOX_NANO_SHA = "c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d"
RELEASE = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0"


def write_model(
    models_dir: Path, dirname: str, /, *, label_lines: list[str] | None = None, **fields: object
) -> Path:
    """Write <models_dir>/<dirname>/model.yaml (plus labels.txt when given); return the dir."""
    d = models_dir / dirname
    d.mkdir(parents=True, exist_ok=True)
    desc: dict[str, object] = {
        "name": dirname,
        "file": f"{dirname}.onnx",
        "labels": "labels.txt",
        "input_size": [64, 64],
        "postprocess": "yolox",
    }
    desc.update(fields)
    (d / "model.yaml").write_text(yaml.safe_dump(desc, sort_keys=False), encoding="utf-8")
    if label_lines is not None:
        (d / "labels.txt").write_text("\n".join(label_lines) + "\n", encoding="utf-8")
    return d


# --- built-in descriptors -------------------------------------------------------------


def test_constants() -> None:
    assert DEFAULT_MODEL == "yolox-s"
    assert SUPPORTED_POSTPROCESS == ("yolox",)


def test_builtin_dir_ships_descriptors_and_coco_but_no_model_files() -> None:
    root = builtin_models_dir()
    assert (root / "coco.txt").is_file()
    assert (root / "yolox-s" / "model.yaml").is_file()
    assert (root / "yolox-nano" / "model.yaml").is_file()
    assert list(root.rglob("*.onnx")) == []


def test_builtin_yolox_s_descriptor(tmp_path: Path) -> None:
    desc = load_descriptor("yolox-s", tmp_path / "models")
    assert desc.name == "yolox-s"
    assert desc.file == "yolox_s.onnx"
    assert desc.labels == "coco.txt"
    assert desc.input_size == (640, 640)
    assert desc.postprocess == "yolox"
    assert desc.sha256 == YOLOX_S_SHA
    assert desc.url == f"{RELEASE}/yolox_s.onnx"
    assert desc.base_dir == builtin_models_dir() / "yolox-s"


def test_builtin_yolox_nano_descriptor(tmp_path: Path) -> None:
    desc = load_descriptor("yolox-nano", tmp_path / "models")
    assert desc.file == "yolox_nano.onnx"
    assert desc.input_size == (416, 416)
    assert desc.sha256 == YOLOX_NANO_SHA
    assert desc.url == f"{RELEASE}/yolox_nano.onnx"


@pytest.mark.parametrize("name", ["yolox-s", "yolox-nano"])
def test_builtin_labels_are_the_80_coco_names_in_order(tmp_path: Path, name: str) -> None:
    desc = load_descriptor(name, tmp_path)
    labels = load_labels(desc)
    assert labels == load_class_names()
    assert len(labels) == 80
    assert labels[0] == "person"
    assert labels[15] == "cat"
    assert labels_path(desc) == builtin_models_dir() / "coco.txt"


def test_lookup_never_creates_models_dir(tmp_path: Path) -> None:
    models_dir = tmp_path / "not-created"
    load_descriptor("yolox-s", models_dir)
    list_models(models_dir)
    assert not models_dir.exists()


def test_model_file_path_is_under_models_dir(tmp_path: Path) -> None:
    desc = load_descriptor("yolox-s", tmp_path)
    assert model_file_path(desc, tmp_path) == tmp_path / "yolox-s" / "yolox_s.onnx"


# --- user models ------------------------------------------------------------------------


def test_user_model_loads_from_models_dir(tmp_path: Path) -> None:
    write_model(tmp_path, "critters", label_lines=["raccoon", "fox"])
    desc = load_descriptor("critters", tmp_path)
    assert desc.base_dir == tmp_path / "critters"
    assert desc.input_size == (64, 64)
    assert load_labels(desc) == ["raccoon", "fox"]


def test_models_dir_descriptor_overrides_builtin(tmp_path: Path) -> None:
    write_model(tmp_path, "yolox-s", label_lines=["person", "cat"], input_size=[320, 320])
    desc = load_descriptor("yolox-s", tmp_path)
    assert desc.base_dir == tmp_path / "yolox-s"
    assert desc.input_size == (320, 320)
    assert load_labels(desc) == ["person", "cat"]


def test_user_model_can_use_the_builtin_coco_labels(tmp_path: Path) -> None:
    write_model(tmp_path, "my-yolox", labels="coco.txt")
    desc = load_descriptor("my-yolox", tmp_path)
    assert labels_path(desc) == builtin_models_dir() / "coco.txt"
    assert len(load_labels(desc)) == 80


def test_labels_file_skips_blank_lines_and_strips(tmp_path: Path) -> None:
    write_model(tmp_path, "m", label_lines=["person", "", "  car  "])
    assert load_labels(load_descriptor("m", tmp_path)) == ["person", "car"]


def test_missing_labels_file_raises(tmp_path: Path) -> None:
    write_model(tmp_path, "m")
    with pytest.raises(ModelDescriptorError, match="labels file 'labels.txt' of model 'm'"):
        load_labels(load_descriptor("m", tmp_path))


def test_empty_labels_file_raises(tmp_path: Path) -> None:
    write_model(tmp_path, "m", label_lines=["", "  "])
    with pytest.raises(ModelDescriptorError, match="is empty"):
        load_labels(load_descriptor("m", tmp_path))


def test_unknown_model_lists_available_models(tmp_path: Path) -> None:
    write_model(tmp_path, "critters", label_lines=["raccoon"])
    with pytest.raises(ModelNotFound) as ei:
        load_descriptor("yolox-m", tmp_path)
    assert str(ei.value) == "unknown model 'yolox-m'; available: critters, yolox-nano, yolox-s"


@pytest.mark.parametrize("name", ["../etc", "a/b", "", ".hidden"])
def test_invalid_model_name_rejected(tmp_path: Path, name: str) -> None:
    with pytest.raises(ModelNotFound, match="invalid model name"):
        load_descriptor(name, tmp_path)


def test_list_models_merges_builtin_and_user_dirs(tmp_path: Path) -> None:
    assert list_models(tmp_path) == ["yolox-nano", "yolox-s"]
    write_model(tmp_path, "critters", label_lines=["raccoon"])
    (tmp_path / "junk").mkdir()  # no model.yaml: not a model
    assert list_models(tmp_path) == ["critters", "yolox-nano", "yolox-s"]
    assert list_models(tmp_path / "missing") == ["yolox-nano", "yolox-s"]


def test_unknown_postprocess_lists_supported_values(tmp_path: Path) -> None:
    write_model(tmp_path, "ssd-model", label_lines=["a"], postprocess="ssd")
    with pytest.raises(ModelDescriptorError) as ei:
        load_descriptor("ssd-model", tmp_path)
    assert "unknown postprocess 'ssd'; supported: yolox" in str(ei.value)


def test_descriptor_rejects_unknown_keys(tmp_path: Path) -> None:
    write_model(tmp_path, "m", label_lines=["a"], threshold=0.5)
    with pytest.raises(ModelDescriptorError, match="threshold: Extra inputs are not permitted"):
        load_descriptor("m", tmp_path)


def test_descriptor_name_must_match_its_directory(tmp_path: Path) -> None:
    write_model(tmp_path, "a", label_lines=["x"], name="b")
    with pytest.raises(ModelDescriptorError, match="declares name 'b' but lives in directory 'a'"):
        load_descriptor("a", tmp_path)


def test_descriptor_name_defaults_to_its_directory(tmp_path: Path) -> None:
    d = tmp_path / "noname"
    d.mkdir()
    (d / "model.yaml").write_text(
        "file: m.onnx\nlabels: coco.txt\ninput_size: [64, 64]\npostprocess: yolox\n",
        encoding="utf-8",
    )
    assert load_descriptor("noname", tmp_path).name == "noname"


@pytest.mark.parametrize(
    ("field", "value", "fragment"),
    [
        ("input_size", [640, 600], "positive multiples of 32"),
        ("input_size", [0, 640], "positive multiples of 32"),
        ("sha256", "abc", "64 hexadecimal characters"),
        ("url", "ftp://example.invalid/m.onnx", "must start with https://"),
        ("file", "sub/m.onnx", "bare file name"),
        ("file", "..", "bare file name"),
    ],
)
def test_descriptor_field_validation(
    tmp_path: Path, field: str, value: object, fragment: str
) -> None:
    write_model(tmp_path, "m", label_lines=["a"], **{field: value})
    with pytest.raises(ModelDescriptorError, match=fragment):
        load_descriptor("m", tmp_path)


def test_sha256_is_normalised_to_lowercase(tmp_path: Path) -> None:
    write_model(tmp_path, "m", label_lines=["a"], sha256=YOLOX_S_SHA.upper())
    assert load_descriptor("m", tmp_path).sha256 == YOLOX_S_SHA


def test_malformed_yaml_raises_descriptor_error(tmp_path: Path) -> None:
    d = tmp_path / "bad"
    d.mkdir()
    (d / "model.yaml").write_text("name: [\n", encoding="utf-8")
    with pytest.raises(ModelDescriptorError, match="cannot read model descriptor"):
        load_descriptor("bad", tmp_path)


def test_non_mapping_yaml_raises_descriptor_error(tmp_path: Path) -> None:
    d = tmp_path / "list"
    d.mkdir()
    (d / "model.yaml").write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ModelDescriptorError, match="must be a YAML mapping"):
        load_descriptor("list", tmp_path)


def test_registry_module_imports_no_heavy_dependencies() -> None:
    """Config validation imports this module: no cv2, numpy, onnx or onnxruntime."""
    tree = ast.parse(Path(mr.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.split(".")[0])
    assert imported.isdisjoint({"cv2", "numpy", "onnx", "onnxruntime"})


def test_unknown_labels_message_suggests_and_lists_every_model() -> None:
    msg = unknown_labels_message(
        "yard", "detect_classes", ["persn", "zzz"], {"a": ["person", "car"], "b": ["fox"]}
    )
    assert msg == (
        "camera 'yard': unknown detect_classes ['persn', 'zzz'] "
        "(did you mean 'persn' -> 'person'?); a labels: person, car; b labels: fox"
    )


def test_every_registry_error_is_a_model_error() -> None:
    for cls in (ModelNotFound, ModelDescriptorError, ModelDownloadError, ModelVerifyError):
        assert issubclass(cls, ModelError)


# --- default models_dir -------------------------------------------------------------------


def test_default_models_dir_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("WARDEN_MODELS_DIR", str(tmp_path / "m"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert default_models_dir() == tmp_path / "m"


def test_default_models_dir_uses_xdg_cache_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("WARDEN_MODELS_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert default_models_dir() == tmp_path / "xdg" / "rtsp-warden" / "models"


def test_default_models_dir_falls_back_to_home_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("WARDEN_MODELS_DIR", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert default_models_dir() == tmp_path / ".cache" / "rtsp-warden" / "models"
    assert not (tmp_path / ".cache").exists()


# --- ensure_model_file ----------------------------------------------------------------------


def tiny_desc(tmp_path: Path, payload: bytes, *, with_sha: bool = True) -> ModelDescriptor:
    return ModelDescriptor(
        name="tiny",
        file="tiny.onnx",
        labels="labels.txt",
        input_size=(64, 64),
        postprocess="yolox",
        sha256=hashlib.sha256(payload).hexdigest() if with_sha else None,
        url="https://example.invalid/tiny.onnx",
        base_dir=tmp_path / "descriptors" / "tiny",
    )


class FakeResponse(io.BytesIO):
    """A response body; ``on_close`` runs when the download's ``with`` block exits."""

    def __init__(self, payload: bytes, on_close: Callable[[], None]) -> None:
        super().__init__(payload)
        self._on_close = on_close

    def close(self) -> None:
        if not self.closed:
            self._on_close()
        super().close()


class BrokenResponse(io.BytesIO):
    """A response whose body read times out half way."""

    def read(self, size: int | None = -1) -> bytes:
        raise TimeoutError("timed out")


class FakeOpener:
    def __init__(
        self,
        payload: bytes = b"",
        *,
        error: Exception | None = None,
        broken: bool = False,
        on_close: Callable[[], None] | None = None,
    ) -> None:
        self.payload = payload
        self.error = error
        self.broken = broken
        self.on_close = on_close or (lambda: None)
        self.calls: list[tuple[str, float]] = []

    def __call__(self, url: str, timeout_s: float) -> io.BytesIO:
        self.calls.append((url, timeout_s))
        if self.error is not None:
            raise self.error
        if self.broken:
            return BrokenResponse(b"")
        return FakeResponse(self.payload, self.on_close)


def test_download_writes_part_then_renames(tmp_path: Path) -> None:
    payload = b"fake onnx bytes"
    models_dir = tmp_path / "models"
    desc = tiny_desc(tmp_path, payload)
    final = model_file_path(desc, models_dir)
    part = final.with_name("tiny.onnx.part")
    seen: dict[str, bool] = {}

    def on_close() -> None:
        seen["part"] = part.is_file()
        seen["final"] = final.exists()

    opener = FakeOpener(payload, on_close=on_close)
    path = ensure_model_file(desc, models_dir, opener=opener, timeout_s=12.5)

    assert path == final
    assert final.read_bytes() == payload
    assert not part.exists()
    assert seen == {"part": True, "final": False}
    assert opener.calls == [("https://example.invalid/tiny.onnx", 12.5)]


def test_existing_file_with_matching_sha_is_not_downloaded(tmp_path: Path) -> None:
    payload = b"already here"
    desc = tiny_desc(tmp_path, payload)
    final = model_file_path(desc, tmp_path)
    final.parent.mkdir(parents=True)
    final.write_bytes(payload)
    opener = FakeOpener(error=AssertionError("must not download"))

    assert ensure_model_file(desc, tmp_path, opener=opener) == final
    assert opener.calls == []


def test_download_sha_mismatch_discards_the_file(tmp_path: Path) -> None:
    desc = tiny_desc(tmp_path, b"expected bytes")
    final = model_file_path(desc, tmp_path)
    with pytest.raises(ModelVerifyError, match="the download was discarded"):
        ensure_model_file(desc, tmp_path, opener=FakeOpener(b"tampered bytes"))
    assert not final.exists()
    assert not final.with_name("tiny.onnx.part").exists()


def test_download_failure_leaves_nothing_and_the_next_call_retries(tmp_path: Path) -> None:
    """(review focus 1) offline home lab: a typed error, no partial file, retry works later."""
    payload = b"model"
    desc = tiny_desc(tmp_path, payload)
    final = model_file_path(desc, tmp_path)

    with pytest.raises(ModelDownloadError, match="OSError: network is unreachable") as ei:
        ensure_model_file(
            desc, tmp_path, opener=FakeOpener(error=OSError("network is unreachable"))
        )
    assert isinstance(ei.value, ModelError)
    assert not final.exists()
    assert not final.with_name("tiny.onnx.part").exists()

    assert ensure_model_file(desc, tmp_path, opener=FakeOpener(payload)) == final
    assert final.read_bytes() == payload


def test_download_failure_mid_stream_removes_the_part_file(tmp_path: Path) -> None:
    desc = tiny_desc(tmp_path, b"model")
    final = model_file_path(desc, tmp_path)
    with pytest.raises(ModelDownloadError, match="TimeoutError: timed out"):
        ensure_model_file(desc, tmp_path, opener=FakeOpener(broken=True))
    assert not final.exists()
    assert not final.with_name("tiny.onnx.part").exists()


def test_existing_file_with_wrong_sha_is_kept_and_reported(tmp_path: Path) -> None:
    desc = tiny_desc(tmp_path, b"official bytes")
    final = model_file_path(desc, tmp_path)
    final.parent.mkdir(parents=True)
    final.write_bytes(b"user re-export")
    opener = FakeOpener(error=AssertionError("must not download"))

    with pytest.raises(ModelVerifyError, match="delete it to download it again"):
        ensure_model_file(desc, tmp_path, opener=opener)
    assert final.read_bytes() == b"user re-export"
    assert opener.calls == []


def test_missing_file_without_url_raises_not_found(tmp_path: Path) -> None:
    desc = tiny_desc(tmp_path, b"x").model_copy(update={"url": None})
    with pytest.raises(ModelNotFound, match="has no url"):
        ensure_model_file(desc, tmp_path, opener=FakeOpener(b"x"))


def test_download_without_sha_is_accepted_with_a_warning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(mr.log, "warning", lambda msg, *args: warnings.append(msg % args))
    desc = tiny_desc(tmp_path, b"x", with_sha=False)

    path = ensure_model_file(desc, tmp_path, opener=FakeOpener(b"unverified"))

    assert path.read_bytes() == b"unverified"
    assert warnings == ["model tiny has no sha256 in its descriptor; the download was not verified"]


def test_unwritable_models_dir_raises_download_error(tmp_path: Path) -> None:
    blocker = tmp_path / "models"
    blocker.write_text("a file where the directory should be", encoding="utf-8")
    desc = tiny_desc(tmp_path, b"x")
    with pytest.raises(ModelDownloadError, match="cannot create"):
        ensure_model_file(desc, blocker, opener=FakeOpener(b"x"))


def test_concurrent_callers_download_once(tmp_path: Path) -> None:
    payload = b"shared model"
    desc = tiny_desc(tmp_path, payload)
    release = threading.Event()
    calls: list[str] = []

    def slow_opener(url: str, timeout_s: float) -> io.BytesIO:
        calls.append(url)
        release.wait(5)
        return io.BytesIO(payload)

    results: list[Path] = []

    def worker() -> None:
        results.append(ensure_model_file(desc, tmp_path, opener=slow_opener))

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    time.sleep(0.1)  # both threads are inside ensure_model_file; one waits on the lock
    release.set()
    for t in threads:
        t.join(5)

    assert calls == ["https://example.invalid/tiny.onnx"]
    assert results == [model_file_path(desc, tmp_path)] * 2
