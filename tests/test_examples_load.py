"""Shipped example configs, the init-config template and their commented example blocks.

The init-config template (``cli.SAMPLE_CONFIG_YAML``) is RW-2's file on this branch, so it
gets the same detection / actions blocks as ``examples/config.yaml`` only after the RW-2
rebase; until then the template is checked to load as written.

A block runs from a ``# --- <name> example: ...`` line to ``# --- end of <name> example ---``
and holds commented YAML only. ``uncomment_examples`` removes one leading ``# `` from each of its
lines, which is exactly what the marker tells a user to do, so the tests validate what a user
gets after following it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from rtsp_warden.cli import SAMPLE_CONFIG_YAML
from rtsp_warden.config import AppConfig, expand_env, load_config

REPO = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO / "examples"
EXAMPLE_FILES = sorted(EXAMPLES_DIR.rglob("*.yaml"))
EXAMPLE_IDS = [p.relative_to(EXAMPLES_DIR).as_posix() for p in EXAMPLE_FILES]
FOSCAM = EXAMPLES_DIR / "configs" / "config-Foscam-C1-V3.yaml"
EXAMPLE_ENV = {
    "CAM_USER": "u",
    "CAM_PASS": "p",
    "NTFY_TOPIC": "t",
    "NTFY_TOKEN": "tk_example",
}
EXAMPLES_CONFIG = EXAMPLES_DIR / "config.yaml"
BLOCK_START = re.compile(r"^\s*# --- (\w+) example: ")
BLOCK_END = re.compile(r"^\s*# --- end of (\w+) example ---\s*$")


def _scan(text: str) -> list[tuple[str, str | None]]:
    """Pair every line with the name of the example block it is in (None outside, markers)."""
    rows: list[tuple[str, str | None]] = []
    inside: str | None = None
    seen: set[str] = set()
    for line in text.splitlines():
        end = BLOCK_END.match(line)
        if inside is not None and end:
            assert end.group(1) == inside, f"block {inside!r} is closed as {end.group(1)!r}"
            inside = None
            rows.append((line, None))
            continue
        start = BLOCK_START.match(line)
        if inside is None and start:
            inside = start.group(1)
            assert inside not in seen, f"example block {inside!r} appears twice"
            seen.add(inside)
            rows.append((line, None))
            continue
        if inside is not None:
            assert line.lstrip().startswith("#"), f"uncommented line in {inside!r}: {line!r}"
        rows.append((line, inside))
    assert inside is None, f"example block {inside!r} is never closed"
    return rows


def example_blocks(text: str) -> dict[str, list[str]]:
    """The commented lines of every example block, by block name, in file order."""
    blocks: dict[str, list[str]] = {}
    for line, name in _scan(text):
        if name is not None:
            blocks.setdefault(name, []).append(line)
    return blocks


def uncomment_examples(text: str) -> str:
    """*text* with one leading '# ' removed from every line inside an example block."""
    lines = [
        re.sub(r"^(\s*)# ?", r"\1", line, count=1) if name is not None else line
        for line, name in _scan(text)
    ]
    return "\n".join(lines) + "\n"


def _validate(text: str) -> AppConfig:
    return AppConfig.model_validate(expand_env(yaml.safe_load(text), EXAMPLE_ENV))


@pytest.fixture(autouse=True)
def _example_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for name, value in EXAMPLE_ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("WARDEN_MODELS_DIR", str(tmp_path / "models"))


def test_uncomment_examples_touches_only_example_blocks() -> None:
    text = (
        "# keep: this comment\n"
        "a:\n"
        "  # --- demo example: delete the leading '# ' on the lines below ---\n"
        "  # b: 1\n"
        "  # c:\n"
        "  #   - 2\n"
        "  # --- end of demo example ---\n"
    )
    assert example_blocks(text) == {"demo": ["  # b: 1", "  # c:", "  #   - 2"]}
    assert yaml.safe_load(uncomment_examples(text)) == {"a": {"b": 1, "c": [2]}}
    assert uncomment_examples(text).startswith("# keep: this comment\n")


def test_examples_directory_has_every_shipped_config() -> None:
    assert {
        "config.yaml",
        "configs/config-Foscam-C1-V3.yaml",
        "configs/config-NC230-C1-V3-2Cams.yaml",
        "configs/config-TP-Link-NC230.yaml",
    } <= set(EXAMPLE_IDS)


@pytest.mark.parametrize("path", EXAMPLE_FILES, ids=EXAMPLE_IDS)
def test_example_file_loads(path: Path) -> None:
    assert load_config(path).cameras


@pytest.mark.parametrize("path", EXAMPLE_FILES, ids=EXAMPLE_IDS)
def test_example_file_validates_with_its_blocks_turned_on(path: Path) -> None:
    assert _validate(uncomment_examples(path.read_text(encoding="utf-8"))).cameras


def test_template_loads_as_written() -> None:
    cfg = _validate(SAMPLE_CONFIG_YAML)
    assert [cam.name for cam in cfg.cameras] == ["front"]
    assert cfg.cameras[0].detectors == []
    assert cfg.actions == []


def test_examples_config_blocks_turn_on_detection_rules_and_actions() -> None:
    text = EXAMPLES_CONFIG.read_text(encoding="utf-8")
    assert list(example_blocks(text)) == ["detection", "actions"]
    cfg = _validate(uncomment_examples(text))
    (cam,) = cfg.cameras
    assert cam.detect_fps == 5.0
    assert [(s.type, s.fps) for s in cam.detectors] == [("motion", 5.0), ("onnx", 2.0)]
    onnx = cam.detectors[1]
    assert (onnx.model, onnx.device, onnx.min_confidence) == ("yolox-s", "auto", 0.5)
    assert [(r.name, r.labels, r.actions, r.clip) for r in cam.rules] == [
        ("person-any-time", ["person"], ["phone"], True)
    ]
    assert [(a.name, a.type) for a in cfg.actions] == [("phone", "ntfy")]
    assert cfg.actions[0].topic == EXAMPLE_ENV["NTFY_TOPIC"]
    assert cfg.actions[0].token == EXAMPLE_ENV["NTFY_TOKEN"]


def test_no_example_publishes_to_a_literal_ntfy_topic() -> None:
    """ntfy.sh topics are public: every shipped example takes the topic from .env."""
    for path in EXAMPLE_FILES:
        for line in path.read_text(encoding="utf-8").splitlines():
            if "topic:" in line:
                assert "${NTFY_TOPIC}" in line, f"{path.name}: {line.strip()}"


def test_foscam_example_runs_detection_on_the_main_stream_only() -> None:
    cfg = load_config(FOSCAM)
    (cam,) = cfg.cameras
    assert cam.sub_url is None
    assert cam.main_url == "rtsp://u:p@192.168.1.72:554/videoMain"
    assert cam.proxy.stream == "main"
    assert cam.detect_fps == 5.0
    assert [(s.type, s.enabled) for s in cam.detectors] == [("motion", True), ("onnx", True)]
    onnx = cam.detectors[1]
    assert (onnx.model, onnx.device, onnx.fps) == ("yolox-s", "auto", 2.0)
    assert cam.effective_fps(onnx) == 2.0
    assert cam.rules == []
    assert cfg.actions == []


def test_foscam_example_blocks_add_a_rule_and_its_action() -> None:
    text = FOSCAM.read_text(encoding="utf-8")
    assert list(example_blocks(text)) == ["rules", "actions"]
    cfg = _validate(uncomment_examples(text))
    (cam,) = cfg.cameras
    assert [(r.name, r.actions) for r in cam.rules] == [("person-any-time", ["phone"])]
    assert [a.name for a in cfg.actions] == ["phone"]
