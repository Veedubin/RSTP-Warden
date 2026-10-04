"""Deployment files and README facts that nothing else tests.

No test here builds an image or starts a service (the CUDA image needs the NVIDIA Container
Toolkit and about 1.5 GB of wheels). They pin the files' contracts instead: Dockerfile.cuda is
the slim Dockerfile plus the gpu extra, the compose overlay reserves the GPU, compose keeps models
in a persistent writable directory and the README tells systemd installs where to put theirs
(the unit file itself is not RW-3's to edit on this branch), the README's marked config
examples validate against the real schema, and the README's detection / events / actions route
rows match the routers.
"""

from __future__ import annotations

import functools
import importlib
import pkgutil
import re
from pathlib import Path

import pytest
import yaml

import rtsp_warden.web.routes as routes_pkg
from rtsp_warden.config import AppConfig, expand_env
from rtsp_warden.detectors.builtin.onnx import CUDA_PROVIDER, GPU_FIX_COMMAND

REPO = Path(__file__).resolve().parent.parent
README = REPO / "README.md"
GPU_EXTRA_FLAGS = "--extra gpu --no-install-package onnxruntime"
GPU_COMPOSE_UP = "docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build"
MODELS_DIR_DOCKER = "/app/data/models"
MODELS_DIR_SYSTEMD = "/var/lib/rtsp-warden/models"
CUDA_LIB_PATH = (
    "/app/.venv/lib/python3.13/site-packages/nvidia/cu13/lib"
    ":/app/.venv/lib/python3.13/site-packages/nvidia/cudnn/lib"
)
DOC_ENV = {
    "CAM_USER": "u",
    "CAM_PASS": "p",
    "NTFY_TOPIC": "topic_doc",
    "NTFY_TOKEN": "tk_doc",
    "ONVIF_PASS": "op",
    "SMTP_USER": "su",
    "SMTP_PASS": "sp",
}
README_EXAMPLES = ("camera-reference", "global-sections", "detection-example")
# Detection, events and actions routes the README documents, written as in the README.
RW3_ROUTES = (
    ("GET", "/events"),
    ("GET", "/events/{id}"),
    ("GET", "/events/{id}/thumbnail.jpg"),
    ("GET", "/events/{id}/clip"),
    ("GET", "/events/{id}/clip.m3u8"),
    ("GET", "/actions"),
    ("POST", "/actions/{name}/test"),
    ("GET", "/cameras/{name}/detection"),
    ("GET", "/cameras/{name}/detectors"),
    ("POST", "/cameras/{name}/detectors/{index}/enabled"),
    ("POST", "/cameras/{name}/detectors/{index}/fps"),
    ("POST", "/cameras/{name}/detectors/{index}/when"),
    ("POST", "/cameras/{name}/detectors/{index}/classes"),
    ("POST", "/cameras/{name}/detection"),
    ("POST", "/cameras/{name}/rules/test"),
    ("GET", "/cameras/{name}/live-boxes.mjpeg"),
    ("GET", "/cameras/{name}/sensitivity"),
    ("POST", "/cameras/{name}/sensitivity"),
    ("GET", "/cameras/{name}/detection-classes"),
    ("POST", "/cameras/{name}/detection-classes"),
    ("POST", "/cameras/{name}/reload"),
)
# Routes this release removed (rulings R5, R14, R15).
REMOVED_ROUTES = (
    ("GET", "/recordings"),
    ("GET", "/recordings/{id}"),
    ("GET", "/api/recordings/{id}/timeline"),
    ("POST", "/events/{id}/clip"),
    ("GET", "/clips/{id}"),
    ("GET", "/clips/{id}/download"),
    ("GET", "/alerts"),
    ("POST", "/alerts/{name}/test"),
    ("POST", "/cameras/{name}/detectors/{type}/enabled"),
)
ROUTE_IN_README = re.compile(r"`(GET|POST|PUT|PATCH|DELETE) (/[^`\s]*)`")


def instructions(path: Path) -> list[str]:
    """Dockerfile instructions, comments and blank lines dropped, continuations joined."""
    out: list[str] = []
    buf = ""
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("#") or (not buf and not line):
            continue
        if line.endswith("\\"):
            buf += line[:-1].strip() + " "
            continue
        out.append((buf + line).strip())
        buf = ""
    assert not buf, f"{path.name} ends inside a line continuation"
    return out


def _readme() -> str:
    return README.read_text(encoding="utf-8")


def readme_example(name: str) -> str:
    """The ```yaml block right after the README marker <!-- config-example: name -->."""
    text = _readme()
    marker = f"<!-- config-example: {name} -->"
    assert text.count(marker) == 1, f"{marker} must appear exactly once in README.md"
    match = re.match(r"\s*```yaml\n(.*?)\n```", text.split(marker, 1)[1], re.S)
    assert match, f"{marker} must be followed by a ```yaml block"
    return match.group(1)


def _shape(path: str) -> str:
    return re.sub(r"\{[^}]*\}", "{}", path)


@functools.lru_cache(maxsize=1)
def served_routes() -> frozenset[tuple[str, str]]:
    """(method, path shape) of every route on every router in rtsp_warden.web.routes."""
    served: set[tuple[str, str]] = set()
    for info in pkgutil.iter_modules(routes_pkg.__path__):
        module = importlib.import_module(f"{routes_pkg.__name__}.{info.name}")
        for route in getattr(getattr(module, "router", None), "routes", ()):
            for method in getattr(route, "methods", None) or ():
                served.add((method, _shape(route.path)))
    return frozenset(served)


def readme_routes() -> set[tuple[str, str]]:
    return {(method, _shape(path)) for method, path in ROUTE_IN_README.findall(_readme())}


def _validate(text: str) -> AppConfig:
    raw = yaml.safe_load(text)
    raw.setdefault("cameras", [{"name": "cam", "main_url": "rtsp://u:p@h/m"}])
    return AppConfig.model_validate(expand_env(raw, DOC_ENV))


@pytest.fixture(autouse=True)
def _models_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("WARDEN_MODELS_DIR", str(tmp_path / "models"))


# --- Dockerfile.cuda ---------------------------------------------------------------------------


def test_cuda_image_is_the_slim_image_with_the_gpu_extra() -> None:
    slim = instructions(REPO / "Dockerfile")
    cuda = instructions(REPO / "Dockerfile.cuda")

    def common(lines: list[str]) -> list[str]:
        return [x for x in lines if "uv sync" not in x and not x.startswith("ENV ")]

    assert common(cuda) == common(slim)
    slim_syncs = [x for x in slim if "uv sync" in x]
    assert len(slim_syncs) == 2
    assert [x for x in cuda if "uv sync" in x] == [
        x.replace("--no-dev", f"--no-dev {GPU_EXTRA_FLAGS}") for x in slim_syncs
    ]


def test_cuda_image_env_adds_only_the_nvidia_settings() -> None:
    (slim_env,) = [x for x in instructions(REPO / "Dockerfile") if x.startswith("ENV ")]
    (cuda_env,) = [x for x in instructions(REPO / "Dockerfile.cuda") if x.startswith("ENV ")]
    assert cuda_env.startswith(slim_env + " ")
    assert cuda_env[len(slim_env) + 1 :].split() == [
        "NVIDIA_VISIBLE_DEVICES=all",
        "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
        f"LD_LIBRARY_PATH={CUDA_LIB_PATH}",
    ]


# --- compose -----------------------------------------------------------------------------------


def test_gpu_overlay_builds_the_cuda_image_and_reserves_the_gpu_through_cdi() -> None:
    overlay = yaml.safe_load((REPO / "docker-compose.gpu.yml").read_text(encoding="utf-8"))
    assert list(overlay["services"]) == ["warden"]
    warden = overlay["services"]["warden"]
    assert warden["build"] == {"context": ".", "dockerfile": "Dockerfile.cuda"}
    assert warden["image"] == "rtsp-warden:cuda"
    assert warden["deploy"]["resources"]["reservations"]["devices"] == [
        {"driver": "cdi", "device_ids": ["nvidia.com/gpu=all"], "capabilities": ["gpu"]}
    ]
    # the classic nvidia-runtime form stays documented as the alternative
    assert "driver: nvidia" in (REPO / "docker-compose.gpu.yml").read_text(encoding="utf-8")


def test_gpu_overlay_and_readme_give_the_same_command() -> None:
    assert GPU_COMPOSE_UP in (REPO / "docker-compose.gpu.yml").read_text(encoding="utf-8")
    assert GPU_COMPOSE_UP in _readme()


def test_compose_keeps_models_in_the_data_volume() -> None:
    compose = yaml.safe_load((REPO / "docker-compose.yml").read_text(encoding="utf-8"))
    warden = compose["services"]["warden"]
    assert warden["environment"]["WARDEN_MODELS_DIR"] == MODELS_DIR_DOCKER
    data_mounts = [v for v in warden["volumes"] if v.split(":")[1] == "/app/data"]
    assert [v.split(":")[0] for v in data_mounts] == ["./data"]
    assert MODELS_DIR_DOCKER.startswith("/app/data/")


# --- systemd -----------------------------------------------------------------------------------


def test_readme_puts_systemd_models_inside_the_units_writable_paths() -> None:
    """The unit's HOME is not writable (ProtectHome), so the default ~/.cache models_dir fails
    under systemd: the README tells installs to set WARDEN_MODELS_DIR in warden.env, to a
    directory the unit can write."""
    assert f"WARDEN_MODELS_DIR={MODELS_DIR_SYSTEMD}" in _readme()
    unit = (REPO / "packaging" / "systemd" / "rtsp-warden.service").read_text(encoding="utf-8")
    settings = [
        line.strip()
        for line in unit.splitlines()
        if "=" in line and not line.lstrip().startswith("#")
    ]
    writable = [
        path
        for line in settings
        if line.startswith("ReadWritePaths=")
        for path in line.split("=", 1)[1].split()
    ]
    assert any(MODELS_DIR_SYSTEMD.startswith(path.rstrip("/") + "/") for path in writable)


# --- README ------------------------------------------------------------------------------------


def test_readme_gpu_section_gives_the_working_commands() -> None:
    text = _readme()
    assert GPU_FIX_COMMAND in text
    assert "uv run --no-sync rtsp-warden serve" in text
    assert f"provider: {CUDA_PROVIDER}" in text
    assert "nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml" in text
    assert "PrivateDevices=false" in text


def test_readme_documents_the_models_dir_variable() -> None:
    text = _readme()
    assert "| `WARDEN_MODELS_DIR` |" in text
    assert MODELS_DIR_DOCKER in text


@pytest.mark.parametrize("name", README_EXAMPLES)
def test_readme_config_example_validates(name: str) -> None:
    cfg = _validate(readme_example(name))
    assert cfg.cameras


def test_readme_detection_example_wires_rules_to_actions() -> None:
    cfg = _validate(readme_example("detection-example"))
    (cam,) = cfg.cameras
    assert cam.detect_fps == 5.0
    assert [s.type for s in cam.detectors] == ["motion", "onnx"]
    assert {z.name: z.kind for z in cam.zones} == {"driveway": "area", "road": "ignore"}
    assert [(r.name, r.zones, r.between, r.actions) for r in cam.rules] == [
        ("person-any-time", [], None, ["phone"]),
        ("car-in-driveway-at-night", ["driveway"], "22:00-06:00", ["phone", "homeassistant"]),
    ]
    assert [(a.name, a.type) for a in cfg.actions] == [
        ("phone", "ntfy"),
        ("homeassistant", "webhook"),
    ]
    assert str(cfg.runtime.public_url).rstrip("/") == "http://nvr.lan:8080"


@pytest.mark.parametrize(("method", "path"), RW3_ROUTES, ids=[f"{m} {p}" for m, p in RW3_ROUTES])
def test_readme_route_row_is_served(method: str, path: str) -> None:
    assert f"`{method} {path}`" in _readme(), f"README.md does not document `{method} {path}`"
    assert (method, _shape(path)) in served_routes(), f"no router serves {method} {path}"


@pytest.mark.parametrize(
    ("method", "path"), REMOVED_ROUTES, ids=[f"{m} {p}" for m, p in REMOVED_ROUTES]
)
def test_removed_route_is_gone(method: str, path: str) -> None:
    assert f"`{method} {path}`" not in _readme(), f"README.md still documents {method} {path}"
    key = (method, _shape(path))
    if key in {(m, _shape(p)) for m, p in RW3_ROUTES}:
        return  # same shape as a documented route with a renamed path parameter
    assert key not in readme_routes(), f"README.md still documents {method} {path}"
    assert key not in served_routes(), f"{method} {path} is still served"
