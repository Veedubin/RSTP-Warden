"""Install-time guarantees for the ONNX Runtime dependency (ruling R2).

The ONNX detector imports ``onnxruntime`` lazily, and its tests build tiny graphs with
``onnx.helper`` and run them on the CPU provider. These tests pin what that code relies on:
the CPU runtime is a core dependency, ``onnx`` is in the dev group, the CPU and GPU builds of
ONNX Runtime are never installed side by side (they share one ``onnxruntime`` import package),
and the Python floor is 3.11 while ruff still targets py310.
"""

from __future__ import annotations

import importlib.metadata
import sys
from pathlib import Path

import numpy as np
import pytest
import tomllib  # ruff sorts it as third party: target-version py310 predates tomllib

REPO_ROOT = Path(__file__).resolve().parent.parent
MIN_ORT = (1, 28)
ORT_DISTRIBUTIONS = ("onnxruntime", "onnxruntime-gpu")
GPU_SYNC = (
    "uv sync --extra gpu --no-install-package onnxruntime --reinstall-package onnxruntime-gpu"
)
CPU_SYNC = "uv sync --reinstall-package onnxruntime"


def _pyproject() -> dict:
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _installed(dist: str) -> bool:
    try:
        importlib.metadata.distribution(dist)
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


def _identity_model_bytes() -> bytes:
    """An Identity graph with the detector's tensor names, IR 10 and opset 17."""
    import onnx
    from onnx import TensorProto, helper

    graph = helper.make_graph(
        [helper.make_node("Identity", ["images"], ["output"])],
        "identity",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 4, 4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 4, 4])],
    )
    model = helper.make_model(graph, ir_version=10, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    return model.SerializeToString()


def test_python_floor_is_311_and_ruff_target_stays_py310() -> None:
    data = _pyproject()
    assert sys.version_info >= (3, 11)
    assert data["project"]["requires-python"] == ">=3.11"
    # py311 would turn the 10-line E501 baseline into 93 errors (UP017, UP042).
    assert data["tool"]["ruff"]["target-version"] == "py310"


def test_onnxruntime_is_core_gpu_build_is_an_extra_and_onnx_is_dev_only() -> None:
    data = _pyproject()
    core = data["project"]["dependencies"]
    assert "onnxruntime>=1.28" in core
    assert not any(dep.startswith("onnxruntime-gpu") for dep in core)
    assert data["project"]["optional-dependencies"] == {
        "gpu": ["onnxruntime-gpu[cuda,cudnn]>=1.28"]
    }
    assert "onnx>=1.17" in data["dependency-groups"]["dev"]
    assert not any(dep.startswith("onnx>") for dep in core)


def test_gpu_marker_is_registered_for_strict_markers() -> None:
    markers = _pyproject()["tool"]["pytest"]["ini_options"].get("markers", [])
    assert any(marker.startswith("gpu:") for marker in markers), markers


def test_lock_has_no_python_310_fork_and_carries_both_builds() -> None:
    text = (REPO_ROOT / "uv.lock").read_text(encoding="utf-8")
    lock = tomllib.loads(text)
    assert lock["requires-python"] == ">=3.11"
    assert "python_full_version < '3.11'" not in text
    versions = {pkg["name"]: pkg["version"] for pkg in lock["package"]}
    for name in ("onnxruntime", "onnxruntime-gpu", "onnx"):
        assert name in versions, f"{name} missing from uv.lock; run `uv lock`"
    locked = tuple(int(part) for part in versions["onnxruntime"].split(".")[:2])
    assert locked >= MIN_ORT, versions["onnxruntime"]


def test_exactly_one_onnxruntime_distribution_is_installed() -> None:
    from onnxruntime.capi import build_and_package_info

    installed = [dist for dist in ORT_DISTRIBUTIONS if _installed(dist)]
    assert len(installed) == 1, (
        f"installed: {installed}. Both builds share the onnxruntime/ package and the CPU "
        f"files win. Fix: `{CPU_SYNC}` (CPU) or `{GPU_SYNC}` (GPU)."
    )
    assert build_and_package_info.package_name == installed[0]


def test_onnxruntime_imports_with_the_cpu_provider() -> None:
    import onnxruntime as ort

    version = tuple(int(part) for part in ort.__version__.split(".")[:2])
    assert version >= MIN_ORT, ort.__version__
    assert "CPUExecutionProvider" in ort.get_available_providers()


def test_ir10_opset17_test_graph_runs_on_the_cpu_provider() -> None:
    import onnxruntime as ort

    session = ort.InferenceSession(_identity_model_bytes(), providers=["CPUExecutionProvider"])
    frame = np.arange(48, dtype=np.float32).reshape(1, 3, 4, 4)
    (out,) = session.run(["output"], {"images": frame})
    assert session.get_providers()[0] == "CPUExecutionProvider"
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, frame)


@pytest.mark.gpu
def test_cuda_provider_is_used_when_the_gpu_build_is_installed() -> None:
    import onnxruntime as ort
    from onnxruntime.capi import build_and_package_info

    if build_and_package_info.package_name != "onnxruntime-gpu":
        pytest.skip(f"CPU build of onnxruntime installed; for the GPU check run `{GPU_SYNC}`")
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
    session = ort.InferenceSession(
        _identity_model_bytes(),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    # A CUDA provider that cannot load its libraries silently falls back to CPU.
    assert session.get_providers()[0] == "CUDAExecutionProvider", session.get_providers()
