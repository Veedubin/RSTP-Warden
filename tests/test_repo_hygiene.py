"""Repository hygiene guards.

A fresh checkout (a clone, CI, or a ``git worktree``) contains only what git tracks. These tests
fail when an ignore rule hides a source file, when runtime output stops being ignored, when the
``.worktrees`` directory used for parallel branches is not ignored, and when the Docker build
context would include it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Directories whose files must always be committable, and the file types that live in them.
SOURCE_DIRS = ("src", "tests", "migrations")
SOURCE_SUFFIXES = frozenset({".py", ".html", ".js", ".css", ".mako", ".cfg", ".names", ".xml"})

# What `serve` writes under `record.output_dir` (paths need not exist; `git check-ignore
# --no-index` matches the patterns only).
RUNTIME_PATHS = (
    "recordings/front/main/front_main_20261002-120000.ts",
    "recordings/front/thumbnails/12.jpg",
    "recordings/front/clips/12.mp4",
    "clips/front_12.mp4",
    "clips/concat_list.txt",
    "examples/recordings/front/clips/12.mp4",
    "examples/recordings/front/thumbnails/12.jpg",
    "data/recordings/front/main/front_main_20261002-120000.ts",
)

requires_git_checkout = pytest.mark.skipif(
    shutil.which("git") is None or not (REPO_ROOT / ".git").exists(),
    reason="needs the git CLI and a git checkout (not an sdist or a git-archive copy)",
)


def _git(*args: str, stdin: str | None = None) -> subprocess.CompletedProcess[str]:
    # Hermetic: the user's global/system git config and excludes must not change the outcome.
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), "-c", f"core.excludesFile={os.devnull}", *args],
        input=stdin,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def _source_files() -> list[str]:
    files: list[str] = []
    for top in SOURCE_DIRS:
        base = REPO_ROOT / top
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if "__pycache__" in path.parts or not path.is_file():
                continue
            if path.suffix in SOURCE_SUFFIXES:
                files.append(path.relative_to(REPO_ROOT).as_posix())
    return sorted(files)


@requires_git_checkout
def test_no_source_file_is_hidden_by_an_ignore_rule() -> None:
    files = _source_files()
    assert "src/rtsp_warden/web/templates/base.html" in files
    # --no-index checks the ignore rules themselves, so a tracked file still counts: a rule that
    # matches it would silently skip every new sibling file a later `git add` should pick up.
    result = _git("check-ignore", "--no-index", "--stdin", stdin="\n".join(files) + "\n")
    assert result.returncode in (0, 1), result.stderr
    hidden = result.stdout.split()
    assert hidden == [], f"source files matched by an ignore rule: {hidden}"


@requires_git_checkout
def test_runtime_output_stays_ignored() -> None:
    result = _git("check-ignore", "--no-index", "--stdin", stdin="\n".join(RUNTIME_PATHS) + "\n")
    assert result.returncode == 0, result.stderr
    assert sorted(result.stdout.split()) == sorted(RUNTIME_PATHS)


@requires_git_checkout
def test_worktrees_dir_is_ignored_before_it_exists() -> None:
    # A ".worktrees/" pattern (trailing slash) does not match while the directory is absent.
    result = _git("check-ignore", "-q", ".worktrees")
    assert result.returncode == 0, "'.worktrees' is not ignored; add a '.worktrees' line"


def test_dockerignore_excludes_worktrees() -> None:
    text = (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8")
    lines = [line.strip() for line in text.splitlines()]
    assert ".worktrees" in lines
    glued = [line for line in lines if line.endswith(".worktrees") and line != ".worktrees"]
    assert glued == [], f"pattern glued onto the previous line: {glued}"
    assert text.endswith("\n")
