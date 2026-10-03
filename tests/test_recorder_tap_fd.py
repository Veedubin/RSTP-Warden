"""The frame tap pipe: ffmpeg is told the fd it really gets, and no tap fd leaks.

``subprocess.Popen`` is replaced by ``_FakeChild``, which behaves like the ffmpeg child.
Popen hands a child only fds 0-2 and ``pass_fds``, under their parent numbers, so the fake
writes two JPEG frames to the fd named by its last argument (the tap is always the last
ffmpeg output) only when that fd is in ``pass_fds``, and otherwise fails the way ffmpeg
does (rc 247, "Bad file descriptor"). Nothing real is spawned.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from rtsp_warden.config import RuntimeConfig, StreamRecordConfig
from rtsp_warden.frame_tap import FrameTapDispatcher
from rtsp_warden.recorder import StreamIngestor

FRAME_ONE = b"\xff\xd8one\xff\xd9"
FRAME_TWO = b"\xff\xd8two\xff\xd9"


class _Collect:
    """FrameConsumer that keeps every frame it is given."""

    name = "collect"

    def __init__(self) -> None:
        self.frames: list[tuple[str, str, bytes]] = []

    def on_frame(self, camera: str, stream: str, jpeg_bytes: bytes, ts_unix: float) -> None:
        self.frames.append((camera, stream, jpeg_bytes))


class _FakeChild:
    """subprocess.Popen stand-in for an ffmpeg that writes two tap frames, then exits."""

    def __init__(self, args: list[str], *, pass_fds: tuple[int, ...] = (), **_kw: Any) -> None:
        self.args = list(args)
        self.pass_fds = tuple(pass_fds)
        self.stdout = None
        self.stderr = None
        self.pid = 4242
        self.returncode = 0
        target = self.args[-1]
        if target.startswith("pipe:"):
            fd = int(target.split(":", 1)[1])
            if fd in self.pass_fds:
                os.write(fd, FRAME_ONE + FRAME_TWO)
            else:
                self.returncode = 247  # the fd is not open in the child

    def poll(self) -> int:
        return self.returncode


@dataclass
class _Spy:
    pipes: list[tuple[int, int]] = field(default_factory=list)
    closed: list[int] = field(default_factory=list)
    children: list[_FakeChild] = field(default_factory=list)


@pytest.fixture
def spy(monkeypatch: pytest.MonkeyPatch) -> _Spy:
    """Record every os.pipe() pair, every os.close() and every (fake) child spawned."""
    seen = _Spy()
    real_pipe, real_close = os.pipe, os.close

    def pipe() -> tuple[int, int]:
        pair = real_pipe()
        seen.pipes.append(pair)
        return pair

    def close(fd: int) -> None:
        seen.closed.append(fd)
        real_close(fd)

    def popen(args: list[str], **kw: Any) -> _FakeChild:
        child = _FakeChild(args, **kw)
        seen.children.append(child)
        return child

    monkeypatch.setattr(os, "pipe", pipe)
    monkeypatch.setattr(os, "close", close)
    monkeypatch.setattr(subprocess, "Popen", popen)
    return seen


def _tap_only(sink: _Collect) -> StreamIngestor:
    """An ingestor whose only ffmpeg output is the frame tap."""
    return StreamIngestor(
        camera_name="cam",
        stream_name="main",
        upstream_url="rtsp://u:p@h/m",
        runtime=RuntimeConfig(),
        record_cfg=StreamRecordConfig(enabled=False, container="ts"),
        frame_tap_enabled=True,
        frame_tap_dispatcher=FrameTapDispatcher(consumers=(sink,)),
    )


def _join_reader(ing: StreamIngestor) -> None:
    reader = ing._frame_tap_thread
    assert reader is not None
    reader.join(timeout=2.0)
    assert not reader.is_alive()


def test_tap_argv_names_the_fd_the_child_receives(spy: _Spy) -> None:
    sink = _Collect()
    ing = _tap_only(sink)

    ing.start()
    try:
        child = spy.children[0]
        assert child.args[-1] == f"pipe:{child.pass_fds[0]}"
        assert child.returncode == 0
        _join_reader(ing)
        assert sink.frames == [("cam", "main", FRAME_ONE), ("cam", "main", FRAME_TWO)]
    finally:
        ing.stop()


def test_parent_drops_the_write_end_and_the_reader_closes_the_read_end(spy: _Spy) -> None:
    ing = _tap_only(_Collect())

    ing.start()
    try:
        r_fd, w_fd = spy.pipes[0]
        assert ing.frame_tap_write_fd is None
        assert w_fd in spy.closed
        # EOF reaches the reader only once no write end is open anywhere, so the reader
        # exiting proves the parent let go of its copy as soon as ffmpeg was spawned.
        _join_reader(ing)
        assert r_fd in spy.closed
    finally:
        ing.stop()

    # Nothing is closed twice: a freed fd number may already belong to another camera.
    assert spy.closed.count(w_fd) == 1
    assert spy.closed.count(r_fd) == 1


def test_restarts_do_not_leak_tap_fds(spy: _Spy) -> None:
    ing = _tap_only(_Collect())

    for _ in range(3):  # what the supervisor does after every ffmpeg crash
        ing.start()
        _join_reader(ing)
        ing.stop()

    assert len(spy.pipes) == 3
    # fd numbers are reused from one cycle to the next, so count per number: every
    # opening of a tap fd is matched by exactly one close.
    opened = [fd for pair in spy.pipes for fd in pair]
    for fd in set(opened):
        assert spy.closed.count(fd) == opened.count(fd)


def test_a_failed_spawn_closes_both_tap_fds(spy: _Spy, monkeypatch: pytest.MonkeyPatch) -> None:
    def no_ffmpeg(args: list[str], **_kw: Any) -> _FakeChild:
        raise FileNotFoundError(2, "No such file or directory", "ffmpeg")

    monkeypatch.setattr(subprocess, "Popen", no_ffmpeg)
    ing = _tap_only(_Collect())

    with pytest.raises(FileNotFoundError):
        ing.start()

    r_fd, w_fd = spy.pipes[0]
    assert r_fd in spy.closed
    assert w_fd in spy.closed
    assert ing.frame_tap_write_fd is None
    assert ing._frame_tap_read_fd is None
    assert ing._frame_tap_thread is None
    ing.stop()
    assert spy.closed.count(r_fd) == 1
    assert spy.closed.count(w_fd) == 1


def test_a_tap_without_a_dispatcher_opens_no_pipe(spy: _Spy, tmp_path: Path) -> None:
    ing = StreamIngestor(
        camera_name="cam",
        stream_name="main",
        upstream_url="rtsp://u:p@h/m",
        runtime=RuntimeConfig(),
        record_cfg=StreamRecordConfig(enabled=True, container="ts"),
        record_output_dir=tmp_path,
        frame_tap_enabled=True,
        frame_tap_dispatcher=None,
    )

    ing.start()
    try:
        child = spy.children[0]
        assert spy.pipes == []
        assert child.pass_fds == ()
        assert "mjpeg" not in child.args
        assert ing._frame_tap_thread is None
    finally:
        ing.stop()
