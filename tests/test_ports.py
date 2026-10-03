"""Tests for rtsp_warden.ports (moved out of cli.py so the web layer can use it)."""

import socket

from rtsp_warden import cli, ports


def test_listening_port_is_not_free():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        port = s.getsockname()[1]
        assert ports.port_is_free("127.0.0.1", port) is False


def test_unused_port_is_free():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    assert ports.port_is_free("127.0.0.1", port) is True


def test_cli_uses_the_shared_helper():
    assert cli._port_is_free is ports.port_is_free
