"""TCP port helpers shared by the CLI and the web layer."""

from __future__ import annotations

import socket


def port_is_free(host: str, port: int) -> bool:
    """Return True when a TCP socket can bind ``(host, port)`` right now.

    SO_REUSEADDR is set the way the servers set it, so a listening socket counts as
    taken and a port left in TIME_WAIT counts as free. A host that is not an IPv4
    address of this machine also returns False.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, int(port)))
        return True
    except OSError:
        return False
