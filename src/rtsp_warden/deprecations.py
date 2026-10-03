"""One-time deprecation warnings for config keys and detector types.

Config models are validated more than once per process (startup, web
write-back round trips, hot reload), so every deprecation goes through
``warn_once``: the first call with a given key logs, later calls are silent.
This module imports nothing from the package, so both ``config.py`` and
``detectors/registry.py`` can use it without an import cycle.
"""

from __future__ import annotations

import logging
import threading

log = logging.getLogger(__name__)

_WARNED: set[str] = set()
_LOCK = threading.Lock()


def warn_once(key: str, msg: str, *args: object) -> None:
    """Log ``msg % args`` as a warning the first time ``key`` is seen in this process."""
    with _LOCK:
        if key in _WARNED:
            return
        _WARNED.add(key)
    log.warning(msg, *args)
