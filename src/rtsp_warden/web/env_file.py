"""Locked, atomic KEY="VALUE" updates to a .env file (camera credentials).

The add/edit camera routes store per-camera credentials as ``${CAM_<SLUG>_USER}`` /
``${CAM_<SLUG>_PASS}`` references in config.yaml and write the values here, in the
``.env`` next to config.yaml. ``cli._load_dotenv`` reads the file back at startup.
"""

from __future__ import annotations

import fcntl
import logging
import os
import re
from collections.abc import Mapping
from pathlib import Path

log = logging.getLogger(__name__)

_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _quote_value(value: str) -> str:
    """Double-quote ``value`` the way ``cli._parse_dotenv_value`` unquotes it."""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _line_key(line: str) -> str | None:
    """Return the variable name a .env line defines, or None for blanks and comments."""
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in stripped:
        return None
    return stripped.partition("=")[0].strip()


def upsert_env_vars(path: Path, values: Mapping[str, str]) -> None:
    """Set ``values`` in the .env file at ``path``; other lines are kept as they are.

    An existing ``KEY=`` line is replaced in place (later duplicates of that key are
    dropped); new keys are appended in the order given. The file is rewritten through
    a temp file and ``os.replace`` under an exclusive ``flock`` on
    ``path.with_suffix(".lock")`` and always ends up with mode 0600. Values are never
    logged.

    Raises ValueError for a key that is not a valid variable name or a value that
    contains a line break or NUL; the file is not touched in that case.
    """
    for key, value in values.items():
        if not _KEY_RE.match(key):
            raise ValueError(f"invalid environment variable name: {key!r}")
        if any(ch in value for ch in ("\n", "\r", "\0")):
            raise ValueError(f"value for {key} contains a line break or NUL character")
    if not values:
        return

    lock_path = path.with_suffix(".lock")
    lock_fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
        pending = dict(values)
        written: set[str] = set()
        out: list[str] = []
        for line in lines:
            key = _line_key(line)
            if key is not None and key in values:
                if key in written:
                    continue  # drop a later duplicate of a key we already wrote
                out.append(f"{key}={_quote_value(values[key])}")
                written.add(key)
                pending.pop(key, None)
                continue
            out.append(line)
        for key, value in pending.items():
            out.append(f"{key}={_quote_value(value)}")

        tmp = path.with_name(path.name + ".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.fchmod(fd, 0o600)
            os.write(fd, ("\n".join(out) + "\n").encode("utf-8"))
        finally:
            os.close(fd)
        os.replace(tmp, path)
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
    log.info("Updated %s: %s", path, ", ".join(sorted(values)))


def read_env_file(path: Path) -> dict[str, str]:
    """Return the variables a .env file defines, parsed the way ``serve`` reads them.

    The first definition of a key wins, as in ``cli._load_dotenv_file``. A missing file
    reads as empty.
    """
    from ..cli import _parse_dotenv_value  # imported late: cli imports the web layer

    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key = _line_key(line)
        if key and key not in values:
            values[key] = _parse_dotenv_value(line.strip().partition("=")[2])
    return values
