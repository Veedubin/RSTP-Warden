"""Atomic config.yaml writer with file locking.

Used by web UI handlers that mutate config.yaml (zones, sensitivity, presets).
Wraps a fcntl.flock exclusive lock + os.replace() for crash safety.
"""

from __future__ import annotations

import fcntl
import os
from pathlib import Path

import yaml


def _locked_write_yaml(path: Path, data: dict) -> None:
    """Atomically write data to path with an exclusive file lock.

    1. Open a lock file in the same directory (path.parent / f".{path.name}.lock")
    2. Acquire exclusive lock
    3. Write data as YAML to a temp file
    4. os.replace(tmp, path) for atomic swap
    5. Release lock
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.parent / f".{path.name}.lock"
    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        try:
            tmp = path.with_suffix(path.suffix + ".tmp")
            with open(tmp, "w") as f:
                yaml.safe_dump(data, f, sort_keys=False)
            os.replace(tmp, path)
        finally:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
