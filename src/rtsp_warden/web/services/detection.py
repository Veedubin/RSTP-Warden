"""Detection helpers for the web UI (RW-3).

Task 15 grows this module (Detection panel, rules, detector write-back); the
config.yaml write-failure message lives here so every RW-3 write-back route
(zones included) words a failed write the same way (ruling R9).
"""

from __future__ import annotations

from pathlib import Path


def write_failed_message(config_path: Path, exc: OSError) -> str:
    """User-facing text for a config.yaml write that failed (read-only mount, permissions)."""
    reason = exc.strerror or str(exc) or type(exc).__name__
    return f"Could not save {config_path}: {reason}. The change is active until the next restart."
