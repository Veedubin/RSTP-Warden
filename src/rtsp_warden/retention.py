from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import RetentionConfig

log = logging.getLogger(__name__)


def _sweep_roots(root: Path, only_subdirs: tuple[str, ...] | None) -> list[Path]:
    """The directories a sweep walks: the camera root, or only the named sub-directories."""
    if only_subdirs is None:
        return [root]
    return [root / name for name in only_subdirs]


def _iter_files(root: Path, only_subdirs: tuple[str, ...] | None = None) -> list[Path]:
    files: list[Path] = []
    for base in _sweep_roots(root, only_subdirs):
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if p.is_file():
                files.append(p)
    return files


# Recorded segments live directly in these sub-directories of a camera root. Only they
# count toward keep_last_n; thumbnails and clips are swept by max_days / max_gb (ruling R20).
_SEGMENT_DIRS = frozenset({"main", "sub"})
# Never pruned when empty, so the thumbnail and clip writers never race an rmdir.
_KEPT_DIRS = frozenset({"thumbnails", "clips"})


def _is_segment(root: Path, p: Path) -> bool:
    """True for a file directly inside ``<root>/main`` or ``<root>/sub``."""
    return p.parent.parent == root and p.parent.name in _SEGMENT_DIRS


def _safe_unlink(p: Path) -> bool:
    try:
        p.unlink()
        return True
    except FileNotFoundError:
        return False
    except Exception as e:
        log.warning(f"[retention] failed to delete {p}: {e!r}")
        return False


def _prune_empty_dirs(root: Path, only_subdirs: tuple[str, ...] | None = None) -> None:
    dirs: list[Path] = []
    for base in _sweep_roots(root, only_subdirs):
        if base.exists():
            dirs.extend(d for d in base.rglob("*") if d.is_dir())
    # bottom-up: remove empty leaf dirs
    for p in sorted(dirs, key=lambda x: len(str(x)), reverse=True):
        if p.parent == root and p.name in _KEPT_DIRS:
            continue
        try:
            if not any(p.iterdir()):
                p.rmdir()
        except Exception:
            pass


@dataclass
class RetentionManager:
    camera_name: str
    camera_root: Path  # e.g. recordings/<camera>
    cfg: RetentionConfig
    # When set, only these sub-directories of camera_root are swept. A camera with
    # recording off sweeps ("thumbnails", "clips") only, so segments a user kept from
    # when it recorded are never deleted behind their back.
    only_subdirs: tuple[str, ...] | None = None

    _next_run: float = field(default=0.0, init=False)

    def maybe_run(self) -> None:
        now = time.time()
        if self.cfg.cleanup_interval_seconds <= 0:
            return
        if self._next_run and now < self._next_run:
            return
        self._next_run = now + float(self.cfg.cleanup_interval_seconds)
        self.run()

    def run(self) -> None:
        # Nothing configured: skip.
        if (
            (self.cfg.max_days is None)
            and (self.cfg.max_gb is None)
            and (self.cfg.keep_last_n <= 0)
        ):
            return

        files = _iter_files(self.camera_root, self.only_subdirs)
        if not files:
            return

        entries = []
        for f in files:
            try:
                st = f.stat()
            except FileNotFoundError:
                continue
            entries.append((f, st.st_mtime, st.st_size))

        # newest first
        entries.sort(key=lambda x: x[1], reverse=True)

        protected = set()
        if self.cfg.keep_last_n and self.cfg.keep_last_n > 0:
            segments = [e for e in entries if _is_segment(self.camera_root, e[0])]
            for f, _, _ in segments[: self.cfg.keep_last_n]:
                protected.add(f)

        removed = 0
        removed_bytes = 0

        # Rule 1: max_days
        if self.cfg.max_days is not None:
            cutoff = time.time() - (self.cfg.max_days * 86400)
            # oldest first for deletion
            for f, mtime, size in sorted(entries, key=lambda x: x[1]):
                if f in protected:
                    continue
                if mtime < cutoff:
                    if _safe_unlink(f):
                        removed += 1
                        removed_bytes += size

        # Refresh after deletions
        files2 = []
        total = 0
        for f in _iter_files(self.camera_root, self.only_subdirs):
            try:
                st = f.stat()
            except FileNotFoundError:
                continue
            total += st.st_size
            files2.append((f, st.st_mtime, st.st_size))
        files2.sort(key=lambda x: x[1], reverse=True)

        # Rule 2: max_gb
        if self.cfg.max_gb is not None:
            limit = int(self.cfg.max_gb * (1024**3))
            if total > limit:
                # delete oldest until within limit, skipping protected
                for f, _, size in sorted(files2, key=lambda x: x[1]):
                    if f in protected:
                        continue
                    if total <= limit:
                        break
                    if _safe_unlink(f):
                        removed += 1
                        removed_bytes += size
                        total -= size

        if removed:
            log.info(
                "[retention] %s: deleted %d files (%.1f MiB)",
                self.camera_name,
                removed,
                removed_bytes / 1024 / 1024,
            )
        _prune_empty_dirs(self.camera_root, self.only_subdirs)
