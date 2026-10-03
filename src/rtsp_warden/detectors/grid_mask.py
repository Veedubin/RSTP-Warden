"""Grid-based detection zone mask.

A GridMask divides the frame into an N x M grid of cells. Cells in
`blocked_cells` have detections SUPPRESSED (the "block the road" use case).
Cells not in blocked_cells are ACTIVE -- detections there are kept.

Cells are fractions of the frame a point was measured in: pass that frame's
size (the decoded tap frame at runtime). The zone's saved frame size is only
the fallback divisor when no frame size is given.

The same mask also describes a named ``area`` zone: its active cells are the
area (see `zone_for_point`). Area zones never filter detections.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .base import Detection

if TYPE_CHECKING:
    from ..config import GridZoneConfig


@dataclass(slots=True)
class GridMask:
    """Grid-based zone mask.

    Attributes:
        grid_cols: Number of columns (e.g. 16).
        grid_rows: Number of rows (e.g. 16).
        blocked_cells: Set of (col, row) tuples that are BLOCKED.
        frame_width: Frame width the zone was drawn at (fallback divisor only).
        frame_height: Frame height the zone was drawn at (fallback divisor only).
        name: Optional label.
    """

    grid_cols: int
    grid_rows: int
    blocked_cells: set[tuple[int, int]] = field(default_factory=set)
    frame_width: int = 1920
    frame_height: int = 1080
    name: str = ""

    def __post_init__(self) -> None:
        if self.grid_cols < 2 or self.grid_cols > 64:
            raise ValueError(f"grid_cols must be 2-64, got {self.grid_cols}")
        if self.grid_rows < 2 or self.grid_rows > 64:
            raise ValueError(f"grid_rows must be 2-64, got {self.grid_rows}")
        for c, r in self.blocked_cells:
            if not (0 <= c < self.grid_cols and 0 <= r < self.grid_rows):
                raise ValueError(
                    f"blocked cell ({c},{r}) out of bounds for {self.grid_cols}x{self.grid_rows}"
                )

    @classmethod
    def from_zone(cls, zone: GridZoneConfig) -> GridMask:
        """Build a mask from a camera zone. The zone's kind is the caller's business."""
        return cls(
            grid_cols=zone.grid_cols,
            grid_rows=zone.grid_rows,
            blocked_cells=set(zone.blocked_cells),
            frame_width=zone.frame_width,
            frame_height=zone.frame_height,
            name=zone.name,
        )

    def is_cell_blocked(self, col: int, row: int) -> bool:
        """Return True if the given cell is blocked."""
        return (col, row) in self.blocked_cells

    def cell_for_point(
        self,
        x: float,
        y: float,
        frame_w: int | None = None,
        frame_h: int | None = None,
    ) -> tuple[int, int]:
        """Map a point to a grid cell (col, row).

        ``frame_w``/``frame_h`` are the size of the frame the point was measured
        in. When missing or not positive, the zone's saved size is used.
        """
        width = frame_w if frame_w is not None and frame_w > 0 else self.frame_width
        height = frame_h if frame_h is not None and frame_h > 0 else self.frame_height
        col = int(x * self.grid_cols / width)
        row = int(y * self.grid_rows / height)
        col = max(0, min(self.grid_cols - 1, col))
        row = max(0, min(self.grid_rows - 1, row))
        return (col, row)

    def contains_point(
        self,
        x: float,
        y: float,
        frame_w: int | None = None,
        frame_h: int | None = None,
    ) -> bool:
        """Return True when the point lies in an active (not blocked) cell."""
        col, row = self.cell_for_point(x, y, frame_w, frame_h)
        return not self.is_cell_blocked(col, row)

    def filter_detections(
        self,
        detections: list[Detection],
        frame_w: int | None = None,
        frame_h: int | None = None,
    ) -> list[Detection]:
        """Remove detections whose bbox center is in a blocked cell.

        ``frame_w``/``frame_h`` are the size of the frame the boxes were found in.
        """
        result = []
        for det in detections:
            if det.bbox is None:
                continue  # No bbox = can't determine cell = drop
            x, y, w, h = det.bbox
            if self.contains_point(x + w // 2, y + h // 2, frame_w, frame_h):
                result.append(det)
        return result

    def cells_to_polygons(self) -> list[list[tuple[int, int]]]:
        """Convert blocked cells to polygons (for visualization in UI)."""
        polygons = []
        cell_w = self.frame_width / self.grid_cols
        cell_h = self.frame_height / self.grid_rows
        for c, r in self.blocked_cells:
            x1 = int(c * cell_w)
            y1 = int(r * cell_h)
            x2 = int((c + 1) * cell_w)
            y2 = int((r + 1) * cell_h)
            polygons.append([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
        return polygons


def zone_for_point(
    area_masks: Sequence[tuple[str, GridMask]],
    x: float,
    y: float,
    frame_w: int,
    frame_h: int,
) -> str:
    """Name of the first area zone (config order) whose active cells contain (x, y), else ""."""
    for name, mask in area_masks:
        if mask.contains_point(x, y, frame_w, frame_h):
            return name
    return ""


__all__ = ["GridMask", "zone_for_point"]
