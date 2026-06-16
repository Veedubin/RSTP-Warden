"""Grid-based detection zone mask.

A GridMask divides the frame into an N x M grid of cells. Cells in
`blocked_cells` have detections SUPPRESSED (the "block the road" use case).
Cells not in blocked_cells are ACTIVE -- detections there are kept.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .base import Detection


@dataclass(slots=True)
class GridMask:
    """Grid-based zone mask.

    Attributes:
        grid_cols: Number of columns (e.g. 16).
        grid_rows: Number of rows (e.g. 16).
        blocked_cells: Set of (col, row) tuples that are BLOCKED.
        frame_width: Snapshot frame width (used for coordinate math).
        frame_height: Snapshot frame height.
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

    def is_cell_blocked(self, col: int, row: int) -> bool:
        """Return True if the given cell is blocked."""
        return (col, row) in self.blocked_cells

    def cell_for_point(self, x: float, y: float) -> tuple[int, int]:
        """Map a frame coordinate to a grid cell (col, row)."""
        col = int(x * self.grid_cols / self.frame_width)
        row = int(y * self.grid_rows / self.frame_height)
        col = max(0, min(self.grid_cols - 1, col))
        row = max(0, min(self.grid_rows - 1, row))
        return (col, row)

    def filter_detections(self, detections: list[Detection]) -> list[Detection]:
        """Remove detections whose bbox center is in a blocked cell."""
        result = []
        for det in detections:
            if det.bbox is None:
                continue  # No bbox = can't determine cell = drop
            x, y, w, h = det.bbox
            cx = x + w // 2
            cy = y + h // 2
            col, row = self.cell_for_point(cx, cy)
            if not self.is_cell_blocked(col, row):
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


__all__ = ["GridMask"]
