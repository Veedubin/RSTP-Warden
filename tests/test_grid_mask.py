"""Tests for detectors/grid_mask.py: GridMask class."""

from __future__ import annotations

import pytest

from rtsp_warden.detectors.base import Detection
from rtsp_warden.detectors.grid_mask import GridMask, zone_for_point

# ---------------------------------------------------------------------------
# Construction and validation
# ---------------------------------------------------------------------------


def test_grid_mask_default_values() -> None:
    """GridMask can be constructed with just grid dimensions."""
    gm = GridMask(grid_cols=8, grid_rows=8)
    assert gm.grid_cols == 8
    assert gm.grid_rows == 8
    assert gm.blocked_cells == set()
    assert gm.frame_width == 1920
    assert gm.frame_height == 1080
    assert gm.name == ""


def test_grid_mask_with_blocked_cells() -> None:
    """GridMask stores blocked cells correctly."""
    blocked = {(0, 0), (7, 7), (3, 4)}
    gm = GridMask(grid_cols=8, grid_rows=8, blocked_cells=blocked)
    assert gm.blocked_cells == blocked


def test_grid_mask_invalid_cols_too_small() -> None:
    """GridMask raises ValueError for grid_cols < 2."""
    with pytest.raises(ValueError, match="grid_cols must be 2-64"):
        GridMask(grid_cols=1, grid_rows=8)


def test_grid_mask_invalid_cols_too_large() -> None:
    """GridMask raises ValueError for grid_cols > 64."""
    with pytest.raises(ValueError, match="grid_cols must be 2-64"):
        GridMask(grid_cols=65, grid_rows=8)


def test_grid_mask_invalid_rows_too_small() -> None:
    """GridMask raises ValueError for grid_rows < 2."""
    with pytest.raises(ValueError, match="grid_rows must be 2-64"):
        GridMask(grid_cols=8, grid_rows=1)


def test_grid_mask_invalid_rows_too_large() -> None:
    """GridMask raises ValueError for grid_rows > 64."""
    with pytest.raises(ValueError, match="grid_rows must be 2-64"):
        GridMask(grid_cols=8, grid_rows=65)


def test_grid_mask_out_of_bounds_blocked_cell() -> None:
    """GridMask raises ValueError for blocked cells outside grid bounds."""
    with pytest.raises(ValueError, match="out of bounds"):
        GridMask(grid_cols=8, grid_rows=8, blocked_cells={(8, 0)})


def test_grid_mask_out_of_bounds_blocked_cell_negative() -> None:
    """GridMask raises ValueError for negative blocked cell coords."""
    with pytest.raises(ValueError, match="out of bounds"):
        GridMask(grid_cols=8, grid_rows=8, blocked_cells={(-1, 0)})


def test_grid_mask_empty_blocked_cells_valid() -> None:
    """GridMask with empty blocked_cells is valid."""
    gm = GridMask(grid_cols=16, grid_rows=16, blocked_cells=set())
    assert gm.blocked_cells == set()


# ---------------------------------------------------------------------------
# is_cell_blocked
# ---------------------------------------------------------------------------


def test_is_cell_blocked_true() -> None:
    """is_cell_blocked returns True for blocked cells."""
    gm = GridMask(grid_cols=8, grid_rows=8, blocked_cells={(3, 4)})
    assert gm.is_cell_blocked(3, 4) is True


def test_is_cell_blocked_false() -> None:
    """is_cell_blocked returns False for non-blocked cells."""
    gm = GridMask(grid_cols=8, grid_rows=8, blocked_cells={(3, 4)})
    assert gm.is_cell_blocked(0, 0) is False


# ---------------------------------------------------------------------------
# cell_for_point
# ---------------------------------------------------------------------------


def test_cell_for_point_origin() -> None:
    """Point (0, 0) maps to cell (0, 0)."""
    gm = GridMask(grid_cols=8, grid_rows=8, frame_width=800, frame_height=600)
    assert gm.cell_for_point(0, 0) == (0, 0)


def test_cell_for_point_max() -> None:
    """Point at (frame_width, frame_height) maps to last cell."""
    gm = GridMask(grid_cols=8, grid_rows=8, frame_width=800, frame_height=600)
    # Just below the boundary maps to (7, 7)
    assert gm.cell_for_point(799, 599) == (7, 7)


def test_cell_for_point_middle() -> None:
    """Point in the center of the frame maps to the center cell."""
    gm = GridMask(grid_cols=8, grid_rows=8, frame_width=800, frame_height=600)
    # x=400 -> col=400*8/800=4, y=300 -> row=300*8/600=4
    assert gm.cell_for_point(400, 300) == (4, 4)


def test_cell_for_point_clamps_negative() -> None:
    """Negative coordinates are clamped to (0, 0)."""
    gm = GridMask(grid_cols=8, grid_rows=8, frame_width=800, frame_height=600)
    assert gm.cell_for_point(-10, -10) == (0, 0)


def test_cell_for_point_clamps_overflow() -> None:
    """Coordinates beyond frame size are clamped to last cell."""
    gm = GridMask(grid_cols=8, grid_rows=8, frame_width=800, frame_height=600)
    assert gm.cell_for_point(900, 700) == (7, 7)


# ---------------------------------------------------------------------------
# filter_detections
# ---------------------------------------------------------------------------


def test_filter_detections_removes_blocked() -> None:
    """Detections whose bbox center is in a blocked cell are removed."""
    gm = GridMask(
        grid_cols=4,
        grid_rows=4,
        frame_width=400,
        frame_height=400,
        blocked_cells={(1, 1)},
    )
    # Detection at center (100, 100) -> cell (1, 1) -> blocked
    dets = [Detection(kind="motion", confidence=0.9, bbox=(80, 80, 40, 40))]
    result = gm.filter_detections(dets)
    assert len(result) == 0


def test_filter_detections_keeps_non_blocked() -> None:
    """Detections outside blocked cells are kept."""
    gm = GridMask(
        grid_cols=4,
        grid_rows=4,
        frame_width=400,
        frame_height=400,
        blocked_cells={(1, 1)},
    )
    # Detection at center (50, 50) -> cell (0, 0) -> not blocked
    dets = [Detection(kind="motion", confidence=0.9, bbox=(30, 30, 40, 40))]
    result = gm.filter_detections(dets)
    assert len(result) == 1


def test_filter_detections_drops_no_bbox() -> None:
    """Detections without a bbox are dropped."""
    gm = GridMask(grid_cols=4, grid_rows=4, frame_width=400, frame_height=400)
    dets = [Detection(kind="motion", confidence=0.9, bbox=None)]
    result = gm.filter_detections(dets)
    assert len(result) == 0


def test_filter_detections_mixed() -> None:
    """A mix of blocked and non-blocked detections."""
    gm = GridMask(
        grid_cols=4,
        grid_rows=4,
        frame_width=400,
        frame_height=400,
        blocked_cells={(0, 0)},
    )
    dets = [
        Detection(kind="motion", confidence=0.9, bbox=(10, 10, 40, 40)),  # cell (0,0) blocked
        Detection(kind="person", confidence=0.8, bbox=(210, 210, 40, 40)),  # cell (2,2) ok
    ]
    result = gm.filter_detections(dets)
    assert len(result) == 1
    assert result[0].kind == "person"


# ---------------------------------------------------------------------------
# cells_to_polygons
# ---------------------------------------------------------------------------


def test_cells_to_polygons_empty() -> None:
    """No blocked cells -> no polygons."""
    gm = GridMask(grid_cols=4, grid_rows=4, frame_width=400, frame_height=400)
    assert gm.cells_to_polygons() == []


def test_cells_to_polygons_single() -> None:
    """One blocked cell produces one polygon."""
    gm = GridMask(
        grid_cols=4,
        grid_rows=4,
        frame_width=400,
        frame_height=400,
        blocked_cells={(1, 2)},
    )
    polys = gm.cells_to_polygons()
    assert len(polys) == 1
    # Cell (1,2): x1=100, y1=200, x2=200, y2=300
    assert polys[0] == [(100, 200), (200, 200), (200, 300), (100, 300)]


def test_cells_to_polygons_multiple() -> None:
    """Multiple blocked cells produce multiple polygons."""
    gm = GridMask(
        grid_cols=4,
        grid_rows=4,
        frame_width=400,
        frame_height=400,
        blocked_cells={(0, 0), (3, 3)},
    )
    polys = gm.cells_to_polygons()
    assert len(polys) == 2


# ---------------------------------------------------------------------------
# Different grid sizes
# ---------------------------------------------------------------------------


def test_8x8_grid() -> None:
    """8x8 grid works correctly."""
    gm = GridMask(grid_cols=8, grid_rows=8, frame_width=1920, frame_height=1080)
    # Point at (960, 540) -> col=4, row=4
    assert gm.cell_for_point(960, 540) == (4, 4)


def test_16x16_grid() -> None:
    """16x16 grid works correctly."""
    gm = GridMask(grid_cols=16, grid_rows=16, frame_width=1920, frame_height=1080)
    # Point at (120, 67) -> col=int(120*16/1920)=1, row=int(67*16/1080)=0
    assert gm.cell_for_point(120, 67) == (1, 0)
    # Point at (120, 135) -> col=1, row=int(135*16/1080)=2
    assert gm.cell_for_point(120, 135) == (1, 2)


def test_32x32_grid() -> None:
    """32x32 grid works correctly."""
    gm = GridMask(grid_cols=32, grid_rows=32, frame_width=1920, frame_height=1080)
    assert gm.cell_for_point(0, 0) == (0, 0)
    assert gm.cell_for_point(1919, 1079) == (31, 31)


# ---------------------------------------------------------------------------
# Integration with GridZoneConfig
# ---------------------------------------------------------------------------


def test_grid_mask_from_grid_zone_config() -> None:
    """GridMask can be built from a GridZoneConfig."""
    from rtsp_warden.config import GridZoneConfig as GZC

    zc = GZC(
        name="test_zone",
        grid_cols=8,
        grid_rows=8,
        blocked_cells={(0, 0), (1, 1)},
        frame_width=1920,
        frame_height=1080,
    )
    gm = GridMask(
        name=zc.name,
        grid_cols=zc.grid_cols,
        grid_rows=zc.grid_rows,
        blocked_cells=set(zc.blocked_cells),
        frame_width=zc.frame_width,
        frame_height=zc.frame_height,
    )
    assert gm.name == "test_zone"
    assert gm.is_cell_blocked(0, 0)
    assert gm.is_cell_blocked(1, 1)
    assert not gm.is_cell_blocked(2, 2)


# ---------------------------------------------------------------------------
# Runtime frame size, contains_point, from_zone, zone_for_point (RW-3 Task 8)
# ---------------------------------------------------------------------------


def _right_half_blocked() -> GridMask:
    """16x16 zone saved at the editor default 1920x1080 with the right half blocked."""
    return GridMask(
        grid_cols=16,
        grid_rows=16,
        blocked_cells={(c, r) for c in range(8, 16) for r in range(16)},
        frame_width=1920,
        frame_height=1080,
        name="right_half",
    )


def test_cell_for_point_divides_by_the_given_frame_size() -> None:
    """A point in a 320x180 tap frame lands in the same fraction of the grid."""
    gm = _right_half_blocked()
    # (240, 90) is three quarters across and halfway down a 320x180 frame.
    assert gm.cell_for_point(240, 90, 320, 180) == (12, 8)
    # The same picture tapped at 640x360.
    assert gm.cell_for_point(480, 180, 640, 360) == (12, 8)


def test_cell_for_point_falls_back_to_the_saved_size() -> None:
    """Without a frame size the zone's saved frame_width/frame_height are the divisor."""
    gm = _right_half_blocked()
    assert gm.cell_for_point(240, 90) == (2, 1)
    assert gm.cell_for_point(1440, 540) == (12, 8)


def test_cell_for_point_ignores_a_non_positive_frame_size() -> None:
    """A zero frame size falls back to the saved size instead of dividing by zero."""
    gm = GridMask(grid_cols=8, grid_rows=8, frame_width=800, frame_height=600)
    assert gm.cell_for_point(400, 300, 0, 0) == (4, 4)


def test_contains_point_is_true_only_in_active_cells() -> None:
    """contains_point means "not in a blocked cell" of the given frame."""
    gm = _right_half_blocked()
    assert gm.contains_point(60, 90, 320, 180) is True
    assert gm.contains_point(240, 90, 320, 180) is False


def test_filter_detections_uses_the_frame_size_of_the_boxes() -> None:
    """Boxes from a 320x180 frame are judged in that frame, not the 1920x1080 save size."""
    gm = _right_half_blocked()
    right = Detection(kind="person", confidence=0.9, bbox=(220, 70, 40, 40))  # centre (240, 90)
    left = Detection(kind="person", confidence=0.9, bbox=(40, 70, 40, 40))  # centre (60, 90)
    assert gm.filter_detections([right, left], 320, 180) == [left]


def test_filter_detections_without_frame_size_keeps_the_saved_size() -> None:
    """The one-argument call keeps its meaning: boxes in the zone's saved pixels."""
    gm = _right_half_blocked()
    right = Detection(kind="person", bbox=(1420, 520, 40, 40))  # centre (1440, 540)
    assert gm.filter_detections([right]) == []


def test_from_zone_copies_the_zone() -> None:
    """GridMask.from_zone carries name, grid, cells (as a new set) and saved size."""
    from rtsp_warden.config import GridZoneConfig

    zone = GridZoneConfig(
        name="yard",
        kind="area",
        grid_cols=8,
        grid_rows=4,
        blocked_cells={(0, 0), (7, 3)},
        frame_width=1280,
        frame_height=720,
    )
    gm = GridMask.from_zone(zone)
    assert gm.name == "yard"
    assert (gm.grid_cols, gm.grid_rows) == (8, 4)
    assert gm.blocked_cells == {(0, 0), (7, 3)}
    assert gm.blocked_cells is not zone.blocked_cells
    assert (gm.frame_width, gm.frame_height) == (1280, 720)


def test_zone_for_point_returns_the_first_area_in_order() -> None:
    """Overlapping areas resolve by list (config) order."""
    whole = GridMask(grid_cols=2, grid_rows=2, name="whole")
    left = GridMask(grid_cols=2, grid_rows=2, blocked_cells={(1, 0), (1, 1)}, name="left")
    areas = [("left", left), ("whole", whole)]
    assert zone_for_point(areas, 10, 10, 320, 180) == "left"
    assert zone_for_point(areas, 300, 10, 320, 180) == "whole"
    assert zone_for_point(list(reversed(areas)), 10, 10, 320, 180) == "whole"


def test_zone_for_point_is_empty_when_no_area_contains_the_point() -> None:
    """No containing area (or no areas at all) gives the empty string."""
    left = GridMask(grid_cols=2, grid_rows=2, blocked_cells={(1, 0), (1, 1)}, name="left")
    assert zone_for_point([("left", left)], 300, 10, 320, 180) == ""
    assert zone_for_point([], 10, 10, 320, 180) == ""
