# Sprint 6 Implementation Plan -- rtsp-warden v1.2.0

**Baseline:** v1.1.0 (commit `6d7b0d5`), 528 tests passing, 36 web routes, 18 runtime deps.
**Goal:** 5 per-camera tuning features, additive layers, zero regressions.

---

## Feature 1: Per-Camera Data Retention Policies

### 1. Files to Create / Modify

| Action | Path | Purpose |
|--------|------|---------|
| **MODIFY** | `src/rtsp_warden/config.py` | Add `retention` to `CameraConfig`, add global `retention` to `AppConfig` |
| **MODIFY** | `src/rtsp_warden/app.py` | Change `RetentionManager` construction to use per-camera fallback chain |
| **MODIFY** | `src/rtsp_warden/web/routes/cameras.py` | Show per-camera retention in settings page |
| **MODIFY** | `src/rtsp_warden/web/templates/cameras/settings.html` | Display per-camera retention values |
| **CREATE** | `tests/test_per_camera_retention.py` | Unit tests for retention resolution |

### 2. New Pydantic Config Types

In `config.py`, add to `CameraConfig`:

```python
class CameraConfig(BaseModel):
    name: str
    main_url: str
    sub_url: str
    record: RecordConfig = Field(default_factory=RecordConfig)
    proxy: ProxyConfig = Field(default_factory=ProxyConfig)
    detectors: list[DetectorSpec] = Field(default_factory=list)
    events: list[OnvifEventConfig] = Field(default_factory=list)
    presets: list[PTZPresetConfig] = Field(default_factory=list)

    # NEW: per-camera retention override
    retention: RetentionConfig | None = None
```

Add to `AppConfig`:

```python
class AppConfig(BaseModel):
    cameras: list[CameraConfig]
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    alerts: AlertsConfig = Field(default_factory=AlertsConfig)
    onvif: OnvifConfig = Field(default_factory=OnvifConfig)
    clips: ClipsConfig = Field(default_factory=ClipsConfig)

    # NEW: global retention fallback
    retention: RetentionConfig = Field(default_factory=RetentionConfig)
```

**Resolution chain** (documented in docstring):
```
effective_retention = camera.retention or camera.record.retention or app_config.retention
```

This preserves backward compatibility: existing configs with `record.retention` continue to work. New configs can set `camera.retention` directly or rely on the global `retention` at the AppConfig level.

### 3. New Runtime Deps

**None.**

### 4. Public API Surface

No new classes. The change is purely in how `RetentionManager` is constructed.

In `app.py` line 79, change from:

```python
retention = RetentionManager(
    camera_name=cam.name,
    camera_root=cam.record.output_dir / cam.name,
    cfg=cam.record.retention,
)
```

To:

```python
effective_retention = cam.retention or cam.record.retention or self.cfg.retention
retention = RetentionManager(
    camera_name=cam.name,
    camera_root=cam.record.output_dir / cam.name,
    cfg=effective_retention,
)
```

No changes to `retention.py` itself -- `RetentionManager` already accepts any `RetentionConfig`.

### 5. Test Strategy

**File:** `tests/test_per_camera_retention.py`

| Test | What it verifies |
|------|-----------------|
| `test_camera_retention_overrides_record` | `CameraConfig(retention=RetentionConfig(max_days=7))` with `record.retention.max_days=30` -- effective is 7 |
| `test_camera_retention_overrides_global` | `CameraConfig(retention=RetentionConfig(max_days=7))` with `AppConfig.retention.max_days=30` -- effective is 7 |
| `test_record_retention_falls_back` | `CameraConfig(retention=None)` with `record.retention.max_days=30` -- effective is 30 |
| `test_global_retention_fallback` | `CameraConfig(retention=None)`, `record.retention` has no max_days, `AppConfig.retention.max_days=14` -- effective is 14 |
| `test_none_retention_means_no_override` | `CameraConfig(retention=None)` -- resolution chain skips it |
| `test_config_roundtrip` | Serialize config with per-camera retention to YAML, reload, verify field survives |
| `test_settings_page_shows_retention` | GET /cameras/{name}/settings shows per-camera retention values |

### 6. Web UI Changes

**Routes:** No new routes. Existing `/cameras/{name}/settings` already shows retention.

**Templates:** `cameras/settings.html` -- Add a section showing whether retention is per-camera, from `record.retention`, or global. Display the effective values.

### 7. Migration Needs

**None.** Retention is config-driven, not in the database.

### 8. Config Example

```yaml
# Global fallback (applies to cameras without their own retention)
retention:
  max_days: 30
  max_gb: 50.0
  keep_last_n: 100
  cleanup_interval_seconds: 300

cameras:
  - name: front_door
    main_url: rtsp://user:pass@192.168.1.50:554/main
    sub_url: rtsp://user:pass@192.168.1.50:554/sub
    # Per-camera override: keep front door footage longer
    retention:
      max_days: 90
      max_gb: 100.0
      keep_last_n: 500

  - name: driveway
    main_url: rtsp://user:pass@192.168.1.51:554/main
    sub_url: rtsp://user:pass@192.168.1.51:554/sub
    # No per-camera retention -- falls back to global
```

### 9. README Updates

- **Retention section:** Document the three-tier resolution chain (camera > record > global). Add config example showing per-camera override.
- **Config reference:** Add `retention` to `CameraConfig` and `AppConfig` documentation.

### 10. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Existing configs break | Low | High | Resolution chain preserves `record.retention` as fallback. All existing configs continue to work unchanged. |
| Confusion about which retention applies | Medium | Low | Settings page shows effective retention source. Log which config was used at startup. |
| `None` retention on camera with no record.retention and no global | Low | Medium | `RetentionConfig()` default has all fields None/0, so `RetentionManager.run()` is a no-op. Safe. |

---

## Feature 2: Colored Timeline Markers by Object Type

### 1. Files to Create / Modify

| Action | Path | Purpose |
|--------|------|---------|
| **MODIFY** | `src/rtsp_warden/web/services/timeline.py` | Add `object_type` to `TimelineEvent`, add `categorize_object()` function |
| **MODIFY** | `src/rtsp_warden/web/routes/api.py` | Add `object_type` to event JSON in timeline endpoint |
| **MODIFY** | `src/rtsp_warden/web/static/js/timeline.js` | Color markers by `object_type` instead of `severity` |
| **CREATE** | `tests/test_timeline_colors.py` | Unit tests for categorization and color mapping |

### 2. New Pydantic Config Types

**None.** The color map is a module-level constant, not config-driven (consistent UX). A future sprint could add config overrides.

### 3. New Runtime Deps

**None.**

### 4. Public API Surface

```python
# src/rtsp_warden/web/services/timeline.py -- additions

OBJECT_TYPE_COLORS: dict[str, str] = {
    "person": "#ef5350",    # red
    "pet": "#42a5f5",       # blue
    "critter": "#66bb6a",   # green
    "vehicle": "#ffa726",   # orange
    "animal": "#ab47bc",    # purple (zoo/farm animals)
    "other": "#9e9e9e",     # gray
}

def categorize_object(label: str) -> str:
    """Map a detector label (COCO class name or detector kind) to an object type category.

    Categories:
        person: person
        pet: cat, dog
        critter: deer, raccoon, fox, coyote, squirrel, rabbit
        vehicle: car, truck, bus, motorcycle, bicycle
        animal: bird, horse, sheep, cow, elephant, bear, zebra, giraffe
        other: everything else (including "motion" from MotionDetector)

    Args:
        label: Lowercase class name from detector (e.g. "car", "person", "dog").

    Returns:
        One of the six category strings above.
    """
```

**Modified `TimelineEvent`:**

```python
@dataclass(slots=True)
class TimelineEvent:
    """A single detector event on the timeline."""
    id: int
    event_type: str
    severity: str
    ts_unix: float
    object_type: str  # NEW -- category from categorize_object()
```

In `build_timeline()`, when building events from DB rows:

```python
events.append(
    TimelineEvent(
        id=row.id,
        event_type=row.event_type,
        severity=row.severity,
        ts_unix=row.created_at.timestamp(),
        object_type=categorize_object(row.event_type),
    )
)
```

### 5. Test Strategy

**File:** `tests/test_timeline_colors.py`

| Test | What it verifies |
|------|-----------------|
| `test_categorize_person` | `categorize_object("person")` returns `"person"` |
| `test_categorize_pet_dog` | `categorize_object("dog")` returns `"pet"` |
| `test_categorize_pet_cat` | `categorize_object("cat")` returns `"pet"` |
| `test_categorize_critter_deer` | `categorize_object("deer")` returns `"critter"` |
| `test_categorize_critter_raccoon` | `categorize_object("raccoon")` returns `"critter"` |
| `test_categorize_vehicle_car` | `categorize_object("car")` returns `"vehicle"` |
| `test_categorize_vehicle_truck` | `categorize_object("truck")` returns `"vehicle"` |
| `test_categorize_animal_bird` | `categorize_object("bird")` returns `"animal"` |
| `test_categorize_other_motion` | `categorize_object("motion")` returns `"other"` |
| `test_categorize_other_unknown` | `categorize_object("garbage_label")` returns `"other"` |
| `test_categorize_case_insensitive` | `categorize_object("CAR")` returns `"vehicle"` |
| `test_timeline_event_has_object_type` | `build_timeline()` returns events with `object_type` field populated |
| `test_api_returns_object_type` | GET /api/recordings/{id}/timeline includes `object_type` in event JSON |
| `test_color_map_complete` | All 6 categories have hex color entries |

### 6. Web UI Changes

**Routes:** No new routes. The existing `GET /api/recordings/{id}/timeline` now returns `object_type` in each event.

**JavaScript:** `timeline.js` -- Replace the severity-based color logic:

```javascript
// OLD (remove):
const severityColors = {
  info: "#2196f3",
  warn: "#ff9800",
  error: "#f44336",
};
// ... ctx.strokeStyle = severityColors[evt.severity] || "#ffffff";

// NEW:
const objectTypeColors = {
  person: "#ef5350",
  pet: "#42a5f5",
  critter: "#66bb6a",
  vehicle: "#ffa726",
  animal: "#ab47bc",
  other: "#9e9e9e",
};
// ... ctx.strokeStyle = objectTypeColors[evt.object_type] || "#9e9e9e";
```

**Severity as secondary signal:** Use line style to indicate severity:
- `info` -> dashed line (`ctx.setLineDash([4, 4])`)
- `warn` -> solid line (default)
- `error` -> thick solid line (`ctx.lineWidth = 2.5`)

### 7. Migration Needs

**None.** The `Event.event_type` column already stores the label string. No schema changes.

### 8. Config Example

No config changes. The color map is hardcoded for consistent UX.

### 9. README Updates

- **Timeline section:** Document the object type color coding. Include the color legend table.
- **Web UI section:** Note that timeline markers are now color-coded by object type with severity shown as line style.

### 10. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| New COCO classes not in category map | Low | Low | `categorize_object()` defaults unknown labels to `"other"` (gray). Safe fallback. |
| Motion events all become gray | High | Low | MotionDetector sets `kind="motion"`, which maps to `"other"`. This is correct -- motion has no object type. Document this. |
| Colorblind users can't distinguish colors | Medium | Medium | Use distinct line styles for severity as secondary signal. Future: add pattern/texture option. |

---

## Feature 3: Tunable Detection Classes per Camera

### 1. Files to Create / Modify

| Action | Path | Purpose |
|--------|------|---------|
| **MODIFY** | `src/rtsp_warden/config.py` | Add `detect_classes` to `CameraConfig` |
| **MODIFY** | `src/rtsp_warden/detectors/registry.py` | `build_detector` accepts `camera_detect_classes`, computes intersection |
| **MODIFY** | `src/rtsp_warden/app.py` | `_build_detectors` passes `cam.detect_classes` to `build_detector` |
| **CREATE** | `tests/test_detect_classes.py` | Unit tests for intersection logic |

### 2. New Pydantic Config Types

In `config.py`, add to `CameraConfig`:

```python
class CameraConfig(BaseModel):
    # ... existing fields ...

    # NEW: per-camera detection class filter
    detect_classes: list[str] | None = None
    # None = use detector's default allowed_classes
    # ["person", "dog"] = only detect these classes on this camera
```

No new top-level type needed. The field is a simple `list[str] | None`.

### 3. New Runtime Deps

**None.**

### 4. Public API Surface

**Modified `build_detector` signature:**

```python
def build_detector(
    spec: DetectorSpec,
    camera_name: str,
    camera_detect_classes: list[str] | None = None,  # NEW
) -> Detector:
```

**Intersection logic** (applied in the `"dnn"` branch):

```python
if spec.type == "dnn":
    # ... existing code ...
    config = spec.config or {}
    spec_classes = config.get("classes", None)
    if spec_classes is not None:
        spec_classes = list(spec_classes)

    # Compute effective allowed_classes: intersection of camera + spec
    effective_classes: list[str] | None = None
    if camera_detect_classes is not None and spec_classes is not None:
        # Both set: intersection
        cam_set = {c.lower() for c in camera_detect_classes}
        effective_classes = [c for c in spec_classes if c.lower() in cam_set]
    elif camera_detect_classes is not None:
        # Only camera set: use camera's list
        effective_classes = list(camera_detect_classes)
    elif spec_classes is not None:
        # Only spec set: use spec's list
        effective_classes = list(spec_classes)
    # else: both None -> effective_classes stays None (detect all)

    return DNNDetector(
        # ... existing params ...
        allowed_classes=effective_classes,
    )
```

For non-DNN detectors (motion, person, vehicle), `camera_detect_classes` is ignored -- these detectors have fixed detection targets. Only DNN benefits from class filtering.

### 5. Test Strategy

**File:** `tests/test_detect_classes.py`

| Test | What it verifies |
|------|-----------------|
| `test_intersection_both_set` | `camera=["person","dog"]`, `spec=["person","dog","car"]` -> effective `["person","dog"]` |
| `test_intersection_camera_only` | `camera=["person","dog"]`, `spec=None` -> effective `["person","dog"]` |
| `test_intersection_spec_only` | `camera=None`, `spec=["person","dog","car"]` -> effective `["person","dog","car"]` |
| `test_intersection_both_none` | `camera=None`, `spec=None` -> effective `None` (detect all) |
| `test_intersection_no_overlap` | `camera=["person"]`, `spec=["car","truck"]` -> effective `[]` (detect nothing) |
| `test_intersection_case_insensitive` | `camera=["Person","DOG"]`, `spec=["person","dog"]` -> effective `["person","dog"]` |
| `test_motion_ignores_detect_classes` | `camera=["person"]`, `spec.type="motion"` -> MotionDetector built normally |
| `test_config_roundtrip` | Serialize `CameraConfig(detect_classes=["person","dog"])` to YAML, reload, verify |
| `test_build_detector_passes_classes` | `build_detector(spec, "cam", camera_detect_classes=["person"])` -> DNNDetector with `allowed_classes=["person"]` |

### 6. Web UI Changes

No new routes in this feature alone. The camera detail page will show `detect_classes` in the settings section (added in Feature 5's UI work). A future sprint could add a class selector UI.

### 7. Migration Needs

**None.** Config-driven.

### 8. Config Example

```yaml
cameras:
  - name: front_door
    main_url: rtsp://user:pass@192.168.1.50:554/main
    sub_url: rtsp://user:pass@192.168.1.50:554/sub
    detect_classes: [person, dog, cat]  # only care about people and pets at the door
    detectors:
      - type: dnn
        enabled: true
        interval_seconds: 2.0

  - name: driveway
    main_url: rtsp://user:pass@192.168.1.51:554/main
    sub_url: rtsp://user:pass@192.168.1.51:554/sub
    detect_classes: [car, truck, person]  # vehicles and people in driveway
    detectors:
      - type: dnn
        enabled: true
        interval_seconds: 2.0
```

### 9. README Updates

- **Detectors section:** Document `detect_classes` on `CameraConfig`. Explain the intersection logic with `DetectorSpec.config.classes`.
- **Config reference:** Add `detect_classes` to `CameraConfig` documentation.

### 10. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Empty intersection means no detections | Low | Medium | Log a warning when effective_classes is empty. Detector still runs but returns nothing. User can see this in the detector status. |
| User sets detect_classes on motion/person detector | Low | Low | Non-DNN detectors ignore `camera_detect_classes`. Document this clearly. |
| Class name mismatch (typo in config) | Medium | Medium | Intersection is case-insensitive. If a class in `detect_classes` doesn't match any COCO class, it's silently ignored (no error, just no detections for that class). Log unmatched classes at debug level. |

---

## Feature 4: Grid-Based Detection Zones

### 1. Files to Create / Modify

| Action | Path | Purpose |
|--------|------|---------|
| **CREATE** | `src/rtsp_warden/detectors/grid_zone.py` | `GridZone` class with cell-to-polygon conversion |
| **MODIFY** | `src/rtsp_warden/config.py` | Add `GridZoneConfig` pydantic model, add `zones` to `CameraConfig` |
| **MODIFY** | `src/rtsp_warden/detectors/registry.py` | Add `build_grid_zones()` function |
| **MODIFY** | `src/rtsp_warden/app.py` | `_build_detectors` builds GridZones, converts to ROI, merges |
| **MODIFY** | `src/rtsp_warden/web/routes/cameras.py` | Add zone editor routes |
| **CREATE** | `src/rtsp_warden/web/templates/cameras/zones.html` | Grid zone editor page |
| **CREATE** | `src/rtsp_warden/web/static/js/zones.js` | Canvas grid overlay with cell toggling |
| **CREATE** | `tests/test_grid_zones.py` | Unit tests for GridZone and routes |

### 2. New Pydantic Config Types

In `config.py`:

```python
class GridZoneConfig(BaseModel):
    """Grid-based detection zone for a camera.

    Divides the frame into an NxM grid of cells. Only detections whose
    bounding-box center falls within an enabled cell are kept.

    Attributes:
        name: Human-readable zone label.
        grid_cols: Number of columns in the grid (default 8).
        grid_rows: Number of rows in the grid (default 8).
        enabled_cells: Set of (col, row) tuples (0-indexed) that are
            active for detection. Cells not in this set are ignored.
        frame_width: Width of the camera frame in pixels (used to
            compute cell boundaries).
        frame_height: Height of the camera frame in pixels.
    """

    name: str = ""
    grid_cols: int = 8
    grid_rows: int = 8
    enabled_cells: list[tuple[int, int]] = Field(default_factory=list)
    frame_width: int = 1920
    frame_height: int = 1080

    @field_validator("grid_cols", "grid_rows")
    @classmethod
    def _grid_positive(cls, v: int) -> int:
        if v < 2:
            raise ValueError("grid_cols and grid_rows must be >= 2")
        if v > 64:
            raise ValueError("grid_cols and grid_rows must be <= 64")
        return v

    @field_validator("frame_width", "frame_height")
    @classmethod
    def _frame_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("frame dimensions must be > 0")
        return v
```

Add to `CameraConfig`:

```python
class CameraConfig(BaseModel):
    # ... existing fields ...

    # NEW: grid-based detection zones
    zones: list[GridZoneConfig] = Field(default_factory=list)
```

### 3. New Runtime Deps

**None.** Uses existing `cv2` and `numpy` for polygon operations.

### 4. Public API Surface

```python
# src/rtsp_warden/detectors/grid_zone.py

@dataclass(slots=True)
class GridZone:
    """Grid-based detection zone that converts cells to ROI polygons.

    Each enabled cell becomes a rectangular polygon. The union of all
    enabled-cell polygons forms the effective ROI for this zone.

    Args:
        name: Human-readable zone label.
        grid_cols: Number of columns.
        grid_rows: Number of rows.
        enabled_cells: Set of (col, row) tuples that are active.
        frame_width: Camera frame width in pixels.
        frame_height: Camera frame height in pixels.
    """

    name: str
    grid_cols: int
    grid_rows: int
    enabled_cells: set[tuple[int, int]]
    frame_width: int
    frame_height: int

    def __post_init__(self) -> None:
        """Validate grid dimensions and cell bounds."""
        if self.grid_cols < 2 or self.grid_rows < 2:
            raise ValueError("grid must be at least 2x2")
        for col, row in self.enabled_cells:
            if not (0 <= col < self.grid_cols and 0 <= row < self.grid_rows):
                raise ValueError(f"cell ({col},{row}) out of bounds for {self.grid_cols}x{self.grid_rows} grid")

    @property
    def cell_width(self) -> float:
        """Width of a single cell in pixels."""
        return self.frame_width / self.grid_cols

    @property
    def cell_height(self) -> float:
        """Height of a single cell in pixels."""
        return self.frame_height / self.grid_rows

    def cell_polygon(self, col: int, row: int) -> list[tuple[int, int]]:
        """Return the polygon (4 corners) for a single grid cell.

        Returns:
            List of 4 (x, y) integer tuples: top-left, top-right,
            bottom-right, bottom-left.
        """
        x1 = int(col * self.cell_width)
        y1 = int(row * self.cell_height)
        x2 = int((col + 1) * self.cell_width)
        y2 = int((row + 1) * self.cell_height)
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

    def to_roi(self) -> ROI:
        """Convert the grid zone to an ROI for use with filter_by_roi.

        The ROI polygon is the union of all enabled cell polygons.
        For simplicity, we create a single ROI with the first cell's
        polygon. For multi-cell zones, we create a compound ROI by
        checking containment against any enabled cell.

        Returns:
            ROI object whose contains() method checks all enabled cells.
        """
        # Build a MultiCellROI that checks all enabled cells
        return MultiCellROI(
            cells=[self.cell_polygon(c, r) for c, r in self.enabled_cells],
            name=self.name,
        )


class MultiCellROI:
    """ROI-like object that checks containment against multiple cell polygons.

    Compatible with filter_by_roi() -- has a contains(bbox) method.
    """

    def __init__(self, cells: list[list[tuple[int, int]]], name: str = "") -> None:
        self._cells = cells
        self.name = name

    def contains(self, bbox: tuple[int, int, int, int]) -> bool:
        """Return True if bbox center is inside any enabled cell."""
        x, y, w, h = bbox
        if w == 0 or h == 0:
            return False
        cx = x + w // 2
        cy = y + h // 2
        for cell_poly in self._cells:
            pts = np.array(cell_poly, dtype=np.int32)
            if cv2.pointPolygonTest(pts, (float(cx), float(cy)), False) >= 0:
                return True
        return False


# src/rtsp_warden/detectors/registry.py -- new function

def build_grid_zones(zones: list[GridZoneConfig]) -> list[GridZone]:
    """Build GridZone objects from config.

    Args:
        zones: List of GridZoneConfig from CameraConfig.zones.

    Returns:
        List of GridZone objects ready for use.
    """
    result: list[GridZone] = []
    for zc in zones:
        gz = GridZone(
            name=zc.name,
            grid_cols=zc.grid_cols,
            grid_rows=zc.grid_rows,
            enabled_cells=set(zc.enabled_cells),
            frame_width=zc.frame_width,
            frame_height=zc.frame_height,
        )
        result.append(gz)
    return result
```

**Integration in `app.py` `_build_detectors`:**

After building the runner's ROI from detector specs, also build grid zones and merge:

```python
# Build grid zones from camera config
grid_zones = build_grid_zones(cam.zones) if cam.zones else []

# Convert grid zones to ROIs and merge with existing ROI
if grid_zones:
    grid_rois = [gz.to_roi() for gz in grid_zones]
    # If there's already a polygon ROI, combine: check polygon ROI first,
    # then check grid zones. Use a CompoundROI wrapper.
    if runner_roi is not None:
        runner_roi = CompoundROI(primary=runner_roi, secondaries=grid_rois)
    else:
        # If only one grid zone, use it directly; if multiple, compound them
        if len(grid_rois) == 1:
            runner_roi = grid_rois[0]
        else:
            runner_roi = CompoundROI(primary=grid_rois[0], secondaries=grid_rois[1:])
```

**CompoundROI** (in `roi.py` or `grid_zone.py`):

```python
class CompoundROI:
    """ROI that checks multiple sub-ROIs. A detection is kept if it passes
    the primary ROI AND any of the secondary ROIs (grid zones)."""

    def __init__(self, primary: Any, secondaries: list[Any]) -> None:
        self._primary = primary
        self._secondaries = secondaries

    def contains(self, bbox: tuple[int, int, int, int]) -> bool:
        if not self._primary.contains(bbox):
            return False
        if not self._secondaries:
            return True
        return any(s.contains(bbox) for s in self._secondaries)
```

### 5. Test Strategy

**File:** `tests/test_grid_zones.py`

| Test | What it verifies |
|------|-----------------|
| `test_grid_zone_cell_polygon` | 8x8 grid, cell (0,0) on 1920x1080 -> polygon [(0,0),(240,0),(240,135),(0,135)] |
| `test_grid_zone_cell_polygon_last` | Cell (7,7) -> correct bottom-right corner polygon |
| `test_grid_zone_to_roi_single_cell` | One enabled cell -> MultiCellROI with one polygon |
| `test_grid_zone_to_roi_multi_cell` | Two enabled cells -> MultiCellROI with two polygons |
| `test_multi_cell_roi_contains_inside` | Bbox center in enabled cell -> True |
| `test_multi_cell_roi_contains_outside` | Bbox center in disabled cell -> False |
| `test_multi_cell_roi_empty_bbox` | w=0 or h=0 -> False |
| `test_grid_zone_invalid_cell` | Cell (8,0) on 8x8 grid -> ValueError |
| `test_grid_zone_min_grid` | 2x2 grid valid, 1x1 grid -> ValueError |
| `test_build_grid_zones_from_config` | `build_grid_zones([GridZoneConfig(...)])` -> list of GridZone |
| `test_compound_roi_passes_primary_fails_secondary` | Primary passes, no secondary passes -> False |
| `test_compound_roi_passes_both` | Primary passes, one secondary passes -> True |
| `test_config_roundtrip` | Serialize CameraConfig with zones to YAML, reload, verify |
| `test_web_zones_page` | GET /cameras/{name}/zones returns 200 with grid editor |
| `test_web_save_zones` | POST /cameras/{name}/zones saves enabled_cells |

### 6. Web UI Changes

**New routes** (in `src/rtsp_warden/web/routes/cameras.py`):

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/cameras/{name}/zones` | admin | Render the grid zone editor page |
| `POST` | `/cameras/{name}/zones` | admin | Save zone configuration (JSON body with enabled_cells) |

**New template:** `cameras/zones.html`

Page structure:
- A snapshot image from the camera's MJPEG proxy (`/snapshot.jpg` or the MJPEG endpoint)
- A canvas overlay showing the grid lines
- Click a cell to toggle it on/off (green = enabled, transparent = disabled)
- A "Save" button that POSTs the enabled_cells set as JSON
- Display current grid dimensions (cols x rows) and frame size

**New JavaScript:** `web/static/js/zones.js`

```javascript
class ZoneEditor {
  constructor(mountEl) {
    this.mount = mountEl;
    this.cameraName = mountEl.dataset.cameraName;
    this.gridCols = parseInt(mountEl.dataset.gridCols) || 8;
    this.gridRows = parseInt(mountEl.dataset.gridRows) || 8;
    this.frameWidth = parseInt(mountEl.dataset.frameWidth) || 1920;
    this.frameHeight = parseInt(mountEl.dataset.frameHeight) || 1080;
    this.enabledCells = new Set(); // populated from data attribute or API

    this.canvas = mountEl.querySelector("#zone-canvas");
    this.snapshotImg = mountEl.querySelector("#zone-snapshot");

    this._onCanvasClick = this._onCanvasClick.bind(this);
  }

  async init() {
    // Load current zones from API
    // Draw grid overlay on canvas
    // Wire click handler
  }

  _drawGrid() {
    // Draw grid lines, fill enabled cells with semi-transparent green
  }

  _onCanvasClick(e) {
    // Determine which cell was clicked, toggle it, redraw
  }

  async save() {
    // POST enabled cells to /cameras/{name}/zones
  }
}
```

**Modified template:** `cameras/detail.html` -- Add a link/button to the zone editor:

```html
<article>
  <header>Detection Zones</header>
  <p>
    <a href="/cameras/{{ camera.name }}/zones" role="button">
      Edit Grid Zones
    </a>
  </p>
</article>
```

### 7. Migration Needs

**None.** Zones are stored in `config.yaml` under `CameraConfig.zones`.

### 8. Config Example

```yaml
cameras:
  - name: driveway
    main_url: rtsp://user:pass@192.168.1.51:554/main
    sub_url: rtsp://user:pass@192.168.1.51:554/sub
    zones:
      - name: driveway_only
        grid_cols: 8
        grid_rows: 8
        frame_width: 1920
        frame_height: 1080
        enabled_cells:
          - [3, 4]
          - [3, 5]
          - [4, 4]
          - [4, 5]
          - [5, 4]
          - [5, 5]
          # ... only the cells covering the driveway area
```

### 9. README Updates

- **New "Detection Zones" section:** Document grid-based zones. Explain the grid editor UI, how cells map to frame regions, and how zones interact with polygon ROI/masks.
- **Config reference:** Add `zones` and `GridZoneConfig` to documentation.
- **Web UI routes table:** Add the two new zone routes.

### 10. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Frame dimensions mismatch | Medium | Medium | GridZone uses `frame_width`/`frame_height` from config. If camera resolution changes, zones need updating. Show current resolution on the zone editor page. |
| MultiCellROI performance with many cells | Low | Low | `pointPolygonTest` is O(1) per cell. Even 64 cells is negligible. For extreme cases (64x64 grid with all cells enabled), consider spatial indexing in a future optimization. |
| Grid zone + polygon ROI interaction confusion | Medium | Low | Document clearly: polygon ROI is primary filter, grid zones are secondary. Both must pass for a detection to be kept. |
| Config.yaml save conflicts | Low | Medium | Zone editor POSTs to API which writes config.yaml. Use file locking for writes. Accept last-write-wins for v1. |

---

## Feature 5: Per-Camera Sensitivity Adjustment

### 1. Files to Create / Modify

| Action | Path | Purpose |
|--------|------|---------|
| **CREATE** | `src/rtsp_warden/detectors/sensitivity.py` | Sensitivity mapping functions for each detector type |
| **MODIFY** | `src/rtsp_warden/config.py` | Add `sensitivity` to `CameraConfig` |
| **MODIFY** | `src/rtsp_warden/detectors/registry.py` | `build_detector` accepts `camera_sensitivity`, applies mappings |
| **MODIFY** | `src/rtsp_warden/app.py` | `_build_detectors` passes `cam.sensitivity` to `build_detector` |
| **MODIFY** | `src/rtsp_warden/web/routes/cameras.py` | Add sensitivity update route |
| **MODIFY** | `src/rtsp_warden/web/templates/cameras/detail.html` | Add sensitivity slider |
| **CREATE** | `tests/test_sensitivity.py` | Unit tests for mapping functions and integration |

### 2. New Pydantic Config Types

In `config.py`, add to `CameraConfig`:

```python
class CameraConfig(BaseModel):
    # ... existing fields ...

    # NEW: per-camera sensitivity (0-100 scale)
    sensitivity: float = 50.0

    @field_validator("sensitivity")
    @classmethod
    def _sensitivity_range(cls, v: float) -> float:
        if not 0.0 <= v <= 100.0:
            raise ValueError("sensitivity must be between 0.0 and 100.0")
        return v
```

### 3. New Runtime Deps

**None.**

### 4. Public API Surface

```python
# src/rtsp_warden/detectors/sensitivity.py

def motion_sensitivity(camera_sensitivity: float) -> float:
    """Convert 0-100 camera sensitivity to MotionDetector sensitivity (0.0-1.0).

    Higher camera sensitivity = lower varThreshold = more motion detected.
    The MotionDetector's internal _sensitivity_to_var_threshold handles
    the varThreshold mapping.

    Mapping: sensitivity = 1.0 - (camera_sensitivity / 100.0)
    Result is clamped to [0.05, 1.0].

    Args:
        camera_sensitivity: 0-100 value from CameraConfig.

    Returns:
        Float in [0.05, 1.0] for MotionDetector.sensitivity.
    """
    raw = 1.0 - (camera_sensitivity / 100.0)
    return max(0.05, min(1.0, raw))


def person_confidence(camera_sensitivity: float) -> float:
    """Convert 0-100 camera sensitivity to PersonDetector min_confidence.

    Higher sensitivity = lower min_confidence = more detections accepted.

    Mapping: min_confidence = max(0.1, 1.0 - camera_sensitivity / 100.0)

    Args:
        camera_sensitivity: 0-100 value from CameraConfig.

    Returns:
        Float in [0.1, 1.0] for PersonDetector.min_confidence.
    """
    return max(0.1, 1.0 - camera_sensitivity / 100.0)


def dnn_confidence(camera_sensitivity: float) -> float:
    """Convert 0-100 camera sensitivity to DNNDetector confidence threshold.

    Higher sensitivity = lower confidence threshold = more detections.

    Mapping: confidence = max(0.1, 1.0 - camera_sensitivity / 100.0)

    Args:
        camera_sensitivity: 0-100 value from CameraConfig.

    Returns:
        Float in [0.1, 1.0] for DNNDetector.confidence.
    """
    return max(0.1, 1.0 - camera_sensitivity / 100.0)


def dnn_nms(camera_sensitivity: float) -> float:
    """Convert 0-100 camera sensitivity to DNNDetector NMS threshold.

    Higher sensitivity = higher NMS threshold = more overlapping
    detections kept.

    Mapping: nms = clamp(0.3 + (1 - camera_sensitivity/100) * 0.3, 0.3, 0.7)

    Args:
        camera_sensitivity: 0-100 value from CameraConfig.

    Returns:
        Float in [0.3, 0.7] for DNNDetector.nms_threshold.
    """
    raw = 0.3 + (1.0 - camera_sensitivity / 100.0) * 0.3
    return max(0.3, min(0.7, raw))
```

**Modified `build_detector` signature:**

```python
def build_detector(
    spec: DetectorSpec,
    camera_name: str,
    camera_detect_classes: list[str] | None = None,
    camera_sensitivity: float = 50.0,  # NEW
) -> Detector:
```

**Sensitivity application in `build_detector`:**

```python
if spec.type == "motion":
    from .builtin.motion import MotionDetector
    from .sensitivity import motion_sensitivity

    sens = spec.sensitivity if spec.sensitivity is not None else motion_sensitivity(camera_sensitivity)
    return MotionDetector(
        min_area=spec.min_area or 500,
        sensitivity=sens,
    )

if spec.type == "person":
    from .builtin.person import PersonDetector
    from .sensitivity import person_confidence

    conf = spec.min_confidence if spec.min_confidence is not None else person_confidence(camera_sensitivity)
    return PersonDetector(
        min_confidence=conf,
        scale_factor=spec.scale_factor or 1.1,
        min_neighbors=spec.min_neighbors or 3,
    )

if spec.type == "dnn":
    from .builtin.dnn import DNNDetector
    from .sensitivity import dnn_confidence, dnn_nms

    config = spec.config or {}
    conf = float(config.get("confidence_threshold", dnn_confidence(camera_sensitivity)))
    nms = float(config.get("nms_threshold", dnn_nms(camera_sensitivity)))
    # ... rest of DNN construction ...
```

**Design decision:** `DetectorSpec` fields (`sensitivity`, `min_confidence`) take precedence over camera-level sensitivity. If the user explicitly sets `sensitivity: 0.3` on a `DetectorSpec`, that value is used instead of the camera-level mapping. This gives per-detector fine-tuning on top of the per-camera knob.

### 5. Test Strategy

**File:** `tests/test_sensitivity.py`

| Test | What it verifies |
|------|-----------------|
| `test_motion_sensitivity_0` | `motion_sensitivity(0)` -> 1.0 (least sensitive, highest varThreshold) |
| `test_motion_sensitivity_100` | `motion_sensitivity(100)` -> 0.05 (most sensitive, lowest varThreshold) |
| `test_motion_sensitivity_50` | `motion_sensitivity(50)` -> 0.5 (midpoint) |
| `test_person_confidence_0` | `person_confidence(0)` -> 1.0 (only high-confidence detections) |
| `test_person_confidence_100` | `person_confidence(100)` -> 0.1 (accept almost everything) |
| `test_dnn_confidence_50` | `dnn_confidence(50)` -> 0.5 |
| `test_dnn_nms_0` | `dnn_nms(0)` -> 0.6 (more suppression, fewer overlapping boxes) |
| `test_dnn_nms_100` | `dnn_nms(100)` -> 0.3 (less suppression, more overlapping boxes) |
| `test_spec_sensitivity_overrides_camera` | `spec.sensitivity=0.3`, `camera_sensitivity=80` -> MotionDetector gets 0.3 |
| `test_spec_min_confidence_overrides_camera` | `spec.min_confidence=0.8`, `camera_sensitivity=90` -> PersonDetector gets 0.8 |
| `test_build_detector_applies_sensitivity` | `build_detector(motion_spec, "cam", camera_sensitivity=80)` -> MotionDetector.sensitivity ~0.2 |
| `test_config_validation` | `CameraConfig(sensitivity=-1)` raises ValidationError |
| `test_config_validation_max` | `CameraConfig(sensitivity=101)` raises ValidationError |
| `test_web_sensitivity_update` | POST /cameras/{name}/sensitivity with `{"value": 75}` updates config |

### 6. Web UI Changes

**New route** (in `src/rtsp_warden/web/routes/cameras.py`):

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/cameras/{name}/sensitivity` | admin | Update camera sensitivity. Body: `{"value": 75}`. Returns JSON `{"ok": true, "sensitivity": 75.0}`. |

**Modified template:** `cameras/detail.html` -- Add a sensitivity slider section:

```html
<article x-data="{ sensitivity: {{ camera.sensitivity | default(50.0) }}, saving: false }">
  <header>Sensitivity</header>
  <div class="grid">
    <label for="sensitivity-slider">
      Detection sensitivity: <strong x-text="sensitivity"></strong>%
    </label>
    <input type="range" id="sensitivity-slider"
           x-model="sensitivity" min="0" max="100" step="1"
           @change="saving = true;
                    fetch('/cameras/{{ camera.name }}/sensitivity', {
                      method: 'POST',
                      headers: {'Content-Type': 'application/json'},
                      body: JSON.stringify({value: sensitivity})
                    }).then(r => r.json()).then(() => saving = false)">
  </div>
  <footer>
    <small>Higher values = more sensitive = more detections. Changes take effect on next detector restart.</small>
    <small x-show="saving">Saving...</small>
  </footer>
</article>
```

The `camera.sensitivity` value needs to be passed from the route handler. Update `camera_detail()` in `cameras.py` to include the raw `CameraConfig` sensitivity value in the template context.

### 7. Migration Needs

**None.** Config-driven.

### 8. Config Example

```yaml
cameras:
  - name: front_door
    main_url: rtsp://user:pass@192.168.1.50:554/main
    sub_url: rtsp://user:pass@192.168.1.50:554/sub
    sensitivity: 75  # more sensitive -- catch everything at the door
    detectors:
      - type: motion
        enabled: true
        interval_seconds: 1.0
      - type: person
        enabled: true
        interval_seconds: 2.0

  - name: backyard
    main_url: rtsp://user:pass@192.168.1.52:554/main
    sub_url: rtsp://user:pass@192.168.1.52:554/sub
    sensitivity: 30  # less sensitive -- ignore wind-blown trees
    detectors:
      - type: motion
        enabled: true
        interval_seconds: 1.0
        min_area: 1000  # per-detector override: only large motion
```

### 9. README Updates

- **New "Sensitivity" section:** Document the 0-100 per-camera sensitivity knob. Explain the mapping formulas for each detector type. Note that `DetectorSpec` fields override camera sensitivity.
- **Config reference:** Add `sensitivity` to `CameraConfig` documentation.
- **Web UI section:** Document the sensitivity slider on the camera detail page.

### 10. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Sensitivity change requires restart | Medium | Medium | Document that sensitivity changes take effect on next detector restart (or next app restart). Future: hot-reload detectors on config change. |
| Extreme sensitivity causes false positive flood | Medium | Medium | Clamp all mapping outputs to safe ranges (e.g., confidence never below 0.1). Document recommended ranges per camera type. |
| User confusion: camera vs detector sensitivity | Medium | Low | Clear documentation: camera sensitivity is the knob, detector spec fields are fine-tuning overrides. UI shows both. |

---

## Cross-Cutting Concerns

### `categorize_object()` Function

**Location:** `src/rtsp_warden/web/services/timeline.py`

**Categories and their COCO/detector label mappings:**

| Category | Labels |
|----------|--------|
| `person` | `person` |
| `pet` | `cat`, `dog` |
| `critter` | `deer`, `raccoon`, `fox`, `coyote`, `squirrel`, `rabbit`, `skunk`, `opossum` |
| `vehicle` | `car`, `truck`, `bus`, `motorcycle`, `bicycle` |
| `animal` | `bird`, `horse`, `sheep`, `cow`, `elephant`, `bear`, `zebra`, `giraffe` |
| `other` | `motion`, `ingest_lost`, `tamper`, and any unrecognized label |

**Implementation:** A module-level dict `_LABEL_TO_CATEGORY` mapping lowercase label strings to category strings. The function does a dict lookup with `.lower()` on the input, falling back to `"other"`.

### Color Palette

**Location:** `src/rtsp_warden/web/services/timeline.py` (server-side constant) and duplicated in `timeline.js` (client-side).

**Hex colors chosen for dark background (#1a1a2e) readability:**

| Category | Hex | CSS Name | Rationale |
|----------|-----|----------|-----------|
| `person` | `#ef5350` | Red 400 | High urgency, most important detection |
| `pet` | `#42a5f5` | Blue 400 | Distinct from person, friendly color |
| `critter` | `#66bb6a` | Green 400 | Outdoor/wildlife, neutral |
| `vehicle` | `#ffa726` | Orange 400 | Warm, distinct from person/pet |
| `animal` | `#ab47bc` | Purple 400 | Zoo/farm animals, distinct from pets |
| `other` | `#9e9e9e` | Grey 500 | Fallback, low prominence |

All colors are from the Material Design 400 palette, tested for contrast on dark backgrounds.

### Grid Resolution Default and Configurability

**Default:** 8 columns x 8 rows (64 cells). This is the sweet spot between granularity and usability -- enough cells to define meaningful zones without overwhelming the user.

**Configurable per zone:** Each `GridZoneConfig` has its own `grid_cols` and `grid_rows`. Valid range: 2-64. Common presets:
- 4x4 (16 cells) -- coarse zones for low-res cameras
- 8x8 (64 cells) -- standard (default)
- 16x16 (256 cells) -- fine-grained for high-res cameras

**Frame dimensions:** Must match the camera's actual resolution. The zone editor page should display the current frame dimensions (read from the MJPEG snapshot or config).

### Sensitivity Mapping Math

**Formulas (all take `s` = camera_sensitivity in 0-100):**

| Detector | Parameter | Formula | Range |
|----------|-----------|---------|-------|
| MotionDetector | `sensitivity` | `max(0.05, 1.0 - s/100)` | [0.05, 1.0] |
| PersonDetector | `min_confidence` | `max(0.1, 1.0 - s/100)` | [0.1, 1.0] |
| DNNDetector | `confidence` | `max(0.1, 1.0 - s/100)` | [0.1, 1.0] |
| DNNDetector | `nms_threshold` | `clamp(0.3 + (1 - s/100) * 0.3, 0.3, 0.7)` | [0.3, 0.7] |

**Rationale:**
- All use the same core formula: `1.0 - s/100` (higher camera sensitivity = lower threshold = more detections).
- MotionDetector's `sensitivity` parameter is already 0-1 scale with inverse mapping in `_sensitivity_to_var_threshold`.
- Clamp values prevent degenerate cases (confidence=0 would accept everything; nms=1.0 would keep all overlapping boxes).
- The NMS formula is inverted: higher sensitivity = lower NMS = fewer overlapping boxes suppressed = more detections kept.

### `camera.detect_classes` x `detector.allowed_classes` Intersection

**Logic in `build_detector()` (Feature 3):**

```
Let C = camera.detect_classes (list[str] | None)
Let D = detector.allowed_classes (list[str] | None, from DetectorSpec.config.classes)

If C is None and D is None:
    effective = None  (detect all COCO classes)
If C is not None and D is None:
    effective = C  (use camera's list as-is)
If C is None and D is not None:
    effective = D  (use detector's list as-is)
If C is not None and D is not None:
    C_set = {c.lower() for c in C}
    effective = [d for d in D if d.lower() in C_set]
    (intersection -- only classes in both lists)
```

**Case-insensitive:** All comparisons use `.lower()`. The effective list preserves the original casing from D (the detector's COCO class names).

**Empty intersection:** If the intersection is empty (e.g., camera wants `["person"]` but detector only has `["car", "truck"]`), `effective_classes` is `[]`. The DNNDetector will filter out all detections. Log a warning: `"detect_classes intersection is empty for camera {name}; no detections will be reported"`.

**Non-DNN detectors:** `camera_detect_classes` is only applied to `type="dnn"`. Motion, person, and vehicle detectors have fixed detection targets and ignore this field.

---

## Parallelization Strategy

### Independence Analysis

| Feature | Primary files | Overlaps with |
|---------|--------------|---------------|
| F1: Retention | `config.py`, `app.py` (line 79), `cameras.py`, `settings.html` | F3, F4, F5 (config.py), F3, F4, F5 (app.py different method) |
| F2: Timeline Colors | `timeline.py`, `api.py`, `timeline.js` | **NONE** -- completely independent |
| F3: Detect Classes | `config.py`, `registry.py`, `app.py` (_build_detectors) | F1, F4, F5 (config.py), F4, F5 (registry.py), F4, F5 (app.py) |
| F4: Grid Zones | `config.py`, `grid_zone.py`, `roi.py`, `registry.py`, `app.py`, `cameras.py`, `zones.html`, `zones.js` | F1, F3, F5 (config.py), F3, F5 (registry.py), F3, F5 (app.py) |
| F5: Sensitivity | `config.py`, `sensitivity.py`, `registry.py`, `app.py`, `cameras.py`, `detail.html` | F1, F3, F4 (config.py), F3, F4 (registry.py), F3, F4 (app.py) |

**Key insight:** F2 (Timeline Colors) has **zero file overlap** with any other feature. It can run completely in parallel with everything else.

**Conflict zone:** `config.py` is touched by F1, F3, F4, F5. `registry.py` and `app.py` are touched by F3, F4, F5. The solution is a **Config Foundation** phase that adds all new pydantic fields first, then feature coders build on that foundation.

### Recommended Dispatch Order

**Phase 1 -- Config Foundation + F2 (PARALLEL, 2 coders):**

| Order | Coder | Task | Files |
|-------|-------|------|-------|
| 1a | Coder A | **Config Foundation**: Add ALL new fields to `CameraConfig` and `AppConfig` (retention, detect_classes, sensitivity, zones, GridZoneConfig, AppConfig.retention) | `config.py` only |
| 1b | Coder B | **F2: Timeline Colors**: categorize_object, TimelineEvent.object_type, API JSON, JS color rendering | `timeline.py`, `api.py`, `timeline.js` |

**Phase 2 -- Backend Logic (PARALLEL, 2 coders, after Phase 1):**

| Order | Coder | Task | Files |
|-------|-------|------|-------|
| 2a | Coder C | **F1: Retention Wiring**: Change RetentionManager construction in app.py, update settings page | `app.py` (line 79 only), `cameras.py` (settings route), `settings.html` |
| 2b | Coder D | **F3+F4+F5 Backend**: build_detector changes (detect_classes + sensitivity), GridZone class, sensitivity.py, build_grid_zones, CompoundROI, _build_detectors integration | `registry.py`, `app.py` (_build_detectors), `grid_zone.py`, `sensitivity.py`, `roi.py` (CompoundROI) |

**Phase 3 -- Web UI (PARALLEL, 2 coders, after Phase 2):**

| Order | Coder | Task | Files |
|-------|-------|------|-------|
| 3a | Coder E | **F4: Grid Zones Web UI**: Zone editor routes, zones.html template, zones.js | `cameras.py` (zone routes), `zones.html`, `zones.js`, `detail.html` (link) |
| 3b | Coder F | **F5: Sensitivity Web UI**: Sensitivity POST route, detail.html slider | `cameras.py` (sensitivity route), `detail.html` |

**Phase 4 -- Integration (SEQUENTIAL, after all above):**

| Order | Coder | Task |
|-------|-------|------|
| 4a | Coder G | **Integration**: Verify all app.py changes merge cleanly, run full test suite (528 existing + ~60 new), lint all new code, update README with all 5 features, git commit |

### Context Package Template

Each coder sub-agent receives:

```
## Task: [Feature Name] for rtsp-warden Sprint 6

### Baseline
- v1.1.0 (commit 6d7b0d5), 528 tests passing, 18 runtime deps
- Project root: /home/jcharles/Projects/python/rtsp-warden_v0.2.0/

### Files to Create
[List from plan]

### Files to Modify
[List from plan with specific changes]

### Constraints
- No emoji in code or output
- All public functions need type hints
- All new modules need docstrings
- Stdlib-first; deps only when value is real
- Additive layers -- don't destabilize existing detectors/recording
- FrameConsumer / Detector / Notifier / Camera protocols are stable
- Pytest: asyncio_mode = "auto", tests in tests/
- Use uv for package management
- Config: pydantic v2 models in src/rtsp_warden/config.py
- Web UI: FastAPI + Jinja + htmx + Alpine.js + Pico CSS. No React. No build step.
- Alembic for migrations
- Coordinate system for masks/zones: frame pixel space

### Reference Files to Read
[List of existing files the coder should study before writing]

### Expected Output
- All new files created
- All modifications applied
- Tests passing (run pytest on new test file)
- Return: summary of what was done, any issues encountered
```

---

## Summary: Files Changed Across All 5 Features

### New Files (7)
1. `src/rtsp_warden/detectors/grid_zone.py` -- GridZone + MultiCellROI + CompoundROI
2. `src/rtsp_warden/detectors/sensitivity.py` -- Sensitivity mapping functions
3. `src/rtsp_warden/web/templates/cameras/zones.html` -- Grid zone editor page
4. `src/rtsp_warden/web/static/js/zones.js` -- Zone editor JavaScript
5. `tests/test_per_camera_retention.py`
6. `tests/test_timeline_colors.py`
7. `tests/test_detect_classes.py`
8. `tests/test_grid_zones.py`
9. `tests/test_sensitivity.py`

### Modified Files (12)
1. `src/rtsp_warden/config.py` -- 5 new fields on CameraConfig, 1 new field on AppConfig, 1 new model (GridZoneConfig)
2. `src/rtsp_warden/app.py` -- RetentionManager construction (line 79), _build_detectors (grid zones + sensitivity + detect_classes)
3. `src/rtsp_warden/detectors/registry.py` -- build_detector signature + intersection logic + sensitivity application + build_grid_zones
4. `src/rtsp_warden/detectors/roi.py` -- CompoundROI class
5. `src/rtsp_warden/web/services/timeline.py` -- TimelineEvent.object_type + categorize_object + OBJECT_TYPE_COLORS
6. `src/rtsp_warden/web/routes/api.py` -- Add object_type to timeline JSON
7. `src/rtsp_warden/web/routes/cameras.py` -- Zone editor routes + sensitivity route + settings retention display
8. `src/rtsp_warden/web/static/js/timeline.js` -- Color by object_type, severity as line style
9. `src/rtsp_warden/web/templates/cameras/detail.html` -- Sensitivity slider + zone editor link
10. `src/rtsp_warden/web/templates/cameras/settings.html` -- Per-camera retention display
11. `README.md` -- 5 feature documentation sections

### New Runtime Deps
**None.** All 5 features use existing dependencies only.

### New Test Files
- 5 new test files, estimated ~55-65 new test cases

### New Web Routes
- `GET /cameras/{name}/zones` (admin) -- Grid zone editor page
- `POST /cameras/{name}/zones` (admin) -- Save zone configuration
- `POST /cameras/{name}/sensitivity` (admin) -- Update camera sensitivity

---

## Open Questions / Ambiguities (RESOLVED 2026-06-16)

1. **Sensitivity hot-reload:** **RESOLVED** -- User chose **add config-reload endpoint**. Add `POST /cameras/{name}/reload` (admin) that re-reads that camera's config from yaml and rebuilds its detectors in-place. Only affects the affected camera; no app restart required. The reload endpoint is also useful for `detect_classes` and `zones` changes.

2. **Grid zone + polygon ROI semantics:** **RESOLVED** -- User chose **AND (detection kept only if in ROI AND not blocked)**. Polygon ROI defines the area of interest; grid zones within that area further exclude sub-regions. Both must pass for a detection to survive.

3. **Zone editor snapshot source:** **RESOLVED** -- User chose **MJPEG proxy endpoint**. Use `GET /cameras/{name}/snapshot.jpg` (already exists for the legacy stdlib UI). If unavailable, show a placeholder with a notice.

4. **`detect_classes` for non-DNN detectors:** **RESOLVED** -- User chose **all detector types with per-type enable flags**. Extend with a unified structure: each detector gets an `enabled: bool` field, and DNN additionally gets `classes: list[str]`. For motion: `enabled: bool` (off = never fires). For person: `enabled: bool`. For vehicle: `enabled: bool`. For dnn: `enabled: bool` + `classes: list[str]`. This requires extending `DetectorSpec` to have these fields.

5. **Config.yaml write atomicity:** **RESOLVED** -- User chose **atomic os.replace() + simple file lock**. Use `fcntl.flock(fileno, LOCK_EX)` for exclusive locks; `os.replace()` for atomic writes. Helper function `_locked_write_yaml(path, data)` to be shared between zones, sensitivity, presets.
