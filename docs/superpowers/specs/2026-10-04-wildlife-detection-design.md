# RW-5: wildlife detection (cat, fox, raccoon) day and night — design

Owner's ask (2026-10-04): "I want it to know the difference between a cat, fox, and raccoon",
most sightings are at night, "I want to see all the flags for a category until I am happy this
works." No rule gating on day or night for now. Approved in chat 2026-10-04 ("go").

## 1. Problem

The detector pipeline runs YOLOX-S on the 80 COCO labels. COCO has `cat`, `dog` and `bear`
and nothing closer to a fox or a raccoon, so a raccoon on the Foscam at night comes out as
`cat`, `dog`, `bear` or nothing. The model registry accepts only `postprocess: yolox`, and
`detect_classes` and `rules[].labels` are validated against the model's label file, so the
clean way in is a second model whose label file lists the animals.

The camera switches its IR-cut filter at night and the frames become grayscale. Most of the
animals of interest come out at night. The model has to work in both domains.

## 2. Decisions

1. **One model for day and night**, trained on camera-trap night images and daytime photos
   together, with every colour training image also fed as a grayscale copy (brightness and
   noise jitter) so the network cannot learn "grey frame = raccoon, colour frame = cat".
   Two separate day and night models were considered and deferred: they double the training
   and tuning work before we know a single model is weak in one domain.
2. **A second `onnx` detector slot**, not a replacement. The camera keeps `yolox-s` for people
   and vehicles and adds `wildlife-yolox-s` for the animals. The tracker already keeps tracks
   per label; rules can say `labels: [raccoon]`.
3. **A frame-based night flag** with hysteresis, computed by the runner from the decoded tap
   frame. It is a signal (status, event metadata, per-detector `when`), not a pipeline switch.
   Nothing gates rules on it in this release.
4. **`when: always | day | night` on every detector spec**, default `always`, so a
   night-specialised model can be added later with no new plumbing.
5. **`classes` on an `onnx` detector spec**: the labels that slot may report. Without it two
   models both reporting `dog` or `person` for one animal would open two tracks and two events.
6. **Training lives in `tools/wildlife/`**, a separate uv project outside the package and the
   wheel. The model is produced as a user model in `data/models/wildlife-yolox-s/`; publishing
   it as a GitHub release asset with a built-in descriptor is a later, owner-triggered step.
7. **MIT-clean**: YOLOX code and weights (Apache-2.0), ENA24 (CDLA-Permissive 1.0),
   Open Images (CC BY 4.0 annotations, CC BY 2.0 images), Dat Tran's raccoon set (MIT),
   Roboflow "Cat/Raccoons" (CC BY 4.0, optional, needs a login to download). No Ultralytics
   (AGPL) or YOLO-World (GPL) code or weights anywhere.

## 3. Labels and data

### 3.1 `wildlife.txt`

One label per line, index = class id, in this order:

```
cat
dog
fox
raccoon
skunk
opossum
squirrel
rabbit
coyote
bobcat
deer
bear
bird
chipmunk
woodchuck
horse
person
vehicle
```

Red fox and grey fox both become `fox`. `dog`, `coyote`, `bobcat`, `person` and `vehicle` stay
in as hard negatives (a coyote must not be called a fox, a person not a bear); the camera's slot
`classes` decides which of them are reported.

### 3.2 Sources

| Source | Taken | Mapping | License |
|---|---|---|---|
| LILA ENA24-detection (~10k camera-trap images, boxes, 3.6 GB) | every image with a box | ENA24 category → label above; `Human` → `person`, `Chicken` / `Wild Turkey` / `American Crow` / `Bird` → `bird`, squirrels → `squirrel`, foxes → `fox` | CDLA-Permissive 1.0 |
| Open Images V7 boxes | `Cat`, `Fox`, `Raccoon`, `Skunk`, `Squirrel`, `Rabbit`, `Dog`, capped at 1500 images per class, train split, images over plain HTTPS from the public S3 bucket | class name lower-cased | CC BY 4.0 / CC BY 2.0 |
| Dat Tran raccoon set (196 images, Pascal VOC xml) | all | `raccoon` | MIT |
| Roboflow "Cat/Raccoons" (3262 images) | all, **only if the owner places the COCO-format export in `tools/wildlife/data/raw/roboflow-cat-raccoons/`** | `cat`, `raccoon` | CC BY 4.0 |

Unknown ENA24 category names stop the conversion with the list of names, so a renamed
category is never silently dropped. Everything downloads into `tools/wildlife/data/`
(gitignored), resumable, skipping files that exist.

### 3.3 Split and evaluation

90/10 split by image with a fixed seed. Each validation image carries an `is_gray` flag from
the same channel-spread measure the runtime uses (section 5.1), so evaluation reports AP50 per
class for all, colour-only and grayscale-only validation images. The number that matters is
`fox`, `raccoon` and `cat` AP50 on grayscale images.

Soft target: AP50 >= 0.6 for each of `cat`, `fox`, `raccoon` on grayscale validation images.
Below that, at most one more training run (more epochs or more data) before reporting back to
the owner with the numbers.

## 4. Training tool (`tools/wildlife/`)

A uv project (`requires-python >=3.11,<3.13`, because YOLOX's dependency pins are not known to
install on 3.14) with PyTorch CUDA wheels, torchvision, the YOLOX repo pinned to a commit,
pycocotools, onnx, httpx, PyYAML, OpenCV headless. It is never imported by the package.

Commands, each a script with `--help`, documented in `tools/wildlife/README.md`:

- `fetch.py`: download the sources of 3.2 into `data/raw/`.
- `prepare.py`: convert and merge into `data/coco/{train,val}.json` plus image directories
  (symlinks into `data/raw/`), with `is_gray` on validation images and a per-class count table.
- `exp.py`: the YOLOX `Exp` for YOLOX-S: 18 classes, input 640, start from the official
  `yolox_s.pth`, about 50 epochs, batch sized for 16 GB, mosaic and mixup as YOLOX ships them,
  plus the grayscale copy augmentation (probability 0.5 on colour images: convert to gray,
  replicate to 3 channels, random brightness scale 0.4–1.0, Gaussian noise).
- `train.sh`: `python -m yolox.tools.train -f exp.py -d 1 -b <batch> --fp16 -o -c yolox_s.pth`.
- `evaluate.py`: COCO eval on the full, colour and grayscale validation sets, AP50 per class.
- `export.py`: ONNX export with the raw head (`decode_in_inference = False`, the tensor
  contract of the `yolox-s` descriptor: input `images` `[1,3,640,640]`, output `[1,8400,23]`),
  SHA-256, and a `model.yaml` written next to it:

  ```yaml
  name: wildlife-yolox-s
  file: wildlife_yolox_s.onnx
  labels: wildlife.txt
  input_size: [640, 640]
  postprocess: yolox
  sha256: <hex>
  ```

- `verify.py` is run from the **main** venv (`uv run python tools/wildlife/verify.py <dir>`):
  loads the descriptor through `rtsp_warden.detectors.model_registry`, builds an
  `OnnxDetector`, runs a validation image through it and prints the detections. This is the
  proof that the export matches the runtime's decoder.

Pure data functions (label mapping, ENA24 category mapping, Open Images CSV filtering, box
conversion, split, `is_gray`) live in `tools/wildlife/wildlife_data.py`, import only the
standard library, NumPy and OpenCV, and are tested from `tests/test_wildlife_tool.py` by file
path, so the main test suite stays offline and torch-free.

## 5. Runtime changes (package)

### 5.1 Night flag (`detectors/daylight.py`)

- `channel_spread(frame_bgr) -> float`: mean over a 4x-subsampled frame of
  `max(B,G,R) - min(B,G,R)`. A grayscale JPEG from the IR mode measures about 0–2 (chroma
  subsampling noise); a daytime colour frame tens. Default threshold `NIGHT_SPREAD = 4.0`.
  Known limit: a very dark colour frame also measures low and counts as night, which is the
  intended meaning ("IR or too dark for colour").
- `DayNight(threshold=NIGHT_SPREAD, switch_frames=3)`: `update(frame_bgr, ts_unix) -> bool`.
  The first frame sets the state at once; afterwards the state flips only after
  `switch_frames` consecutive frames on the other side (no flapping at dusk). Fields:
  `night: bool | None`, `since_ts: float | None`, `switches: int`, `last_spread: float`.
- `DetectorRunner` owns one `DayNight` and updates it in `_process_job` **before**
  `apply_masks` (masked pixels are black and would bias the measure). The result is stored on
  the runner and set on the event builder (`event_builder.night = ...`) before the motion and
  tracked stages, so `EventBuilder` writes `"night": true|false` into every object and motion
  event's metadata. `DayNight` and the runner accept the threshold for tests.
- Runner status gains `night` (bool or None), `night_since` (unix float or None),
  `night_switches` (int). `status_model.summarize_detection` carries them; `/status.json`
  and the Detection panel show "Night mode: yes / no (since hh:mm)".

### 5.2 `when` on `DetectorSpec`

- `when: Literal["always", "day", "night"] = "always"`.
- `DetectorSlot.when`. In `_process_job`, before the fps check, a slot whose `when` does not
  match the current state is skipped and `_slot_when_skipped[i]` counts it; status
  `when_skipped`. The fps schedule already restarts after a long pause.
- `detector_summary` shows `when=night` when it is not `always`.
- Detection panel: admin gets a `when` select on each detector row posting to
  `/cameras/{name}/detectors/{index}/when` (same shape as the `fps` route: patch only that key
  through `_persist_detector_entry`, update the in-memory spec, rebuild). Non-admins see the
  value in the settings column.

### 5.3 `classes` on an `onnx` spec

- `classes: list[str] | None = None`. `AppConfig._validate_labels` validates each `onnx`
  spec's `classes` against **that model's** labels (`unknown_labels_message`) and rejects
  `classes` on a non-`onnx` spec.
- `_build_onnx_detector`: the detector's class filter is the spec's `classes` intersected
  with the camera's `detect_classes` when both are set, else whichever is set, else all.
- Detection panel: admin gets a `classes` text input (comma-separated) per `onnx` row posting
  to `/cameras/{name}/detectors/{index}/classes`; unknown labels for that model are a 422
  naming them; empty clears the key. `class_groups_for` puts the wildlife names into its
  animals group so the camera-wide classes page groups them sensibly.

### 5.4 Events UI

`event_to_dict` exposes `night: bool | None` from the metadata; `event_card.html` and the
event detail page show a small `night` badge when true.

### 5.5 Config write-back and hot reload

Every write goes through `_persist_detector_entry` → `update_config_yaml` (one lock from read
to write). `when` and `classes` changes are plain rebuilds (`rebuild_camera_detectors`): the
tap settings do not change. The in-memory `CameraConfig` is updated before the rebuild, as the
`fps` route does.

### 5.6 Camera config after the model is in place

```yaml
    detect_classes: [person, car, truck, bicycle, motorcycle, cat, dog, fox, raccoon, skunk, opossum, squirrel, rabbit]
    detectors:
      - type: motion
        fps: 5
      - type: onnx
        model: yolox-s
        classes: [person, car, truck, bicycle, motorcycle]
        fps: 2
        min_confidence: 0.6
      - type: onnx
        model: wildlife-yolox-s
        classes: [cat, dog, fox, raccoon, skunk, opossum, squirrel, rabbit]
        fps: 2
        min_confidence: 0.5
```

`dog` is on the wildlife slot only. The README gets a "Wildlife model" section with this
example, the training commands, and the user-model location (`WARDEN_MODELS_DIR`, compose:
`data/models/`). `examples/config.yaml` and the `init-config` template do **not** reference
`wildlife-yolox-s` until a built-in descriptor exists (their validation reads descriptors).

## 6. Testing

All offline, in the main suite:

- `test_daylight.py`: synthetic grey, colour, dark and mixed frames; hysteresis (2 frames do
  not flip, 3 do); first frame sets state; `since_ts` and `switches`.
- `test_detector_runner.py` additions: night flag on status, set on the event builder, `when`
  skipping with counters, `when` slots resume after a state change.
- `test_event_builder.py`: metadata `night` on object and motion events, None when unknown.
- `test_config_detection.py` / `test_config_labels.py`: `when` default and values, `classes`
  validated per model, `classes` on a motion spec rejected, intersection with `detect_classes`.
- `test_onnx_registry.py` or `test_detector_integration.py`: `_build_onnx_detector` class
  filter from spec and camera.
- `test_detection_panel.py` / `test_web_detectors.py`: `when` and `classes` routes (patch the
  raw YAML, 409 on type mismatch, 422 on unknown label, htmx fragment), panel shows night mode.
- `test_status_detection_surfaces.py`: `night*` fields in `summarize_detection` and
  `/status.json`.
- `test_events_*`: `night` badge.
- `test_wildlife_tool.py`: the pure data functions by file path (label mapping including the
  unknown-category error, Open Images CSV filter and caps, VOC and COCO box conversion, seeded
  split, `is_gray`).
- `tests/test_deploy_docs.py` keeps validating the README's marked config examples; the new
  README example is marked only when the built-in descriptor exists.

Live acceptance is the owner's: a cat, a fox and a raccoon each producing an event with the
right label, by day and by night, with the model dropped into `data/models/wildlife-yolox-s/`.

## 7. Out of scope

- Rules gating on day / night (the flag is there; the rule field is a later task).
- A night-specialised second model (`when: night` makes it a config change later).
- A "this was actually a fox" labelling button on events and the retraining loop (RW-6
  candidate; the thumbnails and the `night` flag already give it its data).
- MegaDetector or any non-YOLOX head in the registry.
- Publishing the model as a release asset and a built-in descriptor (owner-triggered follow-up;
  the README carries the descriptor text ready to paste).

## 8. Release

Version 1.4.0 (`pyproject.toml`, `__init__.py`, the assertion in `tests/test_admin.py`).
Gate: `uv run ruff check src/ tests/` (4 baseline E501 only), `uv run ruff format --check`,
full `uv run pytest`.
