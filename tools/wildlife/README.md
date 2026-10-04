# wildlife-yolox-s training tool

Builds the `wildlife-yolox-s` model that lets rtsp-warden tell a cat, a fox and a raccoon
(and 15 other labels, see `wildlife.txt`) apart on IR night frames as well as by day. It is a
separate uv project: nothing here is imported by the `rtsp_warden` package or shipped in the
wheel. Design: `docs/superpowers/specs/2026-10-04-wildlife-detection-design.md`.

Needs an NVIDIA GPU with about 12 GB free (tested on a 16 GB RTX 4080 SUPER), roughly 7 GB of
disk under `data/`, and one to three hours.

## Steps

All commands run from this directory unless noted.

```bash
./setup.sh                               # uv sync (Python 3.11, PyTorch cu128) + pinned YOLOX checkout
uv run python fetch.py                   # ENA24 (3.6 GB), Open Images boxes + images, raccoon set
uv run python prepare.py                 # one COCO dataset in data/coco/, 90/10 split
./train.sh                               # YOLOX-S fine-tune, about 50 epochs (BATCH=16 if OOM)
uv run python evaluate.py                # AP50 per class: all / colour / grayscale validation
uv run python export.py                  # out/wildlife-yolox-s/{.onnx, wildlife.txt, model.yaml}
```

Then from the repository root, with the main venv:

```bash
uv run python tools/wildlife/verify.py tools/wildlife/out/wildlife-yolox-s <some image>
cp -r tools/wildlife/out/wildlife-yolox-s data/models/      # compose stack; or $WARDEN_MODELS_DIR
```

and add the second detector to the camera (see "Wildlife model" in the main README).

## Data

| Source | Used | License |
|---|---|---|
| [LILA ENA24-detection](https://lila.science/datasets/ena24detection) | all boxed images (camera traps, mostly night IR) | CDLA-Permissive 1.0 |
| [Open Images V7](https://storage.googleapis.com/openimages/web/factsfigures_v7.html) | `Cat`, `Fox`, `Raccoon`, `Skunk`, `Squirrel`, `Rabbit`, `Dog`; up to 1500 images each | CC BY 4.0 (boxes), CC BY 2.0 (images) |
| [Dat Tran's raccoon set](https://github.com/datitran/raccoon_dataset) | all 196 images | MIT |
| [Roboflow "Cat/Raccoons"](https://universe.roboflow.com/cat-feeder-project/cat-raccoons) | optional, see below | CC BY 4.0 |

The Roboflow set needs a login to download. To include it, export it as **COCO JSON** and
unzip into `data/raw/roboflow-cat-raccoons/` (so that `train/_annotations.coco.json` exists);
`prepare.py` picks it up when the directory is there.

Both day and night work because every colour training image is also fed as a grayscale copy
with brightness and noise jitter (`wildlife_data.grayscale_copy`), so the model cannot learn
"grey frame = raccoon, colour frame = cat".

## Why YOLOX is a checkout, not a dependency

YOLOX's `setup.py` imports torch to pre-compile an optional extension, so `uv` cannot build it
as an ordinary dependency. `setup.sh` runs `uv sync` and then fetches the pinned commit
`6ddff4824372906469a7fae2dc3206c7aa4bbaee` of
[Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) (Apache-2.0) into
`.yolox/`; every script here puts that directory on `sys.path` when it exists. The pure-Python
COCO evaluator is used (no compiled extension needed).

## Files

- `wildlife.txt`: the 18 labels in class-id order; copied next to the model by `export.py`.
- `wildlife_data.py`: pure helpers (label mapping, Open Images filtering, box conversion,
  split, the grayscale measure); tested by `tests/test_wildlife_tool.py` in the main suite.
- `fetch.py`, `prepare.py`, `exp.py`, `train.py`, `train.sh`, `evaluate.py`, `export.py`:
  the pipeline, in order. `verify.py` runs in the main venv.
