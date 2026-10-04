# TASKS

One card per unit of work. Status phrases are updated in place; history stays in git and `HANDOFF.md`.

## RW-1 — Stabilize (sub-project 1 of 3)
- Spec: `docs/superpowers/specs/2026-10-02-detection-and-automation-design.md` Appendix A.1
- Plan: `docs/superpowers/plans/2026-10-02-stabilize.md`
- Status: **done**, merged to local `master` at `fb36d99` on 2026-10-02 (fast-forward, branch deleted). Not pushed.
  836 tests green. Deferred minors listed in `HANDOFF.md`.

## RW-0 — Pre-fork: repo hygiene + per-camera lifecycle API (added 2026-10-02 evening)
- Plan: `docs/superpowers/plans/2026-10-02-rw0-prefork.md`
- Status: **done** on `master` (`dcaf660`, `19c075b`). 877 tests, ruff baseline 7.

## RW-2 — UI pass + add-camera flow (sub-project 2 of 3)
- Spec: same document, Appendix A.2 (bounded)
- Plan: `docs/superpowers/plans/2026-10-02-ui-pass.md` (rulings in `2026-10-02-rw2-rw3-decisions.md`)
- Status: **done**, merged to `master` 2026-10-02 night (branch `feat/ui-pass`, 11 task commits; manual task 12 not run, needs the owner). Reviewed 2026-10-03 (`/code-review high`, fixes in `fad1eb8`).

## RW-3 — Detection and automation (sub-project 3 of 3)
- Spec: same document, sections 4-12 and Appendix B
- Plan: `docs/superpowers/plans/2026-10-02-detection.md` (rulings in `2026-10-02-rw2-rw3-decisions.md`)
- Status: **done**, merged to `master` 2026-10-02 night on top of RW-2 (branch `feat/detection`, 18 task commits + merge + integration commit; manual task 19 not run, needs the owner). Reviewed 2026-10-03 (`/code-review high`, fixes in `fad1eb8`). Owner decisions kept: ONNX Runtime + YOLOX, GPU first with CPU fallback, per-camera `detect_fps` plus per-detector `fps`, actions = ntfy/Apprise/webhook, no Pi/GPIO.

## RW-4 — Follow-ups from the first live run (added 2026-10-03 afternoon)
- Plan/record: `docs/superpowers/plans/2026-10-03-rw4-followups.md`
- Scope (owner's "go"): 1. stationary suppression (`stationary_iou`, default 0.6, in the Detection panel);
  2. full-size event thumbnails from the preview frame nearest the detection; 3. a per-camera "Camera
  settings" page over Foscam's HTTP CGI (`vendor: {type: foscam, port: 88}`), replacing the dead browser plugin.
- Status: **done and accepted** on `master` 2026-10-03 (`5b76252`, `498e313`, `63c2697`; pushed). Owner, verbatim:
  "I was able to change the profile just fine. It seems to work. Motion detection and object recognition works."
  Not yet observed live: a real event's full-size thumbnail and the "held back" count growing.

## RW-5 — Wildlife detection: cat / fox / raccoon, day and night (added 2026-10-04)
- Spec: `docs/superpowers/specs/2026-10-04-wildlife-detection-design.md`
- Plan: `docs/superpowers/plans/2026-10-04-rw5-wildlife-detection.md` (12 tasks, one implementer, no plan workflow)
- Scope (owner's "go"): a second `onnx` model `wildlife-yolox-s` (YOLOX-S fine-tuned on ENA24 + Open Images +
  raccoon sets, 18 labels in `tools/wildlife/wildlife.txt`, grayscale augmentation so one model covers day and IR
  night); runtime: frame-based `night` flag (status, event metadata, badge), `when: always|day|night` and
  per-slot `classes` on detector specs with Detection-panel controls; training tool in `tools/wildlife/` (separate
  uv project, never in the wheel). No rule gating on night in this release; the owner wants every event of every
  label visible until the model is trusted. Release 1.4.0.
- Status: **done** 2026-10-04 morning, on local `master` (not pushed). Runtime + web + tool committed (`d19038a` ..
  `bc2c87f`), version 1.4.0, gate 2295 passed / 1 skipped, ruff baseline 4. Model trained (2 h on the 4080: AP50
  on grayscale validation cat 0.958 / fox 0.970 / raccoon 0.920), exported, verified through the runtime and
  installed in `data/models/wildlife-yolox-s/` with the second detector in the live config. Owner: watch the first
  real animal events; optional Roboflow set + retrain; later a release asset + built-in descriptor.

## RW-5b — Event deletion (admin) and README screenshots (added 2026-10-04 afternoon)
- Owner's ask after RW-5: screenshots of the UI, and a way to delete events (their test thumbnails must not appear),
  admin only.
- Status: **done** 2026-10-04 (`9644ea6` + follow-up commits, pushed). Per-event delete (card + detail) and
  "Delete N events" matching the list filters, both verified live (67 events cleared on the owner's word, a test
  event deleted from its card); `tools/screenshots.py`; seven pages in `docs/screenshots/` and the README section.

## Owner actions (not for agents)
- Rotate the Foscam camera password: it was public in `origin` until the 2026-10-03 history rewrite, and GitHub may still serve
  the old commits by SHA (ask GitHub Support to purge them if wanted). Then update `.env`.
- Walk the rest of the manual camera checks (RW-2/12, RW-3/19) in the stack running on http://127.0.0.1:3333/ (GPU
  overlay, Foscam main stream); motion + object detection and the Camera settings page are confirmed working (2026-10-03).
- Look at the next few real events: thumbnails should now be 1280 px wide, and the Detection panel's "held back"
  count should grow instead of microwave/bottle events. Tune Stationary IoU there if a real visitor is ever held.

## Done 2026-10-03
- History rewrite (`git filter-repo`, real credentials → `admin:admin`) and force-push of `master` + `v1.3.0`; nothing
  tracked or in history holds the real login any more.
- `Dockerfile.cuda` built (5.29 GB, onnxruntime-gpu 1.30.0 with the CUDA provider) and `pytest -m gpu` passed on the RTX 4080.
- GPU in Docker verified after the owner installed the container toolkit: the overlay switched to a CDI reservation (the
  nvidia runtime is not registered with Docker here), and the container logged `provider: CUDAExecutionProvider`.
