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

## Owner actions (not for agents)
- Rotate the Foscam camera password: it was public in `origin` until the 2026-10-03 history rewrite, and GitHub may still serve
  the old commits by SHA (ask GitHub Support to purge them if wanted). Then update `.env`.
- Run the manual camera tests (RW-2/12, RW-3/19) once the camera is free.
- Install the NVIDIA Container Toolkit if the GPU should run inside Docker; `Dockerfile.cuda` builds and the GPU test passes
  natively on this host (2026-10-03), only the in-container GPU run is untested.

## Done 2026-10-03
- History rewrite (`git filter-repo`, real credentials → `admin:admin`) and force-push of `master` + `v1.3.0`; nothing
  tracked or in history holds the real login any more.
- `Dockerfile.cuda` built (5.29 GB, onnxruntime-gpu 1.30.0 with the CUDA provider) and `pytest -m gpu` passed on the RTX 4080.
