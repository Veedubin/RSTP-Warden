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
- Status: **in progress** on branch `feat/ui-pass` in `.worktrees/rw-2` (one implementer agent, tasks 1-11; task 12
  manual, gated on owner). Merge first.

## RW-3 — Detection and automation (sub-project 3 of 3)
- Spec: same document, sections 4-12 and Appendix B
- Plan: `docs/superpowers/plans/2026-10-02-detection.md` (rulings in `2026-10-02-rw2-rw3-decisions.md`)
- Status: **in progress** on branch `feat/detection` in `.worktrees/rw-3` (one implementer agent, tasks 1-18; task 19
  manual, gated on owner). Merge after RW-2. Owner decisions kept: ONNX Runtime + YOLOX (no Ultralytics/AGPL), GPU
  first with CPU fallback, per-camera `detect_fps` plus per-detector `fps`, actions = ntfy/Apprise/webhook, no Pi/GPIO.

## Owner actions (not for agents)
- Rotate the Foscam camera password that is in `origin` git history (pre-scrub `examples/configs/`).
- Decide on pushing `master` to `origin`.
