# TASKS

One card per unit of work. Status phrases are updated in place; history stays in git and `HANDOFF.md`.

## RW-1 — Stabilize (sub-project 1 of 3)
- Spec: `docs/superpowers/specs/2026-10-02-detection-and-automation-design.md` Appendix A.1
- Plan: `docs/superpowers/plans/2026-10-02-stabilize.md`
- Status: **done**, merged to local `master` at `fb36d99` on 2026-10-02 (fast-forward, branch deleted). Not pushed.
  836 tests green. Deferred minors listed in `HANDOFF.md`.

## RW-2 — UI pass + add-camera flow (sub-project 2 of 3)
- Spec: same document, Appendix A.2 (bounded)
- Plan: not written
- Status: **not started**. Owner wants it built in parallel with RW-3 by sub-agents in a fresh session, each in its
  own worktree off `master`, merged afterwards, then reviewed once.

## RW-3 — Detection and automation (sub-project 3 of 3)
- Spec: same document, sections 4-12 and Appendix B
- Plan: not written
- Status: **not started**. Same execution instructions as RW-2. Decisions already made by the owner: ONNX Runtime +
  permissively licensed YOLOX (no Ultralytics/AGPL), GPU first with CPU fallback, per-camera `detect_fps` plus
  per-detector `fps`, actions = ntfy/Apprise/webhook only, no Pi/GPIO.

## Owner actions (not for agents)
- Rotate the Foscam camera password that is in `origin` git history (pre-scrub `examples/configs/`).
- Decide on pushing `master` to `origin`.
