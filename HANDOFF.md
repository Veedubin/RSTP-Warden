# HANDOFF

Session state for whoever picks this project up next. Newest block first. Task ids are in `TASKS.md`.

## 2026-10-02 (evening) — START HERE: RW-0 landed on master; RW-2 and RW-3 being built in parallel worktrees

**Where things are.** `master` at `b2ca2a5`: RW-1 + RW-0 (`dcaf660` hygiene, `19c075b` per-camera lifecycle API on
`AppRuntime`: `request_add_camera` / `request_remove_camera` / `request_restart_camera` futures drained by the supervisor,
`CameraExistsError` / `CameraNotFoundError`, `proxy_error`) + the three plans in `docs/superpowers/plans/`
(`2026-10-02-rw0-prefork.md`, `2026-10-02-ui-pass.md`, `2026-10-02-detection.md`) and the rulings file
`2026-10-02-rw2-rw3-decisions.md`. 877 tests, ruff baseline 7 E501, format clean. Nothing pushed.
Worktrees: `.worktrees/rw-2` = branch `feat/ui-pass` (RW-2, 11 tasks + skipped manual task 12), `.worktrees/rw-3` =
branch `feat/detection` (RW-3, 18 tasks + skipped manual task 19), both forked from `19c075b`, each with its own `.venv`.
One implementer agent per worktree was dispatched 2026-10-02 ~20:30 local; check `git -C .worktrees/rw-N log --oneline`
for progress (one commit per task).

**What happened.** Planning session: 13 codebase readers + critic + 20 gap readers mapped the code (maps were in the
session scratchpad, not the repo); the maps proved several spec premises false (frame tap never reaches detectors in a
default `serve`, `pipe:3` fd bug, shared dispatcher fans every camera's frames to every runner, motion has no debounce,
zones are exclusion masks only, clips regex already fixed, `/status.json` leaks ffmpeg stderr with credentials
unauthenticated). Rulings R1-R22 in the decisions file replace those spec sentences. The owner stopped the plan-review
workflow for cost reasons and ordered the build; Critical/Important review findings that had landed were handed to the
implementers. Owner has NOT individually approved the rulings; they are recommended defaults.

**Next, in order.**
1. When both implementers finish: merge `feat/ui-pass` into `master` first, then rebase/merge `feat/detection` on top
   (expected conflicts: `config.py`, `web/app.py`, `base.html`, `dashboard.html`, `cameras/detail.html`, `warden.css`,
   `web/services/cameras.py`, `README.md`, `CLAUDE.md`, `cli.py`, `uv.lock`; RW-2 wins on layout, RW-3 on content).
2. Full gate on master; one whole-branch review (owner said no more review agents mid-build; ask before spending).
3. Manual Foscam tests (plan tasks RW-2/12 and RW-3/19) only when the owner says so; push only with explicit OK.
4. Owner still needs to rotate the Foscam password in origin history.

## 2026-10-02 (afternoon) — START HERE: RW-1 Stabilize merged to local master (not pushed); RW-2 UI pass and RW-3 Detection are planned-in-spec only, owner wants both built in parallel with sub-agents in a fresh session

**Where things are.** Single checkout `/home/jcharles/Projects/python/rtsp-warden_v0.2.0`, branch `master` at
`fb36d99` (17 commits ahead of `origin/master`, nothing pushed), working tree clean. `uv run pytest -q` → 836
passed; `uv run ruff check src/ tests/` → exactly 10 pre-existing E501 (the accepted baseline). The approved design
is `docs/superpowers/specs/2026-10-02-detection-and-automation-design.md`; the finished plan is
`docs/superpowers/plans/2026-10-02-stabilize.md`. The SDD workspace/ledger for RW-1 was deleted after the final
review (git history is the record). Owner's camera facts are in the auto-memory file `test-camera-foscam-c1-v3.md`.

**What happened.** Audited v1.3.0 (CLAUDE.md review, dead code, stale docs, UI map), brainstormed the roadmap,
wrote and committed the spec, then executed RW-1 inline: 15 tasks, TDD, one fresh-reviewer pass, one fix pass
(commit `fb36d99`). Before → after: web settings forms 403 → work; login swapped into card → full page; reload
503 → 200; config write-back skipped → written through the file lock; clips never found segments → found (local
time, real chunk length); retention under `record:` ignored → honored with warning; `sub_url` required → optional;
fresh Docker/systemd had no admin → `serve` bootstraps schema + admin; 780 → 836 tests; 20 → 10 ruff baseline;
`zeep`, `aiofiles`, `health_server.py`, `web_ui.py`, `run`/`ui` commands removed; README/CLAUDE.md/Docker/systemd
docs corrected; example configs use `${CAM_USER}`/`${CAM_PASS}`.

**Known open.**
- The Foscam password that was committed in `examples/configs/` before the scrub is still in public git history
  on `origin`. Only the owner can rotate it. Not done.
- Deferred minors from the review (owner has the list in the session summary): admin password logged at WARNING
  instead of a 0600 file; CSRF middleware doesn't `form.close()` multipart; `_resolve_web_settings` mutates a
  BaseSettings; proxy-stream fallback validator is silent; one worker thread per live MJPEG viewer; `HX-Redirect`
  has no `next`; restart countdown truncates; nine route modules still build their own Jinja2Templates; unused
  `app_secret` in `run_install`.
- Pre-existing, ruled out of scope: supervisor restart-churn on `mode: event` recorders (UI shows "waiting");
  `Camera.sub_url` DB column non-null (table never written); non-constant-time CSRF compare.
- The Foscam's `videoSub` stream sends no video packets (camera-side); use `main_url` only for that camera.

**Next, in order.**
1. Owner decides whether to push `master` (public repo) — needs explicit OK.
2. RW-2 and RW-3: owner's words — "plan out 2 and 3 and do them at the same time with sub-agents? Then merge them
   after the fact and do a code review". Each needs its own brainstorm → plan (`superpowers:writing-plans`) →
   execution (`superpowers:subagent-driven-development`), in separate git worktrees off `master`, then merge,
   then one whole-branch review. Overlap to expect at merge: `web/routes/cameras.py`, `web/services/cameras.py`,
   `web/templates/cameras/*`, `config.py` (both add camera fields), `warden.css`.
3. RW-2 scope is spec Appendix A.2 (bounded: UI cleanup, add-camera flow with ffprobe test + ONVIF URL fill,
   edit/delete, mobile check). RW-3 scope is spec sections 4-12 (ONNX detection, tracker, rules, actions, UI, GPU
   Docker image). The owner dropped Pi/GPIO from scope; keep it out.

## (no earlier blocks)
