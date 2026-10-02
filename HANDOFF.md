# HANDOFF

Session state for whoever picks this project up next. Newest block first. Task ids are in `TASKS.md`.

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
