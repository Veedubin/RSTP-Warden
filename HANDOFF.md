# HANDOFF

Session state for whoever picks this project up next. Newest block first. Task ids are in `TASKS.md`.

## 2026-10-04 (early morning) — START HERE: RW-5 runtime shipped and gated; the wildlife model is training

**Where things are.** `master` is local-only ahead of `origin/master` (`715e8ec`) by the RW-5 commits (`d6ed1c2`
docs .. `4e402be` version 1.4.0); nothing pushed yet (push only on the owner's word). Single checkout, working tree
clean apart from the gitignored `config/`, `data/`, `tools/wildlife/data/`, `tools/wildlife/.yolox/`,
`tools/wildlife/YOLOX_outputs/`. Gate on `4e402be`: **2292 passed / 1 skipped**, ruff exactly the 4 baseline E501,
format clean. The compose stack still runs the pre-RW-5 image against the Foscam.

**What RW-5 shipped (runtime, all tested offline).** Spec `docs/superpowers/specs/2026-10-04-wildlife-detection-design.md`,
plan `docs/superpowers/plans/2026-10-04-rw5-wildlife-detection.md` (its "Deviations" section is the truth where it
differs from the task text).
- `detectors/daylight.py`: `channel_spread` + `DayNight` (first frame sets the state, 3-frame hysteresis, threshold
  4.0). The runner measures every decoded frame before the privacy masks, sets `event_builder.night`, writes
  `"night"` into every event's metadata, reports `night` / `night_since` / `night_switches`; Detection panel says
  "night mode on / off"; events show a `night` badge (old rows: none).
- `DetectorSpec.when` (`always | day | night`, counted as `when_skipped`) and `DetectorSpec.classes` (onnx only,
  validated against that model's labels, intersected with `detect_classes`; two slots sharing a label log a warning).
  Detection panel: a `when` select on every row and a `classes` field on onnx rows
  (`POST /cameras/{name}/detectors/{index}/when|classes`, patch-one-key write-back, 409 / 422 as the fps route).
- `tools/wildlife/` (own uv project, never in the wheel): `setup.sh` (uv sync + pinned YOLOX checkout in `.yolox/`,
  because YOLOX's `setup.py` imports torch at build time), `fetch.py`, `prepare.py`, `exp.py`, `train.sh`,
  `evaluate.py`, `export.py`, `verify.py`; pure helpers pinned by `tests/test_wildlife_tool.py`.

**Training, in flight.** Dataset built: 16473 train / 1830 val images (ENA24 7898+891, Open Images 8393+921, Dat Tran
182+18); train boxes cat 2053, fox 1321, raccoon 865, person 5123, vehicle 4010; 499 of the 1830 val images are
grayscale. The public ENA24 zip has **no human images**, hence Open Images `Person` and `Car` as hard negatives.
`BATCH=32 ./train.sh` started 01:55 on the RTX 4080 SUPER (about 10 GB, 0.29 s/iter, 515 iters/epoch, 50 epochs,
so roughly 2.5 h); log in the session scratchpad (`train.log`), checkpoints in `tools/wildlife/YOLOX_outputs/wildlife_yolox_s/`.
If a fresh session finds `best_ckpt.pth` there: `cd tools/wildlife && uv run --no-sync python evaluate.py` (soft
target: cat / fox / raccoon AP50 >= 0.6 on the `gray` column), `uv run --no-sync python export.py`, then from the
repo root `uv run python tools/wildlife/verify.py tools/wildlife/out/wildlife-yolox-s <a val raccoon image>`,
`cp -r tools/wildlife/out/wildlife-yolox-s data/models/`, add the second detector to `config/config.yaml` (block in
README "Wildlife model"), rebuild + restart the compose stack (`docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build`).

**Next, in order.** 1. Finish training → evaluate → export → verify → install into the live stack (above).
2. HANDOFF/TASKS with the eval table; then the owner's "push". 3. Owner: watch for `fox` / `raccoon` / `cat` events
with the right label by day and by night; optionally drop the Roboflow "Cat/Raccoons" COCO export into
`tools/wildlife/data/raw/roboflow-cat-raccoons/` and retrain for more raccoons. 4. Later: publish the model as a
release asset + built-in descriptor; an event "this was actually a fox" relabel button (RW-6 candidate).

## 2026-10-03 (late afternoon) — START HERE: RW-4 verified by the owner on the live camera; nothing in flight

**Where things are.** `master` = `origin/master` at `17e196b` plus this docs commit; single checkout
`/home/jcharles/Projects/python/rtsp-warden_v0.2.0`, no worktrees, working tree clean apart from the gitignored
`config/`, `data/`, `recordings/`, `.env`. Gate unchanged since the block below: 2228 passed / 1 skipped, ruff exactly the
4 baseline E501, format clean. The compose stack (GPU overlay, http://127.0.0.1:3333/) runs this code against the Foscam.

**What happened.** After the RW-4 push the owner used the new Camera settings page on the real camera and reported,
verbatim: "I was able to change the profile just fine. It seems to work. Motion detection and object recognition works."
That is the owner's acceptance of RW-4/3 (stream-profile switching through the Foscam CGI) and a first pass over the
live detection path of RW-3 (motion and YOLOX on the GPU produce events). No code changed after `63c2697`.

**Known open.**
- Not yet seen by anyone: a real event's full-size (1280 px) thumbnail and a growing "held back" count on the Detection
  panel; both need a real visitor / a light change after the restart. If a real visitor is ever held, lower
  Stationary IoU on the Detection panel (default 0.6; 0 turns it off).
- Pre-existing tracker limit: consecutive boxes must overlap at IoU >= 0.3, so a fast crosser is never tracked; raise
  the detector `fps` on such cameras.
- Foscam resolution codes other than 0 (1280x720) and 3 (640x360) show as numbers (unverified on this model).
- Still the owner's: rotate the Foscam password (then `.env` +
  `docker compose -f docker-compose.yml -f docker-compose.gpu.yml restart warden`); the remaining items of the manual
  checklists RW-2/12 and RW-3/19 (add-camera flow, rules firing an action, clips) were not walked explicitly.

**Next, in order.** 1. Nothing is queued for an agent; the next agent task is whatever the owner reports from using the
UI (bugs or the next feature). 2. Owner: password rotation, then the rest of the manual checklists.

## 2026-10-03 (afternoon) — superseded by the block above: RW-4 done (stationary suppression, full-size thumbnails, Foscam settings page), pushed

**Where things are.** `master` = `origin/master`, single checkout `/home/jcharles/Projects/python/rtsp-warden_v0.2.0`.
Three feature commits on top of the midday state: `5b76252` stationary suppression, `498e313` full-size thumbnails,
and the Foscam camera-settings page (see `git log`). Gate: `uv run pytest -q` → 2228 passed, 1 skipped (`gpu` marker);
`uv run ruff check src/ tests/` → exactly the 4 baseline E501; `uv run ruff format --check` clean. Everything was built
test-first by one implementer after the owner's "go"; the record is `docs/superpowers/plans/2026-10-03-rw4-followups.md`.

**What RW-4 changed.**
- `CameraConfig.stationary_iou` (default 0.6; 0 = off): a track whose box still overlaps where it was first seen by that
  IoU is held, opens no event until it moves (then with the move frame as `created_at`), and is dropped if it ends in
  place. `Track.first_bbox`, `EventBuilder(stationary_iou=)`, `held_count` / `suppressed_total`, runner status and
  `/status.json` (`stationary_held`, `stationary_suppressed`), Detection panel field and count. The component default
  stays 0 so hand-built builders behave as before; the camera default wires 0.6.
- `FrameHub` history (`frame_near`) + `EventBuilder(frame_source=hub.frame_near)`: thumbnails are the full-size preview
  frame nearest the best detection with the box scaled on, never downgraded to a tap-size one afterwards. Cameras
  without an MJPEG hub keep tap-size thumbnails.
- `vendors/foscam.py` + `web/routes/vendor.py` + `cameras/vendor.html` and three partials: `/cameras/{name}/vendor`
  (admin) with device info, main/sub stream profiles (use / edit), image tuning, mirror / flip / infrared / OSD, a
  snapshot from the camera and reboot. Enabled per camera by `vendor: {type: foscam, port: 88}` (the page's own enable
  form patches only that key). Credentials come from `main_url`; errors never carry URL or credentials. Tests inject a
  fake through `vendor_routes._client`.

**Live state on this host.** The compose stack (GPU overlay, port 3333) runs the new code against the Foscam, with
`vendor:` enabled in `./config/config.yaml`, `detect_classes` limited to person/cat/dog/car/truck/bicycle/motorcycle,
`min_confidence` 0.6 and `min_track_frames` 3. The camera's main stream was switched to its profile 0 (2 Mbps, 25 fps,
VBR) through the CGI (`setMainVideoStreamType&streamType=0`; `streamType=1` is the old 1 Mbps/15 fps one). The settings
page was fetched logged-in against the real camera: device, profiles, image values and a 191 KB snapshot render, the
password is nowhere in the page. Not yet observed live: a real event's full-size thumbnail (no event since the restart).

**Known open.**
- The tracker (pre-existing) matches consecutive boxes at IoU >= 0.3, so something crossing the frame faster than its
  own width per detector frame is never tracked; such cameras need a higher detector `fps`.
- Foscam resolution codes: only 0 (1280x720) and 3 (640x360) are named, both read off this C1 V3; the vendor guide
  lists a different order, so other codes show as numbers. Other vendors: none.
- Earlier items still stand: rotate the Foscam password (old commits may still be served by GitHub by SHA), manual
  checks RW-2/12 and RW-3/19 in the running UI.

**Next, in order.** 1. Owner watches the next real events (thumbnail width, "held back" count on the Detection panel;
tune Stationary IoU there if a real visitor is ever held). 2. Owner rotates the camera password, updates `.env`, and
runs `docker compose -f docker-compose.yml -f docker-compose.gpu.yml restart warden`. 3. Manual checks RW-2/12, RW-3/19.

## 2026-10-03 (midday) — history rewritten (password gone), master pushed, CUDA image verified on the GPU in Docker

**Where things are.** `master` = `origin/master` at `90a5d52` (plus this docs commit), single checkout
`/home/jcharles/Projects/python/rtsp-warden_v0.2.0`. On 2026-10-03 the owner said "scrub my password ... then push it", so the
whole history was rewritten with `git filter-repo --replace-text` (one literal rule: the real `user:pass@` in
`config-Foscam-C1-V3.yaml` / `examples/configs/config-Foscam-C1-V3.yaml` became `admin:admin@`; it touched 11 commits and
nothing else, commit messages were clean). Every SHA changed (`1710701` → `90a5d52`, tag `v1.3.0` → `fa5fbf2`). Verified with
`git grep -F <pass> $(git rev-list --all)` = 0 hits across all 67 commits before `git push --force origin master` and
`git push --force origin refs/tags/v1.3.0`. The local branches `feat/ui-pass` and `feat/detection` were deleted (fully merged).
The pre-rewrite bundle was deleted after the push; the only copy of the old history is whatever GitHub still holds. The real
login stays only in the gitignored `.env` (`CAM_USER` / `CAM_PASS`). Gate at `90a5d52` (tree identical to `1710701`):
2156 passed / 1 skipped, ruff exactly 4 baseline E501, format clean.

**CUDA verified on this host (RTX 4080 SUPER), in Docker too.** `docker build -f Dockerfile.cuda -t rtsp-warden:cuda .`
succeeds (5.29 GB); inside the image `onnxruntime-gpu` 1.30.0 is the only onnxruntime distribution. After the owner installed
the NVIDIA Container Toolkit (1.20.0, Docker 29.8.1) the shipped overlay (`driver: nvidia`) failed with "could not select device
driver nvidia": the toolkit's runtime is not registered with Docker on this host, only its CDI spec exists
(`/etc/cdi/nvidia.yaml`, generated by root). `docker-compose.gpu.yml` therefore now reserves the GPU through CDI
(`driver: cdi`, `device_ids: ["nvidia.com/gpu=all"]`), the README's GPU section gained the
`sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml` step, and the overlay comment keeps the classic `driver: nvidia`
form as the alternative; `tests/test_deploy_docs.py` pins both. End-to-end check: the image run with
`--device nvidia.com/gpu=all`, a scratch config with a dead camera URL and `type: onnx, model: yolox-nano, device: cuda`
(model pre-downloaded into the mounted `WARDEN_MODELS_DIR`) logged
`onnx detector onnx (model yolox-nano, device cuda) provider: CUDAExecutionProvider`, `nvidia-smi` in the container showed the
process holding 268 MiB, `/healthz` answered ok and `/status.json` redacted the camera credentials. `pytest -m gpu` also passes
natively in a scratch GPU venv (`UV_PROJECT_ENVIRONMENT=<scratch> uv sync --frozen --extra gpu --no-install-package onnxruntime`).

**Known open.**
- GitHub still serves the old pre-rewrite commits by SHA (e.g. `70fa60a`) until it garbage-collects them; the owner can ask
  GitHub Support to purge them. The password was public for a while either way, so rotating it on the camera is still due.
- Any other clone of the repo must be re-cloned (or `git fetch && git reset --hard origin/master`); old clones re-introduce the
  old history if they push.
- Manual Foscam tests (plan tasks RW-2/12, RW-3/19) still not run (camera not released).
- Pre-existing, out of scope: `record.mode: event` cameras cannot start recording from detection (the tap rides the same ffmpeg).

**Running on this host (2026-10-03 midday).** `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build`
is up: `./config/config.yaml` is a copy of `examples/configs/config-Foscam-C1-V3.yaml` (main stream only, motion + yolox-s on
`device: auto`), credentials and `WARDEN_ADMIN_PASSWORD` come from the repo-root `.env` (gitignored; compose interpolates it),
the DB and models live in `./data`, segments in `./recordings`. Host port 8080 was taken by another rootless container stack
(`rootlessport`), so compose now publishes the web UI on `${WARDEN_HOST_PORT:-3333}` (owner's choice) and `config/`, `data/` are
gitignored. First start: migrations ran to 0003, the Foscam main stream shows RUN, yolox-s downloaded and loaded with
`provider: CUDAExecutionProvider` (about 400 MiB on the GPU), `/healthz` ok at http://127.0.0.1:3333/.

**Next, in order.** 1. Owner rotates the Foscam password (then updates `.env` and `docker compose restart warden`). 2. Owner walks
the manual checks of plan tasks RW-2/12 and RW-3/19 in the running UI (login `admin` with the `.env` password, then change it).

## 2026-10-03 (morning) — RW-2 + RW-3 built, merged, reviewed and fixed on local master; waiting for the owner's push decision

**Where things are.** `master` in the single checkout `/home/jcharles/Projects/python/rtsp-warden_v0.2.0`; code head
`dcadb42` (review fixes in `fad1eb8`, integration in `57fb350`, merge in `80c2323`); 58 commits ahead of
`origin/master` (`70fa60a`), nothing pushed. Gate at `dcadb42`: `uv run pytest -q` → 2156 passed, 1 skipped (the `gpu`
marker); `uv run ruff check src/ tests/` → exactly 4 baseline E501 (`cli.py` x2, `proxy/mjpeg.py:143`, `recorder.py:401`);
`uv run ruff format --check` clean. Python floor is `>=3.11`. Worktrees removed; branches `feat/ui-pass` and
`feat/detection` are fully merged and can be deleted. The owner's real camera login is in the gitignored `.env` at the
repo root (`CAM_USER` / `CAM_PASS`, mode 0600) and nowhere tracked; every committed example uses `admin` / `admin`.

**What happened.** `/code-review high` over `e5623cb..master` (37 commits, ~14k lines): 8 findings, none Critical, all fixed
test-first in `fad1eb8` (retention form answers 422 not 500 and patches only its camera; detection and retention saves answer
409 when the camera left config.yaml; rule cooldowns survive a camera restart; one credential-redaction helper; model hash
and descriptor reads cached; stale AlertManager comments). Then the owner, verbatim: "purge my password from ANYTHING that is
going to be pushed to remote. put mine in a .env file ... Make it the default admin:admin for whatever you commit."
Verified: no unpushed commit ADDS the password (RW-1's scrub commit `02f8a6c` only removes it); the two lines that hold it
are already public in origin's pre-scrub `examples/configs/config-Foscam-C1-V3.yaml`. Placeholders switched to admin/admin
in `dcadb42`.

**Known open.**
- origin's history still contains the old camera password. Only a history rewrite (`git filter-repo` + force push) removes
  it, and it has been public, so the owner must rotate it on the camera either way. Both are the owner's decisions.
- Manual Foscam tests (plan tasks RW-2/12, RW-3/19) and the `Dockerfile.cuda` build never ran (camera not released, no GPU host).
- Pre-existing, out of scope: `record.mode: event` cameras cannot start recording from detection (the tap rides the same ffmpeg).
- Rulings R1-R22 (`docs/superpowers/plans/2026-10-02-rw2-rw3-decisions.md`) were applied as written; the owner reviewed the
  outcome, not each ruling.

**Next, in order.** 1. Owner says "push" → `git push origin master` (explicit OK required; never force-push without it).
2. Owner rotates the Foscam password and decides on the history rewrite. 3. Manual camera verification per the two plan tasks
(camera at 192.168.1.72, main stream only; login in `.env`). 4. Build and test `Dockerfile.cuda` on a GPU host.

## 2026-10-02 (night) — superseded by the block above: RW-2 and RW-3 built and merged; master is the integrated tree, not pushed

**Where things are.** `master` = RW-1 + RW-0 + RW-2 (`feat/ui-pass`, 11 tasks) + RW-3 (`feat/detection`, 18 tasks) +
the merge (`80c2323`) + the integration commit (`chore: integrate RW-2 and RW-3 after the merge`). Gate on master:
`uv run pytest -q` → 2150 passed, 1 skipped (the `gpu` marker); `uv run ruff check src/ tests/` → exactly 4 baseline
E501 (`cli.py` x2, `proxy/mjpeg.py`, `recorder.py`); `uv run ruff format --check` clean. Python floor is now `>=3.11`
(onnxruntime 1.30.0 core, `gpu` extra onnxruntime-gpu[cuda,cudnn], onnx dev). Verified by booting `serve` against a
scratch config: the 0002 database auto-upgraded to 0003 with a `.bak-0002_clips` copy first, login works, all pages
return 200 (dashboard, cameras, add/edit, zones, detection panel, sensitivity, classes, events, actions, health,
settings, users, api-tokens, onvif), `/alerts` and `/recordings` are gone (404), `/status.json` no longer leaks
credentials. Nothing pushed. Branches `feat/ui-pass` and `feat/detection` are fully merged (worktrees removed).

**What was built.** RW-2: shared templates instance, flash messages, nav/page header/CSS primitives, htmx form fixes,
one camera-card partial with polled status and redacted errors (+ `/status.json` redaction), ffprobe connection test,
ONVIF WS-UsernameToken GetStreamUri discovery over ports 80/8080/888/2020, camera config services (env-ref
credentials in `<config dir>/.env`, raw-YAML append/patch/remove, port allocation), add/edit/delete camera routes with
hot add/restart/remove through the RW-0 lifecycle API, working ONVIF page (form routes, locked preset writes,
per-camera `onvif_port`), writable Docker/systemd config, mobile-width pass. RW-3: `detect_fps`, detector
`fps/model/device/events`, rules, deprecations; frame tap fixed (per-camera dispatcher, real pipe fd, proxy stream
only, ordered runner); model registry with YOLOX descriptors + label validation; OnnxDetector (CUDA/CPU); IoU tracker;
migration 0003 + packaged migrations + auto-upgrade; area zones; EventBuilder/MotionBurst/per-detector fps; sync
ntfy/webhook/apprise actions + rule engine + legacy alerts mapping; ActionQueue + delayed ClipScheduler + mp4 clips
with .ts fallback; runtime wiring + integration test; event cards; actions page; camera detection panel; live boxes;
status surfaces; Dockerfile.cuda + GPU compose overlay + docs. Rulings R1-R22 in
`docs/superpowers/plans/2026-10-02-rw2-rw3-decisions.md` were applied as written; the owner has not reviewed them
individually.

**Known open / deferred.**
- Whole-branch code review (2026-10-03, `/code-review high`, e5623cb..master): 8 findings, none Critical, all fixed in
  `fix: address code-review findings` (retention form 422s, 409 when a camera left config.yaml, rule cooldowns survive
  restarts, one redaction helper, model hash cached, descriptor reads cached, per-camera retention patch).
- Manual Foscam tests (plan tasks RW-2/12, RW-3/19) not run: need the owner's OK and the camera.
- Dockerfile.cuda / GPU path not built or run on this host (no NVIDIA container toolkit). CPU path verified offline only.
- YOLOX model SHA-256 values come from mirrors; the first real download verifies them (fails loudly on mismatch).
- `docker/README.md` and `packaging/systemd/README.md` got the minimum edits; README's configuration reference is RW-3's.
- Owner still needs to rotate the Foscam password present in `origin` git history, and decide on pushing `master`.

**Next, in order.** 1. Owner: review `git log master`, decide on push. 2. Optional: one whole-branch review
(`/code-review`) before pushing. 3. Manual camera verification per the two plan tasks. 4. Build the CUDA image on a
GPU host.

## 2026-10-02 (evening) — RW-0 landed on master; RW-2 and RW-3 being built in parallel worktrees

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

## 2026-10-02 (afternoon) — RW-1 Stabilize merged to local master (not pushed); RW-2 UI pass and RW-3 Detection are planned-in-spec only, owner wants both built in parallel with sub-agents in a fresh session

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
