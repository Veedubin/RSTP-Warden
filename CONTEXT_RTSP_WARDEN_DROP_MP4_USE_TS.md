# RTSP Warden — MP4 Segment Failures, Root Cause, and Plan (H.264 RTSP)

## Date / context
- Project: `rtsp-warden_v0.2.0`
- User ran:
  - `RTSP_WARDEN_DEBUG_STDIO=1 RTSP_WARDEN_DEBUG_STDOUT=1 rtsp-warden run --config config-Foscam-C1-V3.yaml`
- Verified execution is from the local repo checkout:
  - `rtsp-warden` path: `/home/jcharles/Projects/python/rtsp-warden_v0.2.0/.venv/bin/rtsp-warden`
  - Python imports:
    - `/home/jcharles/Projects/python/rtsp-warden_v0.2.0/src/rtsp_warden/__init__.py`
    - `/home/jcharles/Projects/python/rtsp-warden_v0.2.0/src/rtsp_warden/ffmpeg.py`

---

## Symptoms observed
### Recording failures (copy/remux into MP4 segments)
FFmpeg repeatedly logs:

- `Could not find codec parameters for stream 0 (Video: h264, none): unspecified size`
- `dimensions not set`
- `Could not write header (incorrect codec parameters ?): Invalid argument`

Supervisor then restarts ingest:
- `ingest for front died; restarting ...`

### Timestamp warnings
- `Timestamps are unset in a packet for stream 0. This is deprecated...`
- `Non-monotonic DTS; previous: ..., current: ...; changing to ...`

### Additional RTSP/H.264 metadata warnings
- `Missing PPS in sprop-parameter-sets, ignoring`
- MJPEG preview path warning:
  - `deprecated pixel format used, make sure you did set range correctly`

---

## Root cause (what is actually broken)
The RTSP camera stream is not delivering *reliable codec metadata at startup* for H.264, specifically:

1) **SPS/PPS issue**
- H.264 width/height and other required parameters are derived from SPS/PPS.
- FFmpeg is explicitly warning that SDP’s `sprop-parameter-sets` is malformed: **PPS missing**.
- If PPS does not arrive quickly in-band (typically on an IDR/keyframe), FFmpeg cannot determine dimensions.

2) **Timestamp issue**
- The camera stream is producing **missing and/or non-monotonic timestamps**.
- With `-c copy`, FFmpeg can only do limited correction; newer FFmpeg builds warn that “unset timestamps” may stop working in the future.

These issues can exist even when VLC “plays” the stream, because VLC is more forgiving than FFmpeg’s muxers and segment writers.

---

## Why MP4 was the point of failure
With `-c copy` into **MP4 segments**, FFmpeg must write a valid container header immediately. That header requires:
- Known codec parameters (including width/height)
- Valid extradata (SPS/PPS) or equivalent
- Reasonable timestamping

When FFmpeg starts segmenting and still has “unspecified size,” it cannot write the MP4 header → segment muxer fails immediately.

This is why the error chain is always:
`unspecified size` → `dimensions not set` → `Could not write header`

---

## Decision: remove MP4 recording
**We will stop trying to record to MP4 for these RTSP camera streams.**
Instead:
- Use **MPEG-TS (.ts)** segments for recording with `-c copy`
- Optionally remux to MP4 later offline if desired (when the stream is stable)

Rationale:
- TS is substantially more tolerant:
  - Does **not** require the same up-front header metadata as MP4
  - Handles imperfect timestamps and late-arriving parameter sets better
- TS is the standard “NVR-style” approach for hostile IP camera streams.

---

## What the remaining warnings mean (and what we do about them)
### `Missing PPS in sprop-parameter-sets`
- Indicates the camera is emitting malformed SDP H.264 parameters.
- TS recording is more tolerant; playback may still be affected until an IDR arrives with proper SPS/PPS in-band.
- Best fix is still on the camera settings (see below), but TS mitigates recording startup failures.

### `deprecated pixel format used`
- Typically impacts the MJPEG preview conversion pipeline (colorspace/range).
- Usually non-fatal; only adjust if the MJPEG preview looks wrong (washed out / crushed blacks).

---

## Next steps (hand-off to another agent)

### 1) Implement TS segment recording (primary change)
Change the recording command builder so that:
- Segment output uses MPEG-TS, not MP4:
  - `-f segment`
  - `-segment_format mpegts`
  - Output extension: `.ts`
- Keep `-c copy` as default (fast, low CPU)

Expected outcome:
- “Could not write header / dimensions not set” should stop killing recording on startup.

### 2) Update config defaults / schema
- Remove or deprecate MP4 as a record container for segment recording.
- Set default container to `ts` (or `mpegts`) for both main and sub streams.

### 3) Handle timestamp slop pragmatically
Even with TS, keep flags that reduce timestamp-related thrash (if present in your builder):
- `-fflags +genpts`
- `-use_wallclock_as_timestamps 1` (helpful for some sources)
- Consider `-avoid_negative_ts make_zero` for certain muxers (optional; evaluate)

### 4) Camera-side settings (if user can change them)
These reduce the probability of broken SPS/PPS and timestamping:
- Force H.264 (not H.265)
- Reduce keyframe/I-frame interval to ~1–2 seconds
- Disable “smart codec / variable GOP / H.264+ / ROI” features
- Enable “Repeat headers / Insert SPS/PPS before each I-frame” if the UI exposes it

### 5) Test plan (smoke tests)
For each stream (main + sub):
- Start Warden and confirm:
  - Recording process stays RUN for > 2–5 minutes
  - `.ts` segments appear on disk and grow as expected
- Play segments:
  - `ffplay segment.ts` or `vlc segment.ts`
- Check that substream now records (previously failing).

### 6) Optional future improvements (if time)
- Add a “warmup / probe” period before declaring ingest failed (useful but not required if TS resolves stability).
- Add per-stream fallback policy:
  - default `copy+ts`
  - if still unstable → `transcode` mode as last resort
- Make MJPEG preview set explicit pixel format/range if preview quality is wrong.

---

## Summary for the hand-off
- The core failure is **not** “we didn’t know the width/height from config.”
- The core failure is malformed / late H.264 parameter sets (missing PPS) and bad timestamps.
- MP4 segmenting fails because it needs clean parameters at startup.
- TS segmenting is the practical, NVR-grade solution: **stop fighting the camera; record TS.**
