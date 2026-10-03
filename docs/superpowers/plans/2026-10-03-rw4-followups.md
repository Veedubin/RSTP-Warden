# RW-4: follow-ups from the first live run (2026-10-03)

Owner-approved after the first evening of the Foscam running on the GPU stack: false "microwave"
events from a static shape on a table, soft event thumbnails, and the wish to manage the camera
without Foscam's dead browser plugin. Built test-first by one implementer, no plan workflow.

## 1. Stationary suppression (`stationary_iou`)

- `Track.first_bbox` (tracker) never changes; `bbox` follows the object.
- `EventBuilder(stationary_iou=...)`: a track the tracker opened whose `iou(first_bbox, bbox)` is
  at least the threshold is *held*: no row, no thumbnail, no callback. On a later update where the
  box has drifted below the threshold it opens with `created_at` = that frame's time (the move,
  not the first sighting, is the event; keeps clips short for a parked car that leaves). A held
  track that closes in place is dropped and counted.
- Component default is off (0.0); `CameraConfig.stationary_iou` defaults to 0.6 and is wired in
  `AppRuntime._make_runner`. Why 0.6: box jitter of a few pixels on a 80x45 box keeps IoU above
  0.75; a person walking laterally drops below 0.6 within a couple of frames.
- Surfaces: `runner.status()["stationary_held" | "stationary_suppressed"]`, `summarize_detection`,
  the Detection panel (field + "N objects that never moved were held back").
- Known limit (pre-existing): the tracker matches consecutive boxes at IoU >= 0.3, so an object
  that crosses the frame faster than its own width per detector frame is never tracked at all.
  Raise the detector `fps` for such cameras.

## 2. Full-size thumbnails

- `FrameHub` keeps a sparse history (one frame per 0.2 s, 5 s deep, ~25 JPEGs) and answers
  `frame_near(ts, tolerance_s=0.75)`.
- `EventBuilder(frame_source=hub.frame_near)`: the thumbnail is the full-size preview frame
  nearest `track.best_ts` with `best_bbox` scaled onto it, when that frame is wider than the tap
  frame; else the tap frame. A rewrite never replaces a full-size thumbnail with a tap-size one
  (`_OpenEvent.full_res`). Row metadata (`bbox`, `frame_size`) stays in tap pixels.
- Cameras without an MJPEG hub (proxy off or rtsp mode) keep tap-size thumbnails.

## 3. Camera settings page (Foscam CGI)

- `vendors/foscam.py`: `FoscamClient` over httpx (one client per call, like actions), XML
  `<CGI_Result>` parsing, result codes -1..-7 as text, `FoscamError` without URL or credentials.
  Commands: getDevInfo, get/set{Main,Sub}VideoStreamType, get/set{,Sub}VideoStreamParam,
  getImageSetting + setBrightness/Contrast/Hue/Saturation/Sharpness/DenoiseLevel,
  getMirrorAndFlipSetting + mirrorVideo/flipVideo, getInfraLedConfig + setInfraLedConfig +
  open/closeInfraLed, get/setOSDSetting, snapPicture2, rebootSystem.
- `CameraConfig.vendor: VendorConfig | None` (`type: foscam`, `port: 88`).
- `web/routes/vendor.py`: `/cameras/{name}/vendor` page + enable/disable (patch only the
  `vendor` key of the raw YAML), stream / image / video fragments (htmx `outerHTML` swaps), snapshot
  and reboot. Credentials come from `main_url` through `split_userinfo`. Tests inject a fake
  through `vendor_routes._client`.
- Resolution codes: only 0 (1280x720) and 3 (640x360) are named; both were read off the owner's
  C1 V3 (recording probe and the sub stream's SDP). Foscam's guide lists another order, so
  unverified codes show as numbers.
