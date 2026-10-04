"""Capture the web UI pages as PNGs for the README (Playwright + the system Chromium).

    WARDEN_SCREENSHOT_USER=... WARDEN_SCREENSHOT_PASSWORD=... \\
        uv run --with playwright python tools/screenshots.py [--url http://127.0.0.1:3333] \\
            [--out docs/screenshots] [--chromium /usr/bin/chromium] [--pages dashboard,camera,...]

Logs in through the normal login form, visits each page at a 1280 px wide viewport and saves
``<out>/<name>.png`` (full page). The pages that need a camera name take it from
``--camera`` (default: the first camera in ``WARDEN_SCREENSHOT_CAMERA`` or ``front``).
Credentials come from the environment only, never from arguments, so they do not land in a
shell history. This is a developer tool: it is not part of the package or the test suite.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PAGES: dict[str, str] = {
    "dashboard": "/",
    "camera": "/cameras/{camera}",
    "events": "/events",
    "health": "/health",
    "actions": "/actions",
    "camera-settings": "/cameras/{camera}/vendor",
    "detection-classes": "/cameras/{camera}/detection-classes",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default=os.environ.get("WARDEN_URL", "http://127.0.0.1:3333"))
    ap.add_argument("--out", type=Path, default=Path("docs/screenshots"))
    ap.add_argument("--chromium", default=os.environ.get("CHROMIUM", "/usr/bin/chromium"))
    ap.add_argument("--camera", default=os.environ.get("WARDEN_SCREENSHOT_CAMERA", "front"))
    ap.add_argument("--pages", default=",".join(PAGES), help="comma-separated subset")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--settle-ms", type=int, default=2500, help="wait for htmx/live preview")
    args = ap.parse_args()

    user = os.environ.get("WARDEN_SCREENSHOT_USER")
    password = os.environ.get("WARDEN_SCREENSHOT_PASSWORD")
    if not user or not password:
        print("set WARDEN_SCREENSHOT_USER and WARDEN_SCREENSHOT_PASSWORD", file=sys.stderr)
        return 2
    wanted = [p.strip() for p in args.pages.split(",") if p.strip()]
    unknown = [p for p in wanted if p not in PAGES]
    if unknown:
        print(f"unknown pages: {unknown}; known: {', '.join(PAGES)}", file=sys.stderr)
        return 2

    from playwright.sync_api import sync_playwright

    args.out.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        launch = {"headless": True}
        if args.chromium and Path(args.chromium).exists():
            launch["executable_path"] = args.chromium
        browser = pw.chromium.launch(**launch)
        page = browser.new_page(viewport={"width": args.width, "height": 800})
        page.goto(f"{args.url}/login")
        page.fill('input[name="username"]', user)
        page.fill('input[name="password"]', password)
        page.click('button[type="submit"]')
        page.wait_for_load_state("networkidle")
        if "/login" in page.url:
            print("login failed", file=sys.stderr)
            return 1
        for name in wanted:
            path = PAGES[name].format(camera=args.camera)
            page.goto(f"{args.url}{path}")
            page.wait_for_load_state("networkidle")
            page.wait_for_timeout(args.settle_ms)
            target = args.out / f"{name}.png"
            page.screenshot(path=str(target), full_page=True)
            print(f"{name:18s} {path:40s} -> {target}")
        browser.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
