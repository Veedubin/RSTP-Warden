"""Mobile-width markup checks for every RW-2 page (RW-2 Task 11).

This project has no browser harness, so a 375 px phone is approximated by the
markup rules that keep the page itself from scrolling sideways:

* every ``<table>`` sits inside an element with class ``table-wrap``
  (``overflow-x: auto`` in warden.css), so a wide table scrolls inside its card;
* no inline ``style`` sets ``width`` / ``min-width`` wider than a phone (a
  percentage or ``vw`` above 100, or a fixed length above 320 px), and no inline
  ``grid-template-columns`` overrides Pico's one-column ``.grid`` below 768 px;
* every full page extends ``base.html``, so it carries the viewport meta tag.

Rendered pages are checked through TestClient. Template sources are checked as
well, because some RW-2 fragments only reach the browser through htmx.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from rtsp_warden import auth
from rtsp_warden.config import load_config
from rtsp_warden.db.schema import get_user_by_username
from rtsp_warden.web.app import create_app
from rtsp_warden.web.config import WebSettings
from rtsp_warden.web.paths import STATIC_DIR, TEMPLATES_DIR

CAMERA = "front"

# 375 px minus Pico's two 1rem gutters is 343 px; 320 px leaves room for card padding.
MAX_FIXED_PX = 320.0

# Every RW-2 page as an admin sees it. The tokens page lives at /api-tokens.
ADMIN_PAGES = (
    "/",
    "/cameras",
    f"/cameras/{CAMERA}",
    "/cameras/new",
    f"/cameras/{CAMERA}/edit",
    f"/cameras/{CAMERA}/zones",
    f"/cameras/{CAMERA}/zones/editor",
    "/health",
    "/settings",
    "/users",
    "/users/new",
    "/users/1/reset-password",
    "/api-tokens",
    "/onvif",
    "/events",
    "/actions",
    f"/cameras/{CAMERA}/sensitivity",
    f"/cameras/{CAMERA}/detection-classes",
)

# Pages whose fixture data renders at least one table, so the wrap check bites.
PAGES_WITH_TABLES = (
    f"/cameras/{CAMERA}",
    f"/cameras/{CAMERA}/zones",
    "/health",
    "/settings",
    "/users",
    "/api-tokens",
)

# Templates RW-3 owns, plus dashboard.html, whose "Recent events" and "Recent
# recordings" blocks belong to RW-3; the dashboard is checked as a rendered page.
NOT_RW2_DIRS = ("actions/", "alerts/", "clips/", "events/", "recordings/")
NOT_RW2_FILES = frozenset(
    {
        "dashboard.html",
        "cameras/_detection_panel.html",
        "cameras/detection_classes.html",
        "cameras/sensitivity.html",
        "partials/_detection_badge.html",
        "partials/action_test_result.html",
        "partials/detector_list.html",
        "partials/event_card.html",
        "partials/event_row.html",
        "partials/hls_player.html",
        "partials/timeline.html",
        "partials/zone_kind.html",
    }
)

_VOID = frozenset(
    {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }
)
_JINJA = re.compile(r"\{#.*?#\}|\{%.*?%\}|\{\{.*?\}\}", re.DOTALL)
_WIDTH_DECL = re.compile(r"(?:^|;)\s*(min-width|width)\s*:\s*([^;]+)", re.IGNORECASE)
_GRID_COLUMNS = re.compile(r"(?:^|;)\s*grid-template-columns\s*:", re.IGNORECASE)
_LENGTH = re.compile(r"^(\d+(?:\.\d+)?)(px|rem|em|vw|%)$")
_MEDIA_600 = re.compile(r"@media\s*\(\s*max-width\s*:\s*600px\s*\)\s*\{")
_GRID_ONE_COLUMN = re.compile(
    r"\.grid(?![\w-])[^{}]*\{[^}]*grid-template-columns\s*:\s*1fr\s*(?:!important\s*)?[;}]"
)
_CSS_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_CSS_RULE = re.compile(r"([^{}]+)\{([^{}]*)\}")


def style_problem(style: str) -> str | None:
    """Return why an inline style is too wide for a phone, or None when it fits."""
    if _GRID_COLUMNS.search(style):
        return "inline grid-template-columns stops Pico stacking the grid on phones"
    for prop, raw in _WIDTH_DECL.findall(style):
        value = raw.strip().lower()
        match = _LENGTH.match(value)
        if match is None:
            continue  # auto, fit-content, calc(), min(), var() are not judged here
        number, unit = float(match.group(1)), match.group(2)
        if unit in ("%", "vw"):
            if number > 100:
                return f"{prop}: {value} is wider than the screen"
        elif number * (16 if unit in ("rem", "em") else 1) > MAX_FIXED_PX:
            return f"{prop}: {value} is wider than a 375 px phone"
    return None


class MarkupScan(HTMLParser):
    """Collect mobile-width problems from one HTML document or template source."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._open: list[tuple[str, frozenset[str]]] = []
        self.tables = 0
        self.unwrapped_tables: list[str] = []
        self.wide_styles: list[str] = []
        self.has_viewport = False

    @property
    def problems(self) -> list[str]:
        return self.unwrapped_tables + self.wide_styles

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {name: value or "" for name, value in attrs}
        line = self.getpos()[0]
        if tag == "table":
            self.tables += 1
            if not any("table-wrap" in classes for _, classes in self._open):
                self.unwrapped_tables.append(f"line {line}: <table> outside .table-wrap")
        problem = style_problem(values.get("style", ""))
        if problem:
            self.wide_styles.append(f'line {line}: <{tag} style="{values["style"]}">: {problem}')
        if tag == "meta" and values.get("name") == "viewport":
            self.has_viewport = True
        if tag not in _VOID:
            self._open.append((tag, frozenset(values.get("class", "").split())))

    def handle_endtag(self, tag: str) -> None:
        for index in range(len(self._open) - 1, -1, -1):
            if self._open[index][0] == tag:
                del self._open[index:]
                return


def scan(html: str) -> MarkupScan:
    """Parse *html* and return the collected problems."""
    parser = MarkupScan()
    parser.feed(html)
    parser.close()
    return parser


def _strip_jinja(source: str) -> str:
    """Drop Jinja tags but keep their newlines, so reported line numbers match the file."""
    return _JINJA.sub(lambda m: "\n" * m.group(0).count("\n"), source)


def _rw2_templates() -> list[Path]:
    found = []
    for path in sorted(TEMPLATES_DIR.rglob("*.html")):
        rel = path.relative_to(TEMPLATES_DIR).as_posix()
        if rel.startswith(NOT_RW2_DIRS) or rel in NOT_RW2_FILES:
            continue
        found.append(path)
    return found


def _media_600_bodies(css: str) -> list[str]:
    bodies = []
    for match in _MEDIA_600.finditer(css):
        depth, index = 1, match.end()
        while depth and index < len(css):
            if css[index] == "{":
                depth += 1
            elif css[index] == "}":
                depth -= 1
            index += 1
        bodies.append(css[match.end() : index - 1])
    return bodies


def _css_rules(css: str) -> list[tuple[set[str], str]]:
    """Return (selectors, body) for every innermost rule, comments removed."""
    return [
        ({part.strip() for part in selectors.split(",")}, body)
        for selectors, body in _CSS_RULE.findall(_CSS_COMMENT.sub("", css))
    ]


def _warden_css() -> str:
    return (STATIC_DIR / "css" / "warden.css").read_text(encoding="utf-8")


def _login(client: TestClient) -> None:
    client.get("/login")
    token = client.cookies.get("warden_csrf", "")
    r = client.post(
        "/login",
        data={"username": "admin", "password": "testpass123", "csrf_token": token},
        follow_redirects=False,
    )
    assert r.status_code == 303


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    recordings = str(tmp_path / "recordings")
    raw = {
        "cameras": [
            {
                "name": CAMERA,
                "main_url": "rtsp://u:p@h/m",
                "record": {"enabled": False, "output_dir": recordings},
                "proxy": {"enabled": True, "mode": "mjpeg", "stream": "main", "port": 9001},
                "zones": [
                    {
                        "name": "driveway",
                        "grid_cols": 16,
                        "grid_rows": 16,
                        "blocked_cells": [[0, 0], [1, 0]],
                        "frame_width": 1920,
                        "frame_height": 1080,
                    }
                ],
            },
            {
                "name": "back",
                "main_url": "rtsp://u:p@h/b",
                "record": {"enabled": False, "output_dir": recordings},
                "proxy": {"enabled": False},
            },
        ],
        "runtime": {"workspace_dir": str(tmp_path / "workspace")},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture
def admin_client(
    db_with_user: str, config_file: Path, monkeypatch: pytest.MonkeyPatch
) -> TestClient:
    monkeypatch.setenv("WARDEN_AUTH_ENABLED", "true")
    cfg = load_config(config_file)
    app = create_app(WebSettings(), cfg=cfg, runtime_provider=lambda: None, config_path=config_file)
    client = TestClient(app)
    _login(client)
    admin = get_user_by_username("admin")
    assert admin is not None
    auth.create_api_token(admin, "phone-check")
    return client


def test_style_problem_rules() -> None:
    assert style_problem("") is None
    assert style_problem("display:inline") is None
    assert style_problem("display:block; max-width:100%;") is None
    assert style_problem("width:100%; height:80px") is None
    assert style_problem("width: 320px") is None
    assert style_problem("min-width: 20rem") is None
    assert style_problem("width:800px") == "width: 800px is wider than a 375 px phone"
    assert style_problem("min-width: 30rem") == "min-width: 30rem is wider than a 375 px phone"
    assert style_problem("width: 120%") == "width: 120% is wider than the screen"
    assert style_problem("grid-template-columns: 1fr auto;") is not None


def test_scan_finds_unwrapped_and_wrapped_tables() -> None:
    result = scan(
        '<div class="table-wrap"><table><tr><td>a</td></tr></table></div>'
        "<article><table><tr><td>b</td></tr></table></article>"
    )
    assert result.tables == 2
    assert result.unwrapped_tables == ["line 1: <table> outside .table-wrap"]


def test_strip_jinja_keeps_line_numbers() -> None:
    source = "{# a\nthree-line\ncomment #}\n<table></table>\n"
    assert scan(_strip_jinja(source)).unwrapped_tables == ["line 4: <table> outside .table-wrap"]


@pytest.mark.parametrize("path", ADMIN_PAGES)
def test_admin_page_fits_a_phone(admin_client: TestClient, path: str) -> None:
    r = admin_client.get(path)
    assert r.status_code == 200, f"GET {path} returned {r.status_code}"
    result = scan(r.text)
    assert result.has_viewport, f"{path} does not extend base.html (no viewport meta tag)"
    assert result.problems == [], f"{path}: {result.problems}"


def test_login_page_fits_a_phone(admin_client: TestClient) -> None:
    anonymous = TestClient(admin_client.app)
    r = anonymous.get("/login")
    assert r.status_code == 200
    result = scan(r.text)
    assert result.has_viewport
    assert result.problems == []


@pytest.mark.parametrize("path", PAGES_WITH_TABLES)
def test_fixture_data_renders_tables(admin_client: TestClient, path: str) -> None:
    """Guard: these pages must render tables, or the wrap check proves nothing."""
    assert scan(admin_client.get(path).text).tables >= 1, path


@pytest.mark.parametrize(
    "template", _rw2_templates(), ids=lambda p: p.relative_to(TEMPLATES_DIR).as_posix()
)
def test_template_source_fits_a_phone(template: Path) -> None:
    result = scan(_strip_jinja(template.read_text(encoding="utf-8")))
    rel = template.relative_to(TEMPLATES_DIR).as_posix()
    assert result.problems == [], f"{rel}: {result.problems}"


def test_css_stacks_grids_and_scrolls_tables_on_phones() -> None:
    css = _warden_css()
    bodies = _media_600_bodies(css)
    assert bodies, "warden.css has no @media (max-width: 600px) block"
    assert any(_GRID_ONE_COLUMN.search(body) for body in bodies), (
        "no @media (max-width: 600px) block stacks .grid into one column"
    )
    assert re.search(r"\.table-wrap\s*\{[^}]*overflow-x\s*:\s*auto", css), "no .table-wrap rule"
    assert "minmax(min(300px, 100%), 1fr)" in css, ".camera-grid overflows a 320 px phone"


@pytest.mark.parametrize(
    "selector", ["main h1", ".camera-card header a", "main code", ".camera-card-body small"]
)
def test_css_breaks_long_names_and_urls(selector: str) -> None:
    """A 32-character camera name with underscores or a long RTSP URL has no break point."""
    assert any(
        selector in selectors and re.search(r"overflow-wrap\s*:\s*anywhere", body)
        for selectors, body in _css_rules(_warden_css())
    ), f"{selector} does not break long words"


def test_css_caps_the_zone_placeholder_at_the_screen_width() -> None:
    bodies = [
        body for selectors, body in _css_rules(_warden_css()) if ".zone-placeholder" in selectors
    ]
    assert any(re.search(r"max-width\s*:\s*100%", body) for body in bodies), (
        ".zone-placeholder is not capped at max-width: 100%"
    )


def test_zone_editor_link_opens_a_full_page(admin_client: TestClient) -> None:
    r = admin_client.get(f"/cameras/{CAMERA}/zones/editor")
    assert r.status_code == 200
    assert scan(r.text).has_viewport, "the zone editor link opens a bare fragment"
    assert "/static/js/alpine.min.js" in r.text
    assert "zoneEditor(" in r.text


def test_zone_editor_htmx_request_gets_the_fragment(admin_client: TestClient) -> None:
    r = admin_client.get(f"/cameras/{CAMERA}/zones/editor", headers={"HX-Request": "true"})
    assert r.status_code == 200
    assert "<html" not in r.text
    assert "zoneEditor(" in r.text


def test_zone_editor_placeholder_has_no_fixed_pixel_size(admin_client: TestClient) -> None:
    html = admin_client.get("/cameras/back/zones/editor").text
    assert 'class="zone-placeholder"' in html
    assert ":style=\"{ aspectRatio: frameWidth + ' / ' + frameHeight }\"" in html
    assert "+ 'px" not in html


@pytest.mark.parametrize("path", ADMIN_PAGES)
def test_admin_page_has_one_shared_page_header(admin_client: TestClient, path: str) -> None:
    """Every RW-2 page opens with partials/page_header.html (consistent layout, spec A.2)."""
    html = admin_client.get(path).text
    assert html.count('<header class="page-header">') == 1, path


def test_css_keeps_the_card_header_inside_the_card() -> None:
    """.camera-card has no padding, so Pico's negative article > header margins must go."""
    rules = _css_rules(_warden_css())
    bodies = [body for selectors, body in rules if ".camera-card > header" in selectors]
    assert any(
        all(re.search(rf"margin-{side}\s*:\s*0\s*;", body) for side in ("top", "right", "left"))
        for body in bodies
    ), ".camera-card > header still sticks out of the card"


def test_css_spaces_the_detail_page_actions() -> None:
    rules = _css_rules(_warden_css())
    bodies = [body for selectors, body in rules if "#camera-actions" in selectors]
    assert any(re.search(r"margin-bottom\s*:", body) for body in bodies)
