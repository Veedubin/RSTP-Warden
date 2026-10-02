# Stabilize Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every existing rtsp-warden feature actually work in a real `serve` run: config write-back, forms, login, status, preview, clips, retention, deployment bootstrap, and remove dead code and wrong docs.

**Architecture:** No new subsystems. Each task fixes one confirmed defect in an existing flow, adds the test that was missing, and commits. The two new modules are `db/bootstrap.py` (first-run admin creation) and `web/routes/_common.py` (shared route helpers). Status and preview move from "link to the MJPEG side-server" to "read the in-process `FrameHub` and `CameraRuntime` directly".

**Tech Stack:** Python 3.13, FastAPI + Starlette, pydantic v2, Jinja2 + htmx, SQLAlchemy + Alembic, pytest. `uv` only.

**Spec:** `docs/superpowers/specs/2026-10-02-detection-and-automation-design.md`, Appendix A.1 (this plan is sub-project 1 of 3).

## Global Constraints

- `uv` only: `uv run pytest`, `uv run ruff check src/ tests/`, `uv run ruff format --check src/ tests/`. The 20 pre-existing E501 errors are the accepted baseline; the gate is "no new ruff errors".
- All 780 existing tests keep passing after every task, minus the ones deleted with dead code in Task 12.
- Live segment recording stays `.ts`. Do not touch `build_ffmpeg_ingest_cmd` segment arguments.
- Secrets only in `.env` or `${ENV_VAR}` references. Never commit a credential; never print one in a commit message or test name.
- `config.yaml` stays authoritative. Write-back goes through `web/config_lock.py` only.
- Commit after every task with the message given in the task, ending with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Branch: `feat/detection-automation`.

## Review Focus

1. A JSON POST (the ONVIF PTZ routes read `request.json()`) with the token in the header must still reach the handler with its body intact after the CSRF middleware learns to read form bodies. Test in Task 3.
2. A camera password containing a literal `$` (for example `pa$$word`) must survive `${VAR}` substitution untouched; only the exact `${NAME}` form is replaced. Test in Task 1.
3. An htmx partial request whose session has expired must not get a login page swapped into a camera card; it must get `HX-Redirect`. Test in Task 4.
4. A camera with recording disabled but the MJPEG proxy enabled is "running" when its ingest runs, and a camera with nothing enabled is "idle", never "failed". Test in Task 11.
5. A camera with no `sub_url` but the default `record.sub.enabled: true` must start with only a main ingestor and no error. Test in Task 8.

---

### Task 1: Scrub committed credentials and add `${ENV_VAR}` substitution

**Files:**
- Modify: `examples/configs/config-Foscam-C1-V3.yaml`, `examples/configs/config-NC230-C1-V3-2Cams.yaml`, `examples/configs/config-TP-Link-NC230.yaml`
- Modify: `src/rtsp_warden/config.py:260-266` (`load_config`)
- Test: `tests/test_config_env.py`

**Interfaces:**
- Produces: `rtsp_warden.config.expand_env(obj: Any, env: Mapping[str, str] | None = None) -> Any`; `load_config` calls it before validation. Missing variable raises `SystemExit` naming the variable.

- [ ] **Step 1: Scrub the example configs**

Edit the three files in `examples/configs/`. Replace every `rtsp://<user>:<password>@` with `rtsp://${CAM_USER}:${CAM_PASS}@`. Leave hosts, ports and paths. Add this comment block at the top of each file:

```yaml
# Credentials come from environment variables (see .env):
#   CAM_USER=admin
#   CAM_PASS=your-camera-password
# rtsp-warden expands ${NAME} references in config.yaml at load time.
```

Verify no credential remains:

Run: `grep -rn 'rtsp://[^$]' examples/`
Expected: no output.

Rotation of the exposed camera passwords is the user's job and is noted in the final summary, not done here.

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_config_env.py
import pytest

from rtsp_warden.config import expand_env


def test_expand_env_replaces_braced_names():
    env = {"CAM_USER": "alice", "CAM_PASS": "s3cret"}
    data = {"cameras": [{"main_url": "rtsp://${CAM_USER}:${CAM_PASS}@10.0.0.5:554/x"}]}
    out = expand_env(data, env)
    assert out["cameras"][0]["main_url"] == "rtsp://alice:s3cret@10.0.0.5:554/x"


def test_expand_env_leaves_literal_dollars_alone():
    env = {"CAM_PASS": "pa$$word"}
    data = {"u": "rtsp://a:${CAM_PASS}@h/x", "v": "cost is $5 and $$"}
    out = expand_env(data, env)
    assert out["u"] == "rtsp://a:pa$$word@h/x"
    assert out["v"] == "cost is $5 and $$"


def test_expand_env_walks_lists_and_nested_dicts_and_keeps_non_strings():
    env = {"N": "front"}
    data = {"a": [{"name": "${N}", "port": 554, "flag": True}], "b": None}
    out = expand_env(data, env)
    assert out == {"a": [{"name": "front", "port": 554, "flag": True}], "b": None}


def test_expand_env_missing_variable_is_fatal():
    with pytest.raises(SystemExit) as ei:
        expand_env({"x": "${NOPE_MISSING}"}, {})
    assert "NOPE_MISSING" in str(ei.value)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_config_env.py -v`
Expected: FAIL with `ImportError: cannot import name 'expand_env'`.

- [ ] **Step 4: Implement `expand_env` and call it from `load_config`**

In `src/rtsp_warden/config.py`, add after the imports:

```python
import os
import re
from collections.abc import Mapping
from typing import Any

_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def expand_env(obj: Any, env: Mapping[str, str] | None = None) -> Any:
    """Replace ``${NAME}`` references in every string of a loaded YAML tree.

    Only the exact ``${NAME}`` form is replaced; a bare ``$`` or ``$NAME`` is
    left alone so passwords containing ``$`` survive. A reference to a
    variable that is not set is a fatal config error.
    """
    source = os.environ if env is None else env

    def _sub(m: re.Match[str]) -> str:
        name = m.group(1)
        if name not in source:
            raise SystemExit(
                f"Config references ${{{name}}} but the environment variable {name} is not set"
            )
        return source[name]

    if isinstance(obj, str):
        return _ENV_REF.sub(_sub, obj)
    if isinstance(obj, list):
        return [expand_env(v, source) for v in obj]
    if isinstance(obj, dict):
        return {k: expand_env(v, source) for k, v in obj.items()}
    return obj
```

Change `load_config`:

```python
def load_config(path: str | Path) -> AppConfig:
    p = Path(path)
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    raw = expand_env(raw)
    try:
        return AppConfig.model_validate(raw)
    except ValidationError as e:
        raise SystemExit(f"Config validation failed:\n{e}") from e
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_config_env.py tests/test_config.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add examples/configs src/rtsp_warden/config.py tests/test_config_env.py
git commit -m "fix: scrub example credentials and expand \${ENV} references in config

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: Wire `config_path` and `runtime` into app state

**Files:**
- Modify: `src/rtsp_warden/web/app.py:41-71`
- Modify: `src/rtsp_warden/web/server.py:35-46`
- Modify: `src/rtsp_warden/cli.py:455-502` (`serve`)
- Test: `tests/test_web_wiring.py`

**Interfaces:**
- Produces: `create_app(settings=None, cfg=None, runtime_provider=None, config_path: str | Path | None = None, runtime: object | None = None)`; sets `app.state.config_path` (str or None) and `app.state.runtime`.
- Produces: `WebUIServer(settings, runtime_provider, cfg=None, config_path=None, runtime=None)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_web_wiring.py
from pathlib import Path

from rtsp_warden.web.app import create_app
from rtsp_warden.web.config import WebSettings
from rtsp_warden.web.server import WebUIServer


def test_create_app_sets_config_path_and_runtime(tmp_path: Path):
    cfg_path = tmp_path / "config.yaml"
    sentinel = object()
    app = create_app(WebSettings(), cfg=None, config_path=cfg_path, runtime=sentinel)
    assert app.state.config_path == str(cfg_path)
    assert app.state.runtime is sentinel


def test_create_app_defaults_to_none():
    app = create_app(WebSettings())
    assert app.state.config_path is None
    assert app.state.runtime is None


def test_web_server_passes_wiring_through(tmp_path: Path):
    sentinel = object()
    server = WebUIServer(
        WebSettings(host="127.0.0.1", port=8099),
        runtime_provider=lambda: sentinel,
        cfg=None,
        config_path=tmp_path / "c.yaml",
        runtime=sentinel,
    )
    assert server.app.state.runtime is sentinel
    assert server.app.state.config_path == str(tmp_path / "c.yaml")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_web_wiring.py -v`
Expected: FAIL with `TypeError: create_app() got an unexpected keyword argument 'config_path'`.

- [ ] **Step 3: Implement**

`src/rtsp_warden/web/app.py`, replace the `create_app` signature and the state block:

```python
from pathlib import Path


def create_app(
    settings: WebSettings | None = None,
    cfg: AppConfig | None = None,
    runtime_provider: RuntimeProvider | None = None,
    config_path: str | Path | None = None,
    runtime: object | None = None,
) -> FastAPI:
    """Build and return a configured FastAPI application.

    config_path:
        Path of the YAML file the config was loaded from. Routes that edit
        camera settings write back to it. None means in-memory only.
    runtime:
        The live ``AppRuntime``. Routes that hot-reload detectors need it.
    """
    if settings is None:
        settings = WebSettings()

    app = FastAPI(
        title="rtsp-warden",
        version=__version__,
        docs_url=None,
        redoc_url=None,
    )

    app.state.cfg = cfg
    app.state.runtime_provider = runtime_provider or (lambda: None)
    app.state.config_path = str(config_path) if config_path is not None else None
    app.state.runtime = runtime
```

`src/rtsp_warden/web/server.py`:

```python
    def __init__(
        self,
        settings: WebSettings,
        runtime_provider: RuntimeProvider,
        cfg: AppConfig | None = None,
        config_path: str | Path | None = None,
        runtime: object | None = None,
    ) -> None:
        self._settings = settings
        self._runtime_provider = runtime_provider
        self._cfg = cfg
        self._app = create_app(
            settings,
            cfg=cfg,
            runtime_provider=runtime_provider,
            config_path=config_path,
            runtime=runtime,
        )
```

Add `from pathlib import Path` to the imports of `server.py`.

`src/rtsp_warden/cli.py` in `serve`, replace the `WebUIServer(...)` construction:

```python
        ws = WebUIServer(
            settings=web_settings,
            cfg=cfg,
            runtime_provider=lambda: rt,
            config_path=config,
            runtime=rt,
        )
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_web_wiring.py tests/test_web_zones.py tests/test_web_sensitivity.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/rtsp_warden/web/app.py src/rtsp_warden/web/server.py src/rtsp_warden/cli.py tests/test_web_wiring.py
git commit -m "fix: set app.state.config_path and app.state.runtime in serve

Config write-back and detector hot reload were unreachable outside tests.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: CSRF accepts form-body tokens; htmx sends the header globally

**Files:**
- Modify: `src/rtsp_warden/web/csrf.py:48-58`
- Modify: `src/rtsp_warden/web/static/js/warden.js`
- Modify: `src/rtsp_warden/web/templates/cameras/detail.html:99` (retention form), `cameras/detection_classes.html:14`, `cameras/zones.html:27,54`
- Test: `tests/test_csrf_forms.py`

**Interfaces:**
- Consumes: `tests/conftest.py` fixtures `db_with_user` (admin `admin` / `testpass123`).
- Produces: nothing new; middleware behavior change only.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_csrf_forms.py
import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from rtsp_warden.web.app import create_app
from rtsp_warden.web.config import WebSettings


@pytest.fixture
def app(db_with_user):
    app = create_app(WebSettings())

    @app.post("/echo-form")
    async def echo_form(request: Request):
        form = await request.form()
        return {"name": form.get("name")}

    @app.post("/echo-json")
    async def echo_json(request: Request):
        return await request.json()

    return app


def _csrf(client: TestClient) -> str:
    client.get("/login")
    return client.cookies.get("warden_csrf", "")


def test_form_body_token_is_accepted_and_body_reaches_handler(app):
    client = TestClient(app)
    token = _csrf(client)
    r = client.post("/echo-form", data={"csrf_token": token, "name": "bob"})
    assert r.status_code == 200
    assert r.json() == {"name": "bob"}


def test_form_without_token_is_rejected(app):
    client = TestClient(app)
    _csrf(client)
    r = client.post("/echo-form", data={"name": "bob"})
    assert r.status_code == 403


def test_form_with_wrong_token_is_rejected(app):
    client = TestClient(app)
    _csrf(client)
    r = client.post("/echo-form", data={"csrf_token": "nope", "name": "bob"})
    assert r.status_code == 403


def test_json_post_with_header_token_keeps_body(app):
    client = TestClient(app)
    token = _csrf(client)
    r = client.post("/echo-json", json={"pan": 0.5}, headers={"X-CSRF-Token": token})
    assert r.status_code == 200
    assert r.json() == {"pan": 0.5}


def test_multipart_form_token_is_accepted(app):
    client = TestClient(app)
    token = _csrf(client)
    r = client.post(
        "/echo-form",
        data={"csrf_token": token, "name": "multi"},
        files={"blob": ("b.txt", b"x")},
    )
    assert r.status_code == 200
    assert r.json() == {"name": "multi"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_csrf_forms.py -v`
Expected: `test_form_body_token_is_accepted_and_body_reaches_handler` and `test_multipart_form_token_is_accepted` FAIL with 403; the others PASS.

- [ ] **Step 3: Implement the middleware change**

In `src/rtsp_warden/web/csrf.py`, replace the validation block inside `dispatch`:

```python
        if request.method in MUTATING_METHODS:
            provided_token = request.headers.get(CSRF_HEADER_NAME)
            if not provided_token:
                provided_token = request.query_params.get(CSRF_FORM_FIELD)
            if not provided_token:
                provided_token = await _token_from_form(request)

            if not provided_token or provided_token != csrf_cookie:
                return JSONResponse(
                    status_code=403,
                    content={"detail": "CSRF token missing or invalid"},
                )
```

Add the helper at module level:

```python
_FORM_CONTENT_TYPES = ("application/x-www-form-urlencoded", "multipart/form-data")


async def _token_from_form(request: Request) -> str | None:
    """Read ``csrf_token`` from a form body, or None for non-form requests.

    Starlette's BaseHTTPMiddleware caches the body it reads, so the route
    handler can still call ``request.form()`` afterwards.
    """
    content_type = request.headers.get("content-type", "")
    if not content_type.startswith(_FORM_CONTENT_TYPES):
        return None
    try:
        form = await request.form()
    except Exception:
        return None
    value = form.get(CSRF_FORM_FIELD)
    return value if isinstance(value, str) else None
```

Update the module docstring's numbered list to add "3. The ``csrf_token`` form field matches the cookie (form-encoded or multipart bodies)."

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_csrf_forms.py -v`
Expected: all PASS. If `test_form_body_token_is_accepted_and_body_reaches_handler` hangs or returns `{"name": null}`, the installed Starlette predates body caching in `BaseHTTPMiddleware` (0.28.0); run `uv run python -c "import starlette; print(starlette.__version__)"` and report it, do not work around it.

- [ ] **Step 5: Add the global htmx header hook**

Replace the contents of `src/rtsp_warden/web/static/js/warden.js`:

```javascript
// warden.js -- shared browser behavior for the rtsp-warden web UI.

// Attach the CSRF token to every htmx request so templates do not need
// per-form hx-headers. The token is rendered into <meta name="csrf-token">.
document.addEventListener("htmx:configRequest", function (evt) {
  var meta = document.querySelector('meta[name="csrf-token"]');
  if (meta && meta.content) {
    evt.detail.headers["X-CSRF-Token"] = meta.content;
  }
});
```

- [ ] **Step 6: Add hidden token fields to the plain forms that lack one**

In `cameras/detail.html`, directly after `<form method="POST" action="/cameras/{{ camera.name }}/retention">` add:

```html
      <input type="hidden" name="csrf_token" value="{{ request.state.csrf_token }}">
```

Do the same directly after the opening `<form` tag of the forms at `cameras/detection_classes.html:14`, `cameras/zones.html:27` (the delete form) and `cameras/zones.html:54` (the reload form). `cameras/sensitivity.html` already has the field.

- [ ] **Step 7: Run the web suite**

Run: `uv run pytest tests/ -k "web or csrf or zones or sensitivity or classes" -q`
Expected: all PASS.

- [ ] **Step 8: Commit**

```bash
git add src/rtsp_warden/web/csrf.py src/rtsp_warden/web/static/js/warden.js src/rtsp_warden/web/templates tests/test_csrf_forms.py
git commit -m "fix: accept CSRF token from form bodies and send it on every htmx request

Plain-form settings pages were rejected with 403 in a real browser.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Login and logout as plain forms; HTML requests redirect to login

**Files:**
- Modify: `src/rtsp_warden/web/templates/login.html:15-18`, `base.html:42-45`
- Modify: `src/rtsp_warden/web/auth_depends.py`
- Modify: `src/rtsp_warden/web/app.py` (exception handler)
- Test: `tests/test_auth_redirect.py`

**Interfaces:**
- Produces: `rtsp_warden.web.auth_depends.LoginRequired(next_url: str)` exception; `require_user(request: Request, user=Depends(get_current_user))`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_auth_redirect.py
import pytest
from fastapi.testclient import TestClient

from rtsp_warden.web.app import create_app
from rtsp_warden.web.config import WebSettings


@pytest.fixture
def client(db_with_user) -> TestClient:
    return TestClient(create_app(WebSettings()))


def test_html_request_without_session_redirects_to_login(client):
    r = client.get("/cameras", headers={"Accept": "text/html"}, follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/login?next=/cameras"


def test_api_request_without_session_is_401_json(client):
    r = client.get("/cameras", headers={"Accept": "application/json"})
    assert r.status_code == 401
    assert r.json()["detail"] == "Not authenticated"


def test_htmx_request_without_session_gets_hx_redirect(client):
    r = client.get("/cameras/x/status", headers={"HX-Request": "true", "Accept": "text/html"})
    assert r.status_code == 401
    assert r.headers["HX-Redirect"] == "/login"


def test_login_form_has_no_htmx_attributes(client):
    html = client.get("/login").text
    assert "hx-post" not in html


def test_successful_login_redirects(client):
    client.get("/login")
    token = client.cookies.get("warden_csrf", "")
    r = client.post(
        "/login",
        data={"username": "admin", "password": "testpass123", "csrf_token": token},
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert r.headers["location"] == "/"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_auth_redirect.py -v`
Expected: the first, third and fourth tests FAIL.

- [ ] **Step 3: Implement**

`src/rtsp_warden/web/auth_depends.py`:

```python
"""FastAPI dependency injection helpers for authentication."""

from __future__ import annotations

from fastapi import Depends, HTTPException, Request, status

from ..auth import CurrentUser
from .auth_bridge import get_current_user_from_request


class LoginRequired(Exception):
    """Raised for browser requests with no session; handled by a redirect to /login."""

    def __init__(self, next_url: str) -> None:
        super().__init__(next_url)
        self.next_url = next_url


async def get_current_user(request: Request) -> CurrentUser | None:
    return get_current_user_from_request(request)


def _wants_html(request: Request) -> bool:
    accept = request.headers.get("accept", "")
    return "text/html" in accept


async def require_user(
    request: Request,
    user: CurrentUser | None = Depends(get_current_user),
) -> CurrentUser:
    """Require an authenticated user.

    Browser page loads are redirected to /login. htmx partial requests get a
    401 with an ``HX-Redirect`` header so htmx navigates the whole page.
    Everything else gets a plain 401.
    """
    if user is None:
        if request.headers.get("hx-request") == "true":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Not authenticated",
                headers={"HX-Redirect": "/login"},
            )
        if _wants_html(request):
            raise LoginRequired(next_url=request.url.path)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": 'Bearer realm="warden"'},
        )
    return user


async def require_admin(user: CurrentUser = Depends(require_user)) -> CurrentUser:
    if user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required",
        )
    return user
```

In `src/rtsp_warden/web/app.py`, after `install_security(app)`:

```python
    from urllib.parse import quote

    from fastapi.responses import RedirectResponse

    from .auth_depends import LoginRequired

    @app.exception_handler(LoginRequired)
    async def _login_required(request: object, exc: LoginRequired) -> RedirectResponse:
        return RedirectResponse(
            url=f"/login?next={quote(exc.next_url, safe='/')}", status_code=303
        )
```

`login.html`: change the form opening to `<form method="POST" action="/login">` (drop `hx-post` and `hx-headers`). `base.html`: change the logout form opening to `<form method="POST" action="/logout">` (drop `hx-post` and `hx-headers`). Both keep their hidden `csrf_token` inputs.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_auth_redirect.py tests/ -k "auth or login or web" -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/rtsp_warden/web/auth_depends.py src/rtsp_warden/web/app.py src/rtsp_warden/web/templates/login.html src/rtsp_warden/web/templates/base.html tests/test_auth_redirect.py
git commit -m "fix: redirect browser requests to login; make login/logout plain forms

The htmx login swapped the dashboard into the login card.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: `_persist_camera_field` writes only the target camera

**Files:**
- Modify: `src/rtsp_warden/web/routes/cameras.py:812-836` and its two call sites
- Test: `tests/test_persist_camera_field.py`

**Interfaces:**
- Produces: `_persist_camera_field(config_path: Path, camera_name: str, field_name: str, value: object) -> None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_persist_camera_field.py
from pathlib import Path

import yaml

from rtsp_warden.web.routes.cameras import _persist_camera_field


def test_persist_only_touches_named_camera(tmp_path: Path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "cameras": [
                    {"name": "a", "main_url": "rtsp://x/a", "sub_url": "rtsp://x/a2", "sensitivity": 10},
                    {"name": "b", "main_url": "rtsp://x/b", "sub_url": "rtsp://x/b2", "sensitivity": 20},
                ]
            }
        )
    )
    _persist_camera_field(cfg, "b", "sensitivity", 75.0)
    data = yaml.safe_load(cfg.read_text())
    assert data["cameras"][0]["sensitivity"] == 10
    assert data["cameras"][1]["sensitivity"] == 75.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_persist_camera_field.py -v`
Expected: FAIL (`TypeError` on the new signature, or camera `a` rewritten to 75.0 once the signature matches).

- [ ] **Step 3: Implement**

Replace the function:

```python
def _persist_camera_field(
    config_path: Path, camera_name: str, field_name: str, value: object
) -> None:
    """Persist one camera-level field for one camera to config.yaml."""
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    cameras_data = data.get("cameras", [])
    for cam_dict in cameras_data:
        if cam_dict.get("name") == camera_name:
            cam_dict[field_name] = value
            break
    data["cameras"] = cameras_data
    _locked_write_yaml(config_path, data)
```

Find the call sites:

Run: `grep -n "_persist_camera_field(" src/rtsp_warden/web/routes/cameras.py`

Each call currently looks like `_persist_camera_field(config_path, cfg, "sensitivity", <value>)` or `_persist_camera_field(config_path, cfg, "detect_classes", <value>)`. Change each to pass the camera's name instead of `cfg`: `_persist_camera_field(config_path, name, "sensitivity", <value>)` where `name` is the route's path parameter.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_persist_camera_field.py tests/test_web_sensitivity.py tests/test_web_classes.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/rtsp_warden/web/routes/cameras.py tests/test_persist_camera_field.py
git commit -m "fix: persist sensitivity/classes for the edited camera only

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: Clip segment regex matches recorder filenames

**Files:**
- Modify: `src/rtsp_warden/clips.py:23-24, 70-73`
- Test: `tests/test_clips_segments.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_clips_segments.py
from datetime import datetime, timezone
from pathlib import Path

from rtsp_warden.clips import ClipGenerator


def test_find_segments_matches_recorder_naming(tmp_path: Path):
    seg_dir = tmp_path / "foscam_c1" / "main"
    seg_dir.mkdir(parents=True)
    (seg_dir / "foscam_c1_main_20261002_025541.ts").write_bytes(b"")
    (seg_dir / "foscam_c1_main_20261002_025641.ts").write_bytes(b"")
    (seg_dir / "notes.txt").write_bytes(b"")

    gen = ClipGenerator(recordings_dir=tmp_path, clips_dir=tmp_path / "clips")
    start = datetime(2026, 10, 2, 2, 55, 0, tzinfo=timezone.utc)
    end = datetime(2026, 10, 2, 2, 56, 0, tzinfo=timezone.utc)
    found = gen.find_segments("foscam_c1", "main", start, end, segment_duration=60.0)
    assert [p.name for p in found] == ["foscam_c1_main_20261002_025541.ts"]
```

Check the constructor name and arguments first:

Run: `grep -n "def __init__" -A 8 src/rtsp_warden/clips.py`

Adjust the `ClipGenerator(...)` call in the test to the real keyword names if they differ from `recordings_dir` and `clips_dir`.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_clips_segments.py -v`
Expected: FAIL with `assert [] == [...]`.

- [ ] **Step 3: Implement**

```python
# Recorder segments are named {camera}_{stream}_%Y%m%d_%H%M%S.ts (recorder.py).
# Accept a bare timestamp too, for files produced by older builds.
_SEGMENT_RE = re_compile(r"^(?:.+_)?(\d{8}_\d{6})\.ts$")
```

Update the `find_segments` docstring line "Segments are named %Y%m%d_%H%M%S.ts" to "Segments are named {camera}_{stream}_%Y%m%d_%H%M%S.ts".

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_clips_segments.py tests/test_clips.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/rtsp_warden/clips.py tests/test_clips_segments.py
git commit -m "fix: clip generator finds recorder-named segments

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: Honor `record.retention` with a deprecation warning; fix the templates

**Files:**
- Modify: `src/rtsp_warden/retention_resolver.py`
- Modify: `src/rtsp_warden/cli.py:53-58` (`SAMPLE_CONFIG_YAML`), `examples/config.yaml`, `examples/configs/*.yaml`
- Test: `tests/test_retention_resolver_legacy.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_retention_resolver_legacy.py
import logging

from rtsp_warden.config import CameraConfig, RetentionConfig
from rtsp_warden.retention_resolver import resolve_retention


def _cam(**kw) -> CameraConfig:
    return CameraConfig(name="c", main_url="rtsp://h/m", sub_url="rtsp://h/s", **kw)


def test_camera_override_wins():
    cam = _cam(retention=RetentionConfig(max_days=3), record={"retention": {"max_days": 9}})
    assert resolve_retention(cam, RetentionConfig(max_days=30)).max_days == 3


def test_legacy_record_retention_is_honored_with_warning(caplog):
    cam = _cam(record={"retention": {"max_days": 9}})
    with caplog.at_level(logging.WARNING):
        out = resolve_retention(cam, RetentionConfig(max_days=30))
    assert out.max_days == 9
    assert "record.retention" in caplog.text
    assert "cameras[].retention" in caplog.text


def test_global_used_when_nothing_set():
    cam = _cam()
    assert resolve_retention(cam, RetentionConfig(max_days=30)).max_days == 30
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_retention_resolver_legacy.py -v`
Expected: `test_legacy_record_retention_is_honored_with_warning` FAILS with `assert 30 == 9`.

- [ ] **Step 3: Implement**

Replace the body of `resolve_retention` in `src/rtsp_warden/retention_resolver.py`:

```python
import logging

log = logging.getLogger(__name__)


def resolve_retention(camera: CameraConfig, global_cfg: RetentionConfig) -> RetentionConfig:
    """Return the effective retention config for a camera.

    Resolution order:
        1. camera.retention           (per-camera override)
        2. camera.record.retention    (deprecated location; honored with a warning)
        3. global_cfg                 (app-wide fallback)

    A deep copy is always returned so RetentionManager can mutate it.
    """
    if camera.retention is not None:
        return camera.retention.model_copy(deep=True)
    if "retention" in camera.record.model_fields_set:
        log.warning(
            "camera %r sets record.retention, which is deprecated; "
            "move it to cameras[].retention",
            camera.name,
        )
        return camera.record.retention.model_copy(deep=True)
    return global_cfg.model_copy(deep=True)
```

Keep the existing `from .config import CameraConfig, RetentionConfig` import.

- [ ] **Step 4: Move retention in the templates**

In `src/rtsp_warden/cli.py` `SAMPLE_CONFIG_YAML`, remove the `retention:` block nested under `record:` (lines 53-57) and add at camera level, after the `proxy:` block with two-space camera indentation:

```yaml
    retention:
      max_days: 7
      max_gb: 50
      keep_last_n: 10
      cleanup_interval_seconds: 300
```

Make the same move in `examples/config.yaml` and in each file under `examples/configs/`. While there, change every `container: mkv` and `container: mp4` to `container: ts` and delete the comments that recommend MKV or MP4.

Verify: `uv run rtsp-warden init-config --out /tmp/claude-warden-sample.yaml --force && uv run python -c "from rtsp_warden.config import load_config; c=load_config('/tmp/claude-warden-sample.yaml'); print(c.cameras[0].retention)"`
Expected: prints a `RetentionConfig` with `max_days=7`, not `None`.

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_retention_resolver_legacy.py tests/ -k retention -q`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/rtsp_warden/retention_resolver.py src/rtsp_warden/cli.py examples tests/test_retention_resolver_legacy.py
git commit -m "fix: honor record.retention with a deprecation warning; fix sample configs

Every shipped template put retention where nothing read it.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: `sub_url` optional; everything falls back to main

**Files:**
- Modify: `src/rtsp_warden/config.py:140-171` (`CameraConfig`)
- Modify: `src/rtsp_warden/recorder.py:87-148` (`_build_ingestor`)
- Modify: `src/rtsp_warden/web/services/cameras.py:36-37`
- Modify: `src/rtsp_warden/web/templates/cameras/detail.html` (Sub URL row)
- Test: `tests/test_sub_optional.py`

**Interfaces:**
- Produces: `CameraConfig.sub_url: str | None = None`. When None and `proxy.stream == "sub"`, the validator sets `proxy.stream = "main"`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_sub_optional.py
from rtsp_warden.config import CameraConfig, RuntimeConfig
from rtsp_warden.recorder import CameraRecorder
from rtsp_warden.web.services.cameras import list_cameras
from rtsp_warden.config import AppConfig


def test_sub_url_defaults_to_none_and_proxy_falls_back_to_main():
    cam = CameraConfig(name="c", main_url="rtsp://h/m", proxy={"stream": "sub"})
    assert cam.sub_url is None
    assert cam.proxy.stream == "main"


def test_recorder_builds_only_main_when_no_sub(tmp_path):
    cam = CameraConfig(
        name="c",
        main_url="rtsp://h/m",
        record={"enabled": True, "output_dir": str(tmp_path)},
    )
    rec = CameraRecorder(camera=cam, runtime=RuntimeConfig())
    assert rec.main is not None
    assert rec.sub is None
    assert rec.has_any()


def test_list_cameras_handles_missing_sub():
    cfg = AppConfig(cameras=[CameraConfig(name="c", main_url="rtsp://u:p@h/m")])
    row = list_cameras(cfg)[0]
    assert row["sub_url_redacted"] is None
    assert row["main_url_redacted"].startswith("rtsp://***")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_sub_optional.py -v`
Expected: FAIL with a pydantic `ValidationError: sub_url Field required`.

- [ ] **Step 3: Implement**

`src/rtsp_warden/config.py`, in `CameraConfig`:

```python
    name: str
    main_url: str
    sub_url: str | None = None  # optional; every sub-stream consumer falls back to main
```

Add a model validator at the end of the class (import `model_validator` is already there):

```python
    @model_validator(mode="after")
    def _fallback_proxy_stream(self) -> CameraConfig:
        if self.sub_url is None and self.proxy.stream == "sub":
            self.proxy.stream = "main"
        return self
```

`src/rtsp_warden/recorder.py`, at the top of `_build_ingestor`:

```python
        cam = self.camera
        if stream_name == "sub" and cam.sub_url is None:
            return None
```

And change the upstream line to:

```python
        upstream = cam.main_url if stream_name == "main" else cam.sub_url
        assert upstream is not None
```

`src/rtsp_warden/web/services/cameras.py`:

```python
                "sub_url_redacted": redact_rtsp_url(cam.sub_url) if cam.sub_url else None,
```

`cameras/detail.html`, the Sub URL table row: show `{{ camera.sub_url_redacted or 'not configured (using main)' }}`. Find it with `grep -n "sub_url_redacted" src/rtsp_warden/web/templates -r` and fix every occurrence the same way.

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest -q`
Expected: all PASS. If a test fails on `redact_rtsp_url(None)` elsewhere, apply the same `if cam.sub_url else None` guard at that call site.

- [ ] **Step 5: Commit**

```bash
git add src/rtsp_warden/config.py src/rtsp_warden/recorder.py src/rtsp_warden/web/services/cameras.py src/rtsp_warden/web/templates tests/test_sub_optional.py
git commit -m "feat: make sub_url optional; proxy and recording fall back to main

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 9: `serve` bootstraps the database and admin; env drives the web bind

**Files:**
- Create: `src/rtsp_warden/db/bootstrap.py`
- Modify: `src/rtsp_warden/web/config.py` (host default), `src/rtsp_warden/cli.py` (`serve` options and body, `install` strings)
- Modify: `packaging/systemd/rtsp-warden.service` (remove `ExecReload`)
- Test: `tests/test_bootstrap.py`, `tests/test_web_settings_env.py`

**Interfaces:**
- Produces: `rtsp_warden.db.bootstrap.ensure_admin_user(env: Mapping[str, str] | None = None) -> tuple[str, str] | None`. Returns `(username, password)` when it created one, else None.
- Produces: `rtsp_warden.cli._resolve_web_settings(host: str | None, port: int | None) -> WebSettings`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_bootstrap.py
from rtsp_warden.db.bootstrap import ensure_admin_user
from rtsp_warden.db.schema import list_users


def test_creates_admin_from_env_when_no_users(clean_db):
    created = ensure_admin_user({"WARDEN_ADMIN_USERNAME": "ops", "WARDEN_ADMIN_PASSWORD": "pw12345678"})
    assert created == ("ops", "pw12345678")
    users = list_users()
    assert [u.username for u in users] == ["ops"]
    assert users[0].is_admin


def test_generates_password_when_env_absent(clean_db):
    created = ensure_admin_user({})
    assert created is not None
    username, password = created
    assert username == "admin"
    assert len(password) >= 12


def test_noop_when_users_exist(db_with_user):
    assert ensure_admin_user({}) is None
```

```python
# tests/test_web_settings_env.py
from rtsp_warden.cli import _resolve_web_settings
from rtsp_warden.web.config import WebSettings


def test_default_bind_is_loopback(monkeypatch):
    monkeypatch.delenv("WARDEN_WEB_HOST", raising=False)
    monkeypatch.delenv("WARDEN_WEB_PORT", raising=False)
    s = _resolve_web_settings(None, None)
    assert (s.host, s.port) == ("127.0.0.1", 8080)


def test_env_overrides_default(monkeypatch):
    monkeypatch.setenv("WARDEN_WEB_HOST", "0.0.0.0")
    monkeypatch.setenv("WARDEN_WEB_PORT", "9090")
    s = _resolve_web_settings(None, None)
    assert (s.host, s.port) == ("0.0.0.0", 9090)


def test_cli_overrides_env(monkeypatch):
    monkeypatch.setenv("WARDEN_WEB_HOST", "0.0.0.0")
    s = _resolve_web_settings("10.0.0.2", 8081)
    assert (s.host, s.port) == ("10.0.0.2", 8081)


def test_websettings_default_host():
    assert WebSettings(_env_file=None).host == "127.0.0.1"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_bootstrap.py tests/test_web_settings_env.py -v`
Expected: FAIL with `ModuleNotFoundError` / `ImportError`.

- [ ] **Step 3: Implement the bootstrap module**

```python
# src/rtsp_warden/db/bootstrap.py
"""First-run bootstrap used by ``serve``: schema, then an admin user if none exists."""

from __future__ import annotations

import logging
import os
import secrets
from collections.abc import Mapping

from ..auth import hash_password
from .schema import create_admin_user, ensure_schema, list_users

log = logging.getLogger(__name__)


def _generate_password() -> str:
    return secrets.token_urlsafe(12)


def ensure_admin_user(env: Mapping[str, str] | None = None) -> tuple[str, str] | None:
    """Create the first admin user when the users table is empty.

    Username and password come from WARDEN_ADMIN_USERNAME / WARDEN_ADMIN_PASSWORD
    when set; otherwise ``admin`` with a generated password. Returns the
    credentials that were created, or None when users already existed.
    """
    source = os.environ if env is None else env
    if list_users():
        return None
    username = source.get("WARDEN_ADMIN_USERNAME") or "admin"
    password = source.get("WARDEN_ADMIN_PASSWORD") or _generate_password()
    create_admin_user(username, hash_password(password))
    return username, password


def bootstrap_database() -> tuple[str, str] | None:
    """Ensure the schema exists and an admin user exists. Logs created credentials once."""
    ensure_schema()
    created = ensure_admin_user()
    if created is not None:
        username, password = created
        log.warning(
            "No users existed; created admin %r with password %r. "
            "Log in and change it, or set WARDEN_ADMIN_PASSWORD before first start.",
            username,
            password,
        )
    return created
```

Check `hash_password` lives in `rtsp_warden.auth` (`grep -n "def hash_password" src/rtsp_warden/auth.py`); adjust the import if it is elsewhere.

- [ ] **Step 4: Implement the web settings resolution and call bootstrap from `serve`**

`src/rtsp_warden/web/config.py`: change `host: str = "0.0.0.0"` to `host: str = "127.0.0.1"`.

`src/rtsp_warden/cli.py`: add near `_port_is_free`:

```python
def _resolve_web_settings(host: str | None, port: int | None) -> WebSettings:
    """CLI flags override WARDEN_WEB_HOST / WARDEN_WEB_PORT, which override defaults."""
    settings = WebSettings()
    if host:
        settings.host = host
    if port:
        settings.port = port
    return settings
```

In `serve`, change the two options to default to `None`:

```python
    web_host: str | None = typer.Option(
        None, "--web-host", help="Web UI bind host (default: $WARDEN_WEB_HOST or 127.0.0.1)"
    ),
    web_port: int | None = typer.Option(
        None, "--web-port", help="Web UI port (default: $WARDEN_WEB_PORT or 8080)"
    ),
```

In the body, after `_load_dotenv()` and `setup_logging(...)`, add:

```python
    from .db.bootstrap import bootstrap_database

    bootstrap_database()
```

Replace the web block:

```python
    ws: WebUIServer | None = None
    if web:
        web_settings = _resolve_web_settings(web_host, web_port)
        if not _port_is_free(web_settings.host, web_settings.port):
            raise typer.Exit(code=2)
        ws = WebUIServer(
            settings=web_settings,
            cfg=cfg,
            runtime_provider=lambda: rt,
            config_path=config,
            runtime=rt,
        )
```

In `install`: change `"Schema created (7 tables)"` to `"Schema created"` (both in `cli.py` and the `log.info` in `install.py`), and change `rtsp-warden run -c config.yaml` in the "Next steps" to `rtsp-warden serve -c config.yaml`.

`packaging/systemd/rtsp-warden.service`: delete the `ExecReload=` line (HUP has no handler and kills the process).

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_bootstrap.py tests/test_web_settings_env.py tests/test_admin.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/rtsp_warden/db/bootstrap.py src/rtsp_warden/web/config.py src/rtsp_warden/cli.py src/rtsp_warden/install.py packaging/systemd/rtsp-warden.service tests/test_bootstrap.py tests/test_web_settings_env.py
git commit -m "feat: serve bootstraps schema and first admin; env drives web bind

Fresh Docker and systemd deployments had no login and bound to loopback.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 10: Alert notifiers are built at construction; test button is a POST

**Files:**
- Modify: `src/rtsp_warden/alerts/manager.py:30-50`
- Modify: `src/rtsp_warden/web/routes/alerts.py:52`
- Test: `tests/test_alerts_ready.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_alerts_ready.py
import pytest
from fastapi.testclient import TestClient

from rtsp_warden.alerts.manager import AlertManager
from rtsp_warden.config import AlertsConfig, AppConfig
from rtsp_warden.web.app import create_app
from rtsp_warden.web.config import WebSettings


def _alerts_cfg() -> AlertsConfig:
    return AlertsConfig.model_validate(
        {"enabled": True, "notifiers": [{"name": "hook", "type": "webhook", "url": "http://127.0.0.1:9/x"}]}
    )


def test_manager_has_notifiers_without_start():
    mgr = AlertManager(_alerts_cfg())
    assert [n.name for n in mgr.notifiers] == ["hook"]


@pytest.fixture
def admin_client(db_with_user, monkeypatch) -> TestClient:
    cfg = AppConfig(cameras=[], alerts=_alerts_cfg())
    app = create_app(WebSettings(), cfg=cfg)

    async def fake_test(self, name):
        return {"success": True, "notifier": name}

    monkeypatch.setattr(AlertManager, "test_notifier", fake_test)
    client = TestClient(app)
    client.get("/login")
    token = client.cookies.get("warden_csrf", "")
    client.post("/login", data={"username": "admin", "password": "testpass123", "csrf_token": token})
    return client


def test_test_button_is_post(admin_client):
    token = admin_client.cookies.get("warden_csrf", "")
    r = admin_client.post("/alerts/hook/test", headers={"X-CSRF-Token": token})
    assert r.status_code == 200
    assert r.json()["success"] is True
    assert admin_client.get("/alerts/hook/test").status_code == 405
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_alerts_ready.py -v`
Expected: FAIL (`AttributeError: notifiers` and 404/405 mismatch).

- [ ] **Step 3: Implement**

`src/rtsp_warden/alerts/manager.py`: in `__init__`, after the existing attribute setup, call `self._build()`; add:

```python
    def _build(self) -> None:
        self._notifiers = []
        self._spec_map = {}
        for spec in self._cfg.notifiers:
            notifier = build_notifier(spec)
            self._notifiers.append(notifier)
            self._spec_map[spec.name] = spec

    @property
    def notifiers(self) -> list[Any]:
        return list(self._notifiers)

    async def start(self) -> None:
        """Kept for API compatibility; notifiers are built in __init__."""
        logger.info("AlertManager ready with %d notifier(s)", len(self._notifiers))
```

Remove the old body of `start()`.

`src/rtsp_warden/web/routes/alerts.py`: change `@router.get("/{name}/test", response_class=JSONResponse)` to `@router.post("/{name}/test", response_class=JSONResponse)`. The template `alerts/list.html` already uses `hx-post`.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_alerts_ready.py tests/test_alerts.py tests/ -k alert -q`
Expected: all PASS. If `tests/test_alerts.py` asserts that notifiers are empty before `start()`, update that assertion to the new behavior.

- [ ] **Step 5: Commit**

```bash
git add src/rtsp_warden/alerts/manager.py src/rtsp_warden/web/routes/alerts.py tests/test_alerts_ready.py tests/test_alerts.py
git commit -m "fix: build alert notifiers at construction; test route is POST

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 11: Live camera status in the UI and same-origin preview routes

**Files:**
- Modify: `src/rtsp_warden/app.py:28-37` (`CameraRuntime` fields), `:190-210` (supervisor sets them)
- Modify: `src/rtsp_warden/web/services/cameras.py`
- Modify: `src/rtsp_warden/web/routes/cameras.py` (list, detail, status partial, two new routes), `src/rtsp_warden/web/routes/dashboard.py`, `src/rtsp_warden/web/routes/zones.py:118-124`
- Modify: `src/rtsp_warden/web/templates/partials/camera_card.html`, `cameras/detail.html`
- Modify: `src/rtsp_warden/web/static/css/warden.css`
- Test: `tests/test_camera_status.py`, `tests/test_preview_routes.py`

**Interfaces:**
- Produces: `CameraRuntime.next_restart_at: float = 0.0`, `CameraRuntime.last_error: str = ""`.
- Produces: `web.services.cameras.live_status(cam_rt, now: float | None = None) -> dict` with keys `status` (`running|restarting|failed|idle`), `restart_in` (int seconds or None), `last_error` (str), `last_frame_age` (float or None).
- Produces: `list_cameras(cfg, rt=None)` merges `live_status` when `rt` has a matching `CameraRuntime`.
- Produces: `web.services.preview.mjpeg_frames(hub, stop_after: int | None = None) -> Iterator[bytes]`.
- Produces: routes `GET /cameras/{name}/live.mjpeg` and `GET /cameras/{name}/snapshot.jpg`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_camera_status.py
from types import SimpleNamespace

from rtsp_warden.proxy.mjpeg import FrameHub
from rtsp_warden.web.services.cameras import live_status


class _Proc:
    def __init__(self, running: bool, tail=None):
        self._running = running
        self._tail = tail or []

    def poll(self):
        return None if self._running else 1

    def is_running(self):
        return self._running

    def stderr_tail(self):
        return self._tail


def _rt(procs, next_restart_at=0.0, last_error="", hub=None):
    recorder = SimpleNamespace(processes=lambda: [SimpleNamespace(proc=p) for p in procs])
    return SimpleNamespace(
        recorder=recorder, next_restart_at=next_restart_at, last_error=last_error, hub=hub
    )


def test_running():
    assert live_status(_rt([_Proc(True)]), now=100.0)["status"] == "running"


def test_idle_when_no_processes():
    assert live_status(_rt([]), now=100.0)["status"] == "idle"


def test_restarting_with_countdown_and_error():
    st = live_status(_rt([_Proc(False)], next_restart_at=107.4, last_error="boom"), now=100.0)
    assert st["status"] == "restarting"
    assert st["restart_in"] == 8
    assert st["last_error"] == "boom"


def test_failed_when_dead_and_no_restart_scheduled():
    assert live_status(_rt([_Proc(False)]), now=100.0)["status"] == "failed"


def test_last_frame_age_from_hub():
    hub = FrameHub()
    hub.update(b"\xff\xd8\xff\xd9")
    st = live_status(_rt([_Proc(True)], hub=hub))
    assert st["last_frame_age"] is not None and st["last_frame_age"] < 5
```

```python
# tests/test_preview_routes.py
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from rtsp_warden.config import AppConfig, CameraConfig
from rtsp_warden.proxy.mjpeg import FrameHub
from rtsp_warden.web.app import create_app
from rtsp_warden.web.config import WebSettings
from rtsp_warden.web.services.preview import mjpeg_frames

JPEG = b"\xff\xd8\xff\xd9"


def _login(client: TestClient) -> None:
    client.get("/login")
    token = client.cookies.get("warden_csrf", "")
    client.post("/login", data={"username": "admin", "password": "testpass123", "csrf_token": token})


@pytest.fixture
def hub() -> FrameHub:
    h = FrameHub()
    h.update(JPEG)
    return h


@pytest.fixture
def client(db_with_user, hub) -> TestClient:
    cam = CameraConfig(name="cam", main_url="rtsp://u:p@h/m")
    cfg = AppConfig(cameras=[cam])
    runtime = SimpleNamespace(cameras=[SimpleNamespace(camera=cam, hub=hub)])
    app = create_app(WebSettings(), cfg=cfg, runtime_provider=lambda: runtime, runtime=runtime)
    c = TestClient(app)
    _login(c)
    return c


def test_snapshot_route_serves_latest_frame(client):
    r = client.get("/cameras/cam/snapshot.jpg")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"
    assert r.content == JPEG


def test_snapshot_503_without_runtime(db_with_user):
    cfg = AppConfig(cameras=[CameraConfig(name="cam", main_url="rtsp://h/m")])
    c = TestClient(create_app(WebSettings(), cfg=cfg))
    _login(c)
    assert c.get("/cameras/cam/snapshot.jpg").status_code == 503


def test_mjpeg_frames_generator_emits_multipart_parts(hub):
    parts = list(mjpeg_frames(hub, stop_after=1))
    assert parts[0].startswith(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: 4\r\n\r\n")
    assert parts[0].endswith(JPEG + b"\r\n")


def test_detail_page_uses_same_origin_urls(client):
    html = client.get("/cameras/cam").text
    assert "/cameras/cam/live.mjpeg" in html
    assert "127.0.0.1:9001" not in html
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_camera_status.py tests/test_preview_routes.py -v`
Expected: FAIL with `ImportError` for `live_status` and `rtsp_warden.web.services.preview`.

- [ ] **Step 3: Record restart state on `CameraRuntime`**

`src/rtsp_warden/app.py`, add two fields at the end of `CameraRuntime`:

```python
    # Set by the supervisor so the web UI can show restart countdown and last error.
    next_restart_at: float = 0.0
    last_error: str = ""
```

Add a helper above `AppRuntime`:

```python
def _last_stderr_line(procs: list) -> str:
    for sp in procs:
        proc = getattr(sp, "proc", None)
        if proc is None:
            continue
        try:
            tail = [ln for ln in proc.stderr_tail() if ln.strip()]
        except Exception:
            continue
        if tail:
            return tail[-1][:300]
    return ""
```

In `run_forever`, inside the ingest supervision block:

- Where `all_running` is true, after `next_rec_restart[key] = 0.0` add `rt.next_restart_at = 0.0`.
- Where the restart is scheduled (`if sched <= 0.0:`), after `next_rec_restart[key] = now + delay` add:

```python
                            rt.next_restart_at = now + delay
                            rt.last_error = _last_stderr_line(procs)
```

- Where the restart happens (`elif now >= sched:`), after `next_rec_restart[key] = 0.0` add `rt.next_restart_at = 0.0`.

- [ ] **Step 4: Implement `live_status` and merge it in `list_cameras`**

Append to `src/rtsp_warden/web/services/cameras.py`:

```python
import time


def live_status(cam_rt: Any, now: float | None = None) -> dict[str, Any]:
    """Summarize a CameraRuntime for display.

    status: running | restarting | failed | idle
    """
    now = time.time() if now is None else now
    procs = list(cam_rt.recorder.processes())
    last_frame_age: float | None = None
    hub = getattr(cam_rt, "hub", None)
    if hub is not None:
        try:
            _jpeg, _fid, ts = hub.snapshot()
            if ts:
                last_frame_age = max(0.0, now - float(ts))
        except Exception:
            last_frame_age = None

    if not procs:
        status = "idle"
    else:
        any_dead = any(sp.proc is None or sp.proc.poll() is not None for sp in procs)
        if not any_dead:
            status = "running"
        elif getattr(cam_rt, "next_restart_at", 0.0) > now:
            status = "restarting"
        else:
            status = "failed"

    restart_in: int | None = None
    if status == "restarting":
        restart_in = int(round(cam_rt.next_restart_at - now))

    return {
        "status": status,
        "restart_in": restart_in,
        "last_error": getattr(cam_rt, "last_error", "") or "",
        "last_frame_age": last_frame_age,
    }


def _find_runtime(rt: Any, name: str) -> Any | None:
    for cam_rt in getattr(rt, "cameras", []) or []:
        if cam_rt.camera.name == name:
            return cam_rt
    return None
```

Change `list_cameras` to accept the runtime and merge:

```python
def list_cameras(cfg: AppConfig, rt: Any = None) -> list[dict[str, Any]]:
    cameras: list[dict[str, Any]] = []
    for cam in cfg.cameras:
        row: dict[str, Any] = {
            "name": cam.name,
            "enabled": True,
            "record_enabled": cam.record.enabled,
            "proxy_mode": cam.proxy.mode,
            "proxy_port": cam.proxy.port,
            "has_proxy": cam.proxy.enabled,
            "main_url_redacted": redact_rtsp_url(cam.main_url),
            "sub_url_redacted": redact_rtsp_url(cam.sub_url) if cam.sub_url else None,
            "status": "unknown",
            "restart_in": None,
            "last_error": "",
            "last_frame_age": None,
            "stream": cam.proxy.stream,
            "bind_host": cam.proxy.bind_host,
        }
        cam_rt = _find_runtime(rt, cam.name) if rt is not None else None
        if cam_rt is not None:
            row.update(live_status(cam_rt))
        cameras.append(row)
    return cameras


def get_camera_by_name(cfg: AppConfig, name: str, rt: Any = None) -> dict[str, Any] | None:
    for cam_dict in list_cameras(cfg, rt):
        if cam_dict["name"] == name:
            return cam_dict
    return None
```

- [ ] **Step 5: Implement the preview service and routes**

Create `src/rtsp_warden/web/services/preview.py`:

```python
"""Same-origin MJPEG and snapshot helpers backed by the in-process FrameHub."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

BOUNDARY = "frame"
MJPEG_CONTENT_TYPE = f"multipart/x-mixed-replace; boundary={BOUNDARY}"


def mjpeg_frames(hub: Any, stop_after: int | None = None) -> Iterator[bytes]:
    """Yield multipart MJPEG parts as new frames arrive on *hub*.

    stop_after bounds the number of frames for tests; None streams forever.
    """
    last_id = 0
    sent = 0
    while stop_after is None or sent < stop_after:
        jpeg, fid, _ts = hub.wait_for_new(last_id, timeout=2.0)
        if fid == last_id or not jpeg:
            continue
        last_id = fid
        sent += 1
        yield (
            f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpeg)}\r\n\r\n".encode(
                "ascii"
            )
            + jpeg
            + b"\r\n"
        )


def find_hub(runtime: Any, camera_name: str) -> Any | None:
    for cam_rt in getattr(runtime, "cameras", []) or []:
        if cam_rt.camera.name == camera_name:
            return getattr(cam_rt, "hub", None)
    return None
```

In `src/rtsp_warden/web/routes/cameras.py` add the routes (imports: `from fastapi.responses import Response, StreamingResponse` and `from ..services.preview import MJPEG_CONTENT_TYPE, find_hub, mjpeg_frames`):

```python
@router.get("/{name}/snapshot.jpg")
async def camera_snapshot(request: Request, name: str, user=Depends(require_user)) -> Response:
    """Latest JPEG frame from the in-process hub."""
    hub = find_hub(request.app.state.runtime_provider(), name)
    if hub is None:
        raise HTTPException(status_code=503, detail="No live preview for this camera")
    jpeg, _fid, _ts = hub.snapshot()
    if not jpeg:
        raise HTTPException(status_code=503, detail="No frame yet")
    return Response(content=jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@router.get("/{name}/live.mjpeg")
async def camera_live(request: Request, name: str, user=Depends(require_user)) -> StreamingResponse:
    """Same-origin MJPEG stream; replaces links to the side-server on 127.0.0.1."""
    hub = find_hub(request.app.state.runtime_provider(), name)
    if hub is None:
        raise HTTPException(status_code=503, detail="No live preview for this camera")
    return StreamingResponse(
        mjpeg_frames(hub),
        media_type=MJPEG_CONTENT_TYPE,
        headers={"Cache-Control": "no-store"},
    )
```

Register these before the generic `@router.get("/{name}")` detail route so FastAPI does not treat `snapshot.jpg` as a camera name.

In `camera_detail`, replace the MJPEG URL block:

```python
    mjpeg_url = ""
    snapshot_url = ""
    if cam["has_proxy"] and cam["proxy_mode"] == "mjpeg":
        mjpeg_url = f"/cameras/{name}/live.mjpeg"
        snapshot_url = f"/cameras/{name}/snapshot.jpg"
```

Pass the runtime wherever cameras are listed: in `cameras_list`, `camera_detail` and the `/{name}/status` partial route, call `list_cameras(cfg, request.app.state.runtime_provider())` / `get_camera_by_name(cfg, name, request.app.state.runtime_provider())`. In `web/routes/dashboard.py` do the same for its camera list. In `web/routes/zones.py:118-124` replace the `http://127.0.0.1:{port}/snapshot.jpg` background URL with `f"/cameras/{camera_name}/snapshot.jpg"`.

- [ ] **Step 6: Show status on the card and detail page**

`partials/camera_card.html`:

```html
<article class="camera-card" id="camera-card-{{ camera.name }}"
         hx-get="/cameras/{{ camera.name }}/status"
         hx-trigger="every 5s"
         hx-swap="outerHTML">
  <header>
    <a href="/cameras/{{ camera.name }}">{{ camera.name }}</a>
    <span class="status-dot status-{{ camera.status }}" title="{{ camera.status }}"></span>
    <small class="status-label status-{{ camera.status }}">
      {% if camera.status == 'restarting' %}restarting in {{ camera.restart_in }}s
      {% else %}{{ camera.status }}{% endif %}
    </small>
  </header>
  <div class="camera-card-body">
    <small>
      {% if camera.record_enabled %}Recording{% endif %}
      {% if camera.record_enabled and camera.has_proxy %} + {% endif %}
      {% if camera.has_proxy %}{{ camera.proxy_mode | upper }} :{{ camera.proxy_port }}{% endif %}
    </small>
    <br>
    <small>{{ camera.main_url_redacted }}</small>
    {% if camera.last_error and camera.status != 'running' %}
    <br><small class="camera-error" title="{{ camera.last_error }}">{{ camera.last_error[:120] }}</small>
    {% endif %}
  </div>
</article>
```

`cameras/detail.html`: in the Configuration table, change the Status row value to the same label logic as the card, and add a row "Last error" shown only when `camera.last_error` is set.

`warden.css`: add

```css
.status-running { color: var(--pico-ins-color, #2e7d32); }
.status-restarting { color: #b26a00; }
.status-failed { color: var(--pico-del-color, #c62828); }
.status-idle, .status-unknown { color: var(--pico-muted-color); }
.status-dot.status-running { background: #2e7d32; }
.status-dot.status-restarting { background: #b26a00; }
.status-dot.status-failed { background: #c62828; }
.camera-error { color: var(--pico-del-color, #c62828); word-break: break-word; }
```

- [ ] **Step 7: Run tests**

Run: `uv run pytest tests/test_camera_status.py tests/test_preview_routes.py tests/ -k "camera or dashboard or zones" -q`
Expected: all PASS.

- [ ] **Step 8: Commit**

```bash
git add src/rtsp_warden/app.py src/rtsp_warden/web tests/test_camera_status.py tests/test_preview_routes.py
git commit -m "feat: live camera status in the UI and same-origin MJPEG/snapshot routes

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 12: Remove dead code and unused dependencies

**Files:**
- Delete: `src/rtsp_warden/health_server.py`, `tests/test_health_server_auth.py`, `src/rtsp_warden/web_ui.py`, `src/rtsp_warden/web_assets/`, `src/rtsp_warden/web/schemas/`, `src/rtsp_warden/web/templates/macros/icons.html`, `src/rtsp_warden/web/templates/macros/__init__.py`, `src/rtsp_warden/web/templates/partials/__init__.py`
- Modify: `src/rtsp_warden/cli.py` (remove `run` alias and `ui` command and their imports), `src/rtsp_warden/app.py` (remove `run_app`), `src/rtsp_warden/ffmpeg.py` (remove `build_ffmpeg_mjpeg_stdout_cmd`), `src/rtsp_warden/config.py` (remove `DNNDetectorConfig`), `pyproject.toml` (remove `zeep`, `aiofiles`), `uv.lock`

- [ ] **Step 1: Confirm each target has no runtime references**

Run:

```bash
cd /home/jcharles/Projects/python/rtsp-warden_v0.2.0
for sym in health_server web_ui web_assets "web.schemas" "web/schemas" icons.html run_app build_ffmpeg_mjpeg_stdout_cmd DNNDetectorConfig zeep aiofiles; do echo "== $sym"; grep -rn "$sym" src/ tests/ pyproject.toml Dockerfile* docker-compose.yml --include=* | grep -v "^src/rtsp_warden/health_server.py\|^src/rtsp_warden/web_ui.py" ; done
```

Expected: only definition sites, `cli.py` imports for `web_ui`, `tests/test_health_server_auth.py`, and the `pyproject.toml` lines. Anything else is a consumer; stop and report it instead of deleting.

- [ ] **Step 2: Delete and edit**

```bash
git rm -q src/rtsp_warden/health_server.py tests/test_health_server_auth.py src/rtsp_warden/web_ui.py
git rm -rq src/rtsp_warden/web_assets src/rtsp_warden/web/schemas
git rm -q src/rtsp_warden/web/templates/macros/icons.html src/rtsp_warden/web/templates/macros/__init__.py src/rtsp_warden/web/templates/partials/__init__.py
```

`cli.py`: remove `from .web_ui import PreviewTarget, WebUiServer`, the entire `run_command` function with its decorator and the comment above it, and the entire `ui` command. `app.py`: remove `run_app`. `ffmpeg.py`: remove `build_ffmpeg_mjpeg_stdout_cmd`. `config.py`: remove `DNNDetectorConfig`. `pyproject.toml`: remove the `zeep` and `aiofiles` lines from `dependencies`.

If `tests/` references `build_ffmpeg_mjpeg_stdout_cmd` or `DNNDetectorConfig`, delete those tests too (they test dead code).

- [ ] **Step 3: Relock and verify**

Run:

```bash
uv lock && uv sync && uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/ && uv run pytest -q
```

Expected: ruff reports the same 20 E501 baseline or fewer (the `health_server.py` and `web_ui.py` ones are gone), format clean, all remaining tests pass. If `rtsp_warden/web/templates/macros/` is now empty, remove the directory.

- [ ] **Step 4: Commit**

```bash
git add -A src/ tests/ pyproject.toml uv.lock
git commit -m "chore: remove legacy health server, stdlib UI, run alias, unused deps and dead modules

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 13: Shared route helpers

**Files:**
- Create: `src/rtsp_warden/web/routes/_common.py`
- Modify: `src/rtsp_warden/web/routes/cameras.py`, `zones.py`, `onvif.py`, `alerts.py`
- Test: `tests/test_routes_common.py`

**Interfaces:**
- Produces: `get_cfg(request) -> AppConfig` (503 when missing), `get_config_path(request) -> Path | None`, `find_camera(cfg, name) -> CameraConfig | None`, `templates: Jinja2Templates`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_routes_common.py
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from rtsp_warden.config import AppConfig, CameraConfig
from rtsp_warden.web.routes._common import find_camera, get_cfg, get_config_path


def _req(**state):
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(**state)))


def test_get_cfg_raises_503_when_missing():
    with pytest.raises(HTTPException) as ei:
        get_cfg(_req(cfg=None))
    assert ei.value.status_code == 503


def test_get_config_path_none_or_path():
    assert get_config_path(_req(config_path=None)) is None
    assert get_config_path(_req(config_path="/tmp/x.yaml")) == Path("/tmp/x.yaml")


def test_find_camera():
    cfg = AppConfig(cameras=[CameraConfig(name="a", main_url="rtsp://h/a")])
    assert find_camera(cfg, "a").name == "a"
    assert find_camera(cfg, "zzz") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_routes_common.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# src/rtsp_warden/web/routes/_common.py
"""Helpers shared by route modules: config access, config path, camera lookup, templates."""

from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException, Request
from starlette.templating import Jinja2Templates

from ...config import AppConfig, CameraConfig
from ..paths import TEMPLATES_DIR

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def get_cfg(request: Request) -> AppConfig:
    cfg = getattr(request.app.state, "cfg", None)
    if cfg is None:
        raise HTTPException(status_code=503, detail="Server configuration not loaded")
    return cfg


def get_config_path(request: Request) -> Path | None:
    config_path = getattr(request.app.state, "config_path", None)
    return Path(config_path) if config_path else None


def find_camera(cfg: AppConfig, name: str) -> CameraConfig | None:
    for cam in cfg.cameras:
        if cam.name == name:
            return cam
    return None
```

In `cameras.py`, `zones.py`, `onvif.py` and `alerts.py`: delete the local `_get_cfg`, `_get_config_path` and `_find_camera_config` definitions and the local `_templates = Jinja2Templates(...)` line; add `from ._common import find_camera, get_cfg, get_config_path, templates` and rename call sites (`_get_cfg(` to `get_cfg(`, `_get_config_path(` to `get_config_path(`, `_find_camera_config(` to `find_camera(`, `_templates.` to `templates.`). Use `sed` per file, then `uv run ruff check` to catch leftover imports.

- [ ] **Step 4: Run the suite**

Run: `uv run pytest -q && uv run ruff check src/ tests/`
Expected: all PASS, no new ruff errors.

- [ ] **Step 5: Commit**

```bash
git add src/rtsp_warden/web/routes tests/test_routes_common.py
git commit -m "refactor: share route helpers instead of four copies

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 14: Documentation matches the code

**Files:**
- Modify: `README.md`, `CLAUDE.md`, `docker/README.md`, `docker-compose.yml`, `packaging/systemd/README.md`, `packaging/systemd/rtsp-warden.service`, `packaging/systemd/warden.env.example`, `.env.example`, `src/rtsp_warden/install.py` (ENV_EXAMPLE_CONTENT), `src/rtsp_warden/frame_tap.py:10`, `src/rtsp_warden/recorder.py:30`, `src/rtsp_warden/config.py:74-75`, `src/rtsp_warden/detectors/sensitivity.py:30`
- Move: `rtsp_warden_master_context.md` and `CONTEXT_RTSP_WARDEN_DROP_MP4_USE_TS.md` to `docs/archive/`

- [ ] **Step 1: README corrections**

Edit `README.md` so each of these is true. Verify every claim against the code as you go; the line numbers are from the audit and may have shifted.

- Install: `uv sync` (not `uv sync --extra dev`); dev tools are a dependency group.
- Config section: six top-level sections (`cameras`, `runtime`, `alerts`, `onvif`, `clips`, `retention`); `sub_url` optional, falls back to main; retention is `cameras[].retention` over global `retention`, with `record.retention` deprecated; `${ENV_VAR}` substitution with a `.env` example.
- Detector docs: the protocol is `name`, `kind`, `setup()`, `process(frame_bgr, ts_unix) -> list[Detection]`, `teardown()`. DNN options go under `config:` (`model_path`, `confidence_threshold`, `nms_threshold`, `classes`); `classes: null` means vehicles plus animals.
- Remove the per-camera `onvif:` block; document the global `onvif:` block and that discovery, PTZ and events default to off.
- Env vars: delete `WARDEN_SECRET`, `WARDEN_WEB_ENABLED`, `WARDEN_AUTH_HEALTHZ_OPEN`; state that `WARDEN_WEB_HOST` / `WARDEN_WEB_PORT` are overridden by `--web-host` / `--web-port`; Postgres URL form `postgresql+psycopg2://`.
- CLI: `-c` is required; `serve` is not a default command; `init-config` writes a file; remove `ui` and `run`; document that `serve` creates the schema and first admin (password in the log unless `WARDEN_ADMIN_PASSWORD` is set).
- Routes: `POST /cameras/{name}/retention` only; PTZ routes are admin; `POST /alerts/{name}/test`; add `GET /cameras/{name}/live.mjpeg` and `/snapshot.jpg`.
- Docker quick start: config goes to `./config/config.yaml`; no `install` step is needed; first-start admin credentials are in `docker compose logs`.
- Hot reload: rebuilds detectors from the in-memory config after a UI save; it does not re-read YAML.
- Alerts: debounce key is (notifier, camera, event_type); severity values are `info`, `warn`, `error`; detections are not yet wired to notifiers (sub-project 3).
- Recordings are ffmpeg `segment` muxer `.ts` files; HLS playlists are synthesized at request time.
- One `StreamIngestor` per (camera, stream).

- [ ] **Step 2: CLAUDE.md corrections**

- Line 13: replace "The README is accurate and detailed for config schema, CLI, routes, and deployment. Read it for those" with "The README was rewritten on 2026-10-02 to match the code; when they disagree, trust the code and fix the README."
- Docker: "Both run `rtsp-warden serve -c /app/config/config.yaml --web --web-port 8080`; the bind host comes from `WARDEN_WEB_HOST` (compose sets 0.0.0.0)."
- Health endpoints: add `/health` (HTML) and `/health/partial` (htmx).
- "Known wiring gap" paragraph: replace with "`cli.serve` passes `config_path` and `runtime` into `WebUIServer`, which sets `app.state.config_path` and `app.state.runtime`; tests set them directly on `create_app`."
- Repository notes: `examples/configs/` use `${CAM_USER}` / `${CAM_PASS}`; the rules about `uv`, secrets and quality gates live in `~/Projects/AGENTS.md` (two levels up), not the Boomerang roster in `~/Projects/python/AGENTS.md`.
- Commands: `rtsp-warden ui` and `run` no longer exist; `health_server.py` and `web_ui.py` are gone.
- Alerts paragraph: notifiers are built in `AlertManager.__init__`; `dispatch_event` still has no runtime caller until sub-project 3.

- [ ] **Step 3: Other docs and docstrings**

- `git mv rtsp_warden_master_context.md docs/archive/2025-rtsp_warden_master_context.md` and `git mv CONTEXT_RTSP_WARDEN_DROP_MP4_USE_TS.md docs/archive/CONTEXT_RTSP_WARDEN_DROP_MP4_USE_TS.md`; add a one-line note at the top of each: "Archived: historical design note, superseded by CLAUDE.md and docs/superpowers/specs/." Update the two references in `CLAUDE.md` and the one in `.dockerignore`.
- `docker/README.md`: distroless is the default; the UI binds `WARDEN_WEB_HOST`; the DB lives at `WARDEN_DB_URL`. Add `WARDEN_DB_URL: sqlite:////app/data/warden.db` to the compose `environment` so the `./data` mount is used, and fix the `postgresql://` example to `postgresql+psycopg2://`.
- `packaging/systemd/README.md` and `warden.env.example`: document `WARDEN_DB_URL=sqlite:////var/lib/rtsp-warden/data/warden.db` and have `install.sh` create that directory; `WARDEN_WEB_PORT` is overridden by `--web-port` in the unit, so remove `--web-port 8080` from `ExecStart` and let the env drive it. Fix `Documentation=` to `https://github.com/Veedubin/RSTP-Warden`.
- `.env.example` and `ENV_EXAMPLE_CONTENT` in `install.py`: remove `WARDEN_SECRET` and `WARDEN_AUTH_HEALTHZ_OPEN`; stop writing them in `run_install`; add `CAM_USER` / `CAM_PASS` placeholders.
- Docstrings: `frame_tap.py:10` (integration is live, remove "intentionally deferred"); `recorder.py:30` (`.ts`); `config.py:74-75` (`{output_dir}/{camera}/{stream}/{camera}_{stream}_%Y%m%d_%H%M%S.ts`); `detectors/sensitivity.py:30` (50 maps to 0.5).

- [ ] **Step 4: Verify and commit**

Run: `uv run pytest -q && uv run ruff check src/ tests/ && grep -rn "rtsp-warden run\|uv sync --extra\|WARDEN_SECRET\|HEALTHZ_OPEN" README.md CLAUDE.md docker packaging .env.example src/ | grep -v archive`
Expected: tests pass; the grep prints nothing.

```bash
git add -A README.md CLAUDE.md docs docker docker-compose.yml packaging .env.example .dockerignore src/rtsp_warden
git commit -m "docs: make README, CLAUDE.md, Docker and systemd docs match the code

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 15: Verify against the real camera

**Files:** none in the repo. Uses the session scratchpad instance described in memory (`test-camera-foscam-c1-v3`).

- [ ] **Step 1: Run the stabilized app against the Foscam**

Config in the scratchpad: one camera, `main_url` only (no `sub_url`), `proxy.stream` omitted, `record.main.chunk_seconds: 60`, `detectors: [{type: motion}]`, camera-level `retention: {max_days: 1}`. Start with `rtsp-warden serve -c config.yaml --web-port 8099`.

Expected in the log: a line from `bootstrap_database` only if the scratch DB is empty; no `[supervisor] ... died` lines; one `.ts` segment under `recordings/<camera>/main/` within 70 seconds.

- [ ] **Step 2: Browser checks**

- `/` without a session redirects to `/login?next=/`.
- Login lands on the dashboard rendered as a full page.
- Camera card says `running`; the detail page shows a moving image at `/cameras/<name>/live.mjpeg` and `/cameras/<name>/snapshot.jpg` returns a JPEG.
- Sensitivity save returns to the page without a 403 and the value is in `config.yaml`.
- Detector enable toggle and `POST /cameras/<name>/reload` return 200.
- Point `main_url` at `rtsp://127.0.0.1:1/x`, restart, and confirm the card shows `restarting in Ns` with the ffmpeg error line under it.

Record the outcome of each check in the final summary. Fix anything that fails before declaring the plan done.
