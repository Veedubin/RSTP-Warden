# rtsp-warden Sprint 2 Implementation Plan (v0.3.0 → v0.4.0)

**Author:** boomerang-architect  
**Date:** 2026-06-15  
**Status:** User decisions incorporated (2026-06-15) — Ready for coder execution

---

## 0. User Decisions (2026-06-15)

| # | Question | Decision | Impact on plan |
|---|---|---|---|
| 1 | Segments access | Expose entire recordings root to any authenticated user via `/segments/*`. Designed to be upgradeable to per-segment checks later. | Section 7 — `/segments/*` mount, no per-request validation. Add TODO comment. |
| 2 | Camera config source | DB is read-cache. `config.yaml` is authoritative. Restart server to apply changes. Docker-friendly: mount file as volume OR pass via env vars in compose. | Section 3 — Settings pages do NOT write to `config.yaml`. New cameras page is OUT OF SCOPE for Sprint 2. Web UI reads cameras from config.yaml at startup, mirrors to DB cache. |
| 3 | Health server | **ABSORB** `health_server.py` into FastAPI. One app, one port. | Section 10 — `health_server.py` removed. `/health` and `/healthz` added to FastAPI. CLI flag `--health-port` removed. |
| 4 | hls.js delivery | **VENDOR** into `static/js/hls.min.js`. No CDN. | Section 6 — download hls.js 1.5.x during build, commit to repo. |
| 5 | CLI rename | `run` → `serve`. `run` kept as hidden deprecated alias. | Section 10 — typer command `serve` added, `run` aliased with deprecation warning. |
| 6 | Version | v0.3.0 → v0.4.0 (single-digit bump, not v0.5.0). | `pyproject.toml` version. README header. |

### Docker-Ready Design (v1.0.0 target, informed by Sprint 2)

The web UI **does not write to `config.yaml`**. Cameras are loaded from `config.yaml` at startup and mirrored to the `cameras` table (read-cache). To change cameras:
1. Edit `config.yaml` (or env vars in docker-compose)
2. Restart the server
3. Web UI sees new cameras

This eliminates hot-reload complexity, file-locking issues, and YAML mutation bugs.

---

## 1. Framework Decision

### Choice: FastAPI + Jinja2 + htmx + Alpine.js

**Decision:** Replace `web_ui.py` (stdlib `http.server`) with FastAPI. Keep `health_server.py` as stdlib, unchanged.

### Justification Against "Stdlib-First" Constraint

The stdlib-first philosophy is preserved: the recorder, ingest pipeline, health server, and frame tap all remain stdlib+ffmpeg. FastAPI is added **only** for the web UI layer where stdlib `http.server` is genuinely insufficient:

| Requirement | stdlib `http.server` | FastAPI |
|---|---|---|
| 15+ routes with path params | Manual URL parsing per handler | Built-in path param extraction |
| Auth on every request | Manual cookie/header parsing per handler | Dependency injection, one decorator |
| Form handling (login, settings) | Manual multipart parsing | Built-in Form parsing |
| Server-side templates | Manual string concatenation (current approach) | Jinja2 with inheritance, includes, macros |
| CSRF protection | Must build from scratch | Starlette SessionMiddleware + htmx pattern |
| OpenAPI docs | Must build from scratch | Auto-generated |
| Static file serving | Manual per-file handler | `StaticFiles` mount |
| Rate limiting | Must build from scratch | Simple middleware |

The current `web_ui.py` is 406 lines for **3 endpoints** (/, /api/targets, /static/style.css) with no auth, no forms, no DB queries. Extending this pattern to 15+ authenticated, form-heavy, DB-backed pages would produce ~3000+ lines of fragile manual HTTP parsing. FastAPI eliminates this entire class of boilerplate.

### Dependencies Added and Their Real Value

| Dependency | Real Value | Already a dep? |
|---|---|---|
| `fastapi` | Routing, DI, OpenAPI, request validation, background tasks | **NEW** |
| `uvicorn[standard]` | Production ASGI server (uvloop, httptools, websockets) | **NEW** |
| `jinja2` | Server-side templating with inheritance, includes, autoescaping | **NEW** |
| `python-multipart` | Form parsing for login, camera settings, user management | **NEW** |
| `aiofiles` | Async static file serving for ASGI | **NEW** |
| `itsdangerous` | Session signing (pulled by starlette's SessionMiddleware) | **NEW** (transitive) |
| `starlette` | SessionMiddleware, static files (pulled by fastapi) | **NEW** (transitive) |
| `pydantic` | Request/response validation | **ALREADY PRESENT** |
| `sqlalchemy` | DB queries | **ALREADY PRESENT** |

**Total new direct deps: 5** (fastapi, uvicorn, jinja2, python-multipart, aiofiles). The rest are transitive or already present.

### What Stays Stdlib

- `health_server.py` — 3 endpoints, auth-gated, works fine as-is
- `web_ui.py` — deprecated but kept for `rtsp-warden ui` legacy command
- Recorder, ingest, frame tap, retention — all unchanged
- `.env` loader in `cli.py` — already stdlib, no python-dotenv needed

---

## 2. Architecture

### ASGI Server: uvicorn

**Choice:** `uvicorn` with `--workers 1` (single worker, SQLite-safe).

- uvicorn is the standard, battle-tested ASGI server for FastAPI
- Hypercorn is heavier (more deps), granian is newer/less proven in ecosystem
- Single worker because SQLite doesn't handle multi-process well, and we're a self-hosted NVR, not a SaaS

### Process Model: Daemon Thread Inside `serve`

```
┌─────────────────────────────────────────────┐
│  rtsp-warden serve (main process)            │
│                                              │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐  │
│  │ Recorder │  │  Proxy   │  │  Health   │  │
│  │ (ffmpeg) │  │ (mjpeg)  │  │  Server   │  │
│  │ threads  │  │ threads  │  │ (stdlib)  │  │
│  └──────────┘  └──────────┘  └───────────┘  │
│                                              │
│  ┌──────────────────────────────────────┐    │
│  │  FastAPI Web UI (uvicorn thread)     │    │
│  │  Port 8080, daemon thread            │    │
│  └──────────────────────────────────────┘    │
│                                              │
│  Supervisor loop (main thread)               │
└─────────────────────────────────────────────┘
```

- uvicorn runs via `uvicorn.Server` in a daemon thread (not subprocess)
- Started in `AppRuntime.start()`, stopped in `AppRuntime.stop_all()`
- Single process, no IPC needed, shared DB engine works fine
- Supervisor loop continues to monitor ingest/proxy health

### Port Strategy

| Service | Port | Protocol | Auth |
|---|---|---|---|
| Web UI (FastAPI) | **8080** (default) | HTTP | Session cookie + Bearer |
| Health server | **8899** (default) | HTTP | Auth-gated (WARDEN_AUTH_ENABLED) |
| MJPEG proxy (per camera) | **9001, 9002, ...** | HTTP | None (local-only by default) |
| RTSP proxy (per camera) | **9001, 9002, ...** | RTSP | None |

**No mounting.** Each service on its own port. Clean separation, no routing conflicts, no middleware complexity.

### FrameTap ↔ Web UI Interaction

The web UI does **NOT** consume frames from the FrameTap pipeline. FrameTap is for OpenCV consumers only (Sprint 3). The web UI embeds MJPEG streams via `<img>` tags pointing at per-camera proxy ports:

```html
<img src="http://127.0.0.1:9001/mjpeg" alt="front camera live" />
```

This is the same pattern as the legacy `web_ui.py`. No changes to the ingest pipeline needed.

### HLS Playback Architecture

```
Browser                    FastAPI                    Filesystem
  │                          │                          │
  │  GET /recordings/42      │                          │
  │─────────────────────────►│                          │
  │                          │  Query DB for recording  │
  │                          │  path, start_time        │
  │                          │─────────────────────────►│
  │                          │◄─────────────────────────│
  │  HTML page with          │                          │
  │  hls.js player           │                          │
  │◄─────────────────────────│                          │
  │                          │                          │
  │  hls.js requests         │                          │
  │  /htl/1/1718400000/      │                          │
  │  1718403600.m3u8         │                          │
  │─────────────────────────►│                          │
  │                          │  Query recordings in     │
  │                          │  time window             │
  │                          │─────────────────────────►│
  │                          │  Parse existing .m3u8    │
  │                          │  files, build virtual     │
  │                          │  playlist                │
  │                          │◄─────────────────────────│
  │  Virtual M3U8            │                          │
  │◄─────────────────────────│                          │
  │                          │                          │
  │  hls.js requests .ts     │                          │
  │  segments via static     │                          │
  │  file mount              │                          │
  │────────────────────────────────────────────────────►│
  │◄────────────────────────────────────────────────────│
```

- ffmpeg already produces `stream.m3u8` + `.ts` segments in the recording directory
- hls.js (CDN, single `<script>` tag, no build step) handles playback in browser
- `/htl/{camera_id}/{start_ts}/{end_ts}.m3u8` dynamically generates a virtual playlist
- `.ts` segments are served via FastAPI's `StaticFiles` mount on the recordings directory (or a dedicated `/segments/` route for access control)

### Static Assets Structure

```
src/rtsp_warden/web/
├── __init__.py
├── app.py              # FastAPI application factory
├── auth_depends.py     # ASGI auth dependency (bridges auth.py)
├── csrf.py             # CSRF middleware
├── rate_limit.py       # Login rate limiter
├── session.py          # Session config
├── config.py           # WebSettings (pydantic-settings)
├── routes/
│   ├── __init__.py
│   ├── auth.py         # /login, /logout
│   ├── dashboard.py    # /
│   ├── cameras.py      # /cameras, /cameras/{id}, /cameras/new, /cameras/{id}/settings
│   ├── recordings.py   # /recordings, /recordings/{id}
│   ├── events.py       # /events, /events/{id}
│   ├── users.py        # /users, /users/new, /users/{id}/reset-password
│   ├── tokens.py       # /api-tokens
│   ├── settings.py     # /settings
│   ├── health.py       # /health (public health check)
│   └── htl.py          # /htl/{camera_id}/{start_ts}/{end_ts}.m3u8
├── services/
│   ├── __init__.py
│   ├── cameras.py      # Camera DB operations
│   ├── recordings.py   # Recording DB operations
│   ├── events.py       # Event DB operations
│   ├── users.py        # User DB operations
│   └── htl.py          # HLS timeline slicing logic
├── schemas/
│   ├── __init__.py
│   ├── cameras.py      # Pydantic request/response models
│   ├── recordings.py
│   ├── events.py
│   ├── users.py
│   └── common.py        # Page, PageResponse, ErrorResponse
├── templates/
│   ├── base.html        # Base layout (nav, head, scripts)
│   ├── login.html
│   ├── dashboard.html
│   ├── cameras/
│   │   ├── list.html
│   │   ├── detail.html
│   │   ├── new.html
│   │   └── settings.html
│   ├── recordings/
│   │   ├── list.html
│   │   └── detail.html  # HLS player page
│   ├── events/
│   │   ├── list.html
│   │   └── detail.html
│   ├── users/
│   │   ├── list.html
│   │   └── new.html
│   ├── tokens/
│   │   └── list.html
│   ├── settings/
│   │   └── form.html
│   └── partials/
│       ├── camera_card.html    # htmx partial for dashboard grid
│       ├── event_row.html      # htmx partial for event list
│       ├── recording_row.html  # htmx partial for recording list
│       └── health_status.html  # htmx partial for health indicator
└── static/
    ├── css/
    │   ├── pico.min.css        # Pico CSS v2.0.6 (vendored)
    │   └── warden.css          # Custom styles
    └── js/
        └── warden.js           # Alpine.js components, htmx extensions
```

---

## 3. Templates & Routes

### Route Table

| Method | Path | Template | Auth | Description |
|---|---|---|---|---|
| `GET` | `/login` | `login.html` | None | Login form |
| `POST` | `/login` | Redirect | None | Authenticate, set session cookie |
| `POST` | `/logout` | Redirect | Any | Clear session, redirect to /login |
| `GET` | `/` | `dashboard.html` | Any | Camera grid, system stats, recent events |
| `GET` | `/cameras` | `cameras/list.html` | Any | Camera list with status |
| `GET` | `/cameras/{id}` | `cameras/detail.html` | Any | Live preview, recording controls, stats |
| `GET` | `/cameras/new` | `cameras/new.html` | Admin | Add camera form |
| `POST` | `/cameras/new` | Redirect | Admin | Create camera |
| `GET` | `/cameras/{id}/settings` | `cameras/settings.html` | Admin | Edit camera form |
| `PUT` | `/cameras/{id}/settings` | Redirect | Admin | Update camera config |
| `GET` | `/recordings` | `recordings/list.html` | Any | Recording list with date filter |
| `GET` | `/recordings/{id}` | `recordings/detail.html` | Any | HLS playback with timeline |
| `GET` | `/events` | `events/list.html` | Any | Event list with filters |
| `GET` | `/events/{id}` | `events/detail.html` | Any | Event detail with clip preview |
| `GET` | `/users` | `users/list.html` | Admin | User list |
| `GET` | `/users/new` | `users/new.html` | Admin | Create user form |
| `POST` | `/users/new` | Redirect | Admin | Create user |
| `POST` | `/users/{id}/reset-password` | Redirect | Admin | Reset user password |
| `GET` | `/api-tokens` | `tokens/list.html` | Admin | API token list |
| `POST` | `/api-tokens` | htmx partial | Admin | Create token (returns raw token once) |
| `DELETE` | `/api-tokens/{prefix}` | htmx partial | Admin | Revoke token |
| `GET` | `/settings` | `settings/form.html` | Admin | System settings form |
| `PUT` | `/settings` | Redirect | Admin | Update system settings |
| `GET` | `/health` | JSON | None | Public health check (lightweight) |
| `GET` | `/htl/{camera_id}/{start_ts}/{end_ts}.m3u8` | M3U8 playlist | Any | Dynamic HLS slicing |

### Page Details

#### `/login` (GET)
- **Template:** `login.html`
- **Data:** None (static form)
- **Form:** username + password fields, submit button
- **Auth:** None (public)

#### `/login` (POST)
- **Template:** Redirect to `/` on success, back to `/login` with error on failure
- **Data:** Form: username, password
- **Logic:** `auth.py:get_user_by_username()` → `verify_password()` → `create_session()` → set `warden_session` cookie
- **Rate limit:** 5 attempts/minute/IP (429 on exceed)

#### `/logout` (POST)
- **Template:** Redirect to `/login`
- **Logic:** Read `warden_session` cookie → `auth.py:delete_session()` → clear cookie

#### `/` (Dashboard)
- **Template:** `dashboard.html`
- **Data:**
  - All cameras with latest ingest_health (joinedload)
  - Recent events (last 10, ordered by created_at DESC)
  - System stats: total recordings, total size, uptime
- **htmx:** Camera status cards auto-refresh every 10s via `hx-get="/cameras?partial=cards" hx-trigger="every 10s"`
- **Auth:** Any authenticated user

#### `/cameras` (GET)
- **Template:** `cameras/list.html`
- **Data:** All cameras with latest ingest_health
- **Query params:** `?partial=cards` returns only the card grid (htmx partial)
- **Auth:** Any

#### `/cameras/{id}` (GET)
- **Template:** `cameras/detail.html`
- **Data:**
  - Camera row (by id)
  - Latest ingest_health for both streams
  - Recent recordings (last 5)
  - Recent events (last 5)
  - MJPEG URL for live preview embed
- **Auth:** Any

#### `/cameras/new` (GET + POST)
- **Template:** `cameras/new.html`
- **Form fields:** name, main_url, sub_url, enabled, record_enabled, proxy_enabled, proxy_mode, proxy_port
- **POST:** Validates with Pydantic schema, inserts Camera row, redirects to `/cameras/{id}`
- **Auth:** Admin only

#### `/cameras/{id}/settings` (GET + PUT)
- **Template:** `cameras/settings.html`
- **Form:** Pre-filled with current camera config (all fields from CameraConfig)
- **PUT:** Validates, updates Camera row + config_json, redirects to `/cameras/{id}`
- **Auth:** Admin only

#### `/recordings` (GET)
- **Template:** `recordings/list.html`
- **Data:** Paginated recordings (offset-based, 20 per page)
- **Query params:** `?camera_id=X&stream=main&from=2024-01-01&to=2024-01-02&page=1`
- **htmx:** "Load more" button: `hx-get="/recordings?page=2" hx-target="#recording-list" hx-swap="beforeend"`
- **Auth:** Any

#### `/recordings/{id}` (GET)
- **Template:** `recordings/detail.html`
- **Data:**
  - Recording row (by id)
  - Camera name (joined)
  - Adjacent recordings (prev/next for timeline navigation)
- **Player:** hls.js via CDN `<script src="https://cdn.jsdelivr.net/npm/hls.js@1">`
- **HLS source:** `/htl/{camera_id}/{start_ts}/{end_ts}.m3u8` where start_ts/end_ts are the recording's time window
- **Auth:** Any

#### `/events` (GET)
- **Template:** `events/list.html`
- **Data:** Paginated events (offset-based, 20 per page)
- **Query params:** `?camera_id=X&event_type=motion&severity=warn&from=...&to=...&page=1`
- **htmx:** Filter form submits via `hx-get` updating the list, "load more" for pagination
- **Auth:** Any

#### `/events/{id}` (GET)
- **Template:** `events/detail.html`
- **Data:**
  - Event row (by id)
  - Camera name (joined)
  - metadata_json parsed and displayed
  - If event has a recording_id in metadata, link to clip preview
- **Auth:** Any

#### `/users` (GET)
- **Template:** `users/list.html`
- **Data:** All users (id, username, role, is_active, created_at, last_login_at)
- **Auth:** Admin only

#### `/users/new` (GET + POST)
- **Template:** `users/new.html`
- **Form:** username, password, role (admin/viewer dropdown)
- **POST:** `auth.py:hash_password()` → insert User → redirect to `/users`
- **Auth:** Admin only

#### `/users/{id}/reset-password` (POST)
- **Template:** Redirect to `/users`
- **Form:** new_password (or auto-generate)
- **Logic:** `auth.py:hash_password()` → update User.password_hash
- **Auth:** Admin only

#### `/api-tokens` (GET)
- **Template:** `tokens/list.html`
- **Data:** All API tokens for all users (prefix, name, username, expires_at, last_used_at)
- **Auth:** Admin only

#### `/api-tokens` (POST)
- **Template:** htmx partial showing the new raw token (shown once)
- **Form:** user_id, name, ttl_days
- **Logic:** `auth.py:create_api_token()` → return raw token in response
- **Auth:** Admin only

#### `/api-tokens/{prefix}` (DELETE)
- **Template:** htmx partial removing the row
- **Logic:** `auth.py:revoke_api_token(prefix)`
- **Auth:** Admin only

#### `/settings` (GET + PUT)
- **Template:** `settings/form.html`
- **Form:** System settings (retention defaults, notification config placeholders, web UI port)
- **PUT:** Updates system config (stored in a settings table or config file)
- **Auth:** Admin only

#### `/health` (GET)
- **Template:** JSON response
- **Data:** `{"ok": true, "now": <unix_ts>, "version": "0.5.0"}`
- **Auth:** None (public, lightweight — unlike health_server.py which is auth-gated)
- **Note:** This is a quick liveness check. Full status is at health_server:8899/status.json

#### `/htl/{camera_id}/{start_ts}/{end_ts}.m3u8` (GET)
- **Template:** M3U8 playlist (text/plain)
- **Data:** Recordings in time window, parsed .m3u8 files
- **Logic:** See Section 7 (Dynamic HLS Slicing)
- **Auth:** Any

---

## 4. Database Access Layer

### Pattern: SQLAlchemy Direct + Thin Services

Routes use FastAPI's dependency injection for DB sessions:

```python
# web/auth_depends.py
from rtsp_warden.db.engine import get_session

async def get_db():
    with get_session() as session:
        yield session
```

Simple queries happen directly in route handlers:

```python
@router.get("/cameras")
async def list_cameras(
    db: Session = Depends(get_db),
    current_user: CurrentUser = Depends(get_current_user_dependency),
):
    cameras = db.query(Camera).options(joinedload(Camera.ingest_health)).all()
    return templates.TemplateResponse("cameras/list.html", {
        "request": request,
        "cameras": cameras,
        "user": current_user,
    })
```

Complex multi-step operations go in service modules:

```python
# web/services/cameras.py
def create_camera(db: Session, data: CameraCreate) -> Camera:
    camera = Camera(
        name=data.name,
        main_url=data.main_url,
        sub_url=data.sub_url,
        enabled=data.enabled,
        config_json=data.config_json,
    )
    db.add(camera)
    db.commit()
    db.refresh(camera)
    return camera

def get_camera_with_stats(db: Session, camera_id: int) -> dict:
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera:
        return None
    recent_recordings = (
        db.query(Recording)
        .filter(Recording.camera_id == camera_id)
        .order_by(Recording.start_time.desc())
        .limit(5)
        .all()
    )
    recent_events = (
        db.query(Event)
        .filter(Event.camera_id == camera_id)
        .order_by(Event.created_at.desc())
        .limit(5)
        .all()
    )
    health = (
        db.query(IngestHealth)
        .filter(IngestHealth.camera_id == camera_id)
        .all()
    )
    return {
        "camera": camera,
        "recordings": recent_recordings,
        "events": recent_events,
        "health": health,
    }
```

### Pydantic Schemas

```python
# web/schemas/common.py
from pydantic import BaseModel
from typing import Generic, TypeVar

T = TypeVar("T")

class PageResponse(BaseModel, Generic[T]):
    items: list[T]
    total: int
    page: int
    page_size: int
    has_next: bool

# web/schemas/cameras.py
class CameraCreate(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    main_url: str
    sub_url: str
    enabled: bool = True
    record_enabled: bool = True
    proxy_enabled: bool = True
    proxy_mode: Literal["mjpeg", "rtsp"] = "mjpeg"
    proxy_port: int = Field(ge=1, le=65535)

class CameraResponse(BaseModel):
    id: int
    name: str
    main_url: str  # Redacted in response
    sub_url: str   # Redacted in response
    enabled: bool
    created_at: datetime
    health: list[IngestHealthResponse] | None

class CameraUpdate(BaseModel):
    name: str | None = None
    main_url: str | None = None
    sub_url: str | None = None
    enabled: bool | None = None
    config_json: str | None = None

# web/schemas/events.py
class EventResponse(BaseModel):
    id: int
    camera_id: int | None
    camera_name: str | None
    event_type: str
    severity: str
    message: str
    metadata: dict
    created_at: datetime

# web/schemas/recordings.py
class RecordingResponse(BaseModel):
    id: int
    camera_id: int
    camera_name: str
    stream: str
    path: str
    size_bytes: int
    start_time: datetime
    end_time: datetime | None
    container: str
    created_at: datetime
```

### N+1 Prevention

- **Dashboard:** `joinedload(Camera.ingest_health)` on camera list query
- **Event list:** `joinedload(Event.camera)` for camera name
- **Recording list:** `joinedload(Recording.camera)` for camera name
- **Camera detail:** Single query with multiple joinedloads, then slice in Python

### Pagination Strategy

**Offset-based** with `PageResponse` generic:

```python
def paginate(query, page: int = 1, page_size: int = 20) -> PageResponse:
    total = query.count()
    items = query.offset((page - 1) * page_size).limit(page_size).all()
    return PageResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        has_next=(page * page_size) < total,
    )
```

Offset pagination is simple, predictable, and works perfectly for a self-hosted NVR with <10,000 recordings. Cursor pagination is overkill here.

---

## 5. Auth Integration

### Bridging WSGI auth.py to ASGI FastAPI

The existing `auth.py` functions operate on token strings and DB lookups — they don't need to change. We create a thin ASGI adapter:

```python
# web/auth_depends.py
from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from rtsp_warden.auth import (
    SESSION_COOKIE_NAME,
    BEARER_HEADER_PREFIX,
    lookup_session,
    lookup_api_token,
    CurrentUser,
)

bearer_scheme = HTTPBearer(auto_error=False)

async def get_current_user_dependency(
    request: Request,
    bearer: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> CurrentUser:
    # 1. Try session cookie
    session_token = request.cookies.get(SESSION_COOKIE_NAME)
    if session_token:
        auth_session = lookup_session(session_token)
        if auth_session:
            return CurrentUser(
                user_id=auth_session.user_id,
                username=auth_session.username,
                role=auth_session.role,
                auth_method="session",
            )

    # 2. Try bearer token
    if bearer and bearer.credentials.startswith(BEARER_HEADER_PREFIX):
        raw = bearer.credentials[len(BEARER_HEADER_PREFIX):]
        user = lookup_api_token(raw)
        if user and user.is_active:
            return CurrentUser(
                user_id=user.id,
                username=user.username,
                role=user.role,
                auth_method="bearer",
            )

    raise HTTPException(status_code=401, detail="Authentication required")

def require_admin(current_user: CurrentUser = Depends(get_current_user_dependency)) -> CurrentUser:
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user
```

### Login Flow

```python
# web/routes/auth.py
@router.post("/login")
async def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    # Rate limit check
    client_ip = request.client.host
    if not rate_limiter.allow(client_ip):
        raise HTTPException(status_code=429, detail="Too many login attempts")

    user = db.query(User).filter(User.username == username).first()
    if not user or not verify_password(password, user.password_hash):
        rate_limiter.record_failure(client_ip)
        return templates.TemplateResponse("login.html", {
            "request": request,
            "error": "Invalid username or password",
        })

    auth_session = create_session(user)
    response = RedirectResponse(url="/", status_code=303)
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=auth_session.token,
        httponly=True,
        secure=False,  # True in production with TLS
        samesite="lax",
        max_age=SESSION_TTL_SECONDS,
    )
    return response
```

### CSRF Protection

```python
# web/csrf.py
from starlette.middleware.base import BaseHTTPMiddleware
from itsdangerous import URLSafeTimedSerializer
import secrets

class CSRFTokenManager:
    def __init__(self, secret: str):
        self.serializer = URLSafeTimedSerializer(secret)

    def generate_token(self) -> str:
        return secrets.token_hex(32)

    def validate(self, token: str, session_token: str) -> bool:
        try:
            self.serializer.loads(token, max_age=3600)
            return True
        except Exception:
            return False

class CSRFMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if request.method in ("GET", "HEAD", "OPTIONS"):
            return await call_next(request)

        # Check HX-CSRF-Token header (htmx sends this automatically)
        csrf_header = request.headers.get("X-CSRF-Token") or request.headers.get("HX-CSRF-Token")
        if not csrf_header:
            return JSONResponse({"error": "CSRF token missing"}, status_code=403)

        # Validate against session
        session = request.session
        expected = session.get("csrf_token")
        if not expected or csrf_header != expected:
            return JSONResponse({"error": "CSRF token invalid"}, status_code=403)

        return await call_next(request)
```

Base template exposes CSRF token for htmx:

```html
<!-- templates/base.html -->
<meta name="csrf-token" content="{{ csrf_token }}">
<!-- htmx automatically reads this and sends HX-CSRF-Token header -->
```

### Rate Limiting on /login

```python
# web/rate_limit.py
import time
from collections import defaultdict

class RateLimiter:
    def __init__(self, max_attempts: int = 5, window_seconds: int = 60):
        self.max_attempts = max_attempts
        self.window = window_seconds
        self._attempts: dict[str, list[float]] = defaultdict(list)

    def allow(self, key: str) -> bool:
        now = time.time()
        self._attempts[key] = [t for t in self._attempts[key] if now - t < self.window]
        return len(self._attempts[key]) < self.max_attempts

    def record_failure(self, key: str) -> None:
        self._attempts[key].append(time.time())
```

No external dependency. 20 lines. Good enough for a self-hosted NVR.

---

## 6. htmx + Alpine.js Patterns

### Versions and Delivery

| Library | Version | Delivery | Rationale |
|---|---|---|---|
| htmx | 2.0.4 | Vendored in `static/js/htmx.min.js` | ~20KB, no build step, vendoring avoids CDN dependency for core functionality |
| Alpine.js | 3.14.9 | Vendored in `static/js/alpine.min.js` | ~15KB, no build step, used for client-side interactivity (dropdowns, modals, theme toggle) |
| hls.js | 1.6.x | CDN (`cdn.jsdelivr.net`) | ~200KB, only loaded on playback page, CDN is fine for optional heavy deps |

### htmx Patterns Used

| Pattern | Usage |
|---|---|
| `hx-get` | Load camera cards, event rows, recording rows, pagination |
| `hx-post` | Form submissions (login, camera create, user create, token create) |
| `hx-put` | Camera settings update, system settings update |
| `hx-delete` | Revoke API token |
| `hx-trigger="every 10s"` | Auto-refresh dashboard camera status cards |
| `hx-trigger="revealed"` | Lazy-load recording rows on scroll |
| `hx-swap="outerHTML"` | Replace camera card on status change |
| `hx-swap="beforeend"` | Append rows for "load more" pagination |
| `hx-target="#event-list"` | Target specific container for filtered results |
| `hx-confirm="Are you sure?"` | Confirmation for destructive actions (revoke token, delete) |
| `hx-indicator="#spinner"` | Loading spinner during requests |

### htmx + FastAPI Integration

FastAPI routes return HTML fragments for htmx requests:

```python
@router.get("/cameras")
async def list_cameras(
    request: Request,
    partial: str | None = None,
    db: Session = Depends(get_db),
):
    cameras = db.query(Camera).options(joinedload(Camera.ingest_health)).all()

    if partial == "cards":
        # htmx request — return only the card grid fragment
        return templates.TemplateResponse("partials/camera_cards.html", {
            "request": request,
            "cameras": cameras,
        })

    # Full page request
    return templates.TemplateResponse("cameras/list.html", {
        "request": request,
        "cameras": cameras,
    })
```

### HTMX-CSRF Pattern

Base template includes the CSRF meta tag:

```html
<!-- templates/base.html -->
<head>
    <meta name="csrf-token" content="{{ csrf_token }}">
    <script src="/static/js/htmx.min.js"></script>
    <script src="/static/js/alpine.min.js"></script>
</head>
```

htmx automatically:
1. Reads `<meta name="csrf-token">`
2. Sends `HX-CSRF-Token` header on all non-GET requests
3. Our CSRF middleware validates it

### Alpine.js Components

```html
<!-- Theme toggle (dark/light) -->
<div x-data="{ dark: localStorage.getItem('theme') !== 'light' }"
     x-init="$watch('dark', v => { localStorage.setItem('theme', v ? 'dark' : 'light'); document.documentElement.setAttribute('data-theme', v ? 'dark' : 'light') })">
    <button @click="dark = !dark" x-text="dark ? '☀️' : '🌙'"></button>
</div>

<!-- Mobile nav toggle -->
<nav x-data="{ open: false }">
    <button @click="open = !open">Menu</button>
    <div x-show="open" @click.outside="open = false">
        <!-- nav links -->
    </div>
</nav>

<!-- Date range picker for recordings filter -->
<div x-data="{ from: '', to: '' }">
    <input type="date" x-model="from" @change="
        $el.closest('form').dispatchEvent(new Event('submit', {bubbles: true}))
    ">
</div>
```

---

## 7. Dynamic HLS Slicing (htl)

### Design

The `/htl/{camera_id}/{start_ts}/{end_ts}.m3u8` endpoint generates a **virtual M3U8 playlist** that references `.ts` segments from the time window across multiple recorded files.

### How It Works

1. **Query recordings** in the time window:
   ```python
   recordings = db.query(Recording).filter(
       Recording.camera_id == camera_id,
       Recording.start_time <= end_dt,
       or_(Recording.end_time >= start_dt, Recording.end_time.is_(None)),
   ).order_by(Recording.start_time.asc()).all()
   ```

2. **For each recording**, find the `.m3u8` playlist file that ffmpeg's segment muxer produces:
   - Recording path: `recordings/front/main/front_main_20240615_120000.ts`
   - Playlist path: `recordings/front/main/stream.m3u8` (ffmpeg writes this alongside segments)
   - Parse the m3u8 to get segment filenames and durations

3. **Build virtual playlist**:
   - `#EXTM3U`
   - `#EXT-X-VERSION:3`
   - `#EXT-X-TARGETDURATION:{max_segment_duration}`
   - `#EXT-X-MEDIA-SEQUENCE:0`
   - For each segment in the time window:
     - `#EXTINF:{duration},\n{segment_path_relative_to_recordings_root}`
   - `#EXT-X-ENDLIST`

4. **Segment paths** are relative to the recordings root, served via a `/segments/` route that maps to the filesystem.

### Timestamp Mapping

For TS segments, ffmpeg's segment muxer with `-strftime 1` names files with timestamps:
```
front_main_20240615_120000.ts  → start_time = 2024-06-15 12:00:00
front_main_20240615_120500.ts  → start_time = 2024-06-15 12:05:00
```

The `.m3u8` playlist contains:
```
#EXTINF:300.000,
front_main_20240615_120000.ts
#EXTINF:300.000,
front_main_20240615_120500.ts
```

We parse the m3u8, extract segment filenames and durations, and map them to absolute timestamps using the recording's `start_time` as the anchor.

### Implementation

```python
# web/services/htl.py
import re
from datetime import datetime, timezone
from pathlib import Path

def parse_m3u8(m3u8_path: Path) -> list[dict]:
    """Parse an ffmpeg-generated m3u8 playlist.
    Returns list of {filename, duration_seconds}.
    """
    segments = []
    content = m3u8_path.read_text()
    lines = content.splitlines()
    current_duration = None
    for line in lines:
        if line.startswith("#EXTINF:"):
            match = re.match(r"#EXTINF:([\d.]+),", line)
            if match:
                current_duration = float(match.group(1))
        elif line and not line.startswith("#") and current_duration is not None:
            segments.append({
                "filename": line.strip(),
                "duration": current_duration,
            })
            current_duration = None
    return segments

def build_htl_playlist(
    camera_id: int,
    start_ts: float,
    end_ts: float,
    recordings_root: Path,
    db_session,
) -> str | None:
    """Build a virtual M3U8 playlist for a time window."""
    start_dt = datetime.fromtimestamp(start_ts, tz=timezone.utc)
    end_dt = datetime.fromtimestamp(end_ts, tz=timezone.utc)

    recordings = db_session.query(Recording).filter(
        Recording.camera_id == camera_id,
        Recording.start_time <= end_dt,
        or_(Recording.end_time >= start_dt, Recording.end_time.is_(None)),
    ).order_by(Recording.start_time.asc()).all()

    if not recordings:
        return None

    lines = [
        "#EXTM3U",
        "#EXT-X-VERSION:3",
        "#EXT-X-PLAYLIST-TYPE:VOD",
    ]

    max_duration = 0.0
    segment_entries = []

    for rec in recordings:
        rec_path = Path(rec.path)
        m3u8_path = rec_path.parent / "stream.m3u8"
        if not m3u8_path.exists():
            continue

        segments = parse_m3u8(m3u8_path)
        cumulative_time = rec.start_time.timestamp()

        for seg in segments:
            seg_start = cumulative_time
            seg_end = cumulative_time + seg["duration"]

            # Check if segment overlaps with requested window
            if seg_end > start_ts and seg_start < end_ts:
                # Relative path from recordings root
                seg_rel_path = rec_path.parent / seg["filename"]
                try:
                    rel = seg_rel_path.relative_to(recordings_root)
                except ValueError:
                    rel = seg_rel_path

                segment_entries.append({
                    "path": str(rel),
                    "duration": seg["duration"],
                })
                max_duration = max(max_duration, seg["duration"])

            cumulative_time = seg_end

    if not segment_entries:
        return None

    lines.append(f"#EXT-X-TARGETDURATION:{int(max_duration) + 1}")
    lines.append("#EXT-X-MEDIA-SEQUENCE:0")

    for entry in segment_entries:
        lines.append(f"#EXTINF:{entry['duration']:.3f},")
        lines.append(f"/segments/{entry['path']}")

    lines.append("#EXT-X-ENDLIST")
    return "\n".join(lines) + "\n"
```

### Route

```python
# web/routes/htl.py
@router.get("/htl/{camera_id}/{start_ts}/{end_ts}.m3u8")
async def htl_playlist(
    camera_id: int,
    start_ts: float,
    end_ts: float,
    db: Session = Depends(get_db),
):
    playlist = build_htl_playlist(
        camera_id=camera_id,
        start_ts=start_ts,
        end_ts=end_ts,
        recordings_root=Path(settings.recordings_root),
        db_session=db,
    )
    if playlist is None:
        raise HTTPException(status_code=404, detail="No segments in time window")

    return Response(content=playlist, media_type="application/vnd.apple.mpegurl")
```

### Segment Serving

```python
# In web/app.py
from fastapi.staticfiles import StaticFiles

# Mount recordings directory for .ts segment access
app.mount("/segments", StaticFiles(directory=str(recordings_root)), name="segments")
```

### Fallback Behavior

- **No recordings in window:** Return 404 with `{"error": "No recordings found for time window"}`
- **Recordings exist but no .m3u8:** Skip that recording, continue to next
- **No segments at all:** Return 404
- **Empty m3u8 is never returned** — we check `if not segment_entries: return None`

---

## 8. UI Layout & Styling

### CSS Framework: Pico CSS v2 (Classless)

**Choice:** Pico CSS v2.0.6, vendored.

**Rationale:**
- **Classless:** Most styling comes from semantic HTML elements. We add custom classes only where needed.
- **Dark mode built-in:** `data-theme="dark"` attribute, respects `prefers-color-scheme`
- **Responsive by default:** Container queries, fluid typography
- **25KB minified:** Tiny footprint
- **No build step:** Single CSS file, drop in
- **Accessible:** High contrast ratios, focus indicators

**Why not Tailwind?** Requires a build step (or CDN for Play CDN which is 100KB+). User explicitly said "no build step."

**Why not Open Props?** Requires build step for custom properties optimization.

**Why not custom CSS only?** We'd end up rebuilding a framework. Pico gives us 90% of what we need; custom `warden.css` handles the NVR-specific 10%.

### Custom CSS (`warden.css`)

```css
/* Camera grid cards */
.camera-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: 1rem;
}

.camera-card {
    border: 1px solid var(--pico-muted-border-color);
    border-radius: var(--pico-border-radius);
    overflow: hidden;
}

.camera-card .preview {
    width: 100%;
    aspect-ratio: 16/9;
    background: #000;
}

.camera-card .preview img {
    width: 100%;
    height: 100%;
    object-fit: contain;
}

/* Status indicators */
.status-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
}

.status-dot.online { background: var(--pico-color-green-500); }
.status-dot.offline { background: var(--pico-color-red-500); }
.status-dot.warning { background: var(--pico-color-yellow-500); }

/* Timeline scrubber */
.timeline {
    position: relative;
    height: 40px;
    background: var(--pico-secondary-background);
    border-radius: var(--pico-border-radius);
    cursor: pointer;
}

.timeline .played { background: var(--pico-primary-background); }
.timeline .buffered { background: var(--pico-muted-border-color); }

/* Recording player container */
.player-container {
    max-width: 100%;
    aspect-ratio: 16/9;
    background: #000;
}

.player-container video {
    width: 100%;
    height: 100%;
}
```

### Dark Mode Toggle

Alpine.js component in `base.html`:

```html
<html lang="en" data-theme="dark"
      x-data="{ theme: localStorage.getItem('theme') || 'dark' }"
      x-init="$watch('theme', v => {
          localStorage.setItem('theme', v);
          document.documentElement.setAttribute('data-theme', v);
      })">
```

### Responsive Breakpoints

Pico CSS handles this. Key breakpoints:
- `< 576px`: Single column, collapsed nav (hamburger)
- `576-768px`: 2-column camera grid
- `768-992px`: 3-column camera grid
- `> 992px`: 4-column camera grid, full nav

### Icon Library: Lucide Inline SVG

**Choice:** Lucide icons as inline SVGs in a Jinja2 macro template.

**Rationale:**
- No CDN dependency, no font loading
- We need ~10 icons: camera, video, alert-triangle, user, key, settings, log-out, play, chevron-left, chevron-right
- Each icon is ~200-500 bytes of SVG
- Jinja2 macro makes them reusable: `{{ icon('camera') }}`

```html
<!-- templates/macros/icons.html -->
{% macro icon(name) -%}
{% if name == 'camera' -%}
<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z"/><circle cx="12" cy="13" r="3"/></svg>
{% elif name == 'alert-triangle' -%}
<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
{% endif -%}
{% endmacro %}
```

---

## 9. Test Strategy

### Test Infrastructure

- **Framework:** pytest (already in dev deps)
- **FastAPI testing:** `fastapi.testclient.TestClient` (included with fastapi)
- **DB testing:** SQLite in-memory database, `reset_engine()` between tests
- **Coverage target:** 80% on new `web/` modules

### Test Files

```
tests/
├── conftest.py              # (existing, extend with FastAPI fixtures)
├── test_web_auth.py         # Login/logout flow, session validation, bearer token
├── test_web_cameras.py      # Camera CRUD, list, detail, settings
├── test_web_recordings.py   # Recording list, detail, pagination
├── test_web_events.py       # Event list, detail, filters
├── test_web_users.py        # User management (admin only)
├── test_web_tokens.py       # API token create/list/revoke
├── test_web_htl.py          # M3U8 generation correctness
├── test_web_csrf.py         # CSRF middleware enforcement
├── test_web_rate_limit.py   # Rate limiter behavior
└── test_web_dashboard.py    # Dashboard data aggregation
```

### Key Test Cases

**Auth:**
- Login with valid credentials → 303 redirect, session cookie set
- Login with invalid credentials → 200 with error message
- Login rate limit → 429 after 5 failures
- Access protected page without auth → 401
- Access admin page as viewer → 403
- Bearer token auth → 200
- Expired session → 401

**htl:**
- Valid time window with segments → 200, valid M3U8
- Time window with no recordings → 404
- Time window partially overlapping recordings → only overlapping segments included
- M3U8 structure: `#EXTM3U`, `#EXT-X-TARGETDURATION`, `#EXTINF`, `#EXT-X-ENDLIST`
- Segment paths are relative and accessible

**CSRF:**
- POST without CSRF token → 403
- POST with valid CSRF token → 200
- GET requests bypass CSRF check

### TestClient Fixture

```python
# tests/conftest.py (additions)
from fastapi.testclient import TestClient
from rtsp_warden.web.app import create_app
from rtsp_warden.db.engine import get_engine, reset_engine

@pytest.fixture
def client():
    app = create_app()
    with TestClient(app) as c:
        yield c

@pytest.fixture
def admin_client(client):
    """Client pre-authenticated as admin."""
    # Create admin user, get session token, set cookie
    ...
    return client

@pytest.fixture
def viewer_client(client):
    """Client pre-authenticated as viewer."""
    ...
    return client
```

---

## 10. Migration Path

### What Happens to `web_ui.py`?

**Deprecated but kept.** The file stays in `src/rtsp_warden/web_ui.py` with a deprecation notice at the top:

```python
"""
DEPRECATED as of v0.5.0. Use the FastAPI web UI instead (rtsp-warden serve).
This module is kept for backward compatibility via `rtsp-warden ui`.
Will be removed in v1.0.0.
"""
```

### CLI Changes

**New `serve` command** (replaces `run`):

```python
@app.command()
def serve(
    config: Path = typer.Option(...),
    verbosity: str = typer.Option("info"),
    health: bool = typer.Option(True),
    health_host: str = typer.Option("127.0.0.1"),
    health_port: int = typer.Option(8899),
    web: bool = typer.Option(True, "--web/--no-web"),
    web_host: str = typer.Option("127.0.0.1", "--web-host"),
    web_port: int = typer.Option(8080, "--web-port"),
    frame_consumer: list[str] = typer.Option([]),
) -> None:
    """Run the full stack: recording + proxy + health + web UI."""
    # ... starts recorder, health server, AND FastAPI web UI
```

**Legacy `ui` command** (kept, marked deprecated):

```python
@app.command(hidden=True)  # Hidden from help, but still works
def ui(...) -> None:
    """[DEPRECATED] Start the minimal MJPEG grid web UI. Use `serve --web` instead."""
    # ... unchanged
```

**`run` command** (kept as alias):

```python
@app.command(hidden=True)
def run(...) -> None:
    """[DEPRECATED] Alias for `serve --no-web`. Use `serve` instead."""
    # ... delegates to serve with web=False
```

### Port Defaults

| Command | Web UI Port | Health Port | Legacy UI Port |
|---|---|---|---|
| `serve` (new) | 8080 | 8899 | N/A |
| `serve --no-web` | N/A | 8899 | N/A |
| `ui` (legacy) | N/A | N/A | 8080 |

### `health_server.py`

**Unchanged.** Stays stdlib `http.server`. The FastAPI web UI has its own `/health` endpoint (lightweight, public), but the full status/metrics remain on the health server at port 8899.

---

## 11. Dependencies to Add

### `pyproject.toml` Changes

```toml
[project]
version = "0.5.0"
dependencies = [
    # Existing (unchanged)
    "typer>=0.12.0",
    "pydantic>=2.7.0",
    "PyYAML>=6.0.1",
    "rich>=13.7.1",
    "sqlalchemy>=2.0.30",
    "psycopg2-binary>=2.9.9",
    "pydantic-settings>=2.3.0",
    "bcrypt>=4.1.0",
    "opencv-python-headless>=4.10.0",
    "numpy>=1.26.0",

    # Sprint 2 additions
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.30.0",
    "jinja2>=3.1.0",
    "python-multipart>=0.0.9",
    "aiofiles>=24.1.0",
]

[dependency-groups]
dev = [
    "pytest>=8.0.0",
    "pytest-cov>=5.0.0",
    "httpx>=0.27.0",          # For TestClient (fastapi testing)
]
```

### Justification for Each Addition

| Dependency | Why Needed | Stdlib Alternative? |
|---|---|---|
| `fastapi` | Routing, DI, OpenAPI, request validation, background tasks | No — stdlib `http.server` requires manual URL parsing, no DI, no validation |
| `uvicorn[standard]` | Production ASGI server with uvloop, httptools | No — stdlib has no ASGI server |
| `jinja2` | Server-side templating with inheritance, autoescaping, macros | No — stdlib string.Template is too primitive (no inheritance, no autoescaping) |
| `python-multipart` | Form parsing for login, camera settings, user management | No — stdlib has no multipart/form-data parser |
| `aiofiles` | Async static file serving for ASGI | No — stdlib file I/O is synchronous, blocks ASGI event loop |
| `httpx` (dev) | FastAPI TestClient requires httpx | No — but dev-only, not a runtime dep |

### What We Do NOT Add

| Not Added | Why Not |
|---|---|
| `python-dotenv` | Already have stdlib `.env` loader in `cli.py` |
| `alembic` | Deferred to Sprint 4 (v1.0.0) |
| `itsdangerous` | Pulled transitively by starlette's SessionMiddleware |
| `passlib` | Already using `bcrypt` directly |
| `python-jose` | No JWT needed — sessions + bearer tokens are sufficient |
| `celery` / `arq` | No background task queue needed yet |
| `websockets` | No WebSocket endpoints in Sprint 2 |
| `sse-starlette` | No Server-Sent Events in Sprint 2 |

---

## 12. Implementation Order / Batches

### Batch 1: Foundation (FastAPI Skeleton + Uvicorn Integration)

**Goal:** FastAPI app boots, serves a "Hello World" dashboard, uvicorn runs in daemon thread.

**Files created:**
- `src/rtsp_warden/web/__init__.py`
- `src/rtsp_warden/web/app.py` — `create_app()` factory
- `src/rtsp_warden/web/config.py` — `WebSettings` (pydantic-settings)
- `src/rtsp_warden/web/templates/base.html` — Base layout with Pico CSS, htmx, Alpine.js
- `src/rtsp_warden/web/static/css/pico.min.css` — Vendored Pico CSS v2.0.6
- `src/rtsp_warden/web/static/css/warden.css` — Custom styles (empty initially)
- `src/rtsp_warden/web/static/js/htmx.min.js` — Vendored htmx 2.0.4
- `src/rtsp_warden/web/static/js/alpine.min.js` — Vendored Alpine.js 3.14.9
- `src/rtsp_warden/web/static/js/warden.js` — Custom JS (empty initially)

**Files modified:**
- `pyproject.toml` — Add deps, bump version to 0.5.0
- `src/rtsp_warden/cli.py` — Add `serve` command with `--web/--no-web` flag, uvicorn thread start/stop
- `src/rtsp_warden/app.py` — Add `_start_web_ui()` / `_stop_web_ui()` methods

**Testable:** `rtsp-warden serve -c config.yaml --web` boots and shows base template at http://127.0.0.1:8080/

### Batch 2: Auth UI (Login, Logout, Sessions, CSRF)

**Goal:** Users can log in, get a session cookie, access protected pages, log out.

**Files created:**
- `src/rtsp_warden/web/auth_depends.py` — `get_current_user_dependency()`, `require_admin()`
- `src/rtsp_warden/web/session.py` — Starlette SessionMiddleware config
- `src/rtsp_warden/web/csrf.py` — CSRF middleware
- `src/rtsp_warden/web/rate_limit.py` — In-memory rate limiter
- `src/rtsp_warden/web/routes/__init__.py`
- `src/rtsp_warden/web/routes/auth.py` — `/login` GET/POST, `/logout` POST
- `src/rtsp_warden/web/templates/login.html`
- `src/rtsp_warden/web/templates/macros/__init__.py` (empty)
- `src/rtsp_warden/web/templates/macros/icons.html` — Lucide icon macros

**Files modified:**
- `src/rtsp_warden/web/app.py` — Register auth routes, add SessionMiddleware, CSRFMiddleware
- `src/rtsp_warden/web/templates/base.html` — Add nav bar, user menu, CSRF meta tag

**Testable:** Login flow works end-to-end. Protected routes return 401 without auth.

### Batch 3: Core Pages (Dashboard, Cameras, Recordings, Events)

**Goal:** Full dashboard with camera grid, camera detail, recording list, event list.

**Files created:**
- `src/rtsp_warden/web/schemas/__init__.py`
- `src/rtsp_warden/web/schemas/common.py` — `PageResponse`
- `src/rtsp_warden/web/schemas/cameras.py`
- `src/rtsp_warden/web/schemas/recordings.py`
- `src/rtsp_warden/web/schemas/events.py`
- `src/rtsp_warden/web/services/__init__.py`
- `src/rtsp_warden/web/services/cameras.py`
- `src/rtsp_warden/web/services/recordings.py`
- `src/rtsp_warden/web/services/events.py`
- `src/rtsp_warden/web/routes/dashboard.py` — `/`
- `src/rtsp_warden/web/routes/cameras.py` — `/cameras`, `/cameras/{id}`
- `src/rtsp_warden/web/routes/recordings.py` — `/recordings`, `/recordings/{id}`
- `src/rtsp_warden/web/routes/events.py` — `/events`, `/events/{id}`
- `src/rtsp_warden/web/routes/health.py` — `/health`
- `src/rtsp_warden/web/templates/dashboard.html`
- `src/rtsp_warden/web/templates/cameras/list.html`
- `src/rtsp_warden/web/templates/cameras/detail.html`
- `src/rtsp_warden/web/templates/recordings/list.html`
- `src/rtsp_warden/web/templates/recordings/detail.html` — (placeholder, player added in Batch 5)
- `src/rtsp_warden/web/templates/events/list.html`
- `src/rtsp_warden/web/templates/events/detail.html`
- `src/rtsp_warden/web/templates/partials/camera_cards.html`
- `src/rtsp_warden/web/templates/partials/event_rows.html`
- `src/rtsp_warden/web/templates/partials/recording_rows.html`
- `src/rtsp_warden/web/templates/partials/health_status.html`

**Files modified:**
- `src/rtsp_warden/web/app.py` — Register all new route modules
- `src/rtsp_warden/web/static/css/warden.css` — Camera grid, status indicators, timeline styles

**Testable:** Dashboard shows cameras with status. Camera detail shows live MJPEG embed. Recording/event lists with pagination work.

### Batch 4: Admin Pages (Users, API Tokens, Camera Settings, System Settings)

**Goal:** Admin can manage users, API tokens, camera config, system settings.

**Files created:**
- `src/rtsp_warden/web/schemas/users.py`
- `src/rtsp_warden/web/services/users.py`
- `src/rtsp_warden/web/routes/users.py` — `/users`, `/users/new`, `/users/{id}/reset-password`
- `src/rtsp_warden/web/routes/tokens.py` — `/api-tokens`
- `src/rtsp_warden/web/routes/settings.py` — `/settings`
- `src/rtsp_warden/web/templates/users/list.html`
- `src/rtsp_warden/web/templates/users/new.html`
- `src/rtsp_warden/web/templates/tokens/list.html`
- `src/rtsp_warden/web/templates/settings/form.html`
- `src/rtsp_warden/web/templates/cameras/new.html`
- `src/rtsp_warden/web/templates/cameras/settings.html`

**Files modified:**
- `src/rtsp_warden/web/app.py` — Register admin route modules
- `src/rtsp_warden/web/routes/cameras.py` — Add `/cameras/new` GET/POST, `/cameras/{id}/settings` GET/PUT

**Testable:** Admin creates user, creates API token, edits camera settings. Viewer gets 403 on admin routes.

### Batch 5: HLS Playback + htl Endpoint

**Goal:** Recording detail page has working HLS player with timeline scrubbing.

**Files created:**
- `src/rtsp_warden/web/services/htl.py` — `build_htl_playlist()`, `parse_m3u8()`
- `src/rtsp_warden/web/routes/htl.py` — `/htl/{camera_id}/{start_ts}/{end_ts}.m3u8`
- `src/rtsp_warden/web/static/js/player.js` — hls.js initialization, custom timeline UI

**Files modified:**
- `src/rtsp_warden/web/templates/recordings/detail.html` — Add `<video>` element, hls.js CDN script, timeline component
- `src/rtsp_warden/web/app.py` — Register htl route, mount `/segments` static files
- `src/rtsp_warden/web/static/css/warden.css` — Player container, timeline scrubber styles

**Testable:** Navigate to a recording, video plays via HLS. htl endpoint returns valid M3U8. Timeline shows segment boundaries.

---

## 13. Risks & Open Questions

### Risks

| Risk | Mitigation |
|---|---|
| **uvicorn thread conflicts with SQLite** | Single worker, WAL mode, `check_same_thread=False` already set |
| **M3U8 parsing fragile** | ffmpeg's segment muxer output is deterministic; fall back gracefully if .m3u8 missing |
| **Memory usage with many cameras** | MJPEG embeds are `<img>` tags (browser fetches directly from proxy ports, not through FastAPI) |
| **Session cookie theft (no TLS)** | Document that TLS (Caddy/LetsEncrypt) is needed for remote access (Sprint 4) |
| **Rate limiter state lost on restart** | Acceptable for self-hosted NVR; could persist to DB later if needed |

### Open Questions for User Confirmation

1. **Recording directory mount:** Should `/segments/` mount the entire recordings root, or should we restrict to the specific recording's directory? (Security: mounting the whole root exposes all recordings to anyone with a valid session. Is that acceptable, or should we validate per-segment access?)

2. **Camera add/edit via web UI vs config.yaml:** Currently cameras are defined in `config.yaml`. Should the web UI's camera management:
   - (A) Write back to `config.yaml` (sync between DB and YAML)?
   - (B) Be DB-only, with config.yaml as the initial seed on first run?
   - (C) Be read-only from config.yaml (web UI shows but doesn't edit cameras)?

3. **`health_server.py` absorption:** Should we eventually absorb the health server into FastAPI (single port), or keep it separate? The current plan keeps it separate. If absorbed, health_server.py would be deprecated.

4. **Pico CSS version:** v2.0.6 is the latest stable. Confirm this is acceptable vs v1.x (which is smaller but less feature-rich).

5. **hls.js delivery:** CDN (`cdn.jsdelivr.net/npm/hls.js@1`) is the plan. Acceptable, or should we vendor the ~200KB file?

6. **`serve` vs `run` naming:** The plan renames `run` → `serve` (with `run` as hidden deprecated alias). Confirm this naming change is acceptable.

7. **v0.5.0 version bump:** Sprint 2 produces v0.5.0 (skipping v0.4.0 which was the internal auth sprint). Confirm version numbering.

---

## Appendix A: File Tree After Sprint 2

```
src/rtsp_warden/
├── __init__.py
├── app.py                  # Modified: +web UI start/stop
├── auth.py                 # Unchanged
├── cli.py                  # Modified: +serve command, deprecated run/ui
├── config.py               # Unchanged
├── consumers/              # Unchanged
├── db/                     # Unchanged
├── ffmpeg.py               # Unchanged
├── frame_tap.py            # Unchanged
├── health_server.py        # Unchanged
├── install.py              # Unchanged
├── logging_utils.py        # Unchanged
├── proxy/                  # Unchanged
├── recorder.py             # Unchanged
├── retention.py            # Unchanged
├── status_model.py         # Unchanged
├── web_assets/             # Unchanged (legacy)
├── web_ui.py               # Modified: deprecation notice
└── web/                    # NEW
    ├── __init__.py
    ├── app.py
    ├── auth_depends.py
    ├── config.py
    ├── csrf.py
    ├── rate_limit.py
    ├── session.py
    ├── routes/
    │   ├── __init__.py
    │   ├── auth.py
    │   ├── cameras.py
    │   ├── dashboard.py
    │   ├── events.py
    │   ├── health.py
    │   ├── htl.py
    │   ├── recordings.py
    │   ├── settings.py
    │   ├── tokens.py
    │   └── users.py
    ├── schemas/
    │   ├── __init__.py
    │   ├── cameras.py
    │   ├── common.py
    │   ├── events.py
    │   ├── recordings.py
    │   └── users.py
    ├── services/
    │   ├── __init__.py
    │   ├── cameras.py
    │   ├── events.py
    │   ├── htl.py
    │   ├── recordings.py
    │   └── users.py
    ├── static/
    │   ├── css/
    │   │   ├── pico.min.css
    │   │   └── warden.css
    │   └── js/
    │       ├── alpine.min.js
    │       ├── htmx.min.js
    │       ├── player.js
    │       └── warden.js
    └── templates/
        ├── base.html
        ├── dashboard.html
        ├── login.html
        ├── cameras/
        │   ├── detail.html
        │   ├── list.html
        │   ├── new.html
        │   └── settings.html
        ├── events/
        │   ├── detail.html
        │   └── list.html
        ├── macros/
        │   └── icons.html
        ├── partials/
        │   ├── camera_cards.html
        │   ├── event_rows.html
        │   ├── health_status.html
        │   └── recording_rows.html
        ├── recordings/
        │   ├── detail.html
        │   └── list.html
        ├── settings/
        │   └── form.html
        ├── tokens/
        │   └── list.html
        └── users/
            ├── list.html
            └── new.html

tests/
├── conftest.py             # Modified: +FastAPI fixtures
├── test_web_auth.py        # NEW
├── test_web_cameras.py     # NEW
├── test_web_csrf.py        # NEW
├── test_web_dashboard.py   # NEW
├── test_web_events.py      # NEW
├── test_web_htl.py         # NEW
├── test_web_rate_limit.py  # NEW
├── test_web_recordings.py  # NEW
├── test_web_tokens.py      # NEW
├── test_web_users.py       # NEW
└── (existing tests)        # Unchanged
```

## Appendix B: Base Template Reference

```html
<!-- templates/base.html -->
<!doctype html>
<html lang="en" data-theme="dark"
      x-data="{ theme: localStorage.getItem('theme') || 'dark' }"
      x-init="$watch('theme', v => {
          localStorage.setItem('theme', v);
          document.documentElement.setAttribute('data-theme', v);
      })">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta name="csrf-token" content="{{ csrf_token }}">
    <title>{% block title %}RTSP Warden{% endblock %}</title>
    <link rel="stylesheet" href="/static/css/pico.min.css">
    <link rel="stylesheet" href="/static/css/warden.css">
    <script src="/static/js/htmx.min.js" defer></script>
    <script src="/static/js/alpine.min.js" defer></script>
    <script src="/static/js/warden.js" defer></script>
    {% block head_extra %}{% endblock %}
</head>
<body>
    <nav>
        <ul>
            <li><strong><a href="/">RTSP Warden</a></strong></li>
        </ul>
        <ul>
            <li><a href="/cameras">Cameras</a></li>
            <li><a href="/recordings">Recordings</a></li>
            <li><a href="/events">Events</a></li>
            {% if user.role == 'admin' %}
            <li><a href="/users">Users</a></li>
            <li><a href="/api-tokens">Tokens</a></li>
            <li><a href="/settings">Settings</a></li>
            {% endif %}
            <li>
                <form action="/logout" method="post" style="display:inline">
                    <input type="hidden" name="csrf_token" value="{{ csrf_token }}">
                    <button type="submit" class="secondary">Logout ({{ user.username }})</button>
                </form>
            </li>
            <li>
                <button @click="theme = theme === 'dark' ? 'light' : 'dark'"
                        x-text="theme === 'dark' ? '☀️' : '🌙'"
                        class="secondary"></button>
            </li>
        </ul>
    </nav>

    <main class="container">
        {% block content %}{% endblock %}
    </main>

    <div id="htmx-indicator" class="htmx-indicator" style="display:none">
        <progress></progress>
    </div>
</body>
</html>
```
