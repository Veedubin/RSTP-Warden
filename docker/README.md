# Docker Deployment

rtsp-warden ships with two Docker image variants:

| Variant | Base | Size | Build Time | Use When |
|---------|------|------|-----------|----------|
| `Dockerfile` (default) | `python:3.13-slim` | ~1.21 GB | ~3 min | You need a shell in the container, debugging, or want broadest compatibility |
| `Dockerfile.distroless` | `gcr.io/distroless/cc-debian13:nonroot` | **~685 MB** | ~3 min | Production deployments; you want a smaller, more secure image |

**The distroless image is the default in `docker-compose.yml`.**

## Quick Start

```bash
# Build and run the distroless image (default)
docker compose up -d

# OR build and run the larger slim image
docker build -t rtsp-warden:latest .
docker run -d --name rtsp-warden -p 8080:8080 \
    -v $(pwd)/config:/app/config:ro \
    -v $(pwd)/recordings:/app/recordings \
    -v $(pwd)/data:/app/data \
    -e WARDEN_WEB_HOST=0.0.0.0 \
    rtsp-warden:latest
```

## Distroless Architecture

The distroless image uses a 4-stage build:

| Stage | Base | Purpose |
|-------|------|---------|
| 1/4 `builder` | `python:3.13-slim` | Build `.venv` with `uv` |
| 2/4 `python-libs` | `python:3.13-slim` | Collect Python stdlib's system `.so` deps via `ldd` |
| 3/4 `deps` | `debian:13-slim` | Install ffmpeg + OpenCV libs, BFS-collect only needed `.so` files |
| 4/4 `runtime` | `gcr.io/distroless/cc-debian13:nonroot` | Minimal runtime; copies Python, `.venv`, ffmpeg, OpenCV libs |

The BFS dependency collector is at `docker/collect-deps.sh`. It uses `ldd` to walk transitive dependencies of `/usr/bin/ffmpeg` and copies only the `.so` files actually needed.

### Distroless Constraints

- **No shell.** No `bash`, `sh`, `ls`, `cat`. The container runs as the `nonroot` user (uid 65532).
- **No package manager.** No `apt`, `apk`, `pip` (use the venv copied from the builder stage).
- **No `curl`.** The HEALTHCHECK uses `python3 -c "import urllib.request; ..."` instead.
- **No `chown` binary.** Ownership is set via `COPY --chown=nonroot:nonroot` at build time.
- **No `mkdir` at runtime.** The `recordings/`, `config/`, `data/` directories are created by the application on first run, OR mounted as volumes.

### Healthcheck

```dockerfile
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD ["/usr/local/bin/python3", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=3).read()"]
```

## Mounting Volumes

When using **named volumes** or **freshly-created bind mounts**, the nonroot user can write to them. When using bind mounts that already contain files (e.g., recovered from a backup), the host directory must be readable AND writable by uid 65532.

```bash
# Named volumes: works out of the box
docker volume create warden-recordings
docker run -v warden-recordings:/app/recordings rtsp-warden:distroless

# Bind mounts: chown the host directory first
sudo chown -R 65532:65532 ./recordings
docker run -v $(pwd)/recordings:/app/recordings rtsp-warden:distroless
```

## Distroless Compatibility Notes

- **ONVIF discovery** uses UDP multicast to `239.255.255.250:3702`. This does NOT cross Docker bridge networks. Use `network_mode: host` for ONVIF discovery to work.
- **Audio in/out** works normally; uses the same ffmpeg binary copied from the deps stage.
- **Web UI** listens on 0.0.0.0:8080 (set `WARDEN_WEB_HOST=0.0.0.0` explicitly if needed).

## Why Distroless?

Distroless images contain only your application and its runtime dependencies — no package manager, no shell, no other utilities. This means:

1. **Smaller surface area for attacks** — no `bash` to exploit if your app is compromised
2. **Smaller image size** — ~685 MB vs ~1.21 GB (43% smaller)
3. **Faster pulls** — less data to transfer
4. **Better supply-chain hygiene** — fewer packages to audit

The trade-off: harder to debug (no shell). Use the slim image if you need to exec into the container for development.

## Building the Distroless Image Manually

```bash
docker build -f Dockerfile.distroless -t rtsp-warden:distroless .
docker images rtsp-warden:distroless
# Should report ~685 MB
```

## Verifying the Build

```bash
# Check ffmpeg works
docker run --rm rtsp-warden:distroless ffmpeg -version

# Check Python works
docker run --rm rtsp-warden:distroless python3 -c "import rtsp_warden; print(rtsp_warden.__version__)"

# Check the app starts
docker run --rm -p 8080:8080 -v $(pwd)/config:/app/config:ro \
    -v $(pwd)/recordings:/app/recordings \
    -v $(pwd)/data:/app/data \
    -e WARDEN_WEB_HOST=0.0.0.0 \
    rtsp-warden:distroless
# Then curl http://localhost:8080/healthz
```
