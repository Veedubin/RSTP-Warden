# Docker deployment for rtsp-warden

## Quick start

```bash
# 1. Create config
mkdir -p config
cp ../config-NC230-C1-V3-2Cams.yaml config/config.yaml
# Edit config.yaml with your camera URLs

# 2. Run
docker compose up -d

# 3. Open http://localhost:8080

# 4. Find your admin password in logs
docker compose logs warden | grep "admin password"
```

## Using PostgreSQL

Uncomment the `postgres` service in docker-compose.yml AND uncomment the `WARDEN_DB_URL` env var in the `warden` service.

## Custom network for ONVIF

ONVIF discovery uses UDP multicast. If your cameras are on a different network namespace, you need:

```yaml
services:
  warden:
    network_mode: host  # for ONVIF discovery
```

## Volumes

| Path | Purpose |
|------|---------|
| `/app/config` | config.yaml (read-only) |
| `/app/recordings` | video segments |
| `/app/data` | SQLite DB |

## Updating

```bash
docker compose pull
docker compose up -d
```

## Logs

```bash
docker compose logs -f warden
```

## Backups

Back up these directories:
- `/app/recordings` (videos)
- `/app/data` (SQLite)
- `/app/config` (config)