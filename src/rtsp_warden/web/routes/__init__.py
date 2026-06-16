"""Web UI route packages.

Each sub-module registers routes on its own APIRouter:
  - auth: login/logout/session routes
  - dashboard: main dashboard page
  - cameras: camera list, detail, and settings pages
  - recordings: recording list and detail pages
  - events: event list and detail pages
  - health: liveness probes and status endpoint
  - users: user management (admin-only)
  - tokens: API token management
  - settings: system information (admin-only)
"""
