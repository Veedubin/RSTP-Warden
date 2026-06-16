"""Shared path constants for the web UI package.

Centralizes path resolution so that routes and the app factory
can both access the templates directory without circular imports.
"""

from __future__ import annotations

from pathlib import Path

# Resolve paths relative to the web package directory.
WEB_DIR = Path(__file__).resolve().parent
STATIC_DIR = WEB_DIR / "static"
TEMPLATES_DIR = WEB_DIR / "templates"
