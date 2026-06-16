"""ONVIF camera discovery, PTZ control, presets, and event subscription package."""

from __future__ import annotations

from .discovery import DiscoveredCamera, OnvifDiscovery, OnvifError
from .events import OnvifEvent, OnvifEventSubscriber, OnvifEventType
from .presets import PTZPreset, PTZPresetError, PTZPresetStore
from .ptz import OnvifClient, OnvifPTZ

__all__ = [
    "DiscoveredCamera",
    "OnvifClient",
    "OnvifDiscovery",
    "OnvifError",
    "OnvifEvent",
    "OnvifEventSubscriber",
    "OnvifEventType",
    "OnvifPTZ",
    "PTZPreset",
    "PTZPresetError",
    "PTZPresetStore",
]
