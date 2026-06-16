from __future__ import annotations

import logging
from typing import Literal

from rich.logging import RichHandler

Verbosity = Literal["error", "warning", "info", "debug"]

_LEVELS: dict[str, int] = {
    "error": logging.ERROR,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
}


def setup_logging(verbosity: Verbosity = "info") -> None:
    """Configure logging with Rich formatting.

    This config is intentionally simple and library-friendly:
    * root logger configured once
    * log formatting handled by RichHandler
    """
    level = _LEVELS.get(verbosity, logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_time=True, show_level=True)],
    )

    # Reduce common noisy libraries.
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
