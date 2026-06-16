"""In-memory rate limiter for login attempts.

Uses a simple sliding-window counter keyed by client IP.
5 failed login attempts per 60-second window triggers a 429 response.
"""

from __future__ import annotations

import time
from collections import defaultdict


class RateLimiter:
    """Token-bucket rate limiter with a sliding time window.

    Parameters
    ----------
    max_requests:
        Maximum number of requests allowed within the window.
    window_seconds:
        Duration of the sliding window in seconds.
    """

    def __init__(self, max_requests: int = 5, window_seconds: float = 60.0) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._attempts: dict[str, list[float]] = defaultdict(list)

    def check(self, key: str) -> bool:
        """Return True if the request is allowed, False if rate-limited.

        Prunes expired entries before checking.
        """
        now = time.monotonic()
        cutoff = now - self.window_seconds

        # Prune old attempts
        self._attempts[key] = [ts for ts in self._attempts[key] if ts > cutoff]

        if len(self._attempts[key]) >= self.max_requests:
            return False

        # Record this attempt
        self._attempts[key].append(now)
        return True

    def reset(self, key: str) -> None:
        """Clear all attempts for a key (call on successful login)."""
        self._attempts.pop(key, None)


# Module-level singleton for login rate limiting.
_login_limiter = RateLimiter(max_requests=5, window_seconds=60.0)


def get_login_limiter() -> RateLimiter:
    """Return the module-level login rate limiter singleton."""
    return _login_limiter
