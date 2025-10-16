from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class RateLimiter:
    """Simple time-based rate limiter.

    The limiter keeps track of the timestamp of the last granted call and
    ensures a minimum interval between subsequent calls. The implementation is
    intentionally lightweight to avoid external dependencies.
    """

    min_interval_sec: float
    _last_granted: float = field(default=0.0, init=False)

    def allow(self, *, now: Optional[float] = None) -> bool:
        now = time.time() if now is None else now
        if now - self._last_granted >= self.min_interval_sec:
            self._last_granted = now
            return True
        return False

    def reset(self) -> None:
        self._last_granted = 0.0


def now_utc_iso() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""

    return datetime.now(timezone.utc).isoformat()


