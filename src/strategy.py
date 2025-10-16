from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time


@dataclass
class Policy:
    cfg: dict
    _last_bar_time: str | None = field(default=None, init=False)
    _last_order_ts: float = field(default=0.0, init=False)

    @property
    def cooldown_seconds(self) -> float:
        return float(self.cfg.get("cooldowns", {}).get("after_order_seconds", 0))

    def new_bar(self, last_bar_ts: str) -> bool:
        if not self.cfg.get("new_bar_only", True):
            return True
        if self._last_bar_time != last_bar_ts:
            self._last_bar_time = last_bar_ts
            return True
        return False

    def session_allowed(self, now_time: time) -> bool:
        sessions = self.cfg.get("sessions", {})
        if not sessions or not sessions.get("enabled", False):
            return True

        minutes = now_time.hour * 60 + now_time.minute
        for window in sessions.get("windows", []):
            start_s, end_s = window.split("-")
            start = _parse_minutes(start_s)
            end = _parse_minutes(end_s)
            if start <= end:
                if start <= minutes <= end:
                    return True
            else:  # overnight window (e.g. 22:00-06:00)
                if minutes >= start or minutes <= end:
                    return True
        return False

    def cooldown_ready(self, now_ts: float) -> bool:
        if not self.cooldown_seconds:
            return True
        return (now_ts - self._last_order_ts) >= self.cooldown_seconds

    def mark_order_sent(self, now_ts: float) -> None:
        self._last_order_ts = now_ts


def _parse_minutes(hhmm: str) -> int:
    hour, minute = map(int, hhmm.split(":"))
    return hour * 60 + minute
