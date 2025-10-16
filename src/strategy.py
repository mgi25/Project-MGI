from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time
from typing import Any, Dict, Tuple


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


def confirmations_ok(features: Dict[str, Any], cfg: Dict[str, Any]) -> Tuple[bool, str]:
    conf_cfg = cfg.get("confirmations", {})
    if not conf_cfg.get("enabled", True):
        return True, ""

    m1 = features.get("m1", {})
    meta = features.get("meta", {})
    point = float(meta.get("point", 0.0) or 0.0)
    atr_points = m1.get("atr_points")
    if atr_points is None:
        return False, "atr_missing"

    atr_points = float(atr_points)
    atr_min, atr_max = conf_cfg.get("atr_m1_bounds", [0.0, float("inf")])
    if atr_points < float(atr_min) or atr_points > float(atr_max):
        return False, f"atr_bounds:{atr_points:.1f}"

    price = m1.get("price")
    if price is None:
        return False, "price_missing"

    if conf_cfg.get("m5_trend", False):
        m5 = features.get("m5", {})
        ema5 = m5.get("ema50")
        rsi5 = m5.get("rsi")
        if ema5 is None or rsi5 is None or point <= 0:
            return False, "m5_missing"
        diff_points = abs(float(price) - float(ema5)) / point
        if diff_points < 0.2 * atr_points and 45.0 <= float(rsi5) <= 55.0:
            return False, "m5_chop"

    if conf_cfg.get("m15_context", False):
        m15 = features.get("m15", {})
        rsi15 = m15.get("rsi")
        if rsi15 is None:
            return False, "m15_missing"
        if 45.0 <= float(rsi15) <= 55.0:
            return False, "m15_neutral"

    return True, ""


def confirm_action(action: str, features: Dict[str, Any], cfg: Dict[str, Any]) -> Tuple[bool, str]:
    conf_cfg = cfg.get("confirmations", {})
    if not conf_cfg.get("enabled", True):
        return True, ""

    action = action.lower()
    m1 = features.get("m1", {})
    price = m1.get("price")
    if price is None:
        return False, "price_missing"

    if conf_cfg.get("m5_trend", False):
        m5 = features.get("m5", {})
        ema5 = m5.get("ema50")
        rsi5 = m5.get("rsi")
        if ema5 is None or rsi5 is None:
            return False, "m5_missing"
        if action == "buy" and float(price) < float(ema5) and float(rsi5) < 50.0:
            return False, "m5_downtrend"
        if action == "sell" and float(price) > float(ema5) and float(rsi5) > 50.0:
            return False, "m5_uptrend"

    if conf_cfg.get("m15_context", False):
        m15 = features.get("m15", {})
        ema15 = m15.get("ema50")
        rsi15 = m15.get("rsi")
        if ema15 is None or rsi15 is None:
            return False, "m15_missing"
        if action == "buy" and float(price) < float(ema15) and float(rsi15) < 50.0:
            return False, "m15_bearish"
        if action == "sell" and float(price) > float(ema15) and float(rsi15) > 50.0:
            return False, "m15_bullish"

    return True, ""
