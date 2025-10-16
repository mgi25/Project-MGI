from __future__ import annotations

from typing import Callable

from loguru import logger


class Risk:
    def __init__(self, cfg: dict, digits: int, point: float, equity_fn: Callable[[], float]):
        self.cfg = cfg
        self.digits = digits
        self.point = point
        self.equity_fn = equity_fn
        self.contract_size = float(cfg["risk"].get("contract_size", 100.0))

    # -- spread ----------------------------------------------------------
    def within_spread(self, spread_points: int) -> bool:
        max_spread = self.cfg["risk"]["max_spread_points"]
        ok = spread_points <= max_spread
        if not ok:
            logger.info("Blocked: spread {} > max {}", spread_points, max_spread)
        return ok

    # -- volume ----------------------------------------------------------
    def compute_volume(self, entry: float, sl: float) -> float:
        equity = self.equity_fn()
        risk_cash = equity * self.cfg["risk"]["risk_per_trade"]
        sl_points = max(int(abs(entry - sl) / self.point), self.cfg["risk"]["sl_min_points"])
        sl_points = min(sl_points, self.cfg["risk"]["sl_max_points"])
        if sl_points <= 0 or risk_cash <= 0:
            return 0.0

        tick_value_per_lot = self.contract_size * self.point
        lots = risk_cash / (sl_points * tick_value_per_lot)
        lots = max(0.01, round(lots, 2))
        return lots

    # -- rr --------------------------------------------------------------
    def rr_ok(self, entry: float, sl: float, tp: float) -> bool:
        risk = abs(entry - sl)
        reward = abs(tp - entry)
        if risk <= 0 or reward <= 0:
            logger.info("Blocked: invalid SL/TP distances")
            return False
        rr = reward / risk
        if rr < self.cfg["risk"]["rr_min"]:
            logger.info("Blocked: RR {:.2f} below threshold", rr)
            return False
        return True
