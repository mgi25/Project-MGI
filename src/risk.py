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
    def compute_volume(self, sl_points: float) -> float:
        equity = self.equity_fn()
        risk_cash = equity * self.cfg["risk"]["risk_per_trade"]
        if sl_points <= 0 or risk_cash <= 0:
            return 0.0

        usd_per_point_per_lot = self.contract_size * self.point
        lots = risk_cash / (sl_points * usd_per_point_per_lot)
        lot_step = float(self.cfg["risk"].get("lot_step", 0.01))
        lots = max(lot_step, round(lots / lot_step) * lot_step)
        return round(lots, 2)

    def sl_points_ok(self, sl_points: float) -> bool:
        min_sl = float(self.cfg["risk"]["sl_min_points"])
        max_sl = float(self.cfg["risk"]["sl_max_points"])
        if sl_points < min_sl:
            logger.info("Blocked: SL distance {:.1f} < min {}", sl_points, min_sl)
            return False
        if sl_points > max_sl:
            logger.info("Blocked: SL distance {:.1f} > max {}", sl_points, max_sl)
            return False
        return True

    # -- rr --------------------------------------------------------------
    def rr(self, entry: float, sl: float, tp: float) -> float:
        risk = abs(entry - sl)
        reward = abs(tp - entry)
        if risk <= 0 or reward <= 0:
            return 0.0
        return reward / risk

    def rr_ok(self, entry: float, sl: float, tp: float) -> bool:
        ratio = self.rr(entry, sl, tp)
        if ratio <= 0:
            logger.info("Blocked: invalid SL/TP distances")
            return False
        if ratio < self.cfg["risk"]["rr_min"]:
            logger.info("Blocked: RR {:.2f} below threshold", ratio)
            return False
        return True
