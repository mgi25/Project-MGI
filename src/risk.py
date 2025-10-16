from __future__ import annotations

import math
from typing import Any, Callable, Optional

from loguru import logger


class Risk:
    def __init__(self, cfg: dict, digits: int, point: float, equity_fn: Callable[[], float]):
        self.cfg = cfg
        self.digits = digits
        self.point = point
        self.equity_fn = equity_fn
        self.contract_size = float(cfg["risk"].get("contract_size", 100.0))

    def sl_min_points(self) -> int:
        return int(self.cfg.get("risk", {}).get("sl_min_points", 0))

    # -- spread ----------------------------------------------------------
    def within_spread(self, spread_points: int) -> bool:
        max_spread = self.cfg["risk"]["max_spread_points"]
        ok = spread_points <= max_spread
        if not ok:
            logger.info("Blocked: spread {} > max {}", spread_points, max_spread)
        return ok

    # -- volume ----------------------------------------------------------
    def compute_volume(self, entry: float, sl: float, symbol_info: Optional[Any] = None) -> float:
        equity = self.equity_fn()
        risk_cash = equity * float(self.cfg["risk"].get("risk_per_trade", 0.0))
        point = float(getattr(symbol_info, "point", self.point) or self.point)
        if point <= 0 or risk_cash <= 0:
            return 0.0

        sl_points = abs(entry - sl) / point
        if sl_points <= 0:
            return 0.0

        contract_size = float(getattr(symbol_info, "trade_contract_size", 0.0) or self.contract_size)
        if contract_size <= 0:
            return 0.0

        raw_lots = risk_cash / (sl_points * contract_size * point)
        volume_step = float(getattr(symbol_info, "volume_step", 0.01) or 0.01)
        volume_min = float(getattr(symbol_info, "volume_min", volume_step) or volume_step)
        volume_max = float(getattr(symbol_info, "volume_max", raw_lots) or raw_lots)

        lot_step = max(volume_step, 0.01)
        floored = math.floor(raw_lots / lot_step) * lot_step
        lots = max(0.0, round(floored, 2))

        if lots < volume_min:
            logger.info("Blocked: lot {:.2f} below broker minimum {}", lots, volume_min)
            return 0.0

        if lots > volume_max:
            capped = math.floor(volume_max / lot_step) * lot_step
            capped = round(max(capped, 0.0), 2)
            if capped <= 0:
                logger.info("Blocked: lot {:.2f} above broker max {}", lots, volume_max)
                return 0.0
            if capped < volume_min:
                logger.info(
                    "Blocked: lot {:.2f} above broker max {} but below min {}",
                    lots,
                    volume_max,
                    volume_min,
                )
                return 0.0
            lots = capped

        return lots
