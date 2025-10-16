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
    def compute_volume(
        self,
        entry: float,
        sl: float,
        symbol_info: Optional[Any] = None,
        *,
        risk_multiplier: Optional[float] = None,
        risk_pct: Optional[float] = None,
        fixed_lots: Optional[float] = None,
    ) -> float:
        point = float(getattr(symbol_info, "point", self.point) or self.point)
        if point <= 0:
            return 0.0

        sl_points = abs(entry - sl) / point
        if sl_points <= 0:
            return 0.0

        volume_step = float(getattr(symbol_info, "volume_step", 0.01) or 0.01)
        volume_min = float(getattr(symbol_info, "volume_min", volume_step) or volume_step)
        volume_max_attr = float(getattr(symbol_info, "volume_max", 0.0) or 0.0)

        lot_step = max(volume_step, 0.01)

        if fixed_lots is not None and fixed_lots > 0:
            target = max(fixed_lots, 0.0)
            floored = math.floor(target / lot_step) * lot_step
            lots = max(0.0, round(floored, 2))
            volume_max = volume_max_attr if volume_max_attr > 0 else lots
            if lots < volume_min:
                logger.info("Blocked: lot {:.2f} below broker minimum {}", lots, volume_min)
                return 0.0
            if volume_max_attr > 0 and lots > volume_max:
                capped = math.floor(volume_max / lot_step) * lot_step
                capped = round(max(capped, 0.0), 2)
                if capped <= 0:
                    logger.info("Blocked: lot {:.2f} above broker max {}", lots, volume_max_attr)
                    return 0.0
                if capped < volume_min:
                    logger.info(
                        "Blocked: lot {:.2f} above broker max {} but below min {}",
                        lots,
                        volume_max_attr,
                        volume_min,
                    )
                    return 0.0
                lots = capped
            logger.debug("AI requested fixed lots {:.2f}; applied {:.2f}", target, lots)
            return lots

        contract_size = float(getattr(symbol_info, "trade_contract_size", 0.0) or self.contract_size)
        if contract_size <= 0:
            return 0.0

        equity = self.equity_fn()
        base_risk_pct = float(self.cfg["risk"].get("risk_per_trade", 0.0))
        effective_pct = max(base_risk_pct, 0.0)

        if risk_pct is not None and risk_pct >= 0:
            effective_pct = float(risk_pct)
        elif risk_multiplier is not None:
            effective_pct = max(base_risk_pct * float(risk_multiplier), 0.0)

        max_risk_pct = float(self.cfg["risk"].get("max_risk_pct", 0.0))
        if max_risk_pct > 0:
            effective_pct = min(effective_pct, max_risk_pct)

        risk_cash = equity * effective_pct
        if risk_cash <= 0:
            return 0.0

        raw_lots = risk_cash / (sl_points * contract_size * point)
        volume_max = volume_max_attr if volume_max_attr > 0 else raw_lots

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
