from __future__ import annotations

from typing import Optional

from loguru import logger

from .risk import Risk


class Executor:
    def __init__(self, mt5, risk: Risk):
        self.mt5 = mt5
        self.risk = risk

    def place_from_decision(self, symbol: str, decision: dict, digits: int) -> Optional[int]:
        action = decision.get("action", "flat").lower()
        if action == "flat":
            logger.debug("LLM requested flat — no order")
            return None

        try:
            entry = float(decision["entry"])
            sl = float(decision["sl"])
            tp = float(decision["tp"])
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("Decision payload missing price fields: {}", exc)
            return None

        if not self.risk.rr_ok(entry, sl, tp):
            return None

        volume = self.risk.compute_volume(entry, sl)
        if volume <= 0:
            logger.info("Blocked: computed volume <= 0")
            return None

        side = "BUY" if action == "buy" else "SELL"
        result = self.mt5.market_order(symbol, volume, side, sl, tp, comment="gemma-3")
        logger.info("Order sent: {}", result)
        return getattr(result, "order", None)
