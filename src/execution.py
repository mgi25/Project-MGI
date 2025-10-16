from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import MetaTrader5 as mt5
from loguru import logger

from .risk import Risk
from .utils import now_utc_iso


class Executor:
    def __init__(self, mt5, risk: Risk, cfg: Dict[str, Any], digits: int, point: float):
        self.mt5 = mt5
        self.risk = risk
        self.cfg = cfg
        self.digits = digits
        self.point = point

    def place_from_decision(
        self,
        symbol: str,
        decision: Dict[str, Any],
        features: Dict[str, Any],
        *,
        position_open: bool,
    ) -> "ExecutionResult":
        action = str(decision.get("action", "flat")).lower()
        if action not in {"buy", "sell", "flat"}:
            return ExecutionResult(False, None, "invalid_action", action, decision, None, None, None, 0.0, 0.0, 0.0)

        if action == "flat":
            return ExecutionResult(False, None, "flat", action, decision, None, None, None, 0.0, 0.0, 0.0)

        if position_open:
            return ExecutionResult(False, None, "position_open", action, decision, None, None, None, 0.0, 0.0, 0.0)

        info = mt5.symbol_info(symbol)
        tick = mt5.symbol_info_tick(symbol)
        if info is None or tick is None:
            return ExecutionResult(False, None, "symbol_info", action, decision, None, None, None, 0.0, 0.0, 0.0)

        point = float(getattr(info, "point", self.point) or self.point)
        digits = int(getattr(info, "digits", self.digits) or self.digits)
        if point <= 0:
            return ExecutionResult(False, None, "invalid_point", action, decision, None, None, None, 0.0, 0.0, 0.0)

        market_price = float(tick.ask if action == "buy" else tick.bid)
        min_sl_points = max(self.risk.sl_min_points(), int(getattr(info, "trade_stops_level", 0) or 0))
        min_sl_dist = max(min_sl_points * point, point)

        intent_sl = self._to_optional_float(decision.get("sl"))
        intent_tp = self._to_optional_float(decision.get("tp"))

        sl = self._align_sl(action, market_price, intent_sl, min_sl_dist)
        tp = self._align_tp(action, market_price, intent_tp, point)

        rounded_sl = round(sl, digits)
        rounded_tp = round(tp, digits) if tp is not None else None

        lots = self.risk.compute_volume(market_price, rounded_sl, info)
        if lots <= 0:
            return ExecutionResult(False, None, "lot_size", action, decision, market_price, rounded_sl, rounded_tp, 0.0, 0.0, 0.0)

        rr_value = 0.0
        if rounded_tp is not None and rounded_sl is not None:
            risk = abs(market_price - rounded_sl)
            reward = abs(rounded_tp - market_price)
            if risk > 0 and reward > 0:
                rr_value = reward / risk

        sl_points = abs(market_price - rounded_sl) / point if rounded_sl is not None else 0.0

        side = "BUY" if action == "buy" else "SELL"
        try:
            result = self.mt5.market_order(
                symbol,
                lots,
                side,
                rounded_sl,
                rounded_tp,
                comment="gemma-3",
            )
            ticket = getattr(result, "order", None)
            logger.info("Order sent: {}", result)
            return ExecutionResult(
                True,
                ticket,
                "",
                action,
                decision,
                market_price,
                rounded_sl,
                rounded_tp,
                lots,
                rr_value,
                sl_points,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Order send failed: {}", exc)
            return ExecutionResult(
                False,
                None,
                f"order_error:{exc}",
                action,
                decision,
                market_price,
                rounded_sl,
                rounded_tp,
                lots,
                rr_value,
                sl_points,
            )

    def manage_with_ai(self, symbol: str, position, state: Dict[str, Any], ai) -> Optional[str]:
        response = ai.manage_position(state)
        ticket = getattr(position, "ticket", None)
        ts = now_utc_iso()

        if ticket is None:
            logger.warning("Position missing ticket for management")
            return None

        if not response:
            logger.info("mng|ts={} ticket={} decision={} sl={} tp={}", ts, ticket, "skip", state.get("sl"), state.get("tp"))
            return None

        decision = str(response.get("decision", "")).lower()
        if decision not in {"hold", "close", "update"}:
            logger.info("mng|ts={} ticket={} decision={} sl={} tp={}", ts, ticket, "invalid", state.get("sl"), state.get("tp"))
            return None

        if decision == "hold":
            logger.info("mng|ts={} ticket={} decision={} sl={} tp={}", ts, ticket, "hold", state.get("sl"), state.get("tp"))
            return "hold"

        info = mt5.symbol_info(symbol)
        tick = mt5.symbol_info_tick(symbol)
        if info is None or tick is None:
            logger.warning("Management missing symbol info for {}", symbol)
            return None

        point = float(getattr(info, "point", self.point) or self.point)
        digits = int(getattr(info, "digits", self.digits) or self.digits)
        min_sl_points = max(self.risk.sl_min_points(), int(getattr(info, "trade_stops_level", 0) or 0))
        min_sl_dist = max(min_sl_points * point, point)
        current_price = float(tick.bid if position.type == mt5.POSITION_TYPE_BUY else tick.ask)
        side = "buy" if position.type == mt5.POSITION_TYPE_BUY else "sell"

        applied_sl = state.get("sl")
        applied_tp = state.get("tp")

        if decision == "close":
            success = self.mt5.close_position(ticket)
            logger.info("mng|ts={} ticket={} decision={} sl={} tp={}", ts, ticket, "close", applied_sl, applied_tp)
            if not success:
                logger.warning("Failed to close position {}", ticket)
                return None
            return "close"

        # decision == update
        new_sl = response.get("sl")
        new_tp = response.get("tp")

        target_sl = None
        if new_sl is not None:
            target_sl = self._align_sl(side, current_price, float(new_sl), min_sl_dist)
            target_sl = round(target_sl, digits)
            current_sl = position.sl if position.sl not in (None, 0.0) else None
            if current_sl is not None and abs(target_sl - current_sl) < point * 0.5:
                target_sl = None
            else:
                applied_sl = target_sl

        target_tp = None
        if new_tp is not None:
            target_tp = self._align_tp(side, current_price, float(new_tp), point)
            target_tp = round(target_tp, digits)
            current_tp = position.tp if position.tp not in (None, 0.0) else None
            if current_tp is not None and abs(target_tp - current_tp) < point * 0.5:
                target_tp = None
            else:
                applied_tp = target_tp

        if target_sl is None and target_tp is None:
            logger.info("mng|ts={} ticket={} decision={} sl={} tp={}", ts, ticket, "hold", applied_sl, applied_tp)
            return None

        success = self.mt5.modify_position(
            ticket,
            target_sl if target_sl is not None else position.sl,
            target_tp if target_tp is not None else position.tp,
        )
        logger.info("mng|ts={} ticket={} decision={} sl={} tp={}", ts, ticket, "update", applied_sl, applied_tp)
        if not success:
            logger.warning("Failed to modify position {}", ticket)
            return None
        return "update"

    @staticmethod
    def _to_optional_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _align_sl(self, action: str, price: float, proposed: Optional[float], min_offset: float) -> float:
        base_offset = max(min_offset, self.point)
        if action == "buy":
            target = price - base_offset if proposed is None else float(proposed)
            if target >= price:
                target = price - base_offset
            if price - target < base_offset:
                target = price - base_offset
            return max(target, 0.0)

        target = price + base_offset if proposed is None else float(proposed)
        if target <= price:
            target = price + base_offset
        if target - price < base_offset:
            target = price + base_offset
        return target

    def _align_tp(self, action: str, price: float, proposed: Optional[float], min_offset: float) -> Optional[float]:
        if proposed is None:
            if action == "buy":
                return price + max(min_offset, self.point)
            return price - max(min_offset, self.point)

        target = float(proposed)
        if action == "buy" and target <= price:
            return price + max(min_offset, self.point)
        if action == "sell" and target >= price:
            return price - max(min_offset, self.point)
        return target


@dataclass
class ExecutionResult:
    sent: bool
    ticket: Optional[int]
    reason: str
    action: str
    raw_decision: Dict[str, Any]
    entry: Optional[float]
    sl: Optional[float]
    tp: Optional[float]
    lots: float
    rr: float
    sl_points: float
    sl_bounds: Optional[Tuple[float, float]] = None
    tp_bounds: Optional[Tuple[float, float]] = None
