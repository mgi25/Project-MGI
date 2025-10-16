from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import MetaTrader5 as mt5
from loguru import logger

from .risk import Risk


class Executor:
    def __init__(self, mt5, risk: Risk, cfg: Dict[str, Any], digits: int, point: float):
        self.mt5 = mt5
        self.risk = risk
        self.cfg = cfg
        self.digits = digits
        self.point = point
        self.guard = PriceGuard(point, digits, cfg)

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
            return ExecutionResult(False, None, "invalid_action", action, decision, None, None, None, 0.0, 0.0, 0.0, (0.0, 0.0), (0.0, 0.0))

        if action == "flat":
            logger.debug("LLM requested flat — no order")
            return ExecutionResult(False, None, "flat", action, decision, None, None, None, 0.0, 0.0, 0.0, (0.0, 0.0), (0.0, 0.0))

        if position_open:
            return ExecutionResult(False, None, "position_open", action, decision, None, None, None, 0.0, 0.0, 0.0, (0.0, 0.0), (0.0, 0.0))

        atr_points = float(features.get("m1", {}).get("atr_points", 0.0))
        tick = self.mt5.symbol_tick(symbol)
        clamp = self.guard.clamp(action, decision, atr_points, tick)
        if clamp is None:
            return ExecutionResult(False, None, "atr_unavailable", action, decision, None, None, None, 0.0, 0.0, 0.0, (0.0, 0.0), (0.0, 0.0))

        entry, sl, tp, sl_bounds, tp_bounds = clamp

        sl_points = abs(entry - sl) / self.point
        if not self.risk.sl_points_ok(sl_points):
            return ExecutionResult(
                False,
                None,
                "sl_bounds",
                action,
                decision,
                entry,
                sl,
                tp,
                0.0,
                self.risk.rr(entry, sl, tp),
                sl_points,
                sl_bounds,
                tp_bounds,
            )

        rr_value = self.risk.rr(entry, sl, tp)
        min_rr = float(self.cfg["risk"]["rr_min"])
        if rr_value < min_rr:
            tp = self._adjust_tp(action, entry, sl, min_rr, tp_bounds)
            rr_value = self.risk.rr(entry, sl, tp)
            if rr_value < min_rr:
                return ExecutionResult(
                    False,
                    None,
                    "rr_below",
                    action,
                    decision,
                    entry,
                    sl,
                    tp,
                    0.0,
                    rr_value,
                    sl_points,
                    sl_bounds,
                    tp_bounds,
                )

        lots = self.risk.compute_volume(sl_points)
        if lots <= 0:
            return ExecutionResult(
                False,
                None,
                "lot_size",
                action,
                decision,
                entry,
                sl,
                tp,
                0.0,
                rr_value,
                sl_points,
                sl_bounds,
                tp_bounds,
            )

        side = "BUY" if action == "buy" else "SELL"
        try:
            result = self.mt5.market_order(
                symbol,
                lots,
                side,
                round(sl, self.digits),
                round(tp, self.digits),
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
                entry,
                round(sl, self.digits),
                round(tp, self.digits),
                lots,
                rr_value,
                sl_points,
                sl_bounds,
                tp_bounds,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Order send failed: {}", exc)
            return ExecutionResult(
                False,
                None,
                f"order_error:{exc}",
                action,
                decision,
                entry,
                sl,
                tp,
                lots,
                rr_value,
                sl_points,
                sl_bounds,
                tp_bounds,
            )

    def manage_position(self, position, features: Dict[str, Any]) -> Optional[float]:
        m1 = features.get("m1", {})
        atr_points = float(m1.get("atr_points", 0.0))
        if atr_points <= 0:
            return None

        exits_cfg = self.cfg.get("exits", {})
        be_trigger = float(exits_cfg.get("be_trigger_points", 0.0))
        trail_min = float(exits_cfg.get("trail_min_points", 0.0))

        tick = self.mt5.symbol_tick(position.symbol)
        entry = float(position.price_open)
        current_sl = position.sl if position.sl not in (None, 0.0) else None
        trail_points = max(atr_points, trail_min)

        if position.type == mt5.POSITION_TYPE_BUY:
            current_price = tick.bid
            profit_points = (current_price - entry) / self.point
            new_sl = current_sl
            if be_trigger and profit_points >= be_trigger:
                be_level = entry
                if new_sl is None or be_level > new_sl + self.point * 0.5:
                    new_sl = be_level
            trail_level = max(entry, current_price - trail_points * self.point)
            if new_sl is None or trail_level > new_sl + self.point * 0.5:
                new_sl = trail_level
        else:
            current_price = tick.ask
            profit_points = (entry - current_price) / self.point
            new_sl = current_sl
            if be_trigger and profit_points >= be_trigger:
                be_level = entry
                if new_sl is None or be_level < new_sl - self.point * 0.5:
                    new_sl = be_level
            trail_level = min(entry, current_price + trail_points * self.point)
            if new_sl is None or trail_level < new_sl - self.point * 0.5:
                new_sl = trail_level

        if new_sl is None or (current_sl is not None and abs(new_sl - current_sl) < self.point * 0.5):
            return None

        new_sl = round(new_sl, self.digits)
        if self.mt5.modify_position(position.ticket, new_sl, position.tp):
            logger.info("Position {} SL updated to {}", position.ticket, new_sl)
            return new_sl

        logger.warning("Failed to update SL for position {}", position.ticket)
        return None

    def _adjust_tp(
        self,
        action: str,
        entry: float,
        sl: float,
        rr_target: float,
        tp_bounds: Tuple[float, float],
    ) -> float:
        risk = abs(entry - sl)
        desired_reward = rr_target * risk
        if desired_reward <= 0:
            return tp_bounds[0] if action == "sell" else tp_bounds[1]

        if action == "buy":
            desired_tp = entry + desired_reward
            return min(tp_bounds[1], max(desired_tp, tp_bounds[0]))

        desired_tp = entry - desired_reward
        return max(tp_bounds[0], min(desired_tp, tp_bounds[1]))


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
    sl_bounds: Optional[Tuple[float, float]]
    tp_bounds: Optional[Tuple[float, float]]


class PriceGuard:
    def __init__(self, point: float, digits: int, cfg: Dict[str, Any]):
        clamp_cfg = cfg.get("clamps", {})
        self.point = point
        self.digits = digits
        self.atr_mult = float(clamp_cfg.get("atr_mult", 3.0))
        self.min_points = float(clamp_cfg.get("min_points", 150))
        self.max_points = float(clamp_cfg.get("max_points", 1500))

    def clamp(
        self,
        action: str,
        decision: Dict[str, Any],
        atr_points: float,
        tick,
    ) -> Optional[Tuple[float, float, float, Tuple[float, float], Tuple[float, float]]]:
        if atr_points <= 0:
            return None

        max_dist_points = max(self.min_points, min(self.atr_mult * atr_points, self.max_points))
        half_atr = max(atr_points * 0.5, 1e-6)
        tp_min = max(atr_points * 0.8, 1e-6)

        entry_price = tick.ask if action == "buy" else tick.bid
        entry = round(entry_price, self.digits)
        raw_sl = float(decision.get("sl", entry))
        raw_tp = float(decision.get("tp", entry))

        if action == "buy":
            sl_bounds = (
                entry - max_dist_points * self.point,
                entry - half_atr * self.point,
            )
            tp_bounds = (
                entry + tp_min * self.point,
                entry + max_dist_points * self.point,
            )
        else:
            sl_bounds = (
                entry + half_atr * self.point,
                entry + max_dist_points * self.point,
            )
            tp_bounds = (
                entry - max_dist_points * self.point,
                entry - tp_min * self.point,
            )

        sl = max(sl_bounds[0], min(raw_sl, sl_bounds[1]))
        tp = max(tp_bounds[0], min(raw_tp, tp_bounds[1]))

        return entry, sl, tp, sl_bounds, tp_bounds
