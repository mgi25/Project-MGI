from __future__ import annotations

import os
import sys
import time
from collections import deque
from contextlib import ExitStack
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Deque, Dict, Optional

import MetaTrader5 as mt5
import numpy as np
import yaml
from dotenv import load_dotenv
from loguru import logger

from .ai_client import AIClient
from .execution import ExecutionResult, Executor
from .features import build_all_features, build_management_state
from .mt5_client import MT5Connection
from .risk import Risk
from .storage import DecisionRecord, Storage
from .strategy import Policy, confirm_action, confirmations_ok
from .utils import RateLimiter, now_utc_iso
from src.model import EdgeModel


edge = EdgeModel()
loss_streak = 0
post_loss_until = 0.0
equity_ma: list[float] = []
pending_samples: Deque[Dict[str, Any]] = deque(maxlen=1000)
last_logged_bar: Optional[str] = None
active_trade: Optional[Dict[str, Any]] = None


def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def setup_logging(log_dir: str) -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(Path(log_dir) / "bot.log", rotation="10 MB")


def timeframe_from_cfg(cfg: Dict[str, Any]):
    tf_name = cfg["timeframe"].upper()
    try:
        return getattr(mt5, f"TIMEFRAME_{tf_name}")
    except AttributeError as exc:
        raise ValueError(f"Unsupported timeframe: {tf_name}") from exc


def run() -> None:
    global loss_streak, post_loss_until, equity_ma, pending_samples, last_logged_bar, active_trade
    load_dotenv()
    cfg = load_config()
    setup_logging(os.getenv("LOG_DIR", "logs"))

    symbol = cfg["symbol"]
    timeframe = timeframe_from_cfg(cfg)
    tf_m5 = getattr(mt5, "TIMEFRAME_M5", None)
    tf_m15 = getattr(mt5, "TIMEFRAME_M15", None)

    with ExitStack() as stack:
        mt5_conn = MT5Connection(
            terminal_path=os.getenv("MT5_TERMINAL_PATH"),
            login=os.getenv("MT5_LOGIN"),
            password=os.getenv("MT5_PASSWORD"),
            server=os.getenv("MT5_SERVER"),
        )
        stack.enter_context(mt5_conn)
        mt5_conn.ensure_symbol(symbol)
        digits, point = mt5_conn.digits_point(symbol)

        ai = AIClient(url=cfg["ai"]["url"], timeout=cfg["ai"]["timeout"], max_retries=cfg["ai"]["max_retries"])
        risk = Risk(cfg, digits, point, mt5_conn.account_equity)
        executor = Executor(mt5_conn, risk, cfg, digits, point)
        storage = Storage(cfg["persistence"]["db_path"], cfg["persistence"]["csv_path"])
        stack.callback(storage.close)
        policy = Policy(cfg)
        limiter = RateLimiter(cfg["action_interval_seconds"])

        logger.info("Bot started for {}", symbol)

        daily_date: Optional[date] = None
        daily_equity_start: float = 0.0
        kill_switch_active = False

        try:
            while True:
                loop_start = time.time()

                now_dt = datetime.now()
                if daily_date != now_dt.date():
                    daily_date = now_dt.date()
                    daily_equity_start = mt5_conn.account_equity()
                    kill_switch_active = False
                    logger.info("Daily equity reset: {:.2f}", daily_equity_start)

                features = build_all_features(
                    mt5_conn,
                    symbol,
                    timeframe,
                    tf_m5 if cfg.get("confirmations", {}).get("m5_trend", False) else None,
                    tf_m15 if cfg.get("confirmations", {}).get("m15_context", False) else None,
                    cfg,
                )
                if not features or not features.get("m1"):
                    time.sleep(1)
                    continue

                cfgE = cfg.get("expectancy", {})
                cfgR = cfg.get("risk_overlays", {})
                horizon = int(cfgE.get("horizon_bars", 5))

                m1_df = features.get("_m1_df")
                if m1_df is not None and hasattr(m1_df, "empty") and not m1_df.empty:
                    _process_pending_samples(m1_df, horizon)

                positions = mt5_conn.positions(symbol)
                if not positions and active_trade is not None:
                    _handle_trade_close(active_trade, cfgR)

                if positions:
                    active_trade = {
                        "ticket": getattr(positions[0], "ticket", None),
                        "opened": datetime.utcfromtimestamp(getattr(positions[0], "time", 0)),
                        "symbol": symbol,
                        "features": features["m1"].copy(),
                        "bar_time": m1_df.iloc[-1]["time"] if m1_df is not None and not m1_df.empty else None,
                    }
                    state = build_management_state(mt5_conn, positions[0], features)
                    if state:
                        ts_cfg = int(cfgR.get("time_stop_minutes", 0))
                        if ts_cfg > 0:
                            minutes_in_trade = state.get("time_in_trade_sec", 0.0) / 60.0
                            if minutes_in_trade >= ts_cfg:
                                ticket = getattr(positions[0], "ticket", None)
                                if ticket is not None and mt5_conn.close_position(ticket):
                                    logger.info("Time stop triggered for position {}", ticket)
                                time.sleep(1)
                                continue
                        executor.manage_with_ai(symbol, positions[0], state, ai)
                    time.sleep(1)
                    continue

                last_bar_ts = features["m1"].get("time")
                if last_bar_ts and not policy.new_bar(last_bar_ts):
                    time.sleep(1)
                    continue

                if last_bar_ts and last_bar_ts != last_logged_bar:
                    _queue_sample(features["m1"], last_bar_ts)
                    last_logged_bar = last_bar_ts

                spread = int(features.get("meta", {}).get("spread_points", 0))
                if not risk.within_spread(spread):
                    time.sleep(1)
                    continue

                now_time = datetime.now().time()
                if not policy.session_allowed(now_time):
                    time.sleep(1)
                    continue

                if not policy.cooldown_ready(loop_start):
                    time.sleep(1)
                    continue

                conf_ok, conf_reason = confirmations_ok(features, cfg)
                if not conf_ok:
                    logger.debug("Confirmations blocked: {}", conf_reason)
                    time.sleep(1)
                    continue

                equity = mt5_conn.account_equity()
                balance = mt5_conn.account_balance()
                free_margin = mt5_conn.account_free_margin()
                safety_cfg = cfg.get("safety", {})
                max_daily_loss = float(safety_cfg.get("daily_max_loss_pct", 100.0))
                if daily_equity_start > 0:
                    dd_pct = (equity - daily_equity_start) / daily_equity_start * 100.0
                    if dd_pct <= -max_daily_loss:
                        if not kill_switch_active:
                            logger.warning(
                                "Daily loss {:.2f}% beyond limit {:.2f}% — blocking new entries",
                                dd_pct,
                                max_daily_loss,
                            )
                            kill_switch_active = True
                        time.sleep(1)
                        continue
                else:
                    dd_pct = 0.0

                if time.time() < post_loss_until:
                    time.sleep(1)
                    continue

                if cfgR.get("equity_curve_filter", False):
                    equity_ma.append(equity)
                    period = int(cfgR.get("equity_ma_period", 0) or 0)
                    max_keep = max(period * 4, 200) if period > 0 else 200
                    if len(equity_ma) > max_keep:
                        del equity_ma[: len(equity_ma) - max_keep]
                    if period > 0 and len(equity_ma) >= period:
                        ma = float(np.mean(equity_ma[-period:]))
                        if equity < ma:
                            time.sleep(1)
                            continue

                feats = features["m1"]
                gate_enabled = bool(cfgE.get("enabled", True))
                if gate_enabled:
                    p_down, p_up = edge.predict_proba(feats)
                    want_long = p_up >= float(cfgE.get("threshold_long", 0.55))
                    want_short = p_down >= float(cfgE.get("threshold_short", 0.55))
                    min_ev = float(cfgE.get("min_expected_rr", 0.0))
                    edge_spread = max(p_up, p_down) - min(p_up, p_down)
                    if edge_spread <= min_ev:
                        want_long = False
                        want_short = False
                    if not (want_long or want_short):
                        time.sleep(1)
                        continue

                if not limiter.allow(now=loop_start):
                    time.sleep(1)
                    continue

                decision = ai.decide_entry(
                    _ai_payload(
                        features,
                        {
                            "equity": equity,
                            "balance": balance,
                            "free_margin": free_margin,
                        },
                        cfg.get("risk", {}),
                    )
                )
                MIN_CONF = 0.45  # start low to activate; tune to 0.55–0.60 later
                if decision and decision.get("action") in ("buy", "sell"):
                    try:
                        conf = float(decision.get("confidence", 0))
                    except (TypeError, ValueError):
                        conf = 0.0
                    if conf < MIN_CONF:
                        # downgrade to flat to stay safe
                        decision = {"action": "flat", "entry": None, "sl": None, "tp": None, "confidence": conf}
                decision_result: Optional[ExecutionResult] = None
                reason = ""

                if decision is None:
                    reason = "ai_error"
                else:
                    action = decision.get("action", "flat")
                    align_ok, align_reason = confirm_action(action, features, cfg)
                    if not align_ok and action != "flat":
                        reason = f"confirm:{align_reason}"
                        decision_result = ExecutionResult(
                            False,
                            None,
                            reason,
                            action,
                            decision,
                            None,
                            None,
                            None,
                            0.0,
                            0.0,
                            0.0,
                            None,
                            None,
                        )
                    else:
                        try:
                            decision_result = executor.place_from_decision(
                                symbol,
                                decision,
                                features,
                                position_open=False,
                            )
                            if decision_result.sent and decision_result.ticket:
                                policy.mark_order_sent(loop_start)
                        except Exception as exc:  # noqa: BLE001 - keep loop alive
                            logger.error("Execution error: {}", exc)
                            reason = f"execution_error:{exc}"

                record = _build_record(features, decision, decision_result, reason)
                storage.log_decision(record)
                _log_decision_line(record, spread)

                elapsed = time.time() - loop_start
                if elapsed < 1:
                    time.sleep(1 - elapsed)
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            logger.info("Bot stopped")


def _queue_sample(feats: Dict[str, Any], bar_time: str) -> None:
    if not bar_time:
        return
    sample = {
        "time": bar_time,
        "features": feats.copy(),
    }
    pending_samples.append(sample)


def _process_pending_samples(m1_df, horizon: int) -> None:
    if horizon <= 0 or m1_df is None or not hasattr(m1_df, "empty") or m1_df.empty:
        return
    try:
        times = m1_df["time"].astype(str)
        closes = m1_df["close"].to_numpy(dtype=float)
    except Exception:  # noqa: BLE001
        return

    new_queue: Deque[Dict[str, Any]] = deque(maxlen=pending_samples.maxlen)
    for sample in list(pending_samples):
        sample_time = sample.get("time")
        if not sample_time:
            continue
        mask = times == sample_time
        idx = np.flatnonzero(mask.to_numpy()) if hasattr(mask, "to_numpy") else np.array([], dtype=int)
        if idx.size == 0:
            continue
        entry_idx = int(idx[-1])
        future_idx = entry_idx + horizon
        if future_idx < len(closes):
            entry_close = closes[entry_idx]
            future_close = closes[future_idx]
            if entry_close > 0:
                ret_h = (future_close / entry_close) - 1.0
                label_up = 1 if ret_h > 0 else 0
                edge.partial_fit(sample["features"], label_up)
        else:
            new_queue.append(sample)

    pending_samples.clear()
    pending_samples.extend(new_queue)


def _handle_trade_close(trade_ctx: Dict[str, Any], cfgR: Dict[str, Any]) -> None:
    global active_trade, loss_streak, post_loss_until
    ticket = trade_ctx.get("ticket")
    pnl = _fetch_trade_pnl(ticket, trade_ctx.get("opened"))
    if pnl is not None:
        if pnl < 0:
            loss_streak += 1
            max_losses = int(cfgR.get("max_consecutive_losses", 0) or 0)
            if max_losses > 0 and loss_streak >= max_losses:
                cooldown = int(cfgR.get("post_loss_cooldown_sec", 0) or 0)
                if cooldown > 0:
                    post_loss_until = time.time() + cooldown
                loss_streak = 0
        else:
            loss_streak = 0
        logger.info("Trade {} closed with pnl {:.2f}", ticket, pnl)
    active_trade = None


def _fetch_trade_pnl(ticket: Optional[int], opened_time: Optional[datetime]) -> Optional[float]:
    if ticket is None:
        return None
    end = datetime.now()
    start = end - timedelta(days=2)
    if isinstance(opened_time, datetime):
        start = min(start, opened_time - timedelta(hours=1))
    try:
        deals = mt5.history_deals_get(start, end, position=ticket)
    except Exception:  # noqa: BLE001
        return None
    if not deals:
        return None
    pnl = 0.0
    for deal in deals:
        pnl += float(getattr(deal, "profit", 0.0) or 0.0)
    return pnl


def _ai_payload(
    features: Dict[str, Any],
    account: Dict[str, float],
    risk_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "m1": {
            "price": features["m1"].get("price"),
            "open": features["m1"].get("open"),
            "high": features["m1"].get("high"),
            "low": features["m1"].get("low"),
            "atr_points": features["m1"].get("atr_points"),
            "rsi": features["m1"].get("rsi"),
            "ema50": features["m1"].get("ema50"),
            "ema9": features["m1"].get("ema9"),
            "ema21": features["m1"].get("ema21"),
            "ema_gap": features["m1"].get("ema_gap"),
            "volume_rel": features["m1"].get("volume_rel"),
            "skew": features["m1"].get("skew"),
            "r1": features["m1"].get("r1"),
            "r3": features["m1"].get("r3"),
            "r10": features["m1"].get("r10"),
            "vol_z": features["m1"].get("vol_z"),
            "spread_points": features["m1"].get("spread_points"),
        },
        "spread_points": features.get("meta", {}).get("spread_points"),
        "meta": {
            "spread_points": features.get("meta", {}).get("spread_points"),
            "point": features.get("meta", {}).get("point"),
            "digits": features.get("meta", {}).get("digits"),
        },
        "account": {
            "equity": account.get("equity"),
            "balance": account.get("balance"),
            "free_margin": account.get("free_margin"),
        },
        "risk": {
            "base_risk_per_trade": risk_cfg.get("risk_per_trade"),
            "max_spread_points": risk_cfg.get("max_spread_points"),
            "contract_size": risk_cfg.get("contract_size"),
            "max_risk_pct": risk_cfg.get("max_risk_pct"),
        },
    }
    if "m5" in features:
        payload["m5"] = {
            "ema50": features["m5"].get("ema50"),
            "rsi": features["m5"].get("rsi"),
        }
    if "m15" in features:
        payload["m15"] = {
            "ema50": features["m15"].get("ema50"),
            "rsi": features["m15"].get("rsi"),
        }
    return payload


def _build_record(
    features: Dict[str, Any],
    decision: Optional[Dict[str, Any]],
    result: Optional[ExecutionResult],
    fallback_reason: str,
) -> DecisionRecord:
    decision = decision or {}
    m1 = features.get("m1", {})
    meta = features.get("meta", {})
    reason = fallback_reason or (result.reason if result else "")
    return DecisionRecord(
        ts=now_utc_iso(),
        price=m1.get("price"),
        atr_points=m1.get("atr_points"),
        spread_points=meta.get("spread_points"),
        action=decision.get("action"),
        raw_entry=decision.get("entry"),
        raw_sl=decision.get("sl"),
        raw_tp=decision.get("tp"),
        entry=result.entry if result else None,
        sl=result.sl if result else None,
        tp=result.tp if result else None,
        lots=result.lots if result else 0.0,
        rr=result.rr if result else 0.0,
        confidence=decision.get("confidence"),
        reason=reason,
        sl_bound_low=None,
        sl_bound_high=None,
        tp_bound_low=None,
        tp_bound_high=None,
    )


def _log_decision_line(
    record: DecisionRecord,
    spread_points: int,
) -> None:
    price = record.entry if record.entry is not None else record.price
    logger.info(
        "decision|ts={} action={} price={} sl={} tp={} spread_pts={} lots={} reason={}",
        record.ts,
        record.action,
        price,
        record.sl,
        record.tp,
        spread_points,
        record.lots,
        record.reason or "",
    )


if __name__ == "__main__":
    run()
