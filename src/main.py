from __future__ import annotations

import os
import sys
import time
from contextlib import ExitStack
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Optional

import MetaTrader5 as mt5
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

                positions = mt5_conn.positions(symbol)
                if positions:
                    state = build_management_state(mt5_conn, positions[0], features)
                    if state:
                        executor.manage_with_ai(symbol, positions[0], state, ai)
                    time.sleep(1)
                    continue

                last_bar_ts = features["m1"].get("time")
                if last_bar_ts and not policy.new_bar(last_bar_ts):
                    time.sleep(1)
                    continue

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

                if not limiter.allow(now=loop_start):
                    time.sleep(1)
                    continue

                decision = ai.decide_entry(_ai_payload(features))
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


def _ai_payload(features: Dict[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "m1": {
            "price": features["m1"].get("price"),
            "open": features["m1"].get("open"),
            "high": features["m1"].get("high"),
            "low": features["m1"].get("low"),
            "atr_points": features["m1"].get("atr_points"),
            "rsi": features["m1"].get("rsi"),
            "ema50": features["m1"].get("ema50"),
            "volume_rel": features["m1"].get("volume_rel"),
            "skew": features["m1"].get("skew"),
        },
        "spread_points": features.get("meta", {}).get("spread_points"),
        "meta": {
            "spread_points": features.get("meta", {}).get("spread_points"),
            "point": features.get("meta", {}).get("point"),
            "digits": features.get("meta", {}).get("digits"),
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
