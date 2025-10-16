from __future__ import annotations

import os
import sys
import time
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import MetaTrader5 as mt5
import yaml
from dotenv import load_dotenv
from loguru import logger

from .ai_client import AIClient
from .execution import Executor
from .features import build_features
from .mt5_client import MT5Connection
from .risk import Risk
from .storage import Storage
from .strategy import Policy
from .utils import RateLimiter


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
        executor = Executor(mt5_conn, risk)
        storage = Storage(cfg["persistence"]["db_path"], cfg["persistence"]["csv_path"])
        stack.callback(storage.close)
        policy = Policy(cfg)
        limiter = RateLimiter(cfg["action_interval_seconds"])

        logger.info("Bot started for {}", symbol)

        try:
            while True:
                loop_start = time.time()

                bars = mt5_conn.get_bars(symbol, timeframe, cfg["lookback_bars"])
                if bars.empty:
                    time.sleep(1)
                    continue

                last_bar_ts = str(bars.iloc[-1]["time"])
                if not policy.new_bar(last_bar_ts):
                    time.sleep(1)
                    continue

                spread = mt5_conn.current_spread_points(symbol)
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

                features = build_features(bars, spread, digits, point, cfg)
                if not features:
                    time.sleep(1)
                    continue

                if not limiter.allow(now=loop_start):
                    time.sleep(1)
                    continue

                decision = ai.decide(features)
                storage.log_decision(features, decision)

                if decision:
                    try:
                        ticket = executor.place_from_decision(symbol, decision, digits)
                        if ticket:
                            policy.mark_order_sent(loop_start)
                    except Exception as exc:  # noqa: BLE001 - trading errors should not stop the loop
                        logger.error("Execution error: {}", exc)

                elapsed = time.time() - loop_start
                if elapsed < 1:
                    time.sleep(1 - elapsed)
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            logger.info("Bot stopped")


if __name__ == "__main__":
    run()
