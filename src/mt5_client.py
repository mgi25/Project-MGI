from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import MetaTrader5 as mt5
import pandas as pd
from loguru import logger


@dataclass
class MT5Connection(AbstractContextManager["MT5Connection"]):
    """Context manager around the MetaTrader5 module lifecycle."""

    terminal_path: Optional[str]
    login: Optional[int]
    password: Optional[str]
    server: Optional[str]

    def __post_init__(self) -> None:
        if self.login is not None and not isinstance(self.login, int):
            self.login = int(self.login)

    # -- context manager -------------------------------------------------
    def __enter__(self) -> "MT5Connection":
        self.initialize()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        self.shutdown()

    # -- public API ------------------------------------------------------
    def initialize(self) -> None:
        kwargs = {}
        if self.terminal_path:
            kwargs["path"] = self.terminal_path
        if not mt5.initialize(**kwargs):
            raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
        if self.login and self.server and self.password:
            if not mt5.login(self.login, password=self.password, server=self.server):
                raise RuntimeError(f"MT5 login failed: {mt5.last_error()}")
        logger.info("MT5 initialized")

    def shutdown(self) -> None:
        mt5.shutdown()

    # -- symbol helpers --------------------------------------------------
    def ensure_symbol(self, symbol: str) -> None:
        info = mt5.symbol_info(symbol)
        if info is None:
            raise ValueError(f"Symbol not found: {symbol}")
        if not info.visible and not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"Failed to select symbol {symbol}")

    def digits_point(self, symbol: str) -> tuple[int, float]:
        info = mt5.symbol_info(symbol)
        if info is None:
            raise ValueError(f"Symbol not found: {symbol}")
        return info.digits, info.point

    # -- market data -----------------------------------------------------
    def get_ticks(self, symbol: str, n: int = 2000) -> pd.DataFrame:
        ticks = mt5.copy_ticks_from(symbol, datetime.now(), n, mt5.COPY_TICKS_ALL)
        df = pd.DataFrame(ticks)
        if not df.empty:
            df["time"] = pd.to_datetime(df["time"], unit="s")
        return df

    def get_bars(self, symbol: str, timeframe, n: int) -> pd.DataFrame:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n)
        df = pd.DataFrame(rates)
        if not df.empty:
            df["time"] = pd.to_datetime(df["time"], unit="s")
        return df

    # -- orders ----------------------------------------------------------
    def _symbol_tick(self, symbol: str):
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise RuntimeError(f"Failed to fetch tick for {symbol}")
        return tick

    def market_order(
        self,
        symbol: str,
        volume: float,
        order_type: str,
        sl: Optional[float],
        tp: Optional[float],
        comment: str = "",
    ):
        tick = self._symbol_tick(symbol)
        price = tick.ask if order_type == "BUY" else tick.bid
        type_map = {"BUY": mt5.ORDER_TYPE_BUY, "SELL": mt5.ORDER_TYPE_SELL}
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": type_map[order_type],
            "price": price,
            "sl": sl or 0.0,
            "tp": tp or 0.0,
            "deviation": 30,
            "magic": 20251016,
            "comment": comment,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            raise RuntimeError(f"Order failed: {result}")
        return result

    def modify_position(self, ticket: int, sl: Optional[float], tp: Optional[float]) -> bool:
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            return False
        pos = positions[0]
        tick = self._symbol_tick(pos.symbol)
        price = tick.ask if pos.type == mt5.POSITION_TYPE_BUY else tick.bid
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "sl": sl or 0.0,
            "tp": tp or 0.0,
            "price": price,
        }
        r = mt5.order_send(request)
        return bool(r and r.retcode == mt5.TRADE_RETCODE_DONE)

    def close_position(self, ticket: int) -> bool:
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            return False
        pos = positions[0]
        tick = self._symbol_tick(pos.symbol)
        price = tick.bid if pos.type == mt5.POSITION_TYPE_BUY else tick.ask
        order_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": ticket,
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": order_type,
            "price": price,
            "deviation": 30,
            "magic": 20251016,
            "comment": "gemma-3-close",
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        return bool(result and result.retcode == mt5.TRADE_RETCODE_DONE)

    def current_spread_points(self, symbol: str) -> int:
        info = mt5.symbol_info(symbol)
        tick = mt5.symbol_info_tick(symbol)
        if not info or not tick:
            raise RuntimeError(f"Failed to retrieve spread for {symbol}")
        spread_points = int(round((tick.ask - tick.bid) / info.point))
        return spread_points

    def account_equity(self) -> float:
        acc = mt5.account_info()
        return float(acc.equity) if acc else 0.0

    def account_balance(self) -> float:
        acc = mt5.account_info()
        return float(acc.balance) if acc else 0.0

    def account_free_margin(self) -> float:
        acc = mt5.account_info()
        return float(acc.margin_free) if acc else 0.0

    def symbol_tick(self, symbol: str):
        return self._symbol_tick(symbol)

    def positions(self, symbol: Optional[str] = None):
        if symbol:
            positions = mt5.positions_get(symbol=symbol)
        else:
            positions = mt5.positions_get()
        return positions or []

    def has_open_position(self, symbol: str) -> bool:
        return bool(self.positions(symbol))
