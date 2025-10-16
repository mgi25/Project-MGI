from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Mapping

import MetaTrader5 as mt5
import numpy as np
import pandas as pd


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Return a smoothed RSI calculation."""

    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).ewm(alpha=1 / period, adjust=False).mean()
    roll_down = pd.Series(down, index=series.index).ewm(alpha=1 / period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-9)
    return 100.0 - (100.0 / (1.0 + rs))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average true range using an EMA for smoothing."""

    high_low = (df["high"] - df["low"]).abs()
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def volume_spike(vol: pd.Series, window: int = 20) -> pd.Series:
    """Volume relative to the rolling mean."""

    ma = vol.rolling(window).mean()
    return (vol / (ma + 1e-9)).fillna(1.0)


def close_skew(close: pd.Series, window: int = 30) -> pd.Series:
    """Z-score of the close over a rolling window."""

    rolling = close.rolling(window)
    return (close - rolling.mean()) / (rolling.std() + 1e-9)


def _min_required_bars(cfg: Mapping[str, Mapping[str, int]]) -> int:
    feature_cfg = cfg["features"]
    return (
        max(
            feature_cfg["atr_period"],
            feature_cfg["rsi_period"],
            feature_cfg["vol_window"],
            feature_cfg["skew_window"],
        )
        + 5
    )


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _build_m1_features(
    df: pd.DataFrame,
    point: float,
    digits: int,
    cfg: Mapping[str, Mapping[str, int]],
    spread_points: float,
) -> Dict[str, Any]:
    if len(df) < _min_required_bars(cfg):
        return {}

    df = df.copy()
    feature_cfg = cfg["features"]
    df["rsi"] = rsi(df["close"], feature_cfg["rsi_period"])
    df["atr"] = atr(df, feature_cfg["atr_period"])
    df["ema50"] = ema(df["close"], 50)
    df["volume_rel"] = volume_spike(df["tick_volume"], feature_cfg["vol_window"])
    df["skew"] = close_skew(df["close"], feature_cfg["skew_window"])
    # EMAs and short momentum
    df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()
    df["r1"] = df["close"].pct_change(1)
    df["r3"] = df["close"].pct_change(3)
    df["r10"] = df["close"].pct_change(10)
    if "tick_volume" in df.columns:
        df["vol_sma"] = df["tick_volume"].rolling(20).mean()
        df["vol_std"] = df["tick_volume"].rolling(20).std()
        df["vol_z"] = (df["tick_volume"] - df["vol_sma"]) / (df["vol_std"] + 1e-9)
    else:
        df["vol_sma"] = 0.0
        df["vol_std"] = 0.0
        df["vol_z"] = 0.0
    df["volume_rel"] = df["volume_rel"].replace([np.inf, -np.inf], np.nan).fillna(1.0)
    df["skew"] = df["skew"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df[["ema9", "ema21", "r1", "r3", "r10", "vol_sma", "vol_std", "vol_z"]] = df[
        ["ema9", "ema21", "r1", "r3", "r10", "vol_sma", "vol_std", "vol_z"]
    ].replace([np.inf, -np.inf], np.nan)
    df[["ema9", "ema21"]] = df[["ema9", "ema21"]].bfill().ffill()
    df[["r1", "r3", "r10"]] = df[["r1", "r3", "r10"]].fillna(0.0)
    df["vol_z"] = df["vol_z"].fillna(0.0)

    last = df.iloc[-1]
    point_value = point or 10 ** -digits
    atr_points = float(last["atr"]) / point_value

    features_out = {
        "time": str(last["time"]),
        "price": float(last["close"]),
        "open": float(last["open"]),
        "high": float(last["high"]),
        "low": float(last["low"]),
        "atr_points": float(max(atr_points, 0.0)),
        "rsi": float(last["rsi"]),
        "ema50": float(last["ema50"]),
        "volume_rel": float(last["volume_rel"]),
        "skew": float(last["skew"]),
        "spread_points": float(spread_points or 0.0),
    }
    features_out.update(
        {
            "ema9": float(last["ema9"]),
            "ema21": float(last["ema21"]),
            "ema_gap": float(last["ema9"] - last["ema21"]),
            "r1": float(last["r1"]),
            "r3": float(last["r3"]),
            "r10": float(last["r10"]),
            "vol_z": float(last["vol_z"]),
        }
    )
    return features_out


def _build_higher_tf_features(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {}

    df = df.copy()
    df["rsi"] = rsi(df["close"], 14)
    df["ema50"] = ema(df["close"], 50)
    last = df.iloc[-1]
    return {
        "time": str(last["time"]),
        "ema50": float(last["ema50"]),
        "rsi": float(last["rsi"]),
    }


def build_all_features(
    mt5c,
    symbol: str,
    tf_m1,
    tf_m5,
    tf_m15,
    cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    """Fetch market data across timeframes and return engineered features."""

    digits, point = mt5c.digits_point(symbol)
    spread_points = mt5c.current_spread_points(symbol)
    m1_bars = mt5c.get_bars(symbol, tf_m1, cfg.get("lookback_bars", 200))
    if m1_bars.empty:
        return {}

    features: Dict[str, Any] = {
        "m1": _build_m1_features(m1_bars, point, digits, cfg, spread_points),
        "meta": {
            "digits": digits,
            "point": point,
            "spread_points": spread_points,
        },
        "_m1_df": m1_bars,
    }

    if cfg.get("confirmations", {}).get("enabled", True):
        if cfg.get("confirmations", {}).get("m5_trend", False) and tf_m5 is not None:
            m5_bars = mt5c.get_bars(symbol, tf_m5, 120)
            features["m5"] = _build_higher_tf_features(m5_bars)
        if cfg.get("confirmations", {}).get("m15_context", False) and tf_m15 is not None:
            m15_bars = mt5c.get_bars(symbol, tf_m15, 120)
            features["m15"] = _build_higher_tf_features(m15_bars)

    return features


def build_management_state(mt5c, position, features: Mapping[str, Any]) -> Dict[str, Any]:
    try:
        tick = mt5c.symbol_tick(position.symbol)
    except Exception:  # noqa: BLE001
        return {}

    now = datetime.now(timezone.utc)
    opened = datetime.fromtimestamp(getattr(position, "time", 0), tz=timezone.utc)
    time_in_trade = max((now - opened).total_seconds(), 0.0)

    side = "buy" if position.type == mt5.POSITION_TYPE_BUY else "sell"
    sl_value = position.sl if position.sl not in (None, 0.0) else None
    tp_value = position.tp if position.tp not in (None, 0.0) else None

    m1 = features.get("m1", {})
    meta = features.get("meta", {})

    state: Dict[str, Any] = {
        "ticket": getattr(position, "ticket", None),
        "side": side,
        "open_price": float(position.price_open),
        "volume": float(position.volume),
        "current_bid": float(tick.bid),
        "current_ask": float(tick.ask),
        "sl": float(sl_value) if sl_value is not None else None,
        "tp": float(tp_value) if tp_value is not None else None,
        "unrealized_pnl": float(getattr(position, "profit", 0.0)),
        "spread_points": int(meta.get("spread_points", 0) or 0),
        "time_in_trade_sec": time_in_trade,
    }

    if m1:
        state["atr_points"] = float(m1.get("atr_points", 0.0) or 0.0)
        state["rsi"] = float(m1.get("rsi", 0.0) or 0.0)
        state["price"] = float(m1.get("price", 0.0) or 0.0)

    return state
