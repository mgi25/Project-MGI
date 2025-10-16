from __future__ import annotations

from typing import Any, Dict, Mapping

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
) -> Dict[str, Any]:
    if len(df) < _min_required_bars(cfg):
        return {}

    df = df.copy()
    feature_cfg = cfg["features"]
    df["rsi"] = rsi(df["close"], feature_cfg["rsi_period"])
    df["atr"] = atr(df, feature_cfg["atr_period"])
    df["ema50"] = ema(df["close"], 50)

    last = df.iloc[-1]
    point_value = point or 10 ** -digits
    atr_points = float(last["atr"]) / point_value

    return {
        "time": str(last["time"]),
        "price": float(last["close"]),
        "atr_points": float(max(atr_points, 0.0)),
        "rsi": float(last["rsi"]),
        "ema50": float(last["ema50"]),
    }


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
        "m1": _build_m1_features(m1_bars, point, digits, cfg),
        "meta": {
            "digits": digits,
            "point": point,
            "spread_points": spread_points,
        },
    }

    if cfg.get("confirmations", {}).get("enabled", True):
        if cfg.get("confirmations", {}).get("m5_trend", False) and tf_m5 is not None:
            m5_bars = mt5c.get_bars(symbol, tf_m5, 120)
            features["m5"] = _build_higher_tf_features(m5_bars)
        if cfg.get("confirmations", {}).get("m15_context", False) and tf_m15 is not None:
            m15_bars = mt5c.get_bars(symbol, tf_m15, 120)
            features["m15"] = _build_higher_tf_features(m15_bars)

    return features
