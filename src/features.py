from __future__ import annotations

from typing import Dict, Mapping

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


def build_features(
    df: pd.DataFrame,
    spread_points: int,
    digits: int,
    point: float,
    cfg: Mapping[str, Mapping[str, int]],
) -> Dict[str, float]:
    """Construct the feature payload passed to the LLM decision engine."""

    if len(df) < _min_required_bars(cfg):
        return {}

    df = df.copy()
    df["rsi"] = rsi(df["close"], cfg["features"]["rsi_period"])
    df["atr"] = atr(df, cfg["features"]["atr_period"])
    df["vspike"] = volume_spike(df["tick_volume"], cfg["features"]["vol_window"])
    df["skew"] = close_skew(df["close"], cfg["features"]["skew_window"])
    last = df.iloc[-1]

    feature_payload: Dict[str, float] = {
        "price": float(last["close"]),
        "rsi": float(last["rsi"]),
        # Convert ATR from price units into broker points using the precise point value.
        "atr_points": float(last["atr"]) / (point or 10 ** -digits),
        "volume_spike": float(last["vspike"]),
        "skew_z": float(last["skew"]),
        "spread_points": int(spread_points),
    }
    return feature_payload
