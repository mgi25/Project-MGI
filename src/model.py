from __future__ import annotations

import numpy as np
from sklearn.linear_model import SGDClassifier


class EdgeModel:
    def __init__(self, classes=(0, 1)) -> None:
        self.clf = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-5,
            learning_rate="optimal",
        )
        self.is_init = False
        self.classes = np.array(classes)

    def features_to_vec(self, f: dict) -> np.ndarray:
        keys = [
            "r1",
            "r3",
            "r10",
            "ema_gap",
            "rsi",
            "atr_points",
            "spread_points",
            "vol_z",
        ]
        return np.array([float(f.get(k, 0.0)) for k in keys], dtype=np.float64).reshape(1, -1)

    def predict_proba(self, f: dict) -> tuple[float, float]:
        if not self.is_init:
            return 0.5, 0.5
        X = self.features_to_vec(f)
        p = self.clf.predict_proba(X)[0]
        return float(p[0]), float(p[1])

    def partial_fit(self, f: dict, label_up: int) -> None:
        X = self.features_to_vec(f)
        y = np.array([label_up], dtype=np.int32)
        if not self.is_init:
            self.clf.partial_fit(X, y, classes=self.classes)
            self.is_init = True
        else:
            self.clf.partial_fit(X, y)
