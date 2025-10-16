from __future__ import annotations

import csv
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


@dataclass
class Storage:
    db_path: str
    csv_path: str

    def __post_init__(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.csv_path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.db_path)
        self.db.row_factory = sqlite3.Row
        self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS decisions(
                ts TEXT,
                price REAL,
                rsi REAL,
                atr_points REAL,
                volume_spike REAL,
                skew_z REAL,
                spread_points INTEGER,
                action TEXT,
                entry REAL,
                sl REAL,
                tp REAL,
                confidence REAL
            )
            """
        )
        self.db.commit()
        self._csv_header_written = Path(self.csv_path).exists()

    def close(self) -> None:
        try:
            self.db.close()
        except Exception:
            pass

    def log_decision(self, features: Dict[str, float], decision: Optional[Dict[str, float]]) -> None:
        decision = decision or {}
        row = (
            datetime.utcnow().isoformat(),
            features.get("price"),
            features.get("rsi"),
            features.get("atr_points"),
            features.get("volume_spike"),
            features.get("skew_z"),
            features.get("spread_points"),
            decision.get("action"),
            decision.get("entry"),
            decision.get("sl"),
            decision.get("tp"),
            decision.get("confidence"),
        )
        self.db.execute("INSERT INTO decisions VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", row)
        self.db.commit()

        with open(self.csv_path, "a", newline="") as handle:
            writer = csv.writer(handle)
            if not self._csv_header_written:
                headers = [col[1] for col in self.db.execute("PRAGMA table_info(decisions)")]
                writer.writerow(headers)
                self._csv_header_written = True
            writer.writerow(row)

    def __del__(self) -> None:  # pragma: no cover - defensive close
        self.close()
