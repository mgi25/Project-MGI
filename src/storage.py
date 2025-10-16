from __future__ import annotations

import csv
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class Storage:
    db_path: str
    csv_path: str

    def __post_init__(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.csv_path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.db_path)
        self.db.row_factory = sqlite3.Row
        self.columns: Tuple[Tuple[str, str], ...] = (
            ("ts", "TEXT"),
            ("price", "REAL"),
            ("atr_points", "REAL"),
            ("spread_points", "INTEGER"),
            ("action", "TEXT"),
            ("raw_entry", "REAL"),
            ("raw_sl", "REAL"),
            ("raw_tp", "REAL"),
            ("entry", "REAL"),
            ("sl", "REAL"),
            ("tp", "REAL"),
            ("lots", "REAL"),
            ("rr", "REAL"),
            ("confidence", "REAL"),
            ("reason", "TEXT"),
            ("sl_bound_low", "REAL"),
            ("sl_bound_high", "REAL"),
            ("tp_bound_low", "REAL"),
            ("tp_bound_high", "REAL"),
        )
        cols_sql = ",".join(f"{name} {col_type}" for name, col_type in self.columns)
        self.db.execute(f"CREATE TABLE IF NOT EXISTS decisions({cols_sql})")
        self.db.commit()
        self._ensure_columns()
        self._csv_header_written = Path(self.csv_path).exists()

    def close(self) -> None:
        try:
            self.db.close()
        except Exception:
            pass

    def log_decision(self, record: "DecisionRecord") -> None:
        row = tuple(getattr(record, name) for name, _ in self.columns)
        placeholders = ",".join("?" for _ in self.columns)
        self.db.execute(f"INSERT INTO decisions VALUES ({placeholders})", row)
        self.db.commit()

        with open(self.csv_path, "a", newline="") as handle:
            writer = csv.writer(handle)
            if not self._csv_header_written:
                headers = [name for name, _ in self.columns]
                writer.writerow(headers)
                self._csv_header_written = True
            writer.writerow(row)

    def __del__(self) -> None:  # pragma: no cover - defensive close
        self.close()

    def _ensure_columns(self) -> None:
        existing = {row["name"] for row in self.db.execute("PRAGMA table_info(decisions)")}
        for name, col_type in self.columns:
            if name not in existing:
                self.db.execute(f"ALTER TABLE decisions ADD COLUMN {name} {col_type}")
        self.db.commit()


@dataclass
class DecisionRecord:
    ts: str
    price: Optional[float]
    atr_points: Optional[float]
    spread_points: Optional[int]
    action: Optional[str]
    raw_entry: Optional[float]
    raw_sl: Optional[float]
    raw_tp: Optional[float]
    entry: Optional[float]
    sl: Optional[float]
    tp: Optional[float]
    lots: Optional[float]
    rr: Optional[float]
    confidence: Optional[float]
    reason: Optional[str]
    sl_bound_low: Optional[float]
    sl_bound_high: Optional[float]
    tp_bound_low: Optional[float]
    tp_bound_high: Optional[float]
