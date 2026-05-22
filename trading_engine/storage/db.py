from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
from typing import Any, List

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id    TEXT    NOT NULL UNIQUE,
    asset            TEXT    NOT NULL,
    timeframe        TEXT    NOT NULL,
    candle_open_time TEXT    NOT NULL,
    candle_close_time TEXT   NOT NULL,
    direction        TEXT    NOT NULL,
    final_score      REAL    NOT NULL,
    confidence       REAL    NOT NULL,
    regime           TEXT    NOT NULL,
    block_reason     TEXT    NOT NULL,
    is_actionable    INTEGER NOT NULL,
    mc_bias          REAL,
    ml_prob          REAL,
    drift_signal     REAL,
    momentum_signal  REAL,
    ma20             REAL,
    ma50             REAL,
    volatility       REAL,
    vol_zscore       REAL,
    created_at       TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS outcomes (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id    TEXT    NOT NULL REFERENCES predictions(prediction_id),
    actual_direction TEXT    NOT NULL,
    was_correct      INTEGER NOT NULL,
    resolved_at      TEXT    NOT NULL
);
"""


class Database:
    def __init__(self, db_path: str) -> None:
        self._path = db_path
        self._conn: sqlite3.Connection | None = None

    def connect(self) -> None:
        os.makedirs(os.path.dirname(self._path) if os.path.dirname(self._path) else ".", exist_ok=True)
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        logger.info("Database opened: %s", self._path)

    def close(self) -> None:
        if self._conn:
            self._conn.close()

    def execute(self, sql: str, params: tuple = ()) -> None:
        if self._conn is None:
            raise RuntimeError("Database not connected")
        self._conn.execute(sql, params)
        self._conn.commit()

    def fetchall(self, sql: str, params: tuple = ()) -> List[Any]:
        if self._conn is None:
            raise RuntimeError("Database not connected")
        cur = self._conn.execute(sql, params)
        return cur.fetchall()

    def fetchone(self, sql: str, params: tuple = ()) -> Any:
        if self._conn is None:
            raise RuntimeError("Database not connected")
        cur = self._conn.execute(sql, params)
        return cur.fetchone()

    async def execute_async(self, sql: str, params: tuple = ()) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.execute, sql, params)

    async def fetchall_async(self, sql: str, params: tuple = ()) -> List[Any]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.fetchall, sql, params)
