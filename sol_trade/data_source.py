"""Exchange clients and the local candle store.

Candles are persisted in a local SQLite database (``candles_path``); see
``sol_trade.history`` for backfills, incremental updates and data checks.
The bot keeps running on stored bars while the exchange is briefly
unreachable.
"""

import sqlite3
from pathlib import Path
from typing import Any

import ccxt

from sol_trade.config import config

_exchanges: dict[str, Any] = {}

def _candle_dict(
    ts: int, open_: float, high: float, low: float, close: float, volume: float, total_volume: float | None = None
) -> dict:
    """Build the canonical candle shape; totalvolume defaults to volume."""
    return {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "totalvolume": total_volume if total_volume is not None else volume,
        "time": ts,
    }


def get_exchange(exchange_id: str | None = None) -> Any:
    """Return a lazily-created ccxt exchange client, cached per exchange id.

    ``exchange_id`` defaults to the global ``data_exchange``; per-token
    overrides pass their own id so different tokens can use different
    exchanges in the same run.
    """
    exchange_id = exchange_id or config().data_exchange
    if exchange_id not in _exchanges:
        exchange_class = getattr(ccxt, exchange_id, None)
        if exchange_class is None:
            raise ValueError(f"Unknown ccxt exchange: {exchange_id}")
        _exchanges[exchange_id] = exchange_class(
            {"enableRateLimit": True, "timeout": 15000}
        )
    return _exchanges[exchange_id]


def _connect() -> sqlite3.Connection:
    """Open the candle database, creating the schema if needed.

    WAL and a generous busy timeout let the bot and a CLI backfill write at once.
    """
    path = Path(config().candles_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS candles (
            symbol      TEXT    NOT NULL,
            timeframe   TEXT    NOT NULL,
            ts          INTEGER NOT NULL,
            open        REAL,
            high        REAL,
            low         REAL,
            close       REAL,
            volume      REAL,
            totalvolume REAL,
            PRIMARY KEY (symbol, timeframe, ts)
        )
        """
    )
    conn.commit()
    return conn


def fetch_candles(
    base: str,
    quote: str,
    timeframe: str = "1m",
    limit: int = 50,
    exchange_id: str | None = None,
) -> list[dict]:
    """The newest ``limit`` closed candles as ``{open, high, low, close, volume, totalvolume, time}``.

    ``time`` is the bar's open time in seconds. New bars are fetched into the
    local store first; when the exchange is unreachable the stored bars are
    served while they are fresh, and ``ConnectionError`` is raised otherwise.
    """
    from sol_trade import history

    frame, _ = history.window_pair(
        f"{base}/{quote}", exchange_id or config().data_exchange, timeframe, limit
    )
    return [
        _candle_dict(int(row.ts), row.open, row.high, row.low, row.close, row.volume)
        for row in frame.itertuples(index=False)
    ]
