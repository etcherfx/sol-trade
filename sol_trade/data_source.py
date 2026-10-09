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
from sol_trade.log import log_general

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


_CANDLES_COLUMNS = """
        exchange    TEXT    NOT NULL,
        symbol      TEXT    NOT NULL,
        timeframe   TEXT    NOT NULL,
        ts          INTEGER NOT NULL,
        open        REAL,
        high        REAL,
        low         REAL,
        close       REAL,
        volume      REAL,
        totalvolume REAL,
        PRIMARY KEY (exchange, symbol, timeframe, ts)
"""
_CREATE_CANDLES = "CREATE TABLE IF NOT EXISTS candles (" + _CANDLES_COLUMNS + ")"
_CREATE_KEYED = "CREATE TABLE IF NOT EXISTS candles_keyed (" + _CANDLES_COLUMNS + ")"
_COPY_TAGGED = """
    INSERT INTO candles_keyed (exchange, symbol, timeframe, ts, open, high, low, close, volume, totalvolume)
    SELECT ?, symbol, timeframe, ts, open, high, low, close, volume, totalvolume FROM candles WHERE symbol = ?
"""


def _connect() -> sqlite3.Connection:
    """Open the candle database, creating or migrating the schema if needed.

    Bars are keyed by exchange, pair (``BASE/QUOTE``), timeframe and open
    time, so two venues' series never mix. WAL and a generous busy timeout
    let the bot and a CLI backfill write at once.
    """
    path = Path(config().candles_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    if not _has_exchange_column(conn):
        conn.execute("BEGIN IMMEDIATE")
        try:
            if _table_exists(conn) and not _has_exchange_column(conn):
                _migrate_legacy(conn)
            conn.execute(_CREATE_CANDLES)
            conn.commit()
        except BaseException:
            conn.rollback()
            conn.close()
            raise
    return conn


def _table_exists(conn: sqlite3.Connection) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'candles'").fetchone() is not None


def _has_exchange_column(conn: sqlite3.Connection) -> bool:
    return any(row[1] == "exchange" for row in conn.execute("PRAGMA table_info(candles)"))


def _migrate_legacy(conn: sqlite3.Connection) -> None:
    """Tag the rows of a store from before exchanges were part of the key.

    Each pair gets the exchange the current config reads it from: the
    token's ``token_exchanges`` entry when that pair is its configured
    source, else ``data_exchange``.
    """
    from sol_trade import history

    conn.execute(_CREATE_KEYED)
    tagged = {}
    for (pair,) in conn.execute("SELECT DISTINCT symbol FROM candles").fetchall():
        source = history.source_for(pair.split("/", 1)[0])
        exchange = source.exchange if source.pair == pair else config().data_exchange
        conn.execute(_COPY_TAGGED, (exchange, pair))
        tagged[pair] = exchange
    conn.execute("DROP TABLE candles")
    conn.execute("ALTER TABLE candles_keyed RENAME TO candles")
    log_general.warning(f"candle store migrated: rows are now keyed by exchange; tagged {tagged}")


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
