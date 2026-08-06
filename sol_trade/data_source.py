"""Market data abstraction with a local candle store.

Candles are persisted in a local SQLite database (``data/candles.db``). The
bot reads from local data and only contacts the exchange (via ccxt) to refresh
the latest candles, so it keeps running — with slightly stale data — even when
the exchange is unreachable.
"""

import sqlite3
import time
from pathlib import Path
from typing import Any

import ccxt

from sol_trade.config import config
from sol_trade.log import log_general

_exchange: Any | None = None

# Candle period lengths in seconds, used for staleness checks.
_PERIOD_SECONDS = {"1m": 60, "1d": 86400}


def get_exchange() -> Any:
    """Return the configured ccxt exchange client (lazily created)."""
    global _exchange
    if _exchange is None:
        exchange_id = config().data_exchange
        exchange_class = getattr(ccxt, exchange_id, None)
        if exchange_class is None:
            raise ValueError(f"Unknown ccxt exchange: {exchange_id}")
        _exchange = exchange_class({"enableRateLimit": True, "timeout": 15000})
    return _exchange


def _connect() -> sqlite3.Connection:
    """Open the candle database, creating the schema if needed."""
    path = Path(config().candles_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
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


def _store_candles(symbol: str, timeframe: str, candles: list[dict]) -> None:
    """Upsert candles into the local store. Failures are logged, never raised."""
    try:
        conn = _connect()
        try:
            conn.executemany(
                """
                INSERT OR REPLACE INTO candles
                    (symbol, timeframe, ts, open, high, low, close, volume, totalvolume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        symbol,
                        timeframe,
                        candle["time"],
                        candle["open"],
                        candle["high"],
                        candle["low"],
                        candle["close"],
                        candle.get("volume"),
                        candle.get("totalvolume", candle.get("volume")),
                    )
                    for candle in candles
                ],
            )
            conn.commit()
        finally:
            conn.close()
    except Exception as e:  # noqa: BLE001 - storage failure must not break trading
        log_general.warning(f"Candle store: failed to persist candles: {e}")


def _load_candles(symbol: str, timeframe: str, limit: int) -> list[dict]:
    """Return the latest ``limit`` candles from the local store, oldest first."""
    try:
        conn = _connect()
        try:
            rows = conn.execute(
                """
                SELECT ts, open, high, low, close, volume, totalvolume
                FROM candles
                WHERE symbol = ? AND timeframe = ?
                ORDER BY ts DESC
                LIMIT ?
                """,
                (symbol, timeframe, limit),
            ).fetchall()
        finally:
            conn.close()
    except Exception as e:  # noqa: BLE001 - storage failure must not break trading
        log_general.warning(f"Candle store: failed to read candles: {e}")
        return []
    return [
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "totalvolume": total_volume,
            "time": ts,
        }
        for ts, open_, high, low, close, volume, total_volume in rows
    ][::-1]


def fetch_candles(
    base: str, quote: str, timeframe: str = "1m", limit: int = 50
) -> list[dict]:
    """Fetch candles as ``{open, high, low, close, volume, totalvolume, time}``.

    ``time`` is a Unix timestamp in seconds. Refreshes from the exchange via
    ccxt and persists to the local store; on exchange failure serves the local
    cache while it is still fresh, raising ``ConnectionError`` when the cache
    is empty or too stale to trade on.
    """
    symbol = f"{base}/{quote}"
    try:
        exchange = get_exchange()
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        candles = [
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "totalvolume": volume,
                "time": ts // 1000,
            }
            for ts, open_, high, low, close, volume in ohlcv
        ]
        _store_candles(symbol, timeframe, candles)
        return candles
    except Exception as e:
        log_general.warning(
            f"ccxt failed for {symbol} ({type(e).__name__}): {e}; trying local data"
        )
        cached = _load_candles(symbol, timeframe, limit)
        if cached:
            newest_ts = cached[-1]["time"]
            age_seconds = int(time.time()) - newest_ts
            staleness_limit = 3 * _PERIOD_SECONDS.get(timeframe, 60)
            if age_seconds <= staleness_limit:
                log_general.info(
                    f"Candle store: serving {len(cached)} cached candles for {symbol} "
                    f"(age {age_seconds}s)"
                )
                return cached
            log_general.warning(
                f"Candle store: cached data for {symbol} is {age_seconds}s old "
                f"(limit {staleness_limit}s) — refusing to trade on stale data"
            )
        raise ConnectionError(
            f"No fresh candle data available for {symbol} "
            "(exchange unreachable and local cache empty or stale)"
        ) from e
