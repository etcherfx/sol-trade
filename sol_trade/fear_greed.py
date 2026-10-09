"""The daily Crypto Fear & Greed index as a strategy input.

Values (0 to 100) come from the URL in ``sentiment.fear_greed_url`` (the
alternative.me API by default) and are stored per UTC day in the sentiment
database, ``data/sentiment.db``, table ``fear_greed``. The API stamps each
value at 00:00 UTC of the day it describes, so a value is only known once
its day is over: :func:`values` gives a bar closing at ``t`` the value of
the UTC day before ``t``'s day, and NaN when that day is missing. Backtests,
paper books and the live loop all read the index through :func:`values`;
:func:`refresh` keeps it current while the bot runs. No Solana RPC.
"""

import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from sol_trade import sentiment
from sol_trade.config import config

DAY = 86400

_SCHEMA = """
CREATE TABLE IF NOT EXISTS fear_greed (
    day INTEGER PRIMARY KEY,
    value REAL NOT NULL,
    classification TEXT,
    fetched_at REAL NOT NULL
);
"""
_RETRY_SECONDS = 900.0

_next_refresh = 0.0


@contextmanager
def _connect() -> Iterator[Any]:
    with sentiment._connect() as conn:  # the same database as the Reddit recorder
        conn.executescript(_SCHEMA)
        yield conn


def parse(payload: Any) -> list[tuple[int, float, str]]:
    """``(day, value, classification)`` per entry of an API response; malformed entries are skipped."""
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ValueError("fear & greed response has no data list")  # noqa: TRY004 - callers handle ValueError
    error = (payload.get("metadata") or {}).get("error")
    if error:
        raise ValueError(f"fear & greed API error: {error}")
    rows = []
    for entry in payload["data"]:
        try:
            stamp = int(entry["timestamp"])
            value = float(entry["value"])
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= value <= 100:
            rows.append((stamp - stamp % DAY, value, str(entry.get("value_classification") or "")))
    return rows


def _iso(ts: float | None) -> str | None:
    return datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%d") if ts is not None else None


def store(rows: list[tuple[int, float, str]], now: float | None = None) -> dict[str, Any]:
    """Upsert daily values; returns what changed and the stored range."""
    now = time.time() if now is None else now
    with _connect() as conn:
        stored = dict(conn.execute("SELECT day, value FROM fear_greed").fetchall())
        new = [r for r in rows if r[0] not in stored]
        changed = [r for r in rows if r[0] in stored and stored[r[0]] != r[1]]
        conn.executemany(
            """
            INSERT INTO fear_greed (day, value, classification, fetched_at) VALUES (?, ?, ?, ?)
            ON CONFLICT (day) DO UPDATE SET value = excluded.value, classification = excluded.classification,
                fetched_at = excluded.fetched_at
            """,
            [(*r, now) for r in new + changed],
        )
    return {"received": len(rows), "inserted": len(new), "updated": len(changed), **stats()}


def backfill(client: httpx.Client | None = None, now: float | None = None) -> dict[str, Any]:
    """Fetch the whole history from ``sentiment.fear_greed_url`` and store it; safe to repeat."""
    url = str(config().sentiment.get("fear_greed_url") or "")
    if not url:
        raise ValueError("sentiment.fear_greed_url is empty")
    if client is None:
        response = httpx.get(url, timeout=30, follow_redirects=True)
    else:
        response = client.get(url)
    response.raise_for_status()
    return {"url": url, **store(parse(response.json()), now)}


def refresh(now: float | None = None, client: httpx.Client | None = None) -> dict[str, Any] | None:
    """Backfill at most every ``sentiment.fear_greed_refresh_hours``; None when not due or turned off."""
    global _next_refresh
    cfg = config().sentiment
    now = time.time() if now is None else now
    if not cfg.get("fear_greed_url") or now < _next_refresh:
        return None
    _next_refresh = now + _RETRY_SECONDS  # a failed fetch is retried sooner than a full interval
    report = backfill(client, now)
    _next_refresh = now + float(cfg.get("fear_greed_refresh_hours") or 0) * 3600
    return report


def stats() -> dict[str, Any]:
    """Stored days, the first and last day, the newest value and days missing in between."""
    if not Path(config().sentiment["db_path"]).exists():
        return {"days": 0, "first": None, "last": None, "latest": None, "missing_days": 0}
    with _connect() as conn:
        days, first, last = conn.execute("SELECT COUNT(*), MIN(day), MAX(day) FROM fear_greed").fetchone()
        latest = conn.execute("SELECT value FROM fear_greed ORDER BY day DESC LIMIT 1").fetchone()
    missing = int((last - first) // DAY + 1 - days) if days else 0
    return {"days": days, "first": _iso(first), "last": _iso(last), "latest": latest[0] if latest else None,
            "missing_days": missing}


def values(closes: np.ndarray) -> np.ndarray:
    """The index for every bar close ``t``: the value of the UTC day before ``t``'s, else NaN.

    A value describes its whole day, so it's available from the next 00:00
    UTC on; a missing day is NaN, never the day before's value.
    """
    closes = np.asarray(closes, dtype=float)
    flat = closes.ravel()
    out = np.full(flat.shape, np.nan)
    finite = np.isfinite(flat)
    if not finite.any() or not Path(config().sentiment["db_path"]).exists():
        return out.reshape(closes.shape)
    wanted = np.floor(flat[finite] / DAY) * DAY - DAY
    with _connect() as conn:
        rows = conn.execute(
            "SELECT day, value FROM fear_greed WHERE day >= ? AND day <= ? ORDER BY day",
            (float(wanted.min()), float(wanted.max())),
        ).fetchall()
    if rows:
        days = np.array([r[0] for r in rows], dtype=float)
        stored = np.array([r[1] for r in rows], dtype=float)
        index = np.minimum(np.searchsorted(days, wanted), len(days) - 1)
        out[finite] = np.where(days[index] == wanted, stored[index], np.nan)
    return out.reshape(closes.shape)
