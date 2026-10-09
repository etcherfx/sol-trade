"""Historical candles: paged backfill, incremental updates and data checks.

Bars live in the SQLite candle store (``candles_path``) keyed by pair,
timeframe and open time. Only closed bars are stored: a bar whose period
hasn't ended yet (the forming bar) is returned separately and never written.
None of this touches the Solana RPC.
"""

import sqlite3
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from sol_trade import data_source
from sol_trade.config import config
from sol_trade.log import log_general
from sol_trade.spec import timeframe_seconds

# Restart incremental updates this many bars before the newest stored bar, so
# a bar stored while it was still forming (older versions did) gets repaired.
_REPAIR_BARS = 2


@dataclass
class Bar:
    ts: int  # open time, seconds
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class SeriesReport:
    pair: str
    timeframe: str
    exchange: str
    bars_added: int = 0
    total: int = 0
    first: str | None = None
    last: str | None = None
    duplicates: int = 0
    gaps: int = 0
    largest_gap_bars: int = 0
    gap_fraction: float = 0.0
    misaligned: int = 0
    ohlc_errors: int = 0
    pages: int = 0
    # Share of the requested range's bars that are stored, when a range was requested.
    coverage: float | None = None
    notes: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def pair_for(symbol: str) -> str:
    return f"{symbol}/{config().primary_mint_symbol}"


def exchange_id_for(symbol: str) -> str:
    return config().token_exchanges.get(symbol) or config().data_exchange


def _iso(ts: int | None) -> str | None:
    return None if ts is None else datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def check_timeframe(exchange: Any, timeframe: str) -> None:
    supported = getattr(exchange, "timeframes", None) or {}
    if supported and timeframe not in supported:
        raise ValueError(
            f"timeframe {timeframe} is not offered by {exchange.id}; "
            f"supported: {', '.join(supported)}"
        )


# ------------------------------------------------------------------- store


def _stored_range(pair: str, timeframe: str) -> tuple[int | None, int | None, int]:
    conn = data_source._connect()
    try:
        row = conn.execute(
            "SELECT MIN(ts), MAX(ts), COUNT(*) FROM candles WHERE symbol = ? AND timeframe = ?",
            (pair, timeframe),
        ).fetchone()
    finally:
        conn.close()
    return row[0], row[1], int(row[2] or 0)


def _store(pair: str, timeframe: str, bars: list[Bar]) -> None:
    if not bars:
        return
    conn = data_source._connect()
    try:
        conn.executemany(
            """
            INSERT OR REPLACE INTO candles
                (symbol, timeframe, ts, open, high, low, close, volume, totalvolume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (pair, timeframe, b.ts, b.open, b.high, b.low, b.close, b.volume, b.volume)
                for b in bars
            ],
        )
        conn.commit()
    finally:
        conn.close()


def load_bars(
    pair: str, timeframe: str, start: int | None = None, end: int | None = None, limit: int | None = None
) -> pd.DataFrame:
    """Stored closed bars in ``[start, end)`` (seconds), oldest first.

    With ``limit`` the newest ``limit`` bars of that range are returned.
    Columns: ts, open, high, low, close, volume.
    """
    params = (
        pair,
        timeframe,
        int(start) if start is not None else -(2**62),
        int(end) if end is not None else 2**62,
        int(limit) if limit else -1,  # SQLite: a negative LIMIT means no limit
    )
    conn = data_source._connect()
    try:
        rows = conn.execute(
            """
            SELECT ts, open, high, low, close, volume FROM (
                SELECT * FROM candles
                WHERE symbol = ? AND timeframe = ? AND ts >= ? AND ts < ?
                ORDER BY ts DESC LIMIT ?
            ) ORDER BY ts
            """,
            params,
        ).fetchall()
    finally:
        conn.close()
    frame = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    frame["ts"] = frame["ts"].astype("int64")
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = frame[column].astype(float)
    return frame


# ---------------------------------------------------------------- fetching


def _fetch_range(
    exchange: Any, pair: str, timeframe: str, since: int, until: int, report: SeriesReport
) -> tuple[list[Bar], Bar | None]:
    """Page through ``[since, until)``; returns closed bars and the forming bar if seen."""
    period = timeframe_seconds(timeframe)
    page_limit = int(config().backfill.get("page_limit") or 100)
    seen: dict[int, Bar] = {}
    forming: Bar | None = None
    cursor = since
    while cursor < until:
        page = exchange.fetch_ohlcv(pair, timeframe=timeframe, since=cursor * 1000, limit=page_limit)
        report.pages += 1
        if not page:
            # Exchanges return one page from `since` on, so an empty page means
            # the history starts later (listing date or the exchange's depth).
            start = _first_page_with_data(exchange, pair, timeframe, cursor, until, page_limit, report)
            if start is None:
                break
            if cursor == since:
                report.notes.append(f"{exchange_label(exchange)} history starts at {_iso(start)}, after the requested {_iso(since)}")
            cursor = start
            continue
        newest = cursor - 1
        for ts_ms, open_, high, low, close, volume in page:
            ts = int(ts_ms) // 1000
            newest = max(newest, ts)
            bar = Bar(ts, *(float("nan") if v is None else float(v) for v in (open_, high, low, close, volume)))
            if ts >= until:
                if ts == until:
                    forming = bar
                continue
            if ts < since:
                continue
            if ts in seen:
                report.duplicates += 1
            seen[ts] = bar
        if newest < cursor:
            break  # the exchange ignored `since`; no further progress possible
        cursor = newest + period
    return [seen[ts] for ts in sorted(seen)], forming


def exchange_label(exchange: Any) -> str:
    return str(getattr(exchange, "id", "the exchange"))


def _first_page_with_data(
    exchange: Any, pair: str, timeframe: str, low: int, high: int, page_limit: int, report: SeriesReport
) -> int | None:
    """The earliest bar open time in ``[low, high)``, by binary search on ``since``; None if there's none.

    A page starting at ``since`` holds data when the history begins before
    ``since + page_limit`` bars, so emptiness is monotone in ``since``.
    """
    period = timeframe_seconds(timeframe)

    def first_bar(since: int) -> int | None:
        page = exchange.fetch_ohlcv(pair, timeframe=timeframe, since=since * 1000, limit=page_limit)
        report.pages += 1
        stamps = [int(row[0]) // 1000 for row in page if int(row[0]) // 1000 >= since]
        return min(stamps) if stamps else None

    newest = first_bar(max(high - page_limit * period, low))
    if newest is None:
        return None
    lo, hi = low // period, (max(high - page_limit * period, low)) // period  # in bars; hi has data
    found = newest
    while lo < hi:
        mid = (lo + hi) // 2
        bar = first_bar(mid * period)
        if bar is None:
            lo = mid + 1
        else:
            hi, found = mid, bar
    return found


def _validate(pair: str, timeframe: str, report: SeriesReport, start: int | None, end: int | None) -> None:
    period = timeframe_seconds(timeframe)
    frame = load_bars(pair, timeframe, start, end)
    report.total = _stored_range(pair, timeframe)[2]
    if frame.empty:
        return
    ts = frame["ts"].to_numpy()
    report.first, report.last = _iso(int(ts[0])), _iso(int(ts[-1]))
    report.misaligned = int((ts % period != 0).sum())
    steps = np.diff(ts) // period
    missing = steps[steps > 1] - 1
    report.gaps = int(missing.size)
    report.largest_gap_bars = int(missing.max()) if missing.size else 0
    expected = int((ts[-1] - ts[0]) // period) + 1
    report.gap_fraction = round(float(missing.sum()) / expected, 6) if expected else 0.0
    if start is not None and end is not None and end > start:
        report.coverage = round(len(ts) / ((end - start) // period), 6)
    o, h, low, c = (frame[k].to_numpy() for k in ("open", "high", "low", "close"))
    with np.errstate(invalid="ignore"):
        bad = (
            np.isnan(o) | np.isnan(h) | np.isnan(low) | np.isnan(c)
            | (h < np.maximum(o, c)) | (low > np.minimum(o, c)) | (h < low)
            | (o <= 0) | (h <= 0) | (low <= 0) | (c <= 0)
        )
    report.ohlc_errors = int(bad.sum())


def update(
    symbol: str,
    timeframe: str,
    days: float | None = None,
    min_bars: int = 0,
    now: float | None = None,
    validate: bool = True,
) -> tuple[SeriesReport, Bar | None]:
    """Fetch new closed bars for ``symbol`` (quoted in the primary token) into the store.

    Starts two bars before the newest stored bar. ``days`` (a backfill) also
    fetches any older history missing from the window; ``min_bars`` makes
    sure at least that many closed bars are stored. Returns the report and
    the forming bar when the exchange returned it.
    """
    return update_pair(
        pair_for(symbol), exchange_id_for(symbol), timeframe, days, min_bars, now, validate
    )


def update_pair(
    pair: str,
    exchange_id: str,
    timeframe: str,
    days: float | None = None,
    min_bars: int = 0,
    now: float | None = None,
    validate: bool = True,
) -> tuple[SeriesReport, Bar | None]:
    """:func:`update` for any ``BASE/QUOTE`` pair on a given exchange."""
    exchange = data_source.get_exchange(exchange_id)
    check_timeframe(exchange, timeframe)
    period = timeframe_seconds(timeframe)
    now = time.time() if now is None else now
    until = int(now // period) * period  # open time of the forming bar
    report = SeriesReport(pair, timeframe, exchange_id)
    first, last, count_before = _stored_range(pair, timeframe)

    wanted_start = None
    if days:
        wanted_start = until - int(days * 86400) // period * period
    if min_bars:
        floor = until - min_bars * period
        wanted_start = floor if wanted_start is None else min(wanted_start, floor)

    ranges: list[tuple[int, int]] = []
    if first is not None and wanted_start is not None and wanted_start < first:
        ranges.append((wanted_start, first))
    if last is not None:
        ranges.append((max(last - _REPAIR_BARS * period, wanted_start or 0), until))
    else:
        ranges.append((wanted_start if wanted_start is not None else until - 2 * period, until))

    forming: Bar | None = None
    for since, stop in ranges:
        bars, seen_forming = _fetch_range(exchange, pair, timeframe, since, stop, report)
        _store(pair, timeframe, bars)
        forming = seen_forming or forming
    report.bars_added = _stored_range(pair, timeframe)[2] - count_before
    if validate:
        _validate(pair, timeframe, report, wanted_start, until)
    return report, forming


def backfill(
    symbols: list[str], timeframes: list[str], days: float | None = None, now: float | None = None
) -> list[dict[str, Any]]:
    """Backfill every (symbol, timeframe) over ``days`` (default ``backfill.default_days``).

    One report per series, errors included.
    """
    days = days or float(config().backfill["default_days"])
    reports = []
    for symbol in symbols:
        for timeframe in timeframes:
            try:
                report, _ = update(symbol, timeframe, days=days, now=now)
                reports.append(report.as_dict())
            except Exception as e:  # noqa: BLE001 - report and continue with the rest
                log_general.warning(f"backfill {symbol} {timeframe} failed: {e}")
                failed = SeriesReport(pair_for(symbol), timeframe, exchange_id_for(symbol))
                failed.errors.append(f"{type(e).__name__}: {e}")
                reports.append(failed.as_dict())
    return reports


def gap_fraction(frame: pd.DataFrame, timeframe: str) -> float:
    """Share of expected bars missing between the first and last bar of ``frame``."""
    if len(frame) < 2:
        return 0.0
    period = timeframe_seconds(timeframe)
    ts = frame["ts"].to_numpy()
    expected = int((ts[-1] - ts[0]) // period) + 1
    return float(expected - len(ts)) / expected if expected else 0.0


def window(symbol: str, timeframe: str, lookback: int, now: float | None = None) -> tuple[pd.DataFrame, Bar | None]:
    """Refresh and return the newest ``lookback`` closed bars plus the forming bar.

    On exchange errors the stored bars are used when the newest one is at
    most three periods old; otherwise ``ConnectionError`` is raised.
    """
    return window_pair(pair_for(symbol), exchange_id_for(symbol), timeframe, lookback, now)


def window_pair(
    pair: str, exchange_id: str, timeframe: str, lookback: int, now: float | None = None
) -> tuple[pd.DataFrame, Bar | None]:
    """:func:`window` for any ``BASE/QUOTE`` pair on a given exchange."""
    period = timeframe_seconds(timeframe)
    now = time.time() if now is None else now
    forming = None
    try:
        _, forming = update_pair(pair, exchange_id, timeframe, min_bars=lookback, now=now, validate=False)
    except (sqlite3.Error, ValueError):
        raise
    except Exception as e:  # noqa: BLE001 - network errors fall back to the store
        log_general.warning(f"candle update failed for {pair} {timeframe}: {e}; using stored bars")
    frame = load_bars(pair, timeframe, end=int(now // period) * period, limit=lookback)
    if frame.empty or now - (int(frame["ts"].iat[-1]) + period) > 3 * period:
        raise ConnectionError(f"no fresh {timeframe} candles for {pair}")
    return frame, forming
