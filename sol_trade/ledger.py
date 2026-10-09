"""Persistent books: positions, fills and trades in ``data/research.db``.

Each book (``live``, ``dry_run`` or a paper book) keeps one row per symbol
in ``positions`` with the open position, the action queued for the next bar
and the newest bar processed. Fills and closed trades are append-only. The
database runs in WAL mode so the bot and the CLI can share it.
"""

import json
import os
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from sol_trade.config import config
from sol_trade.engine import Action, Book, Fill, Position, SymbolState, Trade
from sol_trade.log import log_general

LIVE = "live"
DRY_RUN = "dry_run"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS positions (
    book TEXT NOT NULL,
    symbol TEXT NOT NULL,
    state TEXT NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (book, symbol)
);
CREATE TABLE IF NOT EXISTS books (
    book TEXT PRIMARY KEY,
    state TEXT NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS fills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    book TEXT NOT NULL,
    symbol TEXT NOT NULL,
    ts INTEGER NOT NULL,
    side TEXT NOT NULL,
    price REAL,
    size REAL,
    notional REAL,
    cost REAL,
    fee REAL,
    modeled_cost REAL,
    quoted_cost REAL,
    reason TEXT,
    strategy TEXT,
    strategy_hash TEXT,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    book TEXT NOT NULL,
    symbol TEXT NOT NULL,
    strategy TEXT,
    strategy_hash TEXT,
    entry_ts INTEGER,
    exit_ts INTEGER,
    entry_price REAL,
    exit_price REAL,
    size REAL,
    cost_basis REAL,
    proceeds REAL,
    pnl REAL,
    reason TEXT,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS trials (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy TEXT NOT NULL,
    strategy_hash TEXT NOT NULL,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    sharpe REAL NOT NULL,
    bars INTEGER NOT NULL,
    trades INTEGER NOT NULL,
    verdict TEXT NOT NULL,
    first_ts INTEGER,
    last_ts INTEGER,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    UNIQUE (strategy_hash, symbol, timeframe)
);
CREATE TABLE IF NOT EXISTS candidates (
    strategy_hash TEXT NOT NULL,
    symbol TEXT NOT NULL,
    strategy TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    state TEXT NOT NULL,
    spec TEXT NOT NULL,
    approved INTEGER NOT NULL DEFAULT 0,
    holdout TEXT,
    created_at REAL NOT NULL,
    entered_at REAL NOT NULL,
    PRIMARY KEY (strategy_hash, symbol)
);
CREATE TABLE IF NOT EXISTS transitions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_hash TEXT NOT NULL,
    symbol TEXT NOT NULL,
    strategy TEXT NOT NULL,
    from_state TEXT,
    to_state TEXT NOT NULL,
    reason TEXT NOT NULL,
    evidence TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS fills_book ON fills (book, symbol, ts);
CREATE INDEX IF NOT EXISTS trades_book ON trades (book, symbol, exit_ts);
"""


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    """A short-lived connection; commits on success, rolls back on error."""
    path = Path(config().research_db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(_SCHEMA)
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def decode(text: str, where: str) -> Any:
    """Parse a stored JSON column; a corrupt row raises naming the row, never passes silently."""
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"corrupt {where} in {config().research_db_path}: {e}") from e


# ---------------------------------------------------------------- positions


def load_state(book: str, symbol: str) -> SymbolState:
    with connect() as conn:
        row = conn.execute(
            "SELECT state FROM positions WHERE book = ? AND symbol = ?", (book, symbol)
        ).fetchone()
    if row is None:
        return SymbolState()
    data = decode(row[0], f"position {book}/{symbol}")
    return SymbolState(
        Position.from_dict(data["position"]) if data.get("position") else None,
        Action.from_dict(data["pending"]) if data.get("pending") else None,
        data.get("last_bar_ts"),
    )


def save_state(book: str, symbol: str, state: SymbolState) -> None:
    data = {
        "position": state.position.to_dict() if state.position else None,
        "pending": state.pending.to_dict() if state.pending else None,
        "last_bar_ts": state.last_bar_ts,
    }
    with connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO positions (book, symbol, state, updated_at) VALUES (?, ?, ?, ?)",
            (book, symbol, json.dumps(data), time.time()),
        )


def delete_state(book: str, symbol: str) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM positions WHERE book = ? AND symbol = ?", (book, symbol))


def open_positions(book: str | None = None) -> list[tuple[str, Position]]:
    """``(book, position)`` for every open position, optionally of one book."""
    with connect() as conn:
        rows = conn.execute(
            "SELECT book, state FROM positions WHERE ? IS NULL OR book = ?", (book, book)
        ).fetchall()
    result = []
    for name, state in rows:
        data = decode(state, f"position {name}")
        if data.get("position"):
            result.append((name, Position.from_dict(data["position"])))
    return result


def open_position(book: str, symbol: str) -> Position | None:
    return load_state(book, symbol).position


def load_book(book: Book, symbols: list[str]) -> None:
    """Fill ``book`` with the stored state of ``symbols`` and its stored cash."""
    meta = load_meta(book.name)
    if "cash" in meta:
        book.cash = float(meta["cash"])
    for symbol in symbols:
        book.symbols[symbol] = load_state(book.name, symbol)


def save_book(book: Book, symbols: list[str] | None = None) -> None:
    for symbol in symbols if symbols is not None else list(book.symbols):
        save_state(book.name, symbol, book.state(symbol))
    meta = load_meta(book.name)
    meta["cash"] = book.cash
    save_meta(book.name, meta)


# --------------------------------------------------------------- book meta


def load_meta(book: str) -> dict[str, Any]:
    with connect() as conn:
        row = conn.execute("SELECT state FROM books WHERE book = ?", (book,)).fetchone()
    return decode(row[0], f"book {book}") if row else {}


def save_meta(book: str, meta: dict[str, Any]) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO books (book, state, updated_at) VALUES (?, ?, ?)",
            (book, json.dumps(meta), time.time()),
        )


def delete_book(book: str) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM books WHERE book = ?", (book,))
        conn.execute("DELETE FROM positions WHERE book = ?", (book,))


# ----------------------------------------------------------- fills, trades


def record_fill(book: Book, fill: Fill) -> None:
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO fills (book, symbol, ts, side, price, size, notional, cost, fee,
                               modeled_cost, quoted_cost, reason, strategy, strategy_hash, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                book.name, fill.symbol, fill.ts, fill.side, fill.price, fill.size, fill.notional,
                fill.cost, fill.fee, fill.modeled_cost, fill.quoted_cost, fill.reason,
                fill.strategy, fill.strategy_hash, time.time(),
            ),
        )


def record_trade(book: Book, trade: Trade) -> None:
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO trades (book, symbol, strategy, strategy_hash, entry_ts, exit_ts, entry_price,
                                exit_price, size, cost_basis, proceeds, pnl, reason, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                book.name, trade.symbol, trade.strategy, trade.strategy_hash, trade.entry_ts,
                trade.exit_ts, trade.entry_price, trade.exit_price, trade.size, trade.cost_basis,
                trade.proceeds, trade.pnl, trade.reason, time.time(),
            ),
        )


def persist(book: Book) -> Book:
    """Wire ``book`` so every fill and trade is written to the database."""
    book.on_fill = record_fill
    book.on_trade = record_trade
    return book


def trades(book: str, symbol: str | None = None, since: int | None = None) -> list[dict[str, Any]]:
    """Closed trades of a book, oldest first."""
    with connect() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT * FROM trades
            WHERE book = ? AND (? IS NULL OR symbol = ?) AND (? IS NULL OR exit_ts >= ?)
            ORDER BY exit_ts, id
            """,
            (book, symbol, symbol, since, since),
        ).fetchall()
    return [dict(r) for r in rows]


def fills(book: str, symbol: str | None = None) -> list[dict[str, Any]]:
    """Fills of a book, oldest first."""
    with connect() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM fills WHERE book = ? AND (? IS NULL OR symbol = ?) ORDER BY ts, id",
            (book, symbol, symbol),
        ).fetchall()
    return [dict(r) for r in rows]


# ------------------------------------------------------------ CSV migration


def _float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(number) else number


def migrate_legacy_csv(book: str = LIVE, data_dir: str = "data") -> list[str]:
    """Import open positions from ``data/{SYMBOL}_data.csv`` once; returns the symbols imported.

    Every legacy CSV is renamed to ``.migrated`` afterwards. A symbol that
    already has a row in the database is left alone.
    """
    imported = []
    directory = Path(data_dir)
    if not directory.is_dir():
        return imported
    for path in sorted(directory.glob("*_data.csv")):
        symbol = path.name[: -len("_data.csv")]
        try:
            df = pd.read_csv(path)
        except (OSError, ValueError) as e:
            log_general.warning(f"legacy position file {path} unreadable: {e}")
            continue
        with connect() as conn:
            exists = conn.execute(
                "SELECT 1 FROM positions WHERE book = ? AND symbol = ?", (book, symbol)
            ).fetchone()
        if not exists and len(df) and "position" in df.columns and _truthy(df["position"].iat[-1]):
            last = df.iloc[-1]
            entry = _float(last.get("entry_price")) or _float(last.get("close")) or 0.0
            size = _float(last.get("position_size")) or 0.0
            highest = _float(last.get("highest_price"))
            trailing = _float(last.get("trailing_stoploss"))
            opened = _legacy_ts(last.get("time"))
            position = Position(
                symbol=symbol,
                strategy=config().token_strategies.get(symbol) or config().strategy or "default",
                strategy_hash="",
                entry_price=entry,
                size=size,
                cost_basis=entry * size,
                entry_ts=opened,
                stop=_float(last.get("stoploss")),
                take_profit=_float(last.get("takeprofit")),
                trailing_active=trailing is not None,
                highest_price=highest if highest is not None else entry,
            )
            save_state(book, symbol, SymbolState(position=position))
            imported.append(symbol)
            log_general.info(f"imported legacy {symbol} position ({size:g} at {entry:g}) into {book}")
        try:
            os.replace(path, path.with_name(path.name + ".migrated"))
        except OSError as e:
            log_general.warning(f"could not rename {path}: {e}")
    return imported


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    try:
        return bool(value) and not pd.isna(value)
    except (TypeError, ValueError):
        return False


def _legacy_ts(value: Any) -> int:
    """Open time of a legacy CSV row (naive UTC), or now when it can't be read."""
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        stamp = None
    if stamp is None or pd.isna(stamp):
        return int(datetime.now(UTC).timestamp())
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    return int(stamp.timestamp())
