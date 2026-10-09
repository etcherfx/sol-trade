"""Community sentiment: a Reddit recorder, point-in-time features, and the old breaker.

The recorder reads the Atom feeds of each configured subreddit (new posts
and new comments) over plain HTTP; no LLM and no Solana RPC. Reddit allows
anonymous feed readers about one request a minute, so each call to
:func:`record` makes at most one request, round-robin over the feeds, and
waits out ``x-ratelimit-reset`` and backs off on 429/403. Raw items go to
``data/sentiment.db`` deduplicated by id; a batch that shares no id with
what's stored is recorded as a possible gap.

:func:`features` turns the stored items into strategy inputs: per bar close,
the number of items mentioning the token in the trailing window and their
mean VADER compound score. Times before recording started are NaN, so a
strategy never mistakes "not recorded" for "no mentions".

The legacy circuit breaker (``sentiment_enabled``, off by default) now reads
the recorded items instead of fetching on its own.
"""

import html
import re
import sqlite3
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from defusedxml import DefusedXmlException
from defusedxml.ElementTree import fromstring

from sol_trade.config import config
from sol_trade.log import log_general
from sol_trade.utils import load_json_data, save_json_data

_ATOM = {"a": "http://www.w3.org/2005/Atom"}
_FEEDS = {"post": "https://www.reddit.com/r/{sub}/new/.rss?limit=100",
          "comment": "https://www.reddit.com/r/{sub}/comments/.rss?limit=100"}
_MAX_BACKOFF = 900.0

_SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    id TEXT PRIMARY KEY,
    subreddit TEXT NOT NULL,
    kind TEXT NOT NULL,
    author TEXT,
    created INTEGER NOT NULL,
    title TEXT,
    text TEXT,
    compound REAL NOT NULL,
    fetched_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS items_created ON items (created);
CREATE TABLE IF NOT EXISTS fetches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    feed TEXT NOT NULL,
    at REAL NOT NULL,
    status INTEGER,
    items INTEGER NOT NULL DEFAULT 0,
    new_items INTEGER NOT NULL DEFAULT 0,
    gap INTEGER NOT NULL DEFAULT 0,
    error TEXT
);
CREATE INDEX IF NOT EXISTS fetches_feed ON fetches (feed, at);
"""


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    path = Path(config().sentiment["db_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


@lru_cache(maxsize=1)
def _analyzer() -> Any:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    return SentimentIntensityAnalyzer()


def score_text(text: str) -> float:
    """VADER compound score in [-1, 1]."""
    return float(_analyzer().polarity_scores(text)["compound"]) if text.strip() else 0.0


# ------------------------------------------------------------------ parsing

_TAGS = re.compile(r"<[^>]+>")
_SUBMITTED = re.compile(r"submitted by\s+/u/\S+.*$", re.DOTALL)
_SPACES = re.compile(r"\s+")


def _plain(markup: str) -> str:
    text = html.unescape(_TAGS.sub(" ", markup or ""))
    return _SPACES.sub(" ", _SUBMITTED.sub("", text)).strip()


def _epoch(value: str | None) -> int | None:
    if not value:
        return None
    try:
        return int(datetime.fromisoformat(value).timestamp())
    except ValueError:
        return None


def parse_feed(xml_text: str, subreddit: str, kind: str) -> list[dict[str, Any]]:
    """Items of one Atom feed; entries without an id or a time are skipped."""
    root = fromstring(xml_text)
    items = []
    for entry in root.findall("a:entry", _ATOM):
        item_id = entry.findtext("a:id", default="", namespaces=_ATOM)
        created = _epoch(entry.findtext("a:published", namespaces=_ATOM)) or _epoch(
            entry.findtext("a:updated", namespaces=_ATOM)
        )
        if not item_id or created is None:
            continue
        title = entry.findtext("a:title", default="", namespaces=_ATOM)
        if kind == "comment":
            title = ""  # "/u/name on <post title>" would count the post's mentions again
        text = _plain(entry.findtext("a:content", default="", namespaces=_ATOM))
        items.append(
            {
                "id": item_id,
                "subreddit": subreddit,
                "kind": kind,
                "author": entry.findtext("a:author/a:name", default="", namespaces=_ATOM),
                "created": created,
                "title": title,
                "text": text,
                "compound": score_text(f"{title} {text}"),
            }
        )
    return items


# ----------------------------------------------------------------- recorder

_next_allowed = 0.0
_backoff = 0.0
_cursor = 0


def feeds() -> list[tuple[str, str, str]]:
    """``(name, subreddit, kind)`` for every configured feed."""
    subs = [s for s in config().sentiment.get("subreddits") or [] if s]
    return [(f"{s}/{kind}", s, kind) for s in subs for kind in _FEEDS]


def _store(feed: str, items: list[dict[str, Any]], status: int, error: str | None, now: float) -> dict[str, Any]:
    with _connect() as conn:
        had_fetch = conn.execute(
            "SELECT 1 FROM fetches WHERE feed = ? AND status = 200 LIMIT 1", (feed,)
        ).fetchone() is not None
        known = sum(
            conn.execute("SELECT 1 FROM items WHERE id = ?", (i["id"],)).fetchone() is not None for i in items
        )
        conn.executemany(
            """
            INSERT OR IGNORE INTO items (id, subreddit, kind, author, created, title, text, compound, fetched_at)
            VALUES (:id, :subreddit, :kind, :author, :created, :title, :text, :compound, :fetched_at)
            """,
            [{**i, "fetched_at": now} for i in items],
        )
        gap = status == 200 and had_fetch and bool(items) and known == 0
        conn.execute(
            "INSERT INTO fetches (feed, at, status, items, new_items, gap, error) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (feed, now, status, len(items), len(items) - known, int(gap), error),
        )
    if gap:
        log_general.warning(f"sentiment: {feed} batch didn't overlap the last one; items may be missing")
    return {"feed": feed, "status": status, "items": len(items), "new": len(items) - known, "gap": gap,
            "error": error}


def record(now: float | None = None, client: httpx.Client | None = None) -> dict[str, Any] | None:
    """Fetch the next due feed if Reddit's rate limit allows; None when nothing was due."""
    global _next_allowed, _backoff, _cursor
    cfg = config().sentiment
    names = feeds()
    now = time.time() if now is None else now
    if not cfg.get("record", True) or not names or now < _next_allowed:
        return None
    name, subreddit, kind = names[_cursor % len(names)]
    _cursor += 1
    interval = float(cfg["request_interval_seconds"])
    headers = {"User-Agent": cfg["user_agent"]}
    try:
        if client is None:
            response = httpx.get(_FEEDS[kind].format(sub=subreddit), headers=headers, timeout=20,
                                 follow_redirects=True)
        else:
            response = client.get(_FEEDS[kind].format(sub=subreddit), headers=headers)
    except httpx.HTTPError as e:
        _next_allowed = now + interval
        return _store(name, [], 0, f"{type(e).__name__}: {e}", now)
    reset = _float(response.headers.get("x-ratelimit-reset"))
    if response.status_code in (403, 429):
        _backoff = min(max(_backoff * 2, interval), _MAX_BACKOFF)
        _next_allowed = now + max(_backoff, reset or 0.0)
        _cursor -= 1  # retry the same feed
        return _store(name, [], response.status_code, f"HTTP {response.status_code}", now)
    _backoff = 0.0
    remaining = _float(response.headers.get("x-ratelimit-remaining"))
    wait = (reset or 0.0) + 1 if remaining is not None and remaining < 1 else 0.0
    _next_allowed = now + max(interval, wait)
    if response.status_code != 200:
        return _store(name, [], response.status_code, f"HTTP {response.status_code}", now)
    try:
        items = parse_feed(response.text, subreddit, kind)
    except (ET.ParseError, DefusedXmlException) as e:
        return _store(name, [], response.status_code, f"bad feed: {e}", now)
    return _store(name, items, 200, None, now)


def _float(value: str | None) -> float | None:
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def stats(now: float | None = None) -> dict[str, Any]:
    """Recorder health for ``status``: item counts, last fetch per feed, recent gaps and errors."""
    now = time.time() if now is None else now
    if not Path(config().sentiment["db_path"]).exists():
        return {"feeds": [f[0] for f in feeds()], "items": 0, "since": None}
    with _connect() as conn:
        items, first = conn.execute("SELECT COUNT(*), MIN(fetched_at) FROM items").fetchone()
        day = now - 86400
        last = {
            row[0]: {"at": _iso(row[1]), "status": row[2]}
            for row in conn.execute(
                "SELECT feed, MAX(at), status FROM fetches GROUP BY feed"
            )
        }
        gaps, errors, fetched = conn.execute(
            "SELECT COALESCE(SUM(gap), 0), COALESCE(SUM(status != 200), 0), COUNT(*) FROM fetches WHERE at >= ?",
            (day,),
        ).fetchone()
        new_day = conn.execute("SELECT COUNT(*) FROM items WHERE fetched_at >= ?", (day,)).fetchone()[0]
    return {
        "feeds": [f[0] for f in feeds()],
        "items": items,
        "since": _iso(first),
        "last_fetch": last,
        "last_24h": {"fetches": fetched, "new_items": new_day, "gaps": gaps, "errors": errors},
    }


def _iso(ts: float | None) -> str | None:
    return datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%dT%H:%M:%SZ") if ts else None


# ----------------------------------------------------------------- features


def keywords(symbol: str) -> list[str]:
    configured = (config().sentiment.get("keywords") or {}).get(symbol)
    return list(configured) if configured else [symbol]


def _mention_pattern(symbol: str) -> re.Pattern[str]:
    words = "|".join(re.escape(k) for k in keywords(symbol))
    return re.compile(rf"(?<![\w$])\$?(?:{words})(?!\w)", re.IGNORECASE)


def _recording_start() -> float | None:
    with _connect() as conn:
        row = conn.execute("SELECT MIN(at) FROM fetches WHERE status = 200").fetchone()
    return row[0] if row else None


def features(symbol: str | None, closes: np.ndarray, window_hours: float) -> dict[str, np.ndarray]:
    """Mentions and mean compound score in ``(t - window, t]`` for every bar close ``t``.

    The arrays match ``closes``. Score is NaN where nothing was mentioned;
    both are NaN for times before the window is fully recorded.
    """
    closes = np.asarray(closes, dtype=float)
    nan = np.full(closes.shape, np.nan)
    if symbol is None or not Path(config().sentiment["db_path"]).exists():
        return {"score": nan, "mentions": nan.copy()}
    window = window_hours * 3600
    start = _recording_start()
    if start is None:
        return {"score": nan, "mentions": nan.copy()}
    finite = closes[np.isfinite(closes)]
    if finite.size == 0:
        return {"score": nan, "mentions": nan.copy()}
    with _connect() as conn:
        rows = conn.execute(
            "SELECT created, compound, title, text FROM items WHERE created > ? AND created <= ? ORDER BY created",
            (float(finite.min()) - window, float(finite.max())),
        ).fetchall()
    pattern = _mention_pattern(symbol)
    hits = [(r[0], r[1]) for r in rows if pattern.search(f"{r[2]} {r[3]}")]
    created = np.array([h[0] for h in hits], dtype=float)
    cumulative = np.concatenate([[0.0], np.cumsum([h[1] for h in hits])])
    flat = closes.ravel()
    hi = np.searchsorted(created, flat, side="right")
    lo = np.searchsorted(created, flat - window, side="right")
    mentions = (hi - lo).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        score = np.where(mentions > 0, (cumulative[hi] - cumulative[lo]) / np.maximum(mentions, 1), np.nan)
    unrecorded = ~np.isfinite(flat) | (flat - window < start)
    mentions[unrecorded] = np.nan
    score[unrecorded] = np.nan
    return {"score": score.reshape(closes.shape), "mentions": mentions.reshape(closes.shape)}


# ------------------------------------------------------- legacy breaker (opt-in)


def _load_sentiment_data() -> dict[str, Any]:
    return load_json_data(config().sentiment_data_path, {})


def _save_sentiment_data(data: dict[str, Any]) -> None:
    save_json_data(config().sentiment_data_path, data)


def _compute_token_sentiment(token_symbol: str, now: float | None = None) -> float:
    """Mean compound score of the last day's recorded mentions; 0 without any."""
    now = time.time() if now is None else now
    result = features(token_symbol, np.array([now]), 24.0)
    score = float(result["score"][0])
    return score if np.isfinite(score) else 0.0


def update_sentiment(token_symbols: list[str]) -> None:
    """Refresh the breaker's per-token scores from recorded items."""
    cfg = config()
    if not cfg.sentiment_enabled or not token_symbols:
        return
    existing_data = _load_sentiment_data()
    timestamp = datetime.now(UTC).isoformat()
    for symbol in token_symbols:
        try:
            score = _compute_token_sentiment(symbol)
            blocked_since = existing_data.get(symbol, {}).get("blocked_since")
            if blocked_since:
                try:
                    expiry = datetime.fromisoformat(blocked_since) + timedelta(hours=cfg.sentiment_pause_hours)
                    if datetime.now(UTC) >= expiry:
                        blocked_since = None
                except (ValueError, TypeError):
                    blocked_since = None
            if blocked_since is None and score < cfg.sentiment_threshold:
                blocked_since = timestamp
                log_general.info(
                    f"Sentiment circuit breaker: {symbol} blocked (score {score:.2f} < {cfg.sentiment_threshold})"
                )
            existing_data[symbol] = {"score": round(score, 4), "blocked_since": blocked_since, "last_check": timestamp}
        except Exception as e:  # noqa: BLE001 - scoring failure; log and continue
            log_general.warning(f"Sentiment: failed to compute score for {symbol}: {e}")
    _save_sentiment_data(existing_data)


def is_token_blocked(token_symbol: str) -> bool:
    """Return True if trading this token is paused due to bad sentiment."""
    cfg = config()
    if not cfg.sentiment_enabled:
        return False
    blocked_since = _load_sentiment_data().get(token_symbol, {}).get("blocked_since")
    if not blocked_since:
        return False
    try:
        expiry = datetime.fromisoformat(blocked_since) + timedelta(hours=cfg.sentiment_pause_hours)
        return datetime.now(UTC) < expiry
    except (ValueError, TypeError):
        return False


def is_market_crash() -> bool:
    """True when every tracked token's score is below ``sentiment_crash_threshold``."""
    cfg = config()
    if not cfg.sentiment_enabled:
        return False
    data = _load_sentiment_data()
    if not data:
        return False
    return all(entry.get("score", 0.0) < cfg.sentiment_crash_threshold for entry in data.values())
