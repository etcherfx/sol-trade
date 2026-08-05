"""Sentiment circuit breaker — monitors social sentiment and pauses trading on crashes."""

import json
import os
from datetime import UTC, datetime, timedelta
from typing import Any

import requests

from sol_trade.config import config
from sol_trade.log import log_general

# Keyword scoring lists
_POSITIVE_KEYWORDS = {
    "moon", "pump", "bullish", "gain", "buy", "accumulation",
    "green", "breakout", "avoided", "survived", "recovering",
}

_NEGATIVE_KEYWORDS = {
    "dump", "crash", "bearish", "sell", "rug", "scam", "loss",
    "red", "liquidation", "death", "collapse", "exit", "panic",
}


def _load_sentiment_data() -> dict[str, Any]:
    """Load sentiment data from disk."""
    path = config().sentiment_data_path
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, KeyError) as e:
        log_general.warning(f"Failed to load sentiment data from {path}: {e}")
        return {}


def _save_sentiment_data(data: dict[str, Any]) -> None:
    """Persist sentiment data to disk."""
    path = config().sentiment_data_path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _score_text(text: str) -> tuple:
    """Score a piece of text, returning (positive_count, negative_count)."""
    words = set(text.lower().split())
    pos = len(words & _POSITIVE_KEYWORDS)
    neg = len(words & _NEGATIVE_KEYWORDS)
    return pos, neg


def _fetch_reddit_posts(subreddit: str, limit: int = 25) -> list:
    """Fetch posts from a Reddit subreddit."""
    url = f"https://www.reddit.com/r/{subreddit}.json"
    params = {"limit": limit}
    headers = {"User-Agent": "SolTrade/2.0"}
    response = requests.get(url, params=params, headers=headers, timeout=30)
    if response.status_code != 200:
        return []
    data = response.json()
    return data.get("data", {}).get("children", [])


def _compute_token_sentiment(token_symbol: str) -> float:
    """Compute sentiment score for a token from Reddit posts.

    Returns a score from -1.0 to +1.0.
    """
    subreddits = ["SolanaTopGainers", "cryptocurrency"]
    total_pos = 0
    total_neg = 0

    for subreddit in subreddits:
        try:
            posts = _fetch_reddit_posts(subreddit)
            for post in posts:
                data = post.get("data", {})
                title = data.get("title", "")
                body = data.get("selftext", "")
                text = f"{title} {body}"

                # Check if token is mentioned
                if token_symbol.upper() not in text.upper():
                    continue

                pos, neg = _score_text(text)
                total_pos += pos
                total_neg += neg
        except Exception as e:  # noqa: BLE001 - fetch failure; log and continue
            log_general.warning(
                f"Sentiment: failed to fetch {subreddit}: {e}"
            )
            continue

    total = total_pos + total_neg
    if total == 0:
        return 0.0

    return (total_pos - total_neg) / total


def update_sentiment(token_symbols: list[str]) -> None:
    """Fetch sentiment data for all tracked tokens."""
    cfg = config()
    if not cfg.sentiment_enabled:
        return

    if not token_symbols:
        return

    existing_data = _load_sentiment_data()
    timestamp = datetime.now(UTC).isoformat()

    for symbol in token_symbols:
        try:
            score = _compute_token_sentiment(symbol)

            # Check if token should be blocked
            entry = existing_data.get(symbol, {})
            blocked_since = entry.get("blocked_since")

            # Check block expiration
            if blocked_since:
                try:
                    block_time = datetime.fromisoformat(blocked_since)
                    expiry = block_time + timedelta(hours=cfg.sentiment_pause_hours)
                    if datetime.now(UTC) >= expiry:
                        # Block expired
                        blocked_since = None
                except (ValueError, TypeError):
                    blocked_since = None

            # Block if sentiment crashes below threshold
            if blocked_since is None and score < cfg.sentiment_threshold:
                blocked_since = timestamp
                log_general.info(
                    f"Sentiment circuit breaker: {symbol} blocked "
                    f"(score {score:.2f} < {cfg.sentiment_threshold})"
                )

            existing_data[symbol] = {
                "score": round(score, 4),
                "blocked_since": blocked_since,
                "last_check": timestamp,
            }
        except Exception as e:  # noqa: BLE001 - scoring failure; log and continue
            log_general.warning(
                f"Sentiment: failed to compute score for {symbol}: {e}"
            )

    _save_sentiment_data(existing_data)


def get_sentiment(token_symbol: str) -> float:
    """Return sentiment score from -1 to +1.

    Returns 0.0 (neutral) if sentiment is disabled or no data available.
    """
    cfg = config()
    if not cfg.sentiment_enabled:
        return 0.0

    data = _load_sentiment_data()
    entry = data.get(token_symbol, {})
    return entry.get("score", 0.0)


def is_token_blocked(token_symbol: str) -> bool:
    """Return True if trading this token is paused due to bad sentiment."""
    cfg = config()
    if not cfg.sentiment_enabled:
        return False

    data = _load_sentiment_data()
    entry = data.get(token_symbol, {})
    blocked_since = entry.get("blocked_since")

    if not blocked_since:
        return False

    # Check expiration
    try:
        block_time = datetime.fromisoformat(blocked_since)
        expiry = block_time + timedelta(hours=cfg.sentiment_pause_hours)
        return datetime.now(UTC) < expiry
    except (ValueError, TypeError):
        return False


def is_market_crash() -> bool:
    """Return True if sentiment crashed across all tokens.

    A market crash is declared when all tracked tokens have sentiment
    below the crash threshold.
    """
    cfg = config()
    if not cfg.sentiment_enabled:
        return False

    data = _load_sentiment_data()
    if not data:
        return False

    # Check if all tokens are below crash threshold
    for entry in data.values():
        score = entry.get("score", 0.0)
        if score >= cfg.sentiment_crash_threshold:
            return False  # At least one token is not crashed

    return len(data) > 0
