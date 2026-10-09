"""Test doubles: deterministic synthetic candles and a ccxt-like exchange."""

import math

import numpy as np
import pandas as pd


def make_ticks(n_bars: int, ticks_per_bar: int = 6, seed: int = 7, start: float = 100.0) -> list[list[float]]:
    """Intrabar price paths: each bar opens at the previous close; some bars wick down or up."""
    rng = np.random.default_rng(seed)
    bars = []
    price = start
    for _ in range(n_bars):
        path = [price]
        for _ in range(ticks_per_bar - 1):
            price *= math.exp(rng.normal(0.0002, 0.006))
            path.append(price)
        if rng.random() < 0.08:  # a wick mid-bar that recovers
            k = int(rng.integers(1, ticks_per_bar - 1))
            path[k] *= 0.97 if rng.random() < 0.6 else 1.03
        bars.append(path)
        price = path[-1]
    return bars


def bars_frame(ticks: list[list[float]], period: int, t0: int = 1_700_000_000) -> pd.DataFrame:
    """Final OHLCV bars of a tick path; ``t0`` is aligned down to the period."""
    t0 -= t0 % period
    rows = []
    for i, path in enumerate(ticks):
        rows.append((t0 + i * period, path[0], max(path), min(path), path[-1], 100.0 + i % 7))
    return pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])


class FakeExchange:
    """ccxt-like ``fetch_ohlcv`` over a tick path, as of ``clock()``.

    Closed bars come back complete; the bar still forming comes back with the
    high, low and close of the ticks seen so far, like a real exchange.
    """

    id = "fake"

    def __init__(
        self, ticks: list[list[float]], period: int, clock, t0: int = 1_700_000_000, windowed: bool = False
    ) -> None:
        self.timeframes = {"1m": "1m", "5m": "5m", "1h": "1h", "1d": "1d"}
        # Like OKX: a page covers [since, since + limit bars) only, so it's empty before the history starts.
        self.windowed = windowed
        self.ticks = ticks
        self.period = period
        self.clock = clock
        self.t0 = t0 - t0 % period
        self.calls: list[tuple] = []

    def fetch_ohlcv(self, pair, timeframe="1m", since=None, limit=100):
        self.calls.append((pair, timeframe, since, limit))
        now = self.clock()
        current = int(now // self.period) * self.period
        requested = int((since or 0) // 1000)
        start = max(requested, self.t0)
        if self.windowed and start >= requested + limit * self.period:
            return []
        first = max((start - self.t0 + self.period - 1) // self.period, 0)
        out = []
        for i in range(first, len(self.ticks)):
            ts = self.t0 + i * self.period
            if ts > current:
                break
            path = self.ticks[i]
            if ts == current:  # forming: only the ticks seen so far
                seen = max(1, min(len(path), int((now - ts) / self.period * len(path)) + 1))
                path = path[:seen]
            out.append([ts * 1000, path[0], max(path), min(path), path[-1], 100.0 + i % 7])
            if len(out) >= limit:
                break
        return out
