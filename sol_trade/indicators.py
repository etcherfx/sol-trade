"""Indicator library for JSON strategies.

Every function takes 2-D float arrays shaped ``(L, N)``: rows are bars in time
order, columns are independent windows. Live trading evaluates one window
(``N == 1``); the backtester evaluates thousands at once with the same code.
Values are NaN until an indicator has enough bars, and an indicator fed by
another one starts counting at that input's first valid bar.

``sma``, ``ema`` and ``rsi`` match TA-Lib's definitions. Rolling means and
standard deviations use pandas' rolling kernels, so they equal
``Series.rolling(...)`` bit for bit.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd

Array = np.ndarray


def _empty(x: Array) -> Array:
    return np.full(x.shape, np.nan)


def _first_valid_groups(x: Array) -> list[tuple[int, Array]]:
    """Group columns by the row of their first non-NaN value."""
    valid = ~np.isnan(x)
    has = valid.any(axis=0)
    first = np.argmax(valid, axis=0)
    groups = []
    for row in np.unique(first[has]):
        groups.append((int(row), np.flatnonzero(has & (first == row))))
    return groups


def _column_mean(block: Array) -> Array:
    """Mean of each column, summed like pandas' ``Series.mean`` on one column."""
    return np.ascontiguousarray(block.T).sum(axis=1) / block.shape[0]


def shift(x: Array, n: int) -> Array:
    out = _empty(x)
    if n == 0:
        return x.copy()
    if n < x.shape[0]:
        out[n:] = x[:-n]
    return out


def sma(x: Array, period: int) -> Array:
    """Simple moving average (pandas rolling mean)."""
    return np.asarray(pd.DataFrame(x).rolling(period).mean(), dtype=float)


def stddev(x: Array, period: int, ddof: int = 1) -> Array:
    """Rolling standard deviation; ``ddof=1`` matches pandas' default."""
    return np.asarray(pd.DataFrame(x).rolling(period).std(ddof=ddof), dtype=float)


def ema(x: Array, period: int) -> Array:
    """Exponential moving average, TA-Lib equivalent.

    Seeded with the SMA of the first ``period`` valid values, then
    ``alpha * x + (1 - alpha) * previous`` with ``alpha = 2 / (period + 1)``.
    """
    out = _empty(x)
    alpha = 2.0 / (period + 1)
    rows = x.shape[0]
    for first, cols in _first_valid_groups(x):
        start = first + period - 1
        if start >= rows:
            continue
        values = x[:, cols]
        previous = _column_mean(values[first : first + period])
        out[start, cols] = previous
        for i in range(start + 1, rows):
            previous = alpha * values[i] + (1 - alpha) * previous
            out[i, cols] = previous
    return out


def rsi(x: Array, period: int = 14) -> Array:
    """Relative Strength Index, TA-Lib equivalent (Wilder smoothing).

    A flat series gives 0, as in TA-Lib, not NaN or 100.
    """
    out = _empty(x)
    rows = x.shape[0]
    for first, cols in _first_valid_groups(x):
        start = first + period
        if start >= rows:
            continue
        values = x[:, cols]
        delta = np.full(values.shape, np.nan)
        delta[1:] = values[1:] - values[:-1]
        gain = np.where(np.isnan(delta), np.nan, np.maximum(delta, 0.0))
        loss = np.where(np.isnan(delta), np.nan, -np.minimum(delta, 0.0))
        avg_gain = _column_mean(gain[first + 1 : start + 1])
        avg_loss = _column_mean(loss[first + 1 : start + 1])
        for i in range(start, rows):
            if i > start:
                avg_gain = (avg_gain * (period - 1) + gain[i]) / period
                avg_loss = (avg_loss * (period - 1) + loss[i]) / period
            total = avg_gain + avg_loss
            with np.errstate(divide="ignore", invalid="ignore"):
                out[i, cols] = np.where(total == 0.0, 0.0, 100.0 * avg_gain / total)
    return out


def _rolling_extreme(x: Array, period: int, reducer: Callable[..., Array]) -> Array:
    out = _empty(x)
    if period <= x.shape[0]:
        view = np.lib.stride_tricks.sliding_window_view(x, period, axis=0)
        out[period - 1 :] = reducer(view, axis=-1)
    return out


def highest(x: Array, period: int) -> Array:
    """Highest value of the last ``period`` bars (NaN while any is missing)."""
    return _rolling_extreme(x, period, np.max)


def lowest(x: Array, period: int) -> Array:
    """Lowest value of the last ``period`` bars (NaN while any is missing)."""
    return _rolling_extreme(x, period, np.min)


def roc(x: Array, period: int) -> Array:
    """Rate of change in percent: ``(x / x[-period] - 1) * 100``."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return (x / shift(x, period) - 1) * 100


def zscore(x: Array, period: int, ddof: int = 1) -> Array:
    with np.errstate(divide="ignore", invalid="ignore"):
        return (x - sma(x, period)) / stddev(x, period, ddof)


def bbands(x: Array, period: int = 20, mult: float = 2.0, ddof: int = 1) -> dict[str, Array]:
    """Bollinger Bands: ``middle`` SMA, ``upper``/``lower`` at ``mult`` deviations, ``width``."""
    middle = sma(x, period)
    deviation = stddev(x, period, ddof)
    upper = middle + mult * deviation
    lower = middle - mult * deviation
    with np.errstate(divide="ignore", invalid="ignore"):
        width = (upper - lower) / middle
    return {"upper": upper, "middle": middle, "lower": lower, "width": width}


def true_range(high: Array, low: Array, close: Array) -> Array:
    """``max(high - low, |high - prev close|, |low - prev close|)``; the first bar is high - low."""
    previous = shift(close, 1)
    return np.fmax(np.fmax(high - low, np.abs(high - previous)), np.abs(low - previous))


def atr(high: Array, low: Array, close: Array, period: int = 14, method: str = "wilder") -> Array:
    """Average true range.

    ``wilder`` (TA-Lib): seeded with the mean true range of bars 1..period,
    then Wilder smoothing. ``sma``: a plain rolling mean of the true range.
    """
    tr = true_range(high, low, close)
    if method == "sma":
        return sma(tr, period)
    out = _empty(tr)
    rows = tr.shape[0]
    if period >= rows:
        return out
    previous = _column_mean(tr[1 : period + 1])
    out[period] = previous
    for i in range(period + 1, rows):
        previous = (previous * (period - 1) + tr[i]) / period
        out[i] = previous
    return out


def macd(x: Array, fast: int = 12, slow: int = 26, signal: int = 9) -> dict[str, Array]:
    """MACD line (fast EMA - slow EMA), its signal EMA and the histogram."""
    line = ema(x, fast) - ema(x, slow)
    signal_line = ema(line, signal)
    return {"macd": line, "signal": signal_line, "hist": line - signal_line}


def stoch(
    high: Array, low: Array, close: Array, k: int = 14, d: int = 3, smooth: int = 3
) -> dict[str, Array]:
    """Stochastic oscillator: %K (smoothed over ``smooth`` bars) and its ``d``-bar mean."""
    low_k = lowest(low, k)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = 100.0 * (close - low_k) / (highest(high, k) - low_k)
    k_line = sma(raw, smooth) if smooth > 1 else raw
    return {"k": k_line, "d": sma(k_line, d)}
