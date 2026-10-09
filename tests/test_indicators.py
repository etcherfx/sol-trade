"""Indicators must keep the values the old pandas implementations produced, bit for bit."""

import numpy as np
import pandas as pd
import pytest

from sol_trade import indicators as ind
from sol_trade.spec import windows


def _reference_ema(close: pd.Series, period: int) -> pd.Series:
    """The pre-JSON implementation (TA-Lib equivalent)."""
    out = np.full(len(close), np.nan)
    if len(close) < period:
        return pd.Series(out, index=close.index)
    alpha = 2.0 / (period + 1)
    out[period - 1] = close.iloc[:period].mean()
    for i in range(period, len(close)):
        out[i] = alpha * close.iloc[i] + (1 - alpha) * out[i - 1]
    return pd.Series(out, index=close.index)


def _reference_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    n = len(close)
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    out = np.full(n, np.nan)
    if n <= period:
        return pd.Series(out, index=close.index)
    avg_gain[period] = gain.iloc[1 : period + 1].mean()
    avg_loss[period] = loss.iloc[1 : period + 1].mean()
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain.iloc[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss.iloc[i]) / period
    total = avg_gain + avg_loss
    for i in range(period, n):
        out[i] = 0.0 if total[i] == 0.0 else 100.0 * avg_gain[i] / total[i]
    return pd.Series(out, index=close.index)


@pytest.fixture(scope="module")
def close() -> pd.Series:
    rng = np.random.default_rng(1)
    return pd.Series(100 + np.cumsum(rng.normal(0, 1, 600)))


def _col(series: pd.Series) -> np.ndarray:
    return series.to_numpy()[:, None]


def _same(a: np.ndarray, b) -> bool:
    return np.array_equal(np.asarray(a).ravel(), np.asarray(b, dtype=float), equal_nan=True)


@pytest.mark.parametrize("period", [5, 9, 21, 50])
def test_ema_matches_reference(close, period):
    assert _same(ind.ema(_col(close), period), _reference_ema(close, period))


@pytest.mark.parametrize("period", [7, 14])
def test_rsi_matches_reference(close, period):
    assert _same(ind.rsi(_col(close), period), _reference_rsi(close, period))


def test_rsi_of_a_flat_series_is_zero():
    flat = np.full((30, 1), 5.0)
    assert ind.rsi(flat, 14)[-1, 0] == 0.0


def test_rolling_indicators_match_pandas(close):
    assert _same(ind.sma(_col(close), 20), close.rolling(20).mean())
    assert _same(ind.stddev(_col(close), 20), close.rolling(20).std())
    assert _same(ind.roc(_col(close), 10), close.pct_change(10) * 100)
    assert _same(ind.highest(_col(close), 5), close.rolling(5).max())
    assert _same(ind.lowest(_col(close), 5), close.rolling(5).min())


def test_batched_windows_equal_single_windows(close):
    batch = windows(close.to_numpy(), 50)
    ema, rsi, sma = ind.ema(batch, 21), ind.rsi(batch, 14), ind.sma(batch, 20)
    for j in range(0, batch.shape[1], 37):
        window = pd.Series(batch[:, j])
        assert _same(ema[:, j], _reference_ema(window, 21))
        assert _same(rsi[:, j], _reference_rsi(window, 14))
        assert _same(sma[:, j], window.rolling(20).mean())


def test_macd_signal_seeds_after_the_macd_warmup(close):
    line = _reference_ema(close, 12) - _reference_ema(close, 26)
    signal = _reference_ema(line.dropna(), 9).reindex(close.index)
    out = ind.macd(_col(close))
    assert _same(out["macd"], line)
    assert _same(out["signal"], signal)


def test_simple_atr_matches_a_pandas_true_range(close):
    high, low = close + 1.5, close - 1.0
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    out = ind.atr(_col(high), _col(low), _col(close), 14, "sma")
    assert _same(out, tr.rolling(14).mean())


def test_wilder_atr_seeds_at_period():
    high = np.arange(1.0, 31.0)[:, None] + 1
    low = high - 2
    close = high - 1
    out = ind.atr(high, low, close, 14, "wilder")
    assert np.isnan(out[13, 0])
    assert out[14, 0] == pytest.approx(2.0)


def test_bbands_outputs(close):
    out = ind.bbands(_col(close), 20, 2.0)
    middle = close.rolling(20).mean()
    deviation = close.rolling(20).std()
    assert _same(out["upper"], middle + 2 * deviation)
    assert _same(out["width"], ((middle + 2 * deviation) - (middle - 2 * deviation)) / middle)
