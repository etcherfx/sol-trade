"""Backtester: replays stored closed bars through the live decision engine.

Signals come from :meth:`Strategy.evaluate` on the same ``lookback``-bar
windows the live loop builds, and every bar runs the engine's three steps
(:meth:`Book.on_open`, :meth:`Book.on_low`, :meth:`Book.on_close`) with the
shared cost model. Each trade books ``risk.max_trade_size`` of notional;
equity is marked at every close. Nothing here calls the Solana RPC.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from sol_trade import costs as cost_module
from sol_trade.config import config
from sol_trade.costs import CostModel
from sol_trade.engine import FORCED, Book, SimExecutor, Trade
from sol_trade.spec import COLUMNS, Strategy, windows

_CHUNK = 4000  # windows evaluated per batch
SECONDS_PER_YEAR = 365.25 * 86400


def compute_signals(
    strategy: Strategy,
    frame: pd.DataFrame,
    symbol: str | None = None,
    sentiment: Callable[..., Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Entry and exit flags per bar of ``frame``; bar ``i`` uses the window ending at ``i``.

    Bars without a full window of history get no signal.
    """
    n = len(frame)
    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    length = strategy.lookback
    if n < length:
        return entry, exit_
    series = {c: frame[c].to_numpy(dtype=float) for c in COLUMNS}
    times = frame["ts"].to_numpy(dtype=float)
    for start in range(0, n - length + 1, _CHUNK):
        stop = min(start + _CHUNK, n - length + 1)  # window starts in [start, stop)
        span = slice(start, stop + length - 1)
        bars = {c: windows(v[span], length) for c, v in series.items()}
        result = strategy.evaluate(bars, windows(times[span], length), symbol, sentiment)
        entry[start + length - 1 : stop + length - 1] = result.entry
        exit_[start + length - 1 : stop + length - 1] = result.exit
    return entry, exit_


@dataclass
class Simulation:
    equity: np.ndarray  # marked at each close of the simulated bars
    trades: list[Trade]
    exposure: float
    costs_paid: float
    capital: float
    ts: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))


def simulate(
    strategy: Strategy,
    symbol: str,
    frame: pd.DataFrame,
    entry: np.ndarray,
    exit_: np.ndarray,
    start: int,
    end: int,
    model: CostModel,
    capital: float,
    force_close: bool = True,
) -> Simulation:
    """Run bars ``[start, end)`` of ``frame`` through a fresh book that starts flat."""
    book = Book("backtest", capital, SimExecutor(lambda _symbol: model), capital, capital)
    risk = strategy.risk
    ts = frame["ts"].to_numpy()
    opens, highs, lows, closes = (frame[c].to_numpy(dtype=float) for c in ("open", "high", "low", "close"))
    equity = np.empty(max(end - start, 0))
    in_position = 0
    for k, i in enumerate(range(start, end)):
        bar_ts = int(ts[i])
        book.on_open(symbol, bar_ts, opens[i], risk)
        book.on_low(symbol, bar_ts, opens[i], lows[i], risk)
        book.on_close(
            symbol, bar_ts, highs[i], closes[i], bool(entry[i]), bool(exit_[i]), risk,
            strategy.name, strategy.hash,
        )
        if book.position(symbol) is not None:
            in_position += 1
        equity[k] = book.equity({symbol: closes[i]})
    if force_close and end > start and book.position(symbol) is not None:
        book.close_position(symbol, int(ts[end - 1]), closes[end - 1], FORCED)
        equity[-1] = book.cash
    return Simulation(
        equity=equity,
        trades=list(book.trades),
        exposure=in_position / max(end - start, 1),
        costs_paid=sum(f.cost for f in book.fills),
        capital=capital,
        ts=ts[start:end].astype(np.int64),
    )


# ----------------------------------------------------------------- metrics


def bar_returns(equity: np.ndarray, capital: float) -> np.ndarray:
    path = np.concatenate([[capital], equity])
    return path[1:] / path[:-1] - 1


def max_drawdown(equity: np.ndarray, capital: float) -> float:
    path = np.concatenate([[capital], equity])
    peaks = np.maximum.accumulate(path)
    return float(np.max(1 - path / peaks)) if path.size else 0.0


def sharpe_per_bar(returns: np.ndarray) -> float:
    if returns.size < 2:
        return 0.0
    sd = float(np.std(returns, ddof=1))
    return float(np.mean(returns)) / sd if sd > 0 else 0.0


def metrics(sim: Simulation, period_seconds: int) -> dict[str, Any]:
    """After-cost metrics of a simulation (returns and drawdown in percent)."""
    returns = bar_returns(sim.equity, sim.capital)
    annual = math.sqrt(SECONDS_PER_YEAR / period_seconds)
    sr = sharpe_per_bar(returns)
    downside = math.sqrt(float(np.mean(np.minimum(returns, 0.0) ** 2))) if returns.size else 0.0
    wins = [t.pnl for t in sim.trades if t.pnl > 0]
    losses = [-t.pnl for t in sim.trades if t.pnl <= 0]
    final = float(sim.equity[-1]) if sim.equity.size else sim.capital
    return {
        "total_return_pct": round((final / sim.capital - 1) * 100, 4),
        "max_drawdown_pct": round(max_drawdown(sim.equity, sim.capital) * 100, 4),
        "sharpe": round(sr * annual, 4),
        "sortino": round(float(np.mean(returns)) / downside * annual, 4) if downside > 0 else 0.0,
        "sharpe_per_bar": sr,
        "trades": len(sim.trades),
        "exposure_pct": round(sim.exposure * 100, 2),
        "win_rate_pct": round(len(wins) / len(sim.trades) * 100, 2) if sim.trades else 0.0,
        "profit_factor": round(float(sum(wins) / sum(losses)), 4) if losses and sum(losses) > 0 else None,
        "costs_paid": round(float(sim.costs_paid), 4),
        "bars": int(sim.equity.size),
        **moments(returns),
    }


def moments(returns: np.ndarray) -> dict[str, float]:
    """Population skewness and (non-excess) kurtosis of bar returns; normal is 0 and 3."""
    if returns.size < 3:
        return {"skew": 0.0, "kurtosis": 3.0}
    centered = returns - returns.mean()
    m2 = float(np.mean(centered**2))
    if m2 <= 0:
        return {"skew": 0.0, "kurtosis": 3.0}
    return {
        "skew": float(np.mean(centered**3)) / m2**1.5,
        "kurtosis": float(np.mean(centered**4)) / m2**2,
    }


def buy_and_hold(frame: pd.DataFrame, start: int, end: int, model: CostModel, capital: float) -> Simulation:
    """Buy at the first open, hold, sell at the last close: one round trip of costs."""
    opens, closes = frame["open"].to_numpy(dtype=float), frame["close"].to_numpy(dtype=float)
    notional = capital - model.network_fee_usd
    entry_cost = model.cost_usd(notional)
    size = notional * (1 - model.bps(notional) / 1e4) / opens[start]
    cash = capital - notional - model.network_fee_usd
    equity = cash + size * closes[start:end]
    if equity.size:
        exit_cost = model.cost_usd(size * closes[end - 1])
        equity[-1] = cash + size * closes[end - 1] - exit_cost
    else:
        exit_cost = 0.0
    return Simulation(equity, [], 1.0, entry_cost + exit_cost, capital)


def benchmarks(frame: pd.DataFrame, start: int, end: int, model: CostModel, capital: float, period_seconds: int) -> dict[str, Any]:
    hold = metrics(buy_and_hold(frame, start, end, model, capital), period_seconds)
    return {
        "buy_and_hold": {k: hold[k] for k in ("total_return_pct", "max_drawdown_pct", "sharpe", "sortino")},
        "cash": {"total_return_pct": 0.0, "max_drawdown_pct": 0.0, "sharpe": 0.0, "sortino": 0.0},
    }


def run(
    strategy: Strategy,
    symbol: str,
    frame: pd.DataFrame,
    start: int,
    end: int,
    model: CostModel | None = None,
    signals: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Backtest bars ``[start, end)`` of ``frame`` and return metrics plus benchmarks.

    ``start`` must leave ``lookback - 1`` bars of history before it.
    """
    model = model or cost_module.model_for(symbol)
    capital = float(config().risk["max_trade_size"])
    entry, exit_ = signals or compute_signals(strategy, frame, symbol)
    sim = simulate(strategy, symbol, frame, entry, exit_, start, end, model, capital)
    return {
        "metrics": metrics(sim, strategy.period_seconds),
        "benchmarks": benchmarks(frame, start, end, model, capital, strategy.period_seconds),
        "trades": [t.to_dict() for t in sim.trades],
        "simulation": sim,
        "costs": model.as_dict(),
    }
