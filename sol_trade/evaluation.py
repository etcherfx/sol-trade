"""Walk-forward evaluation: selection folds, the held-out period, trials and gates.

History is split once: the newest ``gates.holdout_days`` are held out and
only looked at by :func:`holdout` when a strategy is promoted. The rest is
the selection period, run as one continuous backtest whose equity curve is
sliced into sequential folds. Every evaluated hash is registered as a trial
for its (symbol, timeframe), and the deflated Sharpe ratio corrects the
selection Sharpe for the number of trials tried there.

Nothing here calls the Solana RPC; the ``rpc`` block of a result proves it.
"""

import math
import time
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd

from sol_trade import backtest, costs, history, ledger, rpc
from sol_trade.config import config
from sol_trade.costs import CostModel
from sol_trade.engine import FORCED
from sol_trade.spec import Strategy

PASS = "PASS"
FAIL = "FAIL"
INSUFFICIENT = "INSUFFICIENT_EVIDENCE"

COSTS_DEFAULT = "COSTS_DEFAULT"
COSTS_STALE = "COSTS_STALE"
DATA_GAPS = "DATA_GAPS"
PRICE_PROXY = "PRICE_PROXY"  # candles quoted in another token than the one swaps pay with

_EULER_GAMMA = 0.5772156649015329
_NORMAL = NormalDist()


@dataclass(frozen=True)
class Split:
    start: int  # first bar with a full lookback window
    holdout_start: int  # first held-out bar; the selection period is [start, holdout_start)
    end: int

    @property
    def selection_bars(self) -> int:
        return max(self.holdout_start - self.start, 0)

    @property
    def holdout_bars(self) -> int:
        return max(self.end - self.holdout_start, 0)


def split(frame: pd.DataFrame, strategy: Strategy, holdout_days: float | None = None) -> Split:
    days = config().gates["holdout_days"] if holdout_days is None else holdout_days
    n = len(frame)
    start = strategy.lookback - 1
    if n == 0:
        return Split(start, start, 0)
    ts = frame["ts"].to_numpy()
    cutoff = int(ts[-1]) + strategy.period_seconds - int(days * 86400)
    holdout_start = int(np.searchsorted(ts, cutoff, side="left"))
    return Split(start, max(holdout_start, start), n)


# ------------------------------------------------------------ deflated Sharpe


def expected_max_sharpe(n_trials: int, variance: float) -> float:
    """Expected best Sharpe of ``n_trials`` skill-less trials (Bailey & López de Prado)."""
    if n_trials < 2 or variance <= 0:
        return 0.0
    return math.sqrt(variance) * (
        (1 - _EULER_GAMMA) * _NORMAL.inv_cdf(1 - 1 / n_trials)
        + _EULER_GAMMA * _NORMAL.inv_cdf(1 - 1 / (n_trials * math.e))
    )


def deflated_sharpe(
    sharpe: float, observations: int, skew: float, kurtosis: float, n_trials: int, variance: float
) -> float:
    """Probability that the per-bar ``sharpe`` beats the best of ``n_trials`` lucky trials.

    ``kurtosis`` is non-excess (3 for normal returns). With one trial this
    is the probabilistic Sharpe ratio against zero.
    """
    if observations < 2:
        return 0.0
    benchmark = expected_max_sharpe(n_trials, variance)
    spread = 1 - skew * sharpe + (kurtosis - 1) / 4 * sharpe**2
    if spread <= 0:
        return 0.0
    return _NORMAL.cdf((sharpe - benchmark) * math.sqrt(observations - 1) / math.sqrt(spread))


# -------------------------------------------------------------------- trials


def register_trial(
    strategy: Strategy, symbol: str, sharpe: float, bars: int, trades: int, verdict: str, first_ts: int, last_ts: int
) -> dict[str, Any]:
    """Record (or refresh) this hash's trial.

    Returns its number among the (symbol, timeframe) trials, the trial
    count and the variance of their per-bar Sharpe ratios.
    """
    now = time.time()
    with ledger.connect() as conn:
        conn.execute(
            """
            INSERT INTO trials (strategy, strategy_hash, symbol, timeframe, sharpe, bars, trades, verdict,
                                first_ts, last_ts, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (strategy_hash, symbol, timeframe) DO UPDATE SET
                strategy = excluded.strategy, sharpe = excluded.sharpe, bars = excluded.bars,
                trades = excluded.trades, verdict = excluded.verdict, first_ts = excluded.first_ts,
                last_ts = excluded.last_ts, updated_at = excluded.updated_at
            """,
            (strategy.name, strategy.hash, symbol, strategy.timeframe, sharpe, bars, trades, verdict,
             first_ts, last_ts, now, now),
        )
        rows = conn.execute(
            "SELECT strategy_hash, sharpe FROM trials WHERE symbol = ? AND timeframe = ? ORDER BY id",
            (symbol, strategy.timeframe),
        ).fetchall()
    number = next(i for i, row in enumerate(rows, 1) if row[0] == strategy.hash)
    return {"trial": number, **trial_stats([row[1] for row in rows])}


def trial_stats(sharpes: list[float]) -> dict[str, Any]:
    variance = float(np.var(sharpes, ddof=1)) if len(sharpes) > 1 else 0.0
    return {"n_trials": len(sharpes), "trial_sharpe_variance": variance}


def trials(symbol: str | None = None) -> list[dict[str, Any]]:
    with ledger.connect() as conn:
        conn.row_factory = _dict_row
        return conn.execute(
            "SELECT * FROM trials WHERE ? IS NULL OR symbol = ? ORDER BY id", (symbol, symbol)
        ).fetchall()


def _dict_row(cursor: Any, row: tuple) -> dict[str, Any]:
    return {d[0]: row[i] for i, d in enumerate(cursor.description)}


# --------------------------------------------------------------- evaluation


def _gate(name: str, value: Any, threshold: Any, passed: bool) -> dict[str, Any]:
    return {"gate": name, "value": value, "threshold": threshold, "passed": bool(passed)}


def fold_returns(sim: backtest.Simulation, folds: int) -> list[float]:
    """Return (percent) of each sequential slice of the equity curve."""
    returns = []
    previous = sim.capital
    for chunk in np.array_split(sim.equity, folds):
        if chunk.size == 0:
            returns.append(0.0)
            continue
        returns.append(round((float(chunk[-1]) / previous - 1) * 100, 4))
        previous = float(chunk[-1])
    return returns


def _iso(ts: int | None) -> str | None:
    return history._iso(ts)


def _data_info(frame: pd.DataFrame, strategy: Strategy, cut: Split, symbol: str) -> dict[str, Any]:
    selection = frame.iloc[cut.start : cut.holdout_start]
    ts = frame["ts"]
    return {
        "source": history.source_for(symbol).as_dict(),
        "timeframe": strategy.timeframe,
        "bars": len(frame),
        "first": _iso(int(ts.iat[0])) if len(frame) else None,
        "last": _iso(int(ts.iat[-1])) if len(frame) else None,
        "selection_bars": cut.selection_bars,
        "selection_end": _iso(int(ts.iat[cut.holdout_start])) if cut.holdout_start < len(frame) else None,
        "holdout_bars": cut.holdout_bars,
        "gap_fraction": round(history.gap_fraction(selection, strategy.timeframe), 6),
        "capital": backtest.sizing()[0],
        "position_notional": backtest.position_notional(strategy),
    }


def _cost_flags(model: CostModel, symbol: str) -> list[str]:
    if model.source == "default":
        return [COSTS_DEFAULT]
    age = costs.sample_age_days(symbol)
    if age is not None and age > config().costs["max_sample_age_days"]:
        return [COSTS_STALE]
    return []


def load_frame(strategy: Strategy, symbol: str) -> pd.DataFrame:
    return history.load_bars(history.source_for(symbol), strategy.timeframe)


def evaluate(
    strategy: Strategy,
    symbol: str,
    frame: pd.DataFrame | None = None,
    model: CostModel | None = None,
    record: bool = True,
) -> dict[str, Any]:
    """Backtest the selection period and judge it against the gates.

    The verdict is FAIL when any gate fails, INSUFFICIENT_EVIDENCE when the
    gates pass but there are too few trades or the data has gaps, and PASS
    otherwise. The held-out period is not touched.
    """
    before = rpc.usage.snapshot()
    gates = config().gates
    frame = load_frame(strategy, symbol) if frame is None else frame
    model = model or costs.model_for(symbol)
    cut = split(frame, strategy)
    data = _data_info(frame, strategy, cut, symbol)
    flags = _cost_flags(model, symbol)
    if history.source_for(symbol).proxy:
        flags.append(PRICE_PROXY)
    result: dict[str, Any] = {
        "strategy": strategy.name,
        "hash": strategy.hash,
        "symbol": symbol,
        "data": data,
        "costs": model.as_dict(),
        "flags": flags,
    }
    if cut.selection_bars < max(gates["folds"], 2):
        result.update(
            verdict=INSUFFICIENT,
            reasons=[f"only {cut.selection_bars} selection bars of {strategy.timeframe} history; run backfill"],
            rpc=_rpc_delta(before),
        )
        return result

    signals = backtest.compute_signals(strategy, frame, symbol)
    capital, trade_size = backtest.sizing()
    notional = backtest.position_notional(strategy)
    sim = backtest.simulate(strategy, symbol, frame, *signals, cut.start, cut.holdout_start, model, capital, trade_size)
    metrics = backtest.metrics(sim, strategy.period_seconds)
    bench = backtest.benchmarks(frame, cut.start, cut.holdout_start, model, capital, strategy.period_seconds, notional)
    folds = fold_returns(sim, gates["folds"])
    if data["gap_fraction"] > gates["max_gap_fraction"]:
        flags.append(DATA_GAPS)

    first_ts, last_ts = int(frame["ts"].iat[cut.start]), int(frame["ts"].iat[cut.holdout_start - 1])
    if record:
        trial = register_trial(strategy, symbol, metrics["sharpe_per_bar"], metrics["bars"], metrics["trades"],
                               "pending", first_ts, last_ts)
    else:
        trial = {"trial": None, **trial_stats([metrics["sharpe_per_bar"]])}
    dsr = deflated_sharpe(
        metrics["sharpe_per_bar"], metrics["bars"], metrics["skew"], metrics["kurtosis"],
        trial["n_trials"], trial["trial_sharpe_variance"],
    )
    max_dd = float(config().risk["max_drawdown_pct"])
    checks = [
        _gate("max_drawdown_pct", metrics["max_drawdown_pct"], max_dd, metrics["max_drawdown_pct"] <= max_dd),
        _gate("sharpe", metrics["sharpe"], 0, metrics["sharpe"] > 0),
        _gate("sharpe_vs_buy_and_hold", metrics["sharpe"], bench["buy_and_hold"]["sharpe"],
              metrics["sharpe"] > bench["buy_and_hold"]["sharpe"]),
        _gate("positive_folds", sum(r > 0 for r in folds), gates["min_positive_folds"],
              sum(r > 0 for r in folds) >= gates["min_positive_folds"]),
        _gate("deflated_sharpe", round(dsr, 4), gates["min_deflated_sharpe"], dsr >= gates["min_deflated_sharpe"]),
    ]
    evidence = [
        _gate("trades", metrics["trades"], gates["min_trades"], metrics["trades"] >= gates["min_trades"]),
        _gate("gap_fraction", data["gap_fraction"], gates["max_gap_fraction"], DATA_GAPS not in flags),
    ]
    reasons = [_describe(g) for g in checks + evidence if not g["passed"]]
    if not all(g["passed"] for g in checks):
        verdict = FAIL
    elif not all(g["passed"] for g in evidence):
        verdict = INSUFFICIENT
    else:
        verdict = PASS
    if record:
        with ledger.connect() as conn:
            conn.execute(
                "UPDATE trials SET verdict = ? WHERE strategy_hash = ? AND symbol = ? AND timeframe = ?",
                (verdict, strategy.hash, symbol, strategy.timeframe),
            )

    metrics = {k: v for k, v in metrics.items() if k != "sharpe_per_bar"}
    result.update(
        verdict=verdict,
        reasons=reasons,
        metrics=metrics,
        folds=folds,
        benchmarks=bench,
        gates=checks + evidence,
        deflated_sharpe=round(dsr, 4),
        trial=trial["trial"],
        n_trials=trial["n_trials"],
        rpc=_rpc_delta(before),
    )
    return result


def required_holdout_trades(sim: backtest.Simulation, holdout_days: float) -> int:
    """Closed trades a held-out period needs before it can be judged.

    ``gates.holdout_min_trades``, scaled down to what the strategy's trade
    rate in ``sim`` (the selection period) predicts for ``holdout_days``, and
    never below one: a strategy that trades ten times a year can't be expected
    to close eight trades in 60 days.
    """
    floor = int(config().gates["holdout_min_trades"])
    if sim.ts.size < 2:
        return floor
    days = (int(sim.ts[-1]) - int(sim.ts[0])) / 86400
    closed = sum(t.reason != FORCED for t in sim.trades)
    if days <= 0:
        return floor
    return min(floor, max(1, math.ceil(closed / days * holdout_days)))


def holdout(
    strategy: Strategy,
    symbol: str,
    frame: pd.DataFrame | None = None,
    model: CostModel | None = None,
    start_ts: int | None = None,
    end_ts: int | None = None,
) -> dict[str, Any]:
    """Run the held-out period from flat and judge it on promotion.

    The verdict is FAIL when the period breaches the drawdown limit, or closes
    enough trades and loses money; INSUFFICIENT_EVIDENCE when it closes too
    few trades to judge (see :func:`required_holdout_trades`); PASS otherwise.
    Only a FAIL rejects a hash. ``start_ts`` pins the start of an earlier
    attempt, so the period grows as bars arrive; ``end_ts`` (exclusive)
    replays a past window. A position still open at the end is marked at the
    last close for the return and drawdown, but only closed trades count.
    """
    before = rpc.usage.snapshot()
    gates = config().gates
    frame = load_frame(strategy, symbol) if frame is None else frame
    model = model or costs.model_for(symbol)
    cut = split(frame, strategy)
    ts = frame["ts"]
    if start_ts is not None or end_ts is not None:
        stamps = ts.to_numpy()
        first = int(np.searchsorted(stamps, start_ts, side="left")) if start_ts is not None else cut.holdout_start
        last = int(np.searchsorted(stamps, end_ts, side="left")) if end_ts is not None else cut.end
        cut = Split(cut.start, max(first, cut.start), max(last, cut.start))
    result: dict[str, Any] = {"strategy": strategy.name, "hash": strategy.hash, "symbol": symbol}
    if cut.holdout_bars == 0 or cut.selection_bars == 0:
        result.update(
            verdict=INSUFFICIENT, passed=False, reasons=["no held-out bars; run backfill"], start_ts=start_ts,
            rpc=_rpc_delta(before),
        )
        return result
    signals = backtest.compute_signals(strategy, frame, symbol)
    capital, trade_size = backtest.sizing()
    selection = backtest.simulate(
        strategy, symbol, frame, *signals, cut.start, cut.holdout_start, model, capital, trade_size
    )
    required = required_holdout_trades(selection, float(gates["holdout_days"]))
    sim = backtest.simulate(strategy, symbol, frame, *signals, cut.holdout_start, cut.end, model, capital, trade_size)
    metrics = backtest.metrics(sim, strategy.period_seconds)
    closed = sum(t.reason != FORCED for t in sim.trades)
    max_dd = float(config().risk["max_drawdown_pct"])
    drawdown = _gate("holdout_max_drawdown_pct", metrics["max_drawdown_pct"], max_dd,
                     metrics["max_drawdown_pct"] <= max_dd)
    evidence = _gate("holdout_trades", closed, required, closed >= required)
    profit = _gate("holdout_return_pct", metrics["total_return_pct"], 0, metrics["total_return_pct"] > 0)
    if not drawdown["passed"]:
        verdict = FAIL
    elif not evidence["passed"]:
        verdict = INSUFFICIENT
    elif not profit["passed"]:
        verdict = FAIL
    else:
        verdict = PASS
    checks = [evidence, profit, drawdown]
    result.update(
        verdict=verdict,
        passed=verdict == PASS,
        reasons=[_describe(g) for g in checks if not g["passed"]],
        metrics={k: v for k, v in metrics.items() if k != "sharpe_per_bar"},
        benchmarks=backtest.benchmarks(frame, cut.holdout_start, cut.end, model, capital, strategy.period_seconds,
                                       backtest.position_notional(strategy)),
        gates=checks,
        start_ts=int(ts.iat[cut.holdout_start]),
        required_trades=required,
        selection_trades=sum(t.reason != FORCED for t in selection.trades),
        data={
            "first": _iso(int(ts.iat[cut.holdout_start])),
            "last": _iso(int(ts.iat[cut.end - 1])),
            "bars": cut.holdout_bars,
        },
        rpc=_rpc_delta(before),
    )
    return result


_ABOVE = {"max_drawdown_pct", "holdout_max_drawdown_pct", "gap_fraction"}


def _describe(gate: dict[str, Any]) -> str:
    relation = "above" if gate["gate"] in _ABOVE else "below"
    if gate["gate"] in {"sharpe", "holdout_return_pct"}:
        relation = "not above"
    if gate["gate"] == "sharpe_vs_buy_and_hold":
        return f"sharpe {gate['value']} does not beat buy-and-hold {gate['threshold']}"
    return f"{gate['gate']} {gate['value']} is {relation} {gate['threshold']}"


def _rpc_delta(before: tuple[int, int]) -> dict[str, int]:
    calls, cu = rpc.usage.snapshot()
    return {"calls": calls - before[0], "cu": cu - before[1]}
