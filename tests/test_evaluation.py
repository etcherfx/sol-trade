"""Walk-forward evaluation: splits, folds, trials, deflated Sharpe and verdicts."""

import math

import numpy as np
import pandas as pd
import pytest

from sol_trade import evaluation, rpc, spec
from sol_trade.config import config
from sol_trade.costs import CostModel

HOUR = 3600
FREE = CostModel("TKN", "measured", ((0.0, 5.0),), 0.0)
SPEC = {
    "schema": 1,
    "name": "swing",
    "timeframe": "1h",
    "lookback": 40,
    "risk": {"stoploss": 8},
    "indicators": {"mean": {"fn": "sma", "period": 30}},
    "entry": ["close", "<", {"mul": [0.975, "mean"]}],
    "exit": ["close", ">", {"mul": [1.02, "mean"]}],
}


def _market(n=4000, t0=1_700_000_000 - 1_700_000_000 % HOUR) -> pd.DataFrame:
    """A falling market with regular swings: buy-and-hold loses, buying dips wins."""
    t = np.arange(n)
    close = 100 * np.exp(-0.00005 * t) * (1 + 0.04 * np.sin(2 * math.pi * t / 30))
    opens = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame(
        {
            "ts": t0 + t * HOUR,
            "open": opens,
            "high": np.maximum(opens, close) * 1.001,
            "low": np.minimum(opens, close) * 0.999,
            "close": close,
            "volume": np.full(n, 10.0),
        }
    )


@pytest.fixture
def strategy():
    return spec.compile_spec(SPEC)


def test_split_holds_out_the_newest_days(strategy):
    frame = _market(3000)
    cut = evaluation.split(frame, strategy, holdout_days=60)
    assert cut.start == strategy.lookback - 1
    assert cut.holdout_bars == 60 * 24
    assert cut.holdout_start + cut.holdout_bars == len(frame)


def test_expected_max_sharpe_and_deflation():
    assert evaluation.expected_max_sharpe(1, 1.0) == 0.0
    assert evaluation.expected_max_sharpe(10, 1.0) == pytest.approx(1.5745, abs=2e-3)
    psr = evaluation.deflated_sharpe(0.1, 101, 0.0, 3.0, 1, 0.0)
    assert psr == pytest.approx(0.8406, abs=1e-3)  # Φ(0.1·√100 / √1.005)
    deflated = evaluation.deflated_sharpe(0.1, 101, 0.0, 3.0, 20, 0.0004)
    assert deflated < psr  # more trials, higher bar


def test_fat_tails_lower_confidence():
    normal = evaluation.deflated_sharpe(0.1, 500, 0.0, 3.0, 1, 0.0)
    fat = evaluation.deflated_sharpe(0.1, 500, -1.0, 12.0, 1, 0.0)
    assert fat < normal


def test_fold_returns_chain_through_the_equity_curve():
    from sol_trade.backtest import Simulation

    sim = Simulation(np.array([110.0, 121.0, 121.0, 108.9]), [], 0.5, 0.0, 100.0)
    assert evaluation.fold_returns(sim, 4) == [10.0, 10.0, 0.0, -10.0]


def test_good_strategy_passes_with_zero_rpc(strategy):
    before = rpc.usage.snapshot()
    result = evaluation.evaluate(strategy, "TKN", _market(), FREE)

    assert result["verdict"] == evaluation.PASS, result["reasons"]
    assert result["reasons"] == []
    assert result["metrics"]["trades"] >= 30
    assert result["benchmarks"]["buy_and_hold"]["total_return_pct"] < 0
    assert sum(r > 0 for r in result["folds"]) >= 3
    assert result["deflated_sharpe"] >= 0.95
    assert result["trial"] == 1 and result["n_trials"] == 1
    assert result["rpc"] == {"calls": 0, "cu": 0}
    assert rpc.usage.snapshot() == before
    assert result["data"]["holdout_bars"] == 60 * 24
    assert "holdout" not in result  # never peeked at during selection


def test_trials_count_distinct_hashes(strategy):
    frame = _market()
    evaluation.evaluate(strategy, "TKN", frame, FREE)
    again = evaluation.evaluate(strategy, "TKN", frame, FREE)
    assert again["trial"] == 1 and again["n_trials"] == 1  # same hash: same trial

    other = spec.compile_spec({**SPEC, "entry": ["close", "<", {"mul": [0.97, "mean"]}]})
    second = evaluation.evaluate(other, "TKN", frame, FREE)
    assert second["trial"] == 2 and second["n_trials"] == 2
    assert evaluation.evaluate(strategy, "OTHER", frame, FREE)["n_trials"] == 1  # per symbol
    rows = evaluation.trials("TKN")
    assert [r["verdict"] for r in rows] == [again["verdict"], second["verdict"]]


def test_too_few_trades_is_insufficient_evidence(strategy, monkeypatch):
    monkeypatch.setitem(config().gates, "min_trades", 10_000)
    result = evaluation.evaluate(strategy, "TKN", _market(), FREE)
    assert result["verdict"] == evaluation.INSUFFICIENT
    assert result["reasons"][0].startswith("trades ")


def test_failing_gate_is_a_fail(strategy, monkeypatch):
    monkeypatch.setitem(config().risk, "max_drawdown_pct", 0.01)
    monkeypatch.setitem(config().gates, "min_trades", 10_000)
    result = evaluation.evaluate(strategy, "TKN", _market(), FREE)
    assert result["verdict"] == evaluation.FAIL  # a failed gate outranks thin evidence
    assert any(r.startswith("max_drawdown_pct") for r in result["reasons"])


def test_gaps_are_flagged(strategy):
    frame = _market()
    gappy = frame.drop(index=range(500, 700)).reset_index(drop=True)
    result = evaluation.evaluate(strategy, "TKN", gappy, FREE)
    assert evaluation.DATA_GAPS in result["flags"]
    assert result["verdict"] == evaluation.INSUFFICIENT


def test_default_costs_are_flagged(strategy):
    default = CostModel("TKN", "default", ((0.0, 50.0),), 0.0)
    result = evaluation.evaluate(strategy, "TKN", _market(), default)
    assert evaluation.COSTS_DEFAULT in result["flags"]


def test_short_history_is_insufficient(strategy):
    result = evaluation.evaluate(strategy, "TKN", _market(60 * 24 + 41), FREE)
    assert result["verdict"] == evaluation.INSUFFICIENT
    assert "run backfill" in result["reasons"][0]
    assert evaluation.trials() == []


def test_holdout_runs_only_the_held_out_bars(strategy):
    result = evaluation.holdout(strategy, "TKN", _market(), FREE)
    assert result["passed"], result["reasons"]
    assert result["data"]["bars"] == 60 * 24
    assert result["metrics"]["bars"] == 60 * 24
    assert result["rpc"] == {"calls": 0, "cu": 0}


def test_holdout_fails_on_losses(strategy):
    losing = spec.compile_spec({**SPEC, "entry": SPEC["exit"], "exit": SPEC["entry"]})  # buys the tops
    result = evaluation.holdout(losing, "TKN", _market(), FREE)
    assert not result["passed"]
    assert any(r.startswith("holdout_return_pct") for r in result["reasons"])
