"""Walk-forward evaluation: splits, folds, trials, deflated Sharpe and verdicts."""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sol_trade import backtest, evaluation, rpc, spec
from sol_trade.config import config
from sol_trade.costs import CostModel

HOUR = 3600
FREE = CostModel("TKN", "measured", ((0.0, 5.0),), 0.0)
# Real 6h SOL/USDC bars from Coinbase (2022-08-31 to 2026-10-09), exported from a runtime candle store.
SOL_6H = Path(__file__).parent / "fixtures" / "sol_usdc_6h_coinbase.csv.gz"
SOL_COST = CostModel("SOL", "measured", ((25.0, 2.3), (250.0, 2.3)), 0.0)  # as measured that day
# A pre-registered research strategy, verbatim: its hash must not change.
TREND_EMA_6H = {
    "schema": 1,
    "name": "trend_ema_6h",
    "description": "Pre-registered 2026-10-09. Time-series trend following: 6h EMA 21/84 (~5d/21d) crossover, "
    "flat otherwise. Wide disaster stop only; exits on the reverse cross. Basis: TSMOM/trend evidence in crypto.",
    "timeframe": "6h",
    "lookback": 200,
    "risk": {"stoploss": 12, "takeprofit": 0, "trailing_stoploss": 0, "trailing_stoploss_target": 0},
    "indicators": {"fast": {"fn": "ema", "period": 21}, "slow": {"fn": "ema", "period": 84}},
    "entry": {"crosses_above": ["fast", "slow"]},
    "exit": {"crosses_below": ["fast", "slow"]},
}
TREND_EMA_6H_HASH = "89b1801c9ade2a4b0504f963bcc508b3ed0e05bdfbdacc06dee9032c06f5bd9b"
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


@pytest.fixture
def sol_6h() -> pd.DataFrame:
    return pd.read_csv(SOL_6H)


def test_drawdown_is_measured_on_the_capital_live_risks(sol_6h):
    """trend_ema_6h on real SOL: 39% drawdown on one fully invested $100 book, 18% on the live $500.

    The kill switch judges the same 15% limit on risk.max_live_capital, so the
    gates must too; an entry spends risk.max_trade_size of it.
    """
    strategy = spec.compile_spec(TREND_EMA_6H)
    assert strategy.hash == TREND_EMA_6H_HASH  # position_size defaults: existing specs keep their hash

    result = evaluation.evaluate(strategy, "SOL", sol_6h, SOL_COST, record=False)

    assert result["data"]["capital"] == config().risk["max_live_capital"] == 500
    assert result["data"]["position_notional"] == 100
    metrics = result["metrics"]
    assert metrics["trades"] == 36
    assert metrics["max_drawdown_pct"] == pytest.approx(18.2162, abs=1e-3)
    assert result["benchmarks"]["buy_and_hold"]["notional"] == 100  # the same position, held
    sim_capital, trade_size = backtest.sizing()
    assert (sim_capital, trade_size) == (500, 100)


def test_position_size_scales_entries_and_drawdown(sol_6h):
    full = spec.compile_spec(TREND_EMA_6H)
    half = spec.compile_spec({**TREND_EMA_6H, "risk": {**TREND_EMA_6H["risk"], "position_size": 0.5}})
    capital, trade_size = backtest.sizing()
    signals = backtest.compute_signals(full, sol_6h)
    runs = [
        backtest.simulate(s, "SOL", sol_6h, *signals, s.lookback - 1, len(sol_6h), SOL_COST, capital, trade_size)
        for s in (full, half)
    ]

    assert half.hash != TREND_EMA_6H_HASH
    assert len(runs[0].trades) == len(runs[1].trades) > 30
    # Flat costs: half the position is half the P&L on the same capital, bar by bar.
    np.testing.assert_allclose(runs[1].equity, capital + (runs[0].equity - capital) / 2, rtol=1e-9)
    assert backtest.max_drawdown(runs[1].equity, capital) < backtest.max_drawdown(runs[0].equity, capital)
    result = evaluation.evaluate(half, "SOL", sol_6h, SOL_COST, record=False)
    assert result["data"]["position_notional"] == 50
    assert result["benchmarks"]["buy_and_hold"]["notional"] == 50


def _ts(text: str) -> int:
    return int(pd.Timestamp(text, tz="UTC").timestamp())


def test_a_slow_strategys_empty_holdout_is_insufficient_not_a_failure(sol_6h):
    """trend_ema_6h trades ~9 times a year: 60 held-out days with 0 trades can't judge it.

    The old rule demanded 8 trades and rejected the hash for good.
    """
    result = evaluation.holdout(spec.compile_spec(TREND_EMA_6H), "SOL", sol_6h, SOL_COST)

    assert result["verdict"] == evaluation.INSUFFICIENT
    assert not result["passed"]
    assert result["data"] == {"first": "2026-08-10T12:00:00Z", "last": "2026-10-09T06:00:00Z", "bars": 240}
    assert result["required_trades"] == 2  # 35 selection trades in ~3.8 years, scaled to 60 days
    assert result["gates"][0] == {"gate": "holdout_trades", "value": 0, "threshold": 2, "passed": False}


def test_the_held_out_period_grows_from_its_pinned_start(sol_6h):
    strategy = spec.compile_spec(TREND_EMA_6H)
    earlier = sol_6h[sol_6h["ts"] < _ts("2025-12-01")].reset_index(drop=True)
    first = evaluation.holdout(strategy, "SOL", earlier, SOL_COST)
    assert first["verdict"] == evaluation.INSUFFICIENT  # one closed trade

    later = evaluation.holdout(strategy, "SOL", sol_6h, SOL_COST, start_ts=first["start_ts"])

    assert later["data"]["first"] == first["data"]["first"] == "2025-10-02T00:00:00Z"
    assert later["required_trades"] == first["required_trades"]  # the bar doesn't move with time
    assert later["gates"][0]["value"] == 7
    assert later["verdict"] == evaluation.PASS


def test_a_drawdown_breach_fails_even_with_few_trades(strategy, monkeypatch):
    frame = _market()
    start = int(frame["ts"].iat[len(frame) - 60 * 24])
    monkeypatch.setitem(config().risk, "max_drawdown_pct", 0.01)
    result = evaluation.holdout(strategy, "TKN", frame, FREE, start_ts=start, end_ts=start + 40 * HOUR)
    assert result["gates"][0]["passed"] is False  # too few trades to judge the return...
    assert result["verdict"] == evaluation.FAIL  # ...but the drawdown limit is a hard line
    assert result["reasons"][-1].startswith("holdout_max_drawdown_pct")


def test_a_loss_on_too_few_trades_is_insufficient(monkeypatch):
    losing = spec.compile_spec({**SPEC, "entry": SPEC["exit"], "exit": SPEC["entry"]})
    frame = _market()
    start = int(frame["ts"].iat[len(frame) - 60 * 24])
    monkeypatch.setitem(config().risk, "max_drawdown_pct", 100)
    result = evaluation.holdout(losing, "TKN", frame, FREE, start_ts=start, end_ts=start + 40 * HOUR)
    assert result["metrics"]["total_return_pct"] < 0
    assert result["gates"][0]["value"] < result["required_trades"]
    assert result["verdict"] == evaluation.INSUFFICIENT
