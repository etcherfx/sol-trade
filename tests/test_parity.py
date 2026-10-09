"""The backtester and the live loop must make the same trades from the same candles."""

import pytest

from sol_trade import backtest, costs, data_source, history, rpc, spec, trading
from sol_trade.config import config
from sol_trade.costs import CostModel
from sol_trade.engine import Book, SimExecutor
from tests.fakes import FakeExchange, bars_frame, make_ticks

PERIOD = 60
FIXTURE = {
    "schema": 1,
    "name": "parity",
    "timeframe": "1m",
    "lookback": 30,
    "risk": {"stoploss": 2, "takeprofit": 3, "trailing_stoploss": 1, "trailing_stoploss_target": 1.5},
    "indicators": {
        "fast": {"fn": "ema", "period": 5},
        "slow": {"fn": "ema", "period": 13},
        "rsi": {"fn": "rsi", "period": 7},
    },
    "entry": {"any": [{"crosses_above": ["fast", "slow"]}, ["rsi", "<", 25]]},
    "exit": {"crosses_below": ["fast", "slow"]},
}
MODEL = CostModel("TKN", "measured", ((25.0, 10.0), (250.0, 20.0)), 0.01)


def _fills(book: Book, until: int) -> list[tuple]:
    return [
        (f.ts, f.side, float(f.price), float(f.size), float(f.cost), f.reason)
        for f in book.fills
        if f.ts <= until
    ]


def _trades(book: Book, until: int) -> list[dict]:
    return [t.to_dict() for t in book.trades if t.exit_ts <= until]


@pytest.mark.parametrize("seed", [11, 23])
def test_backtest_and_live_loop_trade_identically(monkeypatch, seed):
    n_bars = 200
    ticks = make_ticks(n_bars, ticks_per_bar=6, seed=seed)
    frame = bars_frame(ticks, PERIOD)
    t0 = int(frame["ts"].iat[0])
    clock = {"now": 0.0}
    exchange = FakeExchange(ticks, PERIOD, lambda: clock["now"], t0)
    monkeypatch.setattr(data_source, "get_exchange", lambda *_args, **_kwargs: exchange)
    monkeypatch.setattr(costs, "model_for", lambda _symbol: MODEL)
    monkeypatch.setattr(config(), "primary_mint_symbol", "QUOTE")
    strategy = spec.compile_spec(FIXTURE)

    # Live loop: polls every 20 s, seeing each forming bar three times.
    live = Book("dry_run", 100.0, SimExecutor(costs.model_for), 100.0, 100.0)
    start_bar = 40
    now = t0 + start_bar * PERIOD + 1
    end = t0 + (n_bars - 1) * PERIOD
    while now < end:
        clock["now"] = now
        trading.run_symbol(live, "TKN", strategy, now=now)
        now += 20
    last_closed = int(clock["now"] // PERIOD) * PERIOD - PERIOD

    # Backtest over the same closed bars, starting at the first bar the live loop saw.
    entry, exit_ = backtest.compute_signals(strategy, frame, "TKN")
    first = start_bar - 1
    stop = int((last_closed - t0) // PERIOD) + 1
    sim = Book("backtest", 100.0, SimExecutor(costs.model_for), 100.0, 100.0)
    risk = strategy.risk
    for i in range(first, stop):
        row = frame.iloc[i]
        ts = int(row.ts)
        sim.on_open("TKN", ts, row.open, risk)
        sim.on_low("TKN", ts, row.open, row.low, risk)
        sim.on_close("TKN", ts, row.high, row.close, bool(entry[i]), bool(exit_[i]), risk, strategy.name, strategy.hash)

    assert len(sim.trades) >= 5, "fixture should trade"
    assert {"stoploss", "takeprofit", "trailing_stop"} <= {t.reason for t in sim.trades}
    assert _trades(live, last_closed) == _trades(sim, last_closed)
    assert _fills(live, last_closed) == _fills(sim, last_closed)
    stored = history.load_bars(history.source_for("TKN"), "1m")
    assert int(stored["ts"].iat[-1]) == last_closed  # the forming bar was never stored


def test_backtest_makes_no_rpc_calls(monkeypatch):
    class _NoRpc:
        def __getattr__(self, name):
            raise AssertionError(f"backtest called the RPC: {name}")

    monkeypatch.setattr(config(), "_client", _NoRpc())
    before = rpc.usage.snapshot()
    ticks = make_ticks(500, seed=3)
    frame = bars_frame(ticks, PERIOD)
    strategy = spec.compile_spec(FIXTURE)

    result = backtest.run(strategy, "TKN", frame, strategy.lookback - 1, len(frame))

    assert result["metrics"]["bars"] == len(frame) - strategy.lookback + 1
    assert result["costs"]["source"] == "default"  # no probes stored
    assert rpc.usage.snapshot() == before
