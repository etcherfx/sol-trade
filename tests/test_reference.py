"""Strategies that read another symbol's candles: grammar, alignment, research and the live loop."""

import numpy as np
import pandas as pd
import pytest

from sol_trade import backtest, costs, data_source, evaluation, history, spec, trading
from sol_trade.config import config
from sol_trade.costs import CostModel
from sol_trade.engine import Book, SimExecutor
from tests.fakes import FakeExchange, bars_frame, make_ticks
from tests.test_evaluation import FREE, SPEC, _market

PERIOD = 60
REF_CLOSE = {"ref": {"symbol": "REF", "series": "close"}}
# Relative strength: trade the token while its ratio to REF trends up.
RELATIVE = {
    "schema": 1,
    "name": "relative",
    "timeframe": "1m",
    "lookback": 30,
    "risk": {"stoploss": 2, "takeprofit": 3, "trailing_stoploss": 1, "trailing_stoploss_target": 1.5},
    "indicators": {
        "fast": {"fn": "ema", "period": 4, "input": {"div": ["close", REF_CLOSE]}},
        "slow": {"fn": "ema", "period": 12, "input": {"div": ["close", REF_CLOSE]}},
    },
    "entry": {"crosses_above": ["fast", "slow"]},
    "exit": {"crosses_below": ["fast", "slow"]},
}
MODEL = CostModel("TKN", "measured", ((25.0, 10.0), (250.0, 20.0)), 0.01)


def _frame(ts, close):
    close = np.asarray(close, dtype=float)
    return pd.DataFrame({"ts": ts, "open": close, "high": close, "low": close, "close": close,
                         "volume": np.ones(len(ts))})


# ----------------------------------------------------------------- grammar


def test_a_reference_compiles_and_is_part_of_the_hash():
    strategy = spec.compile_spec(RELATIVE)
    assert strategy.references == ("REF",)
    assert strategy.warmup == 13
    other = spec.compile_spec({**RELATIVE, "indicators": {
        k: {**v, "input": {"div": ["close", {"ref": {"symbol": "OTHER", "series": "close"}}]}}
        for k, v in RELATIVE["indicators"].items()
    }})
    assert other.references == ("OTHER",) and other.hash != strategy.hash
    assert spec.compile_spec({**SPEC}).references == ()


@pytest.mark.parametrize(
    ("ref", "message"),
    [
        ({"symbol": "REF"}, "ref.series: must be one of open, high, low, close, volume"),
        ({"symbol": "REF", "series": "fast"}, "ref.series: must be one of"),
        ({"symbol": "A/B", "series": "close"}, "ref.symbol: must be a token symbol"),
        ({"symbol": "REF", "series": "close", "bars": 1}, "ref.bars: unknown key"),
        ("REF", "ref: must be an object"),
    ],
)
def test_bad_references_name_the_place(ref, message):
    bad = {**SPEC, "entry": [{"ref": ref}, ">", 0]}
    with pytest.raises(spec.SpecError) as e:
        spec.compile_spec(bad)
    assert any(f"entry[0].{message}" in err or f"entry[0]{message[3:]}" in err for err in e.value.errors), e.value.errors


def test_evaluating_without_reference_candles_raises():
    strategy = spec.compile_spec(RELATIVE)
    bars = {c: np.ones((30, 1)) for c in spec.COLUMNS}
    with pytest.raises(ValueError, match="no reference candles"):
        strategy.evaluate(bars, np.arange(30, dtype=float)[:, None] * PERIOD)


# --------------------------------------------------------------- alignment


def test_alignment_never_looks_ahead_and_never_fills_gaps():
    ref = _frame([0, 60, 180, 240], [1.0, 2.0, 4.0, 5.0])  # the bar opening at 120 is missing
    closes = np.array([60, 90, 120, 180, 240, 300, 360], dtype=float)
    aligned = history.align(ref, PERIOD, closes, "close")
    # 90: the bar closing at 120 isn't over yet, so the bar that closed at 60 still counts.
    np.testing.assert_array_equal(aligned, [1.0, 1.0, 2.0, np.nan, 4.0, 5.0, np.nan])


def test_alignment_on_an_offset_grid_reads_the_bar_closed_within_the_period():
    ref = _frame([30, 90], [1.0, 2.0])  # bars close at 90 and 150
    aligned = history.align(ref, PERIOD, np.array([[60.0, 120.0], [150.0, 240.0]]), "close")
    np.testing.assert_array_equal(aligned, [[np.nan, 1.0], [2.0, np.nan]])


def test_a_later_reference_bar_does_not_change_earlier_signals(monkeypatch):
    monkeypatch.setattr(config(), "primary_mint_symbol", "QUOTE")
    strategy = spec.compile_spec(RELATIVE)
    own = bars_frame(make_ticks(120, seed=5), PERIOD)
    ref = bars_frame(make_ticks(120, seed=6), PERIOD)
    cut = 80
    early = history.References("1m", {"REF": ref.iloc[:cut]})
    full = history.References("1m", {"REF": ref})
    entry_early, exit_early = backtest.compute_signals(strategy, own.iloc[:cut], "TKN", references=early)
    entry_full, exit_full = backtest.compute_signals(strategy, own, "TKN", references=full)
    np.testing.assert_array_equal(entry_early, entry_full[:cut])
    np.testing.assert_array_equal(exit_early, exit_full[:cut])
    # A shifted reference is a different input: proof the reference is actually read.
    shifted = history.References("1m", {"REF": ref.assign(close=ref["close"].to_numpy()[::-1])})
    assert not np.array_equal(backtest.compute_signals(strategy, own, "TKN", references=shifted)[0], entry_full)


# ---------------------------------------------------------------- research


def _with_ref_filter(spec_dict):
    return {**spec_dict, "name": "swing_ref", "entry": {"all": [spec_dict["entry"], [REF_CLOSE, ">", 0]]}}


def test_a_missing_reference_is_insufficient_evidence_and_registers_no_trial():
    strategy = spec.compile_spec(_with_ref_filter(SPEC))
    with pytest.raises(history.MissingReference, match="run backfill --symbols REF --timeframes 1h"):
        backtest.compute_signals(strategy, _market(200), "TKN")
    result = evaluation.evaluate(strategy, "TKN", _market(), FREE)
    assert result["verdict"] == evaluation.INSUFFICIENT
    assert result["reasons"] == [str(history.MissingReference("REF", "1h"))]
    assert result["data"]["references"]["REF"]["bars"] == 0
    assert evaluation.trials("TKN") == []
    held = evaluation.holdout(strategy, "TKN", _market(), FREE)
    assert held["verdict"] == evaluation.INSUFFICIENT and not held["passed"]
    assert "reference REF" in held["reasons"][0]


def test_reference_candles_come_from_the_store_and_are_reported():
    market = _market()
    history._store(history.source_for("REF"), "1h", [history.Bar(*row) for row in market.itertuples(index=False)])
    plain = evaluation.evaluate(spec.compile_spec(SPEC), "TKN", market, FREE, record=False)
    filtered = evaluation.evaluate(spec.compile_spec(_with_ref_filter(SPEC)), "TKN", market, FREE, record=False)
    assert filtered["metrics"] == plain["metrics"]  # REF close > 0 always holds
    info = filtered["data"]["references"]["REF"]
    assert info["bars"] == len(market) and info["selection_coverage"] == 1.0
    assert info["source"] == history.source_for("REF").as_dict()
    assert filtered["rpc"] == {"calls": 0, "cu": 0}


# ------------------------------------------------------------- live parity


class _Exchanges:
    """One ccxt-like exchange serving several pairs; ``lag`` delays a pair's closed bars."""

    id = "fake"

    def __init__(self, by_pair, lag=None):
        self.by_pair = by_pair
        self.lag = lag or {}
        self.timeframes = {"1m": "1m"}

    def fetch_ohlcv(self, pair, timeframe="1m", since=None, limit=100):
        exchange = self.by_pair[pair]
        rows = exchange.fetch_ohlcv(pair, timeframe, since, limit)
        if pair in self.lag:  # published `lag` seconds after the close; no forming bar
            published = exchange.clock() - self.lag[pair]
            rows = [r for r in rows if r[0] // 1000 + PERIOD <= published]
        return rows


@pytest.mark.parametrize("seed", [11, 23])
def test_backtest_and_live_loop_trade_identically_with_a_lagging_reference(monkeypatch, seed):
    n_bars = 200
    own_ticks, ref_ticks = make_ticks(n_bars, seed=seed), make_ticks(n_bars, seed=seed + 100)
    own, ref = bars_frame(own_ticks, PERIOD), bars_frame(ref_ticks, PERIOD)
    t0 = int(own["ts"].iat[0])
    clock = {"now": 0.0}
    exchanges = _Exchanges(
        {"TKN/QUOTE": FakeExchange(own_ticks, PERIOD, lambda: clock["now"], t0),
         "REF/QUOTE": FakeExchange(ref_ticks, PERIOD, lambda: clock["now"], t0)},
        lag={"REF/QUOTE": 30},
    )
    monkeypatch.setattr(data_source, "get_exchange", lambda *_args, **_kwargs: exchanges)
    monkeypatch.setattr(costs, "model_for", lambda _symbol: MODEL)
    monkeypatch.setattr(config(), "primary_mint_symbol", "QUOTE")
    strategy = spec.compile_spec(RELATIVE)

    live = Book("dry_run", 100.0, SimExecutor(costs.model_for), 100.0, 100.0)
    start_bar = 40
    now = t0 + start_bar * PERIOD + 1
    end = t0 + (n_bars - 1) * PERIOD
    waited = 0
    while now < end:
        clock["now"] = now
        trading.run_symbol(live, "TKN", strategy, now=now)
        processed = live.state("TKN").last_bar_ts or 0
        waited += processed < int(now // PERIOD) * PERIOD - PERIOD
        now += 20
    last_closed = live.state("TKN").last_bar_ts or 0
    assert waited > 0, "the reference lag should hold bars back"

    entry, exit_ = backtest.compute_signals(strategy, own, "TKN", references=history.References("1m", {"REF": ref}))
    sim = Book("backtest", 100.0, SimExecutor(costs.model_for), 100.0, 100.0)
    for i in range(start_bar - 1, int((last_closed - t0) // PERIOD) + 1):
        row = own.iloc[i]
        ts = int(row.ts)
        sim.on_open("TKN", ts, row.open, strategy.risk)
        sim.on_low("TKN", ts, row.open, row.low, strategy.risk)
        sim.on_close("TKN", ts, row.high, row.close, bool(entry[i]), bool(exit_[i]), strategy.risk,
                     strategy.name, strategy.hash)

    def trades(book):
        return [t.to_dict() for t in book.trades if t.exit_ts <= last_closed]

    def fills(book):
        return [(f.ts, f.side, float(f.price), float(f.size), f.reason) for f in book.fills if f.ts <= last_closed]

    assert len(sim.trades) >= 3, "fixture should trade"
    assert trades(live) == trades(sim)
    assert fills(live) == fills(sim)
    # The stored reference replays to the same signals as the frame the test built.
    stored = backtest.compute_signals(strategy, own, "TKN")
    stop = int((last_closed - t0) // PERIOD) + 1
    np.testing.assert_array_equal(stored[0][:stop], entry[:stop])


def test_a_dead_reference_does_not_hold_the_token_up(monkeypatch):
    ticks = make_ticks(80, seed=4)
    t0 = int(bars_frame(ticks, PERIOD)["ts"].iat[0])
    clock = {"now": float(t0 + 60 * PERIOD + 1)}

    class _Down:
        id = "fake"

        def __init__(self):
            self.timeframes = {"1m": "1m"}
            self.own = FakeExchange(ticks, PERIOD, lambda: clock["now"], t0)

        def fetch_ohlcv(self, pair, timeframe="1m", since=None, limit=100):
            if pair.startswith("REF/"):
                raise OSError("reference venue down")
            return self.own.fetch_ohlcv(pair, timeframe, since, limit)

    monkeypatch.setattr(data_source, "get_exchange", lambda *_args, **_kwargs: _Down())
    monkeypatch.setattr(costs, "model_for", lambda _symbol: MODEL)
    monkeypatch.setattr(config(), "primary_mint_symbol", "QUOTE")
    book = Book("dry_run", 100.0, SimExecutor(costs.model_for), 100.0, 100.0)
    view = trading.run_symbol(book, "TKN", spec.compile_spec(RELATIVE), now=clock["now"])
    assert view["bars"] == 1 and book.state("TKN").last_bar_ts == t0 + 59 * PERIOD
    assert np.isnan(view["values"]["fast"])  # missing reference values stay missing
