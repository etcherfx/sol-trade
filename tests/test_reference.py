"""Strategies that read another symbol's candles: grammar, alignment, research and the live loop."""

import numpy as np
import pandas as pd
import pytest

from sol_trade import (
    backtest,
    costs,
    data_source,
    evaluation,
    history,
    ledger,
    spec,
    trading,
)
from sol_trade.config import config
from sol_trade.costs import CostModel
from sol_trade.engine import Book, Position, SimExecutor
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


def _store_ref(frame, timeframe="1h"):
    history._store(history.source_for("REF"), timeframe, [history.Bar(*row) for row in frame.itertuples(index=False)])


def test_reference_candles_come_from_the_store_and_are_reported():
    market = _market()
    _store_ref(market)
    plain = evaluation.evaluate(spec.compile_spec(SPEC), "TKN", market, FREE, record=False)
    filtered = evaluation.evaluate(spec.compile_spec(_with_ref_filter(SPEC)), "TKN", market, FREE, record=False)
    assert filtered["metrics"] == plain["metrics"]  # REF close > 0 always holds
    info = filtered["data"]["references"]["REF"]
    assert info["bars"] == len(market) and info["coverage"] == 1.0
    assert info["usable_bars"] == filtered["data"]["selection_bars"]
    assert info["source"] == history.source_for("REF").as_dict()
    assert filtered["rpc"] == {"calls": 0, "cu": 0}


def _either(spec_dict):
    """Trades on its own bars alone; the reference branch can only add entries."""
    return {**spec_dict, "name": "swing_either", "entry": {"any": [spec_dict["entry"], [REF_CLOSE, ">", 1e9]]}}


def test_reference_data_outside_the_evaluated_period_is_insufficient_evidence():
    market = _market()
    strategy = spec.compile_spec(_either(SPEC))
    cut = evaluation.split(market, strategy)
    _store_ref(market.iloc[cut.holdout_start :])  # only the held-out period has REF bars
    result = evaluation.evaluate(strategy, "TKN", market, FREE)
    assert result["verdict"] == evaluation.INSUFFICIENT
    assert result["reasons"][0].startswith("reference REF has no run of 1 1h bars ending in the selection period")
    assert result["data"]["references"]["REF"]["coverage"] == 0.0
    assert evaluation.trials("TKN") == []


def test_reference_data_missing_from_the_held_out_period_is_insufficient_evidence():
    market = _market()
    strategy = spec.compile_spec(_either(SPEC))
    cut = evaluation.split(market, strategy)
    _store_ref(market.iloc[: cut.holdout_start])  # REF stops where the held-out period starts
    held = evaluation.holdout(strategy, "TKN", market, FREE)
    assert held["verdict"] == evaluation.INSUFFICIENT and not held["passed"]
    assert held["reasons"][0].startswith("reference REF has no run of 1 1h bars ending in the held-out period")
    assert held["references"]["REF"]["usable_bars"] == 0


def test_reference_indicators_need_their_warm_up_in_the_evaluated_period():
    market = _market()
    _store_ref(market.iloc[::2])  # every other bar: values, but never two in a row
    slow = {**SPEC, "name": "swing_rs", "lookback": 60,
            "indicators": {**SPEC["indicators"], "rs": {"fn": "ema", "period": 50, "input": REF_CLOSE}},
            "entry": {"all": [SPEC["entry"], ["rs", ">", 0]]}}
    strategy = spec.compile_spec(slow)
    assert strategy.reference_warmup == 50
    result = evaluation.evaluate(strategy, "TKN", market, FREE, record=False)
    assert result["verdict"] == evaluation.INSUFFICIENT
    assert result["reasons"][0].startswith("reference REF has no run of 50 1h bars")
    # A rule that reads one reference bar has usable bars in the same data; gaps there stay missing values.
    filtered = evaluation.evaluate(spec.compile_spec(_with_ref_filter(SPEC)), "TKN", market, FREE, record=False)
    info = filtered["data"]["references"]["REF"]
    assert info["coverage"] == pytest.approx(0.5, abs=0.01) and info["usable_bars"] > 0
    assert not any("reference" in r for r in filtered["reasons"])


def test_reference_warmup_follows_the_rules_that_read_a_reference():
    assert spec.compile_spec(RELATIVE).reference_warmup == 13  # ema 12 of the ratio, +1 for the cross
    assert spec.compile_spec(_with_ref_filter(SPEC)).reference_warmup == 1
    assert spec.compile_spec(SPEC).reference_warmup == 0
    shifted = {**SPEC, "entry": [{"shift": [REF_CLOSE, 5]}, ">", 0]}
    assert spec.compile_spec(shifted).reference_warmup == 6


# ------------------------------------------------------- missing reference values


@pytest.mark.parametrize(
    ("entry", "indicators", "blind"),
    [
        ([REF_CLOSE, "!=", 0], {}, [35]),
        ([{"sub": [REF_CLOSE, 1]}, "!=", -1], {}, [35]),
        (["avg", "!=", 0], {"avg": {"fn": "sma", "period": 3, "input": REF_CLOSE}}, [35, 36, 37]),
        ({"not": [REF_CLOSE, "==", 0]}, {}, []),  # not(false) is true: the rule itself is what's false
    ],
)
def test_every_comparison_with_a_missing_reference_is_false(monkeypatch, entry, indicators, blind):
    monkeypatch.setattr(config(), "primary_mint_symbol", "QUOTE")
    ts = T0 + np.arange(50) * PERIOD
    own = _frame(ts, np.full(50, 2.0))
    ref = _frame(np.delete(ts, 35), np.full(49, 3.0))  # the bar closing at ts[35] + PERIOD is missing
    strategy = spec.compile_spec({"schema": 1, "name": "gaps", "timeframe": "1m", "lookback": 5,
                                  "indicators": indicators, "entry": entry})
    flags, _ = backtest.compute_signals(strategy, own, "TKN", references=history.References("1m", {"REF": ref}))
    expected = np.zeros(50, dtype=bool)
    expected[4:] = True
    expected[blind] = False
    np.testing.assert_array_equal(flags, expected)


def test_bars_decided_without_a_reference_stay_missing_after_it_arrives(monkeypatch):
    monkeypatch.setattr(config(), "primary_mint_symbol", "QUOTE")
    stamps = T0 + np.array([0, 1, 3]) * PERIOD  # the bar opening at T0 + 2 periods is late
    _store_ref(_frame(stamps, [1.0, 2.0, 4.0]), "1m")
    closes = T0 + np.array([2, 3, 4], dtype=float) * PERIOD
    refs = history.load_references(("REF",), "1m")
    assert refs.record_unavailable(closes) == {"REF": 1}
    assert refs.record_unavailable(closes) == {}  # recorded once
    _store_ref(_frame([T0 + 2 * PERIOD], [3.0]), "1m")  # the late bar arrives

    later = history.load_references(("REF",), "1m")
    np.testing.assert_array_equal(later("REF", "close", closes), [2.0, np.nan, 4.0])
    unmarked = history.align(later.frames["REF"], PERIOD, closes, "close")
    np.testing.assert_array_equal(unmarked, [2.0, 3.0, 4.0])


# ------------------------------------------------------------- live loop

T0 = 1_700_000_000 - 1_700_000_000 % PERIOD


class _Feed:
    """A ccxt-like exchange for TKN/QUOTE and REF/QUOTE; REF can lag or be down, as a function of the time."""

    id = "fake"

    def __init__(self, own_ticks, ref_ticks, clock, lag=None, down=None):
        self.timeframes = {"1m": "1m"}
        self.clock = clock
        self.own = FakeExchange(own_ticks, PERIOD, clock, T0)
        self.ref = FakeExchange(ref_ticks, PERIOD, clock, T0)
        self.lag = lag or (lambda now: 0)  # seconds after its close a REF bar is published
        self.down = down or (lambda now: False)

    def fetch_ohlcv(self, pair, timeframe="1m", since=None, limit=100):
        if pair.startswith("TKN/"):
            return self.own.fetch_ohlcv(pair, timeframe, since, limit)
        now = self.clock()
        if self.down(now):
            raise OSError("reference venue down")
        rows = self.ref.fetch_ohlcv(pair, timeframe, since, limit)
        lag = self.lag(now)
        return [r for r in rows if r[0] // 1000 + PERIOD <= now - lag] if lag else rows


def _live(monkeypatch, book_name, seed=11, n_bars=300, lag=None, down=None, ticks=None):
    own_ticks = ticks or make_ticks(n_bars, seed=seed)
    clock = {"now": 0.0}
    feed = _Feed(own_ticks, make_ticks(n_bars, seed=seed + 100), lambda: clock["now"], lag, down)
    monkeypatch.setattr(data_source, "get_exchange", lambda *_args, **_kwargs: feed)
    monkeypatch.setattr(costs, "model_for", lambda _symbol: MODEL)
    monkeypatch.setattr(config(), "primary_mint_symbol", "QUOTE")
    book = Book(book_name, 100.0, SimExecutor(costs.model_for), 100.0, 100.0)
    return book, feed, clock


def _poll(book, strategy, clock, start_bar, end_bar, step=20):
    now = T0 + start_bar * PERIOD + 1
    while now < T0 + end_bar * PERIOD:
        clock["now"] = now
        trading.run_symbol(book, "TKN", strategy, now=now)
        yield now
        now += step


def _replay(strategy, first_ts, last_ts):
    """A backtest of the stored bars ``[first_ts, last_ts]``, reading the stored reference."""
    frame = history.load_bars(history.source_for("TKN"), "1m")
    entry, exit_ = backtest.compute_signals(strategy, frame, "TKN")
    sim = Book("backtest", 100.0, SimExecutor(costs.model_for), 100.0, 100.0)
    ts = frame["ts"].to_numpy()
    opens, highs, lows, closes = (frame[c].to_numpy(dtype=float) for c in ("open", "high", "low", "close"))
    for i in np.flatnonzero((ts >= first_ts) & (ts <= last_ts)):
        bar = int(ts[i])
        sim.on_open("TKN", bar, float(opens[i]), strategy.risk)
        sim.on_low("TKN", bar, float(opens[i]), float(lows[i]), strategy.risk)
        sim.on_close("TKN", bar, float(highs[i]), float(closes[i]), bool(entry[i]), bool(exit_[i]), strategy.risk,
                     strategy.name, strategy.hash)
    return sim


@pytest.mark.parametrize("book_name", [ledger.LIVE, "dry_run"])
@pytest.mark.parametrize("seed", [11, 23])
def test_live_and_replay_trade_identically_through_reference_lag_outage_and_recovery(monkeypatch, book_name, seed):
    def lag(now):
        return 90 if T0 + 80 * PERIOD <= now < T0 + 100 * PERIOD else 0  # 1.5 bars late for 20 bars

    def down(now):
        return T0 + 150 * PERIOD <= now < T0 + 165 * PERIOD

    book, feed, clock = _live(monkeypatch, book_name, seed, lag=lag, down=down)
    strategy = spec.compile_spec(RELATIVE)
    for now in _poll(book, strategy, clock, 40, 299):
        assert book.state("TKN").last_bar_ts == int(now // PERIOD) * PERIOD - PERIOD  # never held back
    last = book.state("TKN").last_bar_ts or 0
    marked = history.load_unavailable(history.source_for("REF"), "1m")
    assert marked.size >= 20

    # The venue recovers and the missed bars are backfilled; replays still read them as missing.
    feed.lag, feed.down = (lambda now: 0), (lambda now: False)
    history.update("REF", "1m", min_bars=320, now=clock["now"])
    stored = history.load_bars(history.source_for("REF"), "1m")["ts"].to_numpy() + PERIOD
    assert np.isin(marked, stored).all()
    sim = _replay(strategy, T0 + 39 * PERIOD, last)

    assert len(sim.trades) >= 3, "fixture should trade"
    assert [t.to_dict() for t in book.trades] == [t.to_dict() for t in sim.trades if t.exit_ts <= last]
    assert [(f.ts, f.side, f.price, f.size, f.reason) for f in book.fills] == [
        (f.ts, f.side, f.price, f.size, f.reason) for f in sim.fills if f.ts <= last
    ]
    # The recovered bars hold values; only the record keeps the replay from reading them.
    refs = history.load_references(("REF",), "1m")
    assert np.isnan(refs("REF", "close", marked)).all()
    assert np.isfinite(history.align(refs.frames["REF"], PERIOD, marked, "close")).all()


@pytest.mark.parametrize("book_name", [ledger.LIVE, "dry_run"])
def test_a_reference_lagging_more_than_a_bar_never_holds_a_book_back(monkeypatch, book_name):
    book, _, clock = _live(monkeypatch, book_name, n_bars=80, lag=lambda now: 150)
    strategy = spec.compile_spec(RELATIVE)
    polls = 0
    for now in _poll(book, strategy, clock, 40, 50):
        assert book.state("TKN").last_bar_ts == int(now // PERIOD) * PERIOD - PERIOD
        polls += 1
    assert polls == 30


def test_stops_keep_running_while_the_reference_is_down(monkeypatch):
    ticks = make_ticks(80, seed=4)
    ticks[61][2] = ticks[61][0] * 0.9  # a wick in the forming bar
    book, _, clock = _live(monkeypatch, ledger.LIVE, ticks=ticks, down=lambda now: True)
    strategy = spec.compile_spec(RELATIVE)
    entry = ticks[60][-1]
    state = book.state("TKN")
    state.last_bar_ts = T0 + 60 * PERIOD  # bar 60 already decided
    state.position = Position("TKN", strategy.name, strategy.hash, entry, 1.0, entry, T0 + 60 * PERIOD,
                              stop=entry * 0.97)
    clock["now"] = T0 + 61 * PERIOD + 50  # mid-bar, after the wick
    view = trading.run_symbol(book, "TKN", strategy, now=clock["now"])

    assert view["bars"] == 0
    assert book.position("TKN") is None
    assert [(f.ts, f.reason) for f in book.fills] == [(T0 + 61 * PERIOD, "stoploss")]
    assert "REF" in trading._reference_errors["TKN"]  # an absent reference is reported, not hidden


def test_a_dead_reference_is_reported_and_its_bars_stay_missing(monkeypatch):
    book, _, clock = _live(monkeypatch, "dry_run", n_bars=80, down=lambda now: True)
    clock["now"] = float(T0 + 60 * PERIOD + 1)
    view = trading.run_symbol(book, "TKN", spec.compile_spec(RELATIVE), now=clock["now"])
    assert view["bars"] == 1 and book.state("TKN").last_bar_ts == T0 + 59 * PERIOD
    assert np.isnan(view["values"]["fast"])
    marked = history.load_unavailable(history.source_for("REF"), "1m")
    assert marked[-1] == T0 + 60 * PERIOD and marked.size == RELATIVE["lookback"]
    assert "REF" in trading._reference_errors["TKN"]
    trading._reference_errors.clear()
