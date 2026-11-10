"""Strategies that read another symbol's candles: grammar, alignment, research and the live loop."""

import time

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
    sentiment,
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


@pytest.fixture(autouse=True)
def _no_logged_reference_errors(monkeypatch):
    monkeypatch.setattr(trading, "_reference_errors", {})


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


def test_a_references_depth_follows_its_venue_and_every_reader_switches_with_it(monkeypatch):
    """A symbol only read as a reference can be pointed at a venue with deeper history via token_exchanges."""
    market = _market(2000)
    t0, hour = int(market["ts"].iat[0]), 3600
    clock = {"now": float(t0 + 2000 * hour + 600)}  # bar 2000 is forming
    # The default venue lists REF only for the last 600 bars; the deep one has all of them.
    shallow = FakeExchange(make_ticks(600, seed=3), hour, lambda: clock["now"], t0=t0 + 1400 * hour, windowed=True)
    shallow.id = "shallow"
    deep = FakeExchange(make_ticks(2001, seed=4), hour, lambda: clock["now"], t0=t0)
    deep.id = "deep"
    venues = {"shallow": shallow, "deep": deep}
    monkeypatch.setattr(data_source, "get_exchange", lambda exchange_id, **_: venues[exchange_id])
    monkeypatch.setattr(config(), "data_exchange", "shallow")
    monkeypatch.setattr(config(), "token_exchanges", {})
    strategy = spec.compile_spec(_with_ref_filter(SPEC))
    days = 2000 / 24

    thin = history.backfill(["REF"], ["1h"], days=days, now=clock["now"])[0]
    assert (thin["exchange"], thin["total"], thin["coverage"]) == ("shallow", 600, 0.3)
    assert any("shallow history starts at" in note for note in thin["notes"])
    info = evaluation.evaluate(strategy, "TKN", market, FREE, record=False)["data"]["references"]["REF"]
    assert info["bars"] == 600 and info["coverage"] < 0.35

    monkeypatch.setattr(config(), "token_exchanges", {"REF": {"exchange": "deep", "quote": "USD"}})
    full = history.backfill(["REF"], ["1h"], days=days, now=clock["now"])[0]
    assert (full["exchange"], full["pair"], full["total"], full["coverage"]) == ("deep", "REF/USD", 2000, 1.0)
    assert full["notes"] == []
    result = evaluation.evaluate(strategy, "TKN", market, FREE, record=False)
    info = result["data"]["references"]["REF"]
    assert info["source"] == {"exchange": "deep", "pair": "REF/USD", "price_proxy": True}
    assert (info["bars"], info["coverage"]) == (2000, 1.0)
    assert "PRICE_PROXY" not in result["flags"]  # the traded token's quote decides that, not a reference's
    # One venue per series: the deep series holds only deep bars, and the shallow rows stay where they were.
    stored = history.load_bars(history.source_for("REF"), "1h")
    assert stored["close"].to_numpy() == pytest.approx([bar[-1] for bar in deep.ticks[:2000]])
    assert len(history.load_bars(history.Source("shallow", "REF", config().primary_mint_symbol), "1h")) == 600

    # The live loop reads and extends the same series research just read.
    shallow.calls.clear()
    clock["now"] += hour
    live = history.refresh_references(("REF",), "1h", start=t0 + 1950 * hour, bars=50, now=clock["now"])
    assert shallow.calls == []
    assert live.frames["REF"]["ts"].iat[-1] == t0 + 2000 * hour  # the bar that was forming has closed
    research = history.load_references(("REF",), "1h")
    closes = live.frames["REF"]["ts"].to_numpy(dtype=float) + hour
    np.testing.assert_array_equal(live("REF", "close", closes), research("REF", "close", closes))


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
    assert result["reasons"][0].startswith("reference REF gives the rules no usable value in the selection period")
    assert result["data"]["references"]["REF"]["coverage"] == 0.0
    assert evaluation.trials("TKN") == []


def test_reference_data_missing_from_the_held_out_period_is_insufficient_evidence():
    market = _market()
    strategy = spec.compile_spec(_either(SPEC))
    cut = evaluation.split(market, strategy)
    _store_ref(market.iloc[: cut.holdout_start])  # REF stops where the held-out period starts
    held = evaluation.holdout(strategy, "TKN", market, FREE)
    assert held["verdict"] == evaluation.INSUFFICIENT and not held["passed"]
    assert held["reasons"][0].startswith("reference REF gives the rules no usable value in the held-out period")
    assert held["references"]["REF"]["usable_bars"] == 0


def test_an_indicator_that_never_has_a_value_over_reference_gaps_is_insufficient_evidence():
    market = _market()
    _store_ref(market.drop(index=market.index[::20]))  # a bar missing every 20: present 95% of the time
    gappy = {**SPEC, "name": "swing_rs", "lookback": 30,
             "indicators": {**SPEC["indicators"], "rs": {"fn": "ema", "period": 12, "input": REF_CLOSE}},
             "entry": {"any": [SPEC["entry"], ["rs", ">", 0]]}}  # the first branch trades without REF
    strategy = spec.compile_spec(gappy)
    usable = backtest.reference_usable(strategy, market)["REF"]
    assert not usable.any()  # every 30-bar window holds a gap after the EMA's seed, so it ends missing
    result = evaluation.evaluate(strategy, "TKN", market, FREE, record=False)
    assert result["verdict"] == evaluation.INSUFFICIENT
    assert result["reasons"][0].startswith("reference REF gives the rules no usable value in the selection period")
    info = result["data"]["references"]["REF"]
    assert info["coverage"] == pytest.approx(0.95, abs=0.01) and info["usable_bars"] == 0
    # A rule on the reference bar itself is usable in the same data; the gaps stay missing values.
    filtered = evaluation.evaluate(spec.compile_spec(_with_ref_filter(SPEC)), "TKN", market, FREE, record=False)
    assert filtered["data"]["references"]["REF"]["usable_bars"] > 0
    assert not any("reference" in r for r in filtered["reasons"])


def test_usable_follows_the_bars_each_rule_compares(monkeypatch):
    monkeypatch.setattr(config(), "primary_mint_symbol", "QUOTE")
    ts = T0 + np.arange(40) * PERIOD
    own = _frame(ts, np.full(40, 2.0))
    refs = history.References("1m", {"REF": _frame(np.delete(ts, 20), np.full(39, 3.0))})

    def unusable(entry):
        strategy = spec.compile_spec({"schema": 1, "name": "u", "timeframe": "1m", "lookback": 8, "entry": entry})
        return np.flatnonzero(~backtest.reference_usable(strategy, own, refs)["REF"][7:]) + 7

    np.testing.assert_array_equal(unusable([REF_CLOSE, ">", 0]), [20])
    np.testing.assert_array_equal(unusable([{"shift": [REF_CLOSE, 5]}, ">", 0]), [25])
    np.testing.assert_array_equal(unusable({"crosses_above": [REF_CLOSE, 1]}), [20, 21])  # this bar and the last
    assert backtest.reference_usable(spec.compile_spec(SPEC), _market(100)) == {}


def test_reference_inputs_take_the_validated_parameter_defaults():
    strategy = spec.compile_spec({**SPEC, "name": "rsi_ref", "indicators": {
        **SPEC["indicators"], "r": {"fn": "rsi", "period": None, "input": REF_CLOSE},  # null: period 14
    }, "entry": ["r", ">", 50]})
    market = _market(200)
    usable = backtest.reference_usable(strategy, market, history.References("1h", {"REF": market}))["REF"]
    assert usable[strategy.lookback - 1 :].all()


@pytest.mark.parametrize("input_series", ["close", REF_CLOSE])
def test_shared_indicator_dependencies_compile_once(input_series):
    indicators = {**SPEC["indicators"], "i0": {"fn": "sma", "period": 1, "input": input_series}}
    for i in range(1, 60):  # each level reads the one before twice: 2**59 paths if expanded
        indicators[f"i{i}"] = {"fn": "sma", "period": 1, "input": {"add": [f"i{i - 1}", f"i{i - 1}"]}}
    started = time.perf_counter()
    strategy = spec.compile_spec({**SPEC, "name": "chain", "indicators": indicators, "entry": ["i59", ">", 0]})
    assert time.perf_counter() - started < 2
    assert strategy.references == (() if input_series == "close" else ("REF",))


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


CLOSE_ONLY = [("REF", "close")]


def test_missing_lists_new_gaps_and_marks_read_as_missing(monkeypatch):
    monkeypatch.setattr(config(), "primary_mint_symbol", "QUOTE")
    stamps = T0 + np.array([0, 1, 3]) * PERIOD  # the bar opening at T0 + 2 periods is late
    _store_ref(_frame(stamps, [1.0, 2.0, 4.0]), "1m")
    closes = T0 + np.array([2, 3, 4], dtype=float) * PERIOD
    refs = history.load_references(("REF",), "1m")
    missing = refs.missing(closes, CLOSE_ONLY)
    np.testing.assert_array_equal(missing[("REF", "close")], [T0 + 3 * PERIOD])
    refs.mark(missing)
    assert refs.missing(closes, CLOSE_ONLY) == {}  # already marked
    _store_ref(_frame([T0 + 2 * PERIOD], [3.0]), "1m")  # the late bar arrives

    later = history.load_references(("REF",), "1m")
    np.testing.assert_array_equal(later("REF", "close", closes), [2.0, 3.0, 4.0])  # the store as it is
    later.mark(missing)
    np.testing.assert_array_equal(later("REF", "close", closes), [2.0, np.nan, 4.0])


def test_only_the_series_a_strategy_reads_decide_availability(monkeypatch):
    monkeypatch.setattr(config(), "primary_mint_symbol", "QUOTE")
    ts = T0 + np.arange(60) * PERIOD
    ref = _frame(ts, 3.0 + np.arange(60) % 4)
    ref.loc[30, "volume"] = np.nan  # a bar with a close but no volume
    _store_ref(ref, "1m")
    strategy = spec.compile_spec({"schema": 1, "name": "close_only", "timeframe": "1m", "lookback": 5,
                                  "indicators": {"avg": {"fn": "sma", "period": 3, "input": REF_CLOSE}},
                                  "entry": ["avg", ">", 4]})
    assert strategy.reference_series == (("REF", "close"),)
    closes = ts + PERIOD
    refs = history.load_references(("REF",), "1m")
    assert refs.missing(closes, strategy.reference_series) == {}  # the missing volume isn't read
    volume = refs.missing(closes, [("REF", "volume")])
    np.testing.assert_array_equal(volume[("REF", "volume")], closes[30:31])
    refs.mark(volume)
    assert np.isnan(refs("REF", "volume", closes[30:31])).all()
    assert refs("REF", "close", closes[30:31])[0] == ref["close"].iat[30]


# ------------------------------------------------------------- live loop

T0 = 1_700_000_000 - 1_700_000_000 % PERIOD


class _Feed:
    """A ccxt-like exchange for TKN/QUOTE and REF/QUOTE.

    REF can lag or be down, as a function of the time, and the REF bars
    opening at the times in ``never`` are never published.
    """

    id = "fake"

    def __init__(self, own_ticks, ref_ticks, clock, lag=None, down=None, never=()):
        self.timeframes = {"1m": "1m"}
        self.clock = clock
        self.own = FakeExchange(own_ticks, PERIOD, clock, T0)
        self.ref = FakeExchange(ref_ticks, PERIOD, clock, T0)
        self.lag = lag or (lambda now: 0)  # seconds after its close a REF bar is published
        self.down = down or (lambda now: False)
        self.never = set(never)

    def fetch_ohlcv(self, pair, timeframe="1m", since=None, limit=100):
        if pair.startswith("TKN/"):
            return self.own.fetch_ohlcv(pair, timeframe, since, limit)
        now = self.clock()
        if self.down(now):
            raise OSError("reference venue down")
        rows = [r for r in self.ref.fetch_ohlcv(pair, timeframe, since, limit) if r[0] // 1000 not in self.never]
        lag = self.lag(now)
        return [r for r in rows if r[0] // 1000 + PERIOD <= now - lag] if lag else rows


def _live(monkeypatch, book_name, seed=11, n_bars=300, lag=None, down=None, ticks=None, never=()):
    own_ticks = ticks or make_ticks(n_bars, seed=seed)
    clock = {"now": 0.0}
    feed = _Feed(own_ticks, make_ticks(n_bars, seed=seed + 100), lambda: clock["now"], lag, down, never)
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


def _replay(strategy, first_ts, last_ts, book=None):
    """A backtest of the stored bars ``[first_ts, last_ts]``: a replay of ``book``, or research without one."""
    frame = history.load_bars(history.source_for("TKN"), "1m")
    if book is None:
        entry, exit_ = backtest.compute_signals(strategy, frame, "TKN")
    else:
        entry, exit_ = backtest.book_signals(strategy, frame, "TKN", book)
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


def _same_trades(book, sim, last):
    assert [t.to_dict() for t in book.trades] == [t.to_dict() for t in sim.trades if t.exit_ts <= last]
    assert [(f.ts, f.side, f.price, f.size, f.reason) for f in book.fills] == [
        (f.ts, f.side, f.price, f.size, f.reason) for f in sim.fills if f.ts <= last
    ]


def _book_refs(strategy, book):
    refs = backtest.book_references(strategy, "TKN", book)
    assert refs is not None
    return refs


def _marks(book, strategy):
    return ledger.reference_marks(book, strategy.hash, "TKN").get(("REF", "close"), [])


def _lag_and_outage(now):
    return 90 if T0 + 80 * PERIOD <= now < T0 + 100 * PERIOD else 0  # 1.5 bars late for 20 bars


def _outage(now):
    return T0 + 150 * PERIOD <= now < T0 + 165 * PERIOD


@pytest.mark.parametrize("book_name", [ledger.LIVE, "dry_run"])
@pytest.mark.parametrize("seed", [11, 23])
def test_a_books_replay_matches_it_through_late_reference_bars_while_research_reads_the_store(
    monkeypatch, book_name, seed
):
    book, feed, clock = _live(monkeypatch, book_name, seed, lag=_lag_and_outage, down=_outage)
    strategy = spec.compile_spec(RELATIVE)
    for now in _poll(book, strategy, clock, 40, 299):
        assert book.state("TKN").last_bar_ts == int(now // PERIOD) * PERIOD - PERIOD  # never held back
    last = book.state("TKN").last_bar_ts or 0
    marked = np.array(_marks(book_name, strategy), dtype=float)
    assert marked.size >= 20

    # The venue recovers and the late bars are backfilled.
    feed.lag, feed.down = (lambda now: 0), (lambda now: False)
    history.update("REF", "1m", min_bars=320, now=clock["now"])
    stored = history.load_bars(history.source_for("REF"), "1m")["ts"].to_numpy() + PERIOD
    assert np.isin(marked, stored).all()

    replay = _replay(strategy, T0 + 39 * PERIOD, last, book_name)
    assert len(replay.trades) >= 3, "fixture should trade"
    _same_trades(book, replay, last)
    # Research reads the backfilled values; only the book's own replay reads them as missing.
    assert np.isfinite(history.load_references(("REF",), "1m")("REF", "close", marked)).all()
    assert np.isnan(_book_refs(strategy, book_name)("REF", "close", marked)).all()


@pytest.mark.parametrize("book_name", [ledger.LIVE, "dry_run"])
def test_a_books_replay_matches_it_when_a_reference_bar_never_arrives(monkeypatch, book_name):
    never = [T0 + k * PERIOD for k in (70, 71, 120)]
    book, _, clock = _live(monkeypatch, book_name, 11, never=never)
    strategy = spec.compile_spec(RELATIVE)
    for _ in _poll(book, strategy, clock, 40, 299):
        pass
    last = book.state("TKN").last_bar_ts or 0
    gaps = np.array(never, dtype=float) + PERIOD
    assert set(gaps) <= set(_marks(book_name, strategy))

    replay = _replay(strategy, T0 + 39 * PERIOD, last, book_name)
    assert len(replay.trades) >= 3, "fixture should trade"
    _same_trades(book, replay, last)
    # The bars never arrived, so research sees them as missing values too, and trades as the book did.
    assert np.isnan(history.load_references(("REF",), "1m")("REF", "close", gaps)).all()
    _same_trades(book, _replay(strategy, T0 + 39 * PERIOD, last), last)


def test_one_books_marks_never_reach_another_book_or_hash(monkeypatch):
    book_a, feed, clock = _live(monkeypatch, "dry_run", 11, lag=_lag_and_outage, down=_outage)
    strategy = spec.compile_spec(RELATIVE)
    for _ in _poll(book_a, strategy, clock, 40, 299):
        pass
    assert _marks("dry_run", strategy)
    feed.lag, feed.down = (lambda now: 0), (lambda now: False)
    history.update("REF", "1m", min_bars=320, now=clock["now"])

    other = spec.compile_spec({**RELATIVE, "name": "relative_other"})
    assert other.hash != strategy.hash
    assert _marks("dry_run", other) == [] and _marks("paper_b", strategy) == []
    assert _book_refs(other, "dry_run").unavailable == {}
    assert _book_refs(strategy, "paper_b").unavailable == {}

    # Book B trades the same hash on the recovered store: it decides on what is stored, records no
    # marks, and its replay is research's replay of the same bars.
    book_b = Book("paper_b", 100.0, SimExecutor(costs.model_for), 100.0, 100.0)
    for _ in _poll(book_b, strategy, clock, 40, 299):
        pass
    last = book_b.state("TKN").last_bar_ts or 0
    assert _marks("paper_b", strategy) == []
    research = _replay(strategy, T0 + 39 * PERIOD, last)
    _same_trades(book_b, _replay(strategy, T0 + 39 * PERIOD, last, "paper_b"), last)
    _same_trades(book_b, research, last)


def test_research_on_a_separately_backfilled_store_matches_the_live_store(monkeypatch, tmp_path):
    book, feed, clock = _live(monkeypatch, "dry_run", 23, lag=_lag_and_outage, down=_outage)
    strategy = spec.compile_spec(RELATIVE)
    for _ in _poll(book, strategy, clock, 40, 299):
        pass
    assert _marks("dry_run", strategy)
    feed.lag, feed.down = (lambda now: 0), (lambda now: False)
    history.update("REF", "1m", min_bars=320, now=clock["now"])
    frame = history.load_bars(history.source_for("TKN"), "1m")
    live_store = backtest.compute_signals(strategy, frame, "TKN")
    live_usable = backtest.reference_usable(strategy, frame)

    # A research store backfilled on its own, with no marks anywhere; the same bars read the same.
    monkeypatch.setattr(config(), "candles_path", str(tmp_path / "research_candles.db"))
    for symbol in ("TKN", "REF"):
        history.update(symbol, "1m", min_bars=320, now=clock["now"])
    fresh = history.load_bars(history.source_for("TKN"), "1m", start=int(frame["ts"].iat[0]))
    pd.testing.assert_frame_equal(fresh, frame)
    research_store = backtest.compute_signals(strategy, fresh, "TKN")
    research_usable = backtest.reference_usable(strategy, fresh)
    np.testing.assert_array_equal(np.concatenate(research_store), np.concatenate(live_store))
    np.testing.assert_array_equal(research_usable["REF"], live_usable["REF"])


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


@pytest.mark.parametrize("book_name", [ledger.LIVE, "dry_run"])
def test_an_absent_reference_blocks_entries_but_not_exits(monkeypatch, book_name):
    down = {"on": True}
    book, _, clock = _live(monkeypatch, book_name, n_bars=80, down=lambda now: down["on"])
    # The first branch enters on every bar without REF; offline evaluation would refuse this strategy.
    either = spec.compile_spec({**RELATIVE, "name": "either", "lookback": 30,
                                "entry": {"any": [["close", ">", 0], {"not": [REF_CLOSE, "<", 0]}]},
                                "exit": ["close", ">", 0]})
    for _ in _poll(book, either, clock, 40, 45):
        pass
    assert book.fills == [] and book.state("TKN").pending is None
    assert "REF" in trading._reference_errors["TKN"]
    blocked = ledger.blocked_entries(book_name, either.hash, "TKN")
    assert blocked == [T0 + k * PERIOD for k in range(39, 44)]

    # An open position still gets its exit while REF is absent.
    book.state("TKN").position = Position("TKN", either.name, either.hash, 100.0, 1.0, 100.0, T0 + 40 * PERIOD)
    for _ in _poll(book, either, clock, 45, 47):
        pass
    assert [f.reason for f in book.fills] == ["signal"] and book.position("TKN") is None

    # Once REF has bars again, the same rules enter.
    down["on"] = False
    for _ in _poll(book, either, clock, 47, 50):
        pass
    assert any(f.side == "buy" for f in book.fills)
    assert "TKN" not in trading._reference_errors

    # A replay of this book enters nowhere it was blocked; research would have.
    frame = history.load_bars(history.source_for("TKN"), "1m")
    replayed, _ = backtest.book_signals(either, frame, "TKN", book_name)
    researched, _ = backtest.compute_signals(either, frame, "TKN")
    at = np.isin(frame["ts"].to_numpy(), blocked)
    assert at.sum() == len(blocked) and not replayed[at].any() and researched[at].all()
    assert ledger.blocked_entries("paper_other", either.hash, "TKN") == []


def test_a_dead_reference_is_reported_and_its_bars_stay_missing_for_that_book(monkeypatch):
    book, _, clock = _live(monkeypatch, "dry_run", n_bars=80, down=lambda now: True)
    strategy = spec.compile_spec(RELATIVE)
    clock["now"] = float(T0 + 60 * PERIOD + 1)
    view = trading.run_symbol(book, "TKN", strategy, now=clock["now"])
    assert view["bars"] == 1 and book.state("TKN").last_bar_ts == T0 + 59 * PERIOD
    assert np.isnan(view["values"]["fast"])
    marked = _marks("dry_run", strategy)
    assert marked[-1] == T0 + 60 * PERIOD and len(marked) == RELATIVE["lookback"]
    assert "REF" in trading._reference_errors["TKN"]


def test_restarting_a_book_drops_its_marks():
    ledger.record_reference_marks("paper_x", "h", "TKN", {("REF", "close"): [120, 180]})
    ledger.record_blocked_entries("paper_x", "h", "TKN", [60])
    ledger.record_reference_marks("paper_y", "h", "TKN", {("REF", "close"): [120]})
    assert ledger.reference_marks("paper_x", "h", "TKN") == {("REF", "close"): [120, 180]}
    ledger.delete_book("paper_x")
    assert ledger.reference_marks("paper_x", "h", "TKN") == {} and ledger.blocked_entries("paper_x", "h", "TKN") == []
    assert ledger.reference_marks("paper_y", "h", "TKN") == {("REF", "close"): [120]}


# ------------------------------------------- absence per decision, sentiment


EITHER = {**RELATIVE, "name": "either", "lookback": 30,
          "entry": {"any": [["close", ">", 0], {"not": [REF_CLOSE, "<", 0]}]},
          "exit": ["close", "<", 0]}


def _batched_outage(monkeypatch, book_name):
    """REF stops after the bar opening at bar 80; the book last decided bar 100 and wakes at bar 161."""
    book, _, clock = _live(monkeypatch, book_name, 11, n_bars=200, never=[T0 + k * PERIOD for k in range(81, 200)])
    strategy = spec.compile_spec(EITHER)
    book.state("TKN").last_bar_ts = T0 + 100 * PERIOD
    clock["now"] = float(T0 + 161 * PERIOD + 1)
    view = trading.run_symbol(book, "TKN", strategy, now=clock["now"])
    assert view["bars"] == 60  # bars 101 to 160 in one batch
    return book, strategy


def test_a_batched_live_decision_without_the_reference_in_its_own_lookback_opens_nothing(monkeypatch):
    book, strategy = _batched_outage(monkeypatch, ledger.LIVE)
    stored = history.load_bars(history.source_for("REF"), "1m")["ts"].to_numpy()
    assert stored.max() == T0 + 80 * PERIOD  # the batch still holds REF bars, bar 160's lookback none
    assert book.position("TKN") is None and book.state("TKN").pending is None and book.fills == []
    # Bars whose lookback (the bar and the 29 before it) misses bar 80 are blocked; earlier ones aren't.
    assert ledger.blocked_entries(ledger.LIVE, strategy.hash, "TKN") == [T0 + k * PERIOD for k in range(110, 161)]
    assert "REF" in trading._reference_errors["TKN"]


def test_a_batched_paper_book_and_its_replay_agree_on_blocked_decisions(monkeypatch):
    book, strategy = _batched_outage(monkeypatch, "paper_batch")
    # Not a live book: replayed bars before the blocks may enter; none buys after bar 110's decision.
    buys = [f.ts for f in book.fills if f.side == "buy"]
    assert buys and buys[0] == T0 + 102 * PERIOD and max(buys) <= T0 + 110 * PERIOD
    replay = _replay(strategy, T0 + 101 * PERIOD, T0 + 160 * PERIOD, "paper_batch")
    _same_trades(book, replay, T0 + 160 * PERIOD)
    frame = history.load_bars(history.source_for("TKN"), "1m")
    replayed, _ = backtest.book_signals(strategy, frame, "TKN", "paper_batch")
    researched, _ = backtest.compute_signals(strategy, frame, "TKN")
    late = frame["ts"].to_numpy() >= T0 + 110 * PERIOD
    assert not replayed[late].any() and researched[late].all()


def _mood(spec_dict):
    """The entry also needs REF close plus the token's sentiment score to be positive."""
    return {**spec_dict, "name": "swing_mood",
            "indicators": {**spec_dict["indicators"], "mood": {"fn": "sentiment", "window_hours": 6}},
            "entry": {"all": [spec_dict["entry"], [{"add": [REF_CLOSE, "mood.score"]}, ">", 0]]}}


def test_operands_reading_a_reference_and_the_tokens_sentiment_are_usable(monkeypatch):
    def features(symbol, closes, window_hours, now=None):
        value = np.full(np.shape(closes), 0.5 if symbol == "TKN" else np.nan)
        return {"score": value, "mentions": np.where(np.isnan(value), np.nan, 3.0)}

    monkeypatch.setattr(sentiment, "features", features)
    market = _market()
    _store_ref(market)
    strategy = spec.compile_spec(_mood(SPEC))
    assert strategy.uses_sentiment and strategy.references == ("REF",)
    usable = backtest.reference_usable(strategy, market, symbol="TKN")["REF"]
    assert usable[strategy.lookback - 1 :].all()
    assert not backtest.reference_usable(strategy, market)["REF"].any()  # without the token, no sentiment

    result = evaluation.evaluate(strategy, "TKN", market, FREE, record=False)
    assert result["data"]["references"]["REF"]["usable_bars"] == result["data"]["selection_bars"]
    assert not any("reference" in r for r in result["reasons"])
    assert result["metrics"] == evaluation.evaluate(
        spec.compile_spec(SPEC), "TKN", market, FREE, record=False
    )["metrics"]  # both added inputs are always positive
    held = evaluation.holdout(strategy, "TKN", market, FREE)
    assert "references" not in held and not any("reference" in r for r in held["reasons"])


# ------------------------------------------------- a reference source that refuses


class _HourlyOnly:
    """A reference venue that doesn't offer the strategy's 1m timeframe."""

    id = "hourly"
    timeframes = {"1h": "1h"}  # noqa: RUF012 - read-only fake

    def fetch_ohlcv(self, *_args, **_kwargs):
        raise AssertionError("a refused timeframe must not be fetched")


def _reference_on_another_venue(monkeypatch, ticks):
    book, feed, clock = _live(monkeypatch, ledger.LIVE, ticks=ticks)
    hourly = _HourlyOnly()
    monkeypatch.setattr(data_source, "get_exchange", lambda exchange_id=None, **_: hourly if exchange_id == "hourly" else feed)
    monkeypatch.setattr(config(), "token_exchanges", {"REF": "hourly"})
    warnings = []
    monkeypatch.setattr(history.log_general, "warning", lambda message, *a, **k: warnings.append(str(message)))
    return book, clock, warnings


def test_a_reference_venue_refusing_the_timeframe_keeps_live_stops_running(monkeypatch):
    ticks = make_ticks(80, seed=4)
    ticks[61][2] = ticks[61][0] * 0.9  # a wick in the forming bar
    book, clock, warnings = _reference_on_another_venue(monkeypatch, ticks)
    strategy = spec.compile_spec(EITHER)
    entry = ticks[59][-1]
    state = book.state("TKN")
    state.last_bar_ts = T0 + 59 * PERIOD
    state.position = Position("TKN", strategy.name, strategy.hash, entry, 1.0, entry, T0 + 59 * PERIOD,
                              stop=entry * 0.97)
    clock["now"] = T0 + 61 * PERIOD + 50  # bar 60 closed; bar 61 forming, after the wick
    view = trading.run_symbol(book, "TKN", strategy, now=clock["now"])

    assert view["bars"] == 1 and state.last_bar_ts == T0 + 60 * PERIOD  # bar 60 was still decided
    assert book.position("TKN") is None
    assert [f.reason for f in book.fills] == ["stoploss"]
    assert any("reference REF" in w and "not offered by hourly" in w for w in warnings)
    # The reference has no stored bars: the decision is blocked and recorded for the book's replays.
    assert ledger.blocked_entries(ledger.LIVE, strategy.hash, "TKN") == [T0 + 60 * PERIOD]
    assert "REF" in trading._reference_errors["TKN"]


def test_a_reference_venue_refusing_the_timeframe_reads_stored_bars_like_an_outage(monkeypatch):
    book, clock, warnings = _reference_on_another_venue(monkeypatch, make_ticks(80, seed=4))
    ts = T0 + np.arange(55) * PERIOD
    history._store(history.source_for("REF"), "1m",
                   [history.Bar(*row) for row in _frame(ts, np.full(55, 3.0)).itertuples(index=False)])
    strategy = spec.compile_spec(EITHER)
    clock["now"] = T0 + 60 * PERIOD + 1
    trading.run_symbol(book, "TKN", strategy, now=clock["now"])

    assert warnings  # logged, not raised
    assert ledger.blocked_entries(ledger.LIVE, strategy.hash, "TKN") == []  # bar 59's lookback holds REF bars
    marked = _marks(ledger.LIVE, strategy)
    assert marked == [int(t) for t in T0 + np.arange(55, 60) * PERIOD + PERIOD]  # the bars it never got
    # The independent branch may still enter; the forming bar's open fills it.
    assert [(f.ts, f.side) for f in book.fills] == [(T0 + 60 * PERIOD, "buy")]


def test_the_traded_tokens_own_unsupported_timeframe_still_raises(monkeypatch):
    book, feed, _ = _live(monkeypatch, ledger.LIVE, n_bars=80)
    feed.timeframes = {"1h": "1h"}
    with pytest.raises(ValueError, match="not offered"):
        trading.run_symbol(book, "TKN", spec.compile_spec(EITHER), now=T0 + 60 * PERIOD + 1)
