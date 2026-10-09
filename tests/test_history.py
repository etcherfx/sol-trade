"""History: paged backfill, incremental updates, the forming bar and data checks."""

import pytest

from sol_trade import data_source, history
from sol_trade.config import config
from tests.fakes import FakeExchange, make_ticks

PERIOD = 3600


@pytest.fixture
def market(monkeypatch):
    ticks = make_ticks(300, seed=5)
    clock = {"now": 0.0}
    exchange = FakeExchange(ticks, PERIOD, lambda: clock["now"])
    monkeypatch.setattr(data_source, "get_exchange", lambda *_args, **_kwargs: exchange)
    monkeypatch.setattr(config(), "primary_mint_symbol", "QUOTE")
    monkeypatch.setitem(config().backfill, "page_limit", 50)
    return exchange, clock


def test_backfill_pages_and_skips_the_forming_bar(market):
    exchange, clock = market
    clock["now"] = exchange.t0 + 250 * PERIOD + 600  # bar 250 is forming

    reports = history.backfill(["TKN"], ["1h"], days=8, now=clock["now"])  # 192 bars wanted

    report = reports[0]
    assert report["errors"] == []
    assert report["bars_added"] == 192
    assert report["total"] == 192
    assert report["pages"] >= 4  # 50 bars per page
    assert report["gaps"] == 0
    frame = history.load_bars("TKN/QUOTE", "1h")
    assert int(frame["ts"].iat[-1]) == exchange.t0 + 249 * PERIOD  # newest closed bar
    assert len(frame) == 192


def test_incremental_update_repairs_partial_bars(market):
    exchange, clock = market
    clock["now"] = exchange.t0 + 100 * PERIOD + 10
    history.update("TKN", "1h", days=2, now=clock["now"])
    # A partial bar stored by an older version: wrong close for the newest bar.
    newest = exchange.t0 + 99 * PERIOD
    history._store("TKN/QUOTE", "1h", [history.Bar(newest, 1, 1, 1, 1, 1)])

    clock["now"] = exchange.t0 + 101 * PERIOD + 10
    report, forming = history.update("TKN", "1h", now=clock["now"])

    frame = history.load_bars("TKN/QUOTE", "1h")
    repaired = frame[frame["ts"] == newest]
    assert float(repaired["close"].to_numpy()[0]) == pytest.approx(exchange.ticks[99][-1])
    assert report.bars_added == 1
    assert forming is not None
    assert forming.ts == exchange.t0 + 101 * PERIOD


def test_window_reads_closed_bars_from_the_store(market):
    exchange, clock = market
    clock["now"] = exchange.t0 + 120 * PERIOD + 1800
    frame, forming = history.window("TKN", "1h", 30, now=clock["now"])
    assert len(frame) == 30
    assert int(frame["ts"].iat[-1]) == exchange.t0 + 119 * PERIOD
    assert forming is not None
    assert forming.ts == exchange.t0 + 120 * PERIOD
    assert forming.low == pytest.approx(min(exchange.ticks[120][:4]))  # ticks seen so far


def test_report_counts_gaps_duplicates_and_bad_bars(market):
    _, _ = market
    t = 1_700_000_000 - 1_700_000_000 % PERIOD
    bars = [history.Bar(t + i * PERIOD, 10, 11, 9, 10, 1) for i in range(10) if i not in (3, 4, 7)]
    bars.append(history.Bar(t + 10 * PERIOD, 10, 9, 11, 10, 1))  # high below low
    bars.append(history.Bar(t + 11 * PERIOD + 5, 10, 11, 9, 10, 1))  # misaligned
    history._store("BAD/QUOTE", "1h", bars)
    report = history.SeriesReport("BAD/QUOTE", "1h", "fake")

    history._validate("BAD/QUOTE", "1h", report, None, None)

    assert report.gaps == 2
    assert report.largest_gap_bars == 2
    assert report.ohlc_errors == 1
    assert report.misaligned == 1
    assert report.gap_fraction == pytest.approx(3 / 12)


def test_duplicate_bars_in_pages_are_counted(market, monkeypatch):
    exchange, clock = market
    clock["now"] = exchange.t0 + 40 * PERIOD + 1
    original = exchange.fetch_ohlcv

    def doubled(*args, **kwargs):
        page = original(*args, **kwargs)
        return page + page[:1]

    monkeypatch.setattr(exchange, "fetch_ohlcv", doubled)
    report, _ = history.update("TKN", "1h", days=1, now=clock["now"])
    assert report.duplicates >= 1


def test_unsupported_timeframe_is_refused(market):
    with pytest.raises(ValueError, match="not offered"):
        history.update("TKN", "3h")


def test_candle_store_allows_a_concurrent_writer(market):
    conn = data_source._connect()
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        conn.execute("BEGIN IMMEDIATE")  # another process holding the write lock briefly
        conn.execute("INSERT INTO candles VALUES ('X/Y', '1h', 0, 1, 1, 1, 1, 1, 1)")
        frame = history.load_bars("X/Y", "1h")  # readers aren't blocked by a writer
        assert frame.empty
    finally:
        conn.rollback()
        conn.close()


def test_backfill_finds_where_the_history_starts(monkeypatch):
    """With an OKX-like exchange, pages before the listing are empty; the start is found by bisection."""
    ticks = make_ticks(300, seed=5)
    clock = {"now": 0.0}
    exchange = FakeExchange(ticks, PERIOD, lambda: clock["now"], windowed=True)
    monkeypatch.setattr(data_source, "get_exchange", lambda *_args, **_kwargs: exchange)
    monkeypatch.setattr(config(), "primary_mint_symbol", "QUOTE")
    monkeypatch.setitem(config().backfill, "page_limit", 50)
    clock["now"] = exchange.t0 + 250 * PERIOD + 600

    report = history.backfill(["TKN"], ["1h"], days=100, now=clock["now"])[0]  # 2,400 bars asked, 250 exist

    assert report["errors"] == []
    assert report["total"] == 250
    assert report["first"] == history._iso(exchange.t0)
    assert report["notes"] and "history starts at" in report["notes"][0]
    assert report["coverage"] == pytest.approx(250 / 2400, abs=1e-6)
    assert report["pages"] < 25  # bisection, not 40-odd empty pages


def test_backfill_defaults_to_the_configured_days(market, monkeypatch):
    exchange, clock = market
    clock["now"] = exchange.t0 + 250 * PERIOD + 600
    monkeypatch.setitem(config().backfill, "default_days", 3)
    report = history.backfill(["TKN"], ["1h"], now=clock["now"])[0]
    assert report["total"] == 72


def test_a_token_can_take_its_candles_from_another_quote_and_exchange(market, monkeypatch):
    """E.g. JUP from OKX's JUP/USDT, which goes back years, while swaps still pay in USDC."""
    exchange, clock = market
    asked = []
    monkeypatch.setattr(data_source, "get_exchange", lambda exchange_id=None: asked.append(exchange_id) or exchange)
    monkeypatch.setattr(config(), "token_exchanges", {"TKN": {"exchange": "other", "quote": "USDT"}})
    clock["now"] = exchange.t0 + 250 * PERIOD + 600

    report = history.backfill(["TKN"], ["1h"], days=3, now=clock["now"])[0]
    frame, _ = history.window("TKN", "1h", 30, now=clock["now"])

    source = history.source_for("TKN")
    assert (source.exchange, source.pair, source.proxy) == ("other", "TKN/USDT", True)
    assert (report["pair"], report["exchange"], report["total"]) == ("TKN/USDT", "other", 72)
    assert set(asked) == {"other"}
    assert {call[0] for call in exchange.calls} == {"TKN/USDT"}
    assert len(frame) == 30
    assert history.load_bars("TKN/QUOTE", "1h").empty  # nothing under the primary pair


def test_the_default_source_is_the_primary_pair_on_the_data_exchange(market, monkeypatch):
    monkeypatch.setattr(config(), "token_exchanges", {"TKN": "mexc"})
    assert history.source_for("TKN") == history.Source("mexc", "TKN", "QUOTE")
    assert not history.source_for("TKN").proxy
    assert history.source_for("OTHER") == history.Source(config().data_exchange, "OTHER", "QUOTE")
