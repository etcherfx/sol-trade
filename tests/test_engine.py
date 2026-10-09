"""The shared decision core: stops, take-profit, trailing stop, sizing, replay."""

import pytest

from sol_trade import costs, data_source, spec, trading
from sol_trade.config import config
from sol_trade.costs import CostModel
from sol_trade.engine import (
    BUY,
    SELL,
    Book,
    Position,
    SimExecutor,
    size_order,
    stop_exit,
    update_trailing,
)
from sol_trade.spec import Risk
from tests.fakes import FakeExchange, make_ticks
from tests.test_parity import FIXTURE

FREE = CostModel("TKN", "measured", ((0.0, 0.0),), 0.0)
RISK = Risk(stoploss=5, takeprofit=10, trailing_stoploss=2, trailing_stoploss_target=4)


def _book(cash=100.0):
    return Book("test", cash, SimExecutor(lambda _s: FREE), 100.0, 100.0)


def _enter(book: Book, price=100.0, ts=0, risk=RISK):
    book.on_close("TKN", ts - 60, price, price, True, False, risk)
    book.on_open("TKN", ts, price, risk)
    return book.position("TKN")


def test_entry_fills_at_the_next_open_with_levels():
    book = _book()
    action = book.on_close("TKN", 0, 101, 100, True, False, RISK)
    assert action is not None and action.side == BUY
    assert book.position("TKN") is None  # nothing fills at the signal close
    fill = book.on_open("TKN", 60, 102.0, RISK)
    position = book.position("TKN")
    assert fill is not None and fill.price == 102.0
    assert position is not None
    assert position.stop == pytest.approx(102 * 0.95)
    assert position.take_profit == pytest.approx(102 * 1.10)
    assert book.cash == pytest.approx(0.0)


def test_stop_fills_at_the_level_inside_the_bar():
    book = _book()
    _enter(book, 100.0)
    fill = book.on_low("TKN", 60, 99.0, 94.0, RISK)  # low crosses the 95 stop
    assert fill is not None
    assert fill.price == pytest.approx(95.0)
    assert fill.reason == "stoploss"
    assert book.position("TKN") is None


def test_gap_through_the_stop_fills_at_the_open():
    book = _book()
    _enter(book, 100.0)
    fill = book.on_low("TKN", 60, 90.0, 88.0, RISK)  # opened below the stop
    assert fill is not None
    assert fill.price == pytest.approx(90.0)


def test_take_profit_checks_the_close_and_fills_next_open():
    book = _book()
    _enter(book, 100.0)
    # A wick to 111 doesn't count; only the close does.
    assert book.on_close("TKN", 60, 111.0, 109.0, False, False, RISK) is None
    action = book.on_close("TKN", 120, 112.0, 110.5, False, False, RISK)
    assert action is not None and (action.side, action.reason) == (SELL, "takeprofit")
    fill = book.on_open("TKN", 180, 109.0, RISK)
    assert fill is not None and fill.price == 109.0
    assert book.trades[-1].reason == "takeprofit"


def test_trailing_stop_activates_and_ratchets_on_closed_highs():
    position = Position("TKN", "s", "h", 100.0, 1.0, 100.0, 0, 95.0, None, highest_price=100.0)
    update_trailing(position, RISK, 103.0)
    assert not position.trailing_active  # target is 104
    update_trailing(position, RISK, 106.0)
    assert position.trailing_active and position.highest_price == 106.0
    update_trailing(position, RISK, 105.0)
    assert position.highest_price == 106.0  # never ratchets down
    hit = stop_exit(position, RISK, 104.5, 103.8)
    assert hit == (pytest.approx(106 * 0.98), "trailing_stop")


def test_trailing_state_survives_storage():
    position = Position("TKN", "s", "h", 100.0, 1.0, 100.0, 0, 95.0, 110.0, True, 120.0)
    restored = Position.from_dict(position.to_dict())
    assert restored == position
    assert stop_exit(restored, RISK, 118.0, 117.0) == (pytest.approx(117.6), "trailing_stop")


def test_disabled_exits():
    risk = Risk()
    book = Book("test", 100.0, SimExecutor(lambda _s: FREE), 100.0, 100.0)
    _enter(book, 100.0, risk=risk)
    position = book.position("TKN")
    assert position is not None
    assert position.stop is None and position.take_profit is None
    assert book.on_low("TKN", 60, 50.0, 1.0, risk) is None


def test_size_order():
    assert size_order(cash=500, deployed=0, max_trade_size=100, max_capital=500) == 100
    assert size_order(cash=60, deployed=0, max_trade_size=100, max_capital=500) == 60
    assert size_order(cash=500, deployed=450, max_trade_size=100, max_capital=500) == 50
    assert size_order(cash=500, deployed=0, max_trade_size=100, max_capital=500, weight_cap=30) == 30
    assert size_order(cash=500, deployed=600, max_trade_size=100, max_capital=500) == 0


def test_costs_reduce_the_position_and_proceeds():
    model = CostModel("TKN", "measured", ((0.0, 50.0),), 0.1)  # 50 bps + $0.10 per swap
    book = Book("test", 100.0, SimExecutor(lambda _s: model), 100.0, 100.0)
    book.on_close("TKN", 0, 10, 10, True, False, RISK)
    fill = book.on_open("TKN", 60, 10.0, RISK)
    assert fill is not None
    assert fill.notional == pytest.approx(99.9)  # the fee is reserved first
    assert fill.size == pytest.approx(99.9 * 0.995 / 10)
    assert book.cash == pytest.approx(0.0)
    book.close_position("TKN", 120, 10.0, "signal")
    trade = book.trades[-1]
    assert trade.pnl == pytest.approx(fill.size * 10 * 0.995 - 0.1 - 100.0)


@pytest.fixture
def feed(monkeypatch):
    ticks = make_ticks(200, seed=11)
    clock = {"now": 0.0}
    exchange = FakeExchange(ticks, 60, lambda: clock["now"])
    monkeypatch.setattr(data_source, "get_exchange", lambda *_args, **_kwargs: exchange)
    monkeypatch.setattr(costs, "model_for", lambda _s: FREE)
    monkeypatch.setattr(config(), "primary_mint_symbol", "QUOTE")
    return exchange, clock


def _poll(book, strategy, clock, times):
    for now in times:
        clock["now"] = now
        trading.run_symbol(book, "TKN", strategy, now=now)


def test_missed_bars_are_replayed_after_downtime(feed):
    exchange, clock = feed
    strategy = spec.compile_spec(FIXTURE)
    start = exchange.t0 + 40 * 60 + 1
    every_bar = [start + k * 60 for k in range(140)]
    steady = Book("dry_run", 100.0, SimExecutor(costs.model_for), 100.0, 100.0)
    _poll(steady, strategy, clock, every_bar)

    # Same market, but the bot is off for 50 bars in the middle.
    gappy = Book("dry_run", 100.0, SimExecutor(costs.model_for), 100.0, 100.0)
    _poll(gappy, strategy, clock, every_bar[:40] + every_bar[90:])

    assert [t.to_dict() for t in gappy.trades] == [t.to_dict() for t in steady.trades]


def test_live_book_ignores_entries_from_replayed_bars(feed, monkeypatch):
    exchange, clock = feed
    strategy = spec.compile_spec({**FIXTURE, "entry": ["close", ">", 0], "exit": ["close", "<", 0]})
    bought: list[int] = []

    class _Recorder(SimExecutor):
        def buy(self, symbol, notional, price, ts, reason):
            bought.append(ts)
            return super().buy(symbol, notional, price, ts, reason)

    book = Book("live", 100.0, _Recorder(costs.model_for), 100.0, 100.0)
    state = book.state("TKN")
    state.last_bar_ts = exchange.t0 + 40 * 60
    clock["now"] = exchange.t0 + 50 * 60 + 1  # nine bars missed
    trading.run_symbol(book, "TKN", strategy, now=clock["now"])
    # Only the newest closed bar's signal opens a position, at the forming bar's open.
    assert bought == [exchange.t0 + 50 * 60]
