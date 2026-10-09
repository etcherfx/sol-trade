"""The trading loop: balances, the token guard, dust, live swaps, migration."""

import json
from datetime import UTC, datetime

import pandas as pd
import pytest

from sol_trade import costs, data_source, ledger, trading
from sol_trade.config import config
from sol_trade.costs import CostModel
from sol_trade.engine import Book, Position, SimExecutor, SymbolState
from sol_trade.spec import Risk
from sol_trade.ui import UIState
from tests.fakes import FakeExchange, make_ticks
from tests.test_parity import FIXTURE

FREE = CostModel("SOL", "measured", ((0.0, 0.0),), 0.0)


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.setattr(trading, "_dry_run", False)
    monkeypatch.setattr(trading, "_last_price_map", {})
    monkeypatch.setattr(trading, "_balance_cache", trading.BalanceCache())
    monkeypatch.setattr(trading, "_strategy_errors", {})


def _open(symbol: str, size: float, book: str = ledger.LIVE) -> None:
    position = Position(symbol, "s", "h", 100.0, size, 100.0 * size, 0, 95.0, 110.0)
    ledger.save_state(book, symbol, SymbolState(position=position))


def test_balance_cache_tolerates_rpc_errors(monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("rpc down")

    monkeypatch.setattr(trading, "find_balance", _boom)
    assert trading._balance_cache.get("SOME_MINT") == 0.0


def test_live_balance_reads_expire(monkeypatch):
    reads: list[str] = []
    monkeypatch.setattr(trading, "find_balance", lambda mint: reads.append(mint) or 1.0)
    clock = {"now": 0.0}
    cache = trading.BalanceCache(clock=lambda: clock["now"])

    cache.get("M")
    cache.get("M")
    clock["now"] = config().rpc["balance_refresh_seconds"] + 1
    cache.get("M")

    assert reads == ["M", "M"]  # one read, reused, then refreshed after the TTL


def _prev():
    return {"secondary_mints": ["M1"], "secondary_mint_symbols": ["SOL"], "secondary_weights": []}


def test_token_change_guard_blocks_removal_of_open_position(monkeypatch):
    cfg = config()
    _open("SOL", 1.0)
    monkeypatch.setattr(cfg, "secondary_mints", ["M2"])
    monkeypatch.setattr(cfg, "secondary_mint_symbols", ["JUP"])

    trading._enforce_token_change_guard(_prev())

    assert cfg.secondary_mints == ["M1"]  # rolled back
    assert cfg.secondary_mint_symbols == ["SOL"]


def test_token_change_guard_allows_removal_without_position(monkeypatch):
    cfg = config()
    monkeypatch.setattr(cfg, "secondary_mints", ["M2"])
    monkeypatch.setattr(cfg, "secondary_mint_symbols", ["JUP"])

    trading._enforce_token_change_guard(_prev())

    assert cfg.secondary_mints == ["M2"]


def test_dust_does_not_block_token_removal(monkeypatch):
    cfg = config()
    _open("SOL", 1.4e-8)
    trading._last_price_map["M1"] = 150.0
    monkeypatch.setattr(cfg, "secondary_mints", ["M2"])
    monkeypatch.setattr(cfg, "secondary_mint_symbols", ["JUP"])

    trading._enforce_token_change_guard(_prev())

    assert cfg.secondary_mints == ["M2"]  # the position is dust


def test_dust_is_written_off_without_a_swap(monkeypatch):
    _open("SOL", 1.4e-8)
    _open("JUP", 2.0)
    cfg = config()
    sol, jup = cfg.secondary_mints

    trading._sweep_live_dust({sol: 150.0, jup: 0.5})

    assert ledger.open_position(ledger.LIVE, "SOL") is None
    assert ledger.open_position(ledger.LIVE, "JUP") is not None  # worth $1.00, not dust
    trade = ledger.trades(ledger.LIVE, "SOL")[0]
    assert trade["reason"] == "dust"
    assert ledger.fills(ledger.LIVE) == []  # nothing was swapped


def test_live_executor_records_actual_and_modeled_costs(monkeypatch):
    cfg = config()
    sol = cfg.secondary_mints[0]
    trading._last_price_map[sol] = 100.0
    monkeypatch.setattr(costs, "model_for", lambda _s: CostModel("SOL", "measured", ((0.0, 20.0),), 0.0))
    swaps = []

    async def fake_swap(amount, input_mint, output_mint, *_symbols):
        swaps.append((amount, input_mint, output_mint))
        if input_mint == cfg.primary_mint:
            return {"out_amount": 0.995, "sent_amount": amount}  # $100 -> 0.995 SOL at $100
        return {"out_amount": 99.0, "sent_amount": amount}

    monkeypatch.setattr(trading, "perform_swap", fake_swap)
    executor = trading.LiveExecutor()

    buy = executor.buy("SOL", 100.0, 99.0, 60, "signal")
    trading._balance_cache.set(sol, 0.5)  # the wallet holds less than the position
    sell = executor.sell("SOL", 0.995, 99.0, 120, "signal")

    assert buy is not None and sell is not None
    assert buy.size == pytest.approx(0.995)
    assert buy.cost == pytest.approx(0.5)  # $100 spent for $99.50 of SOL
    assert buy.modeled_cost == pytest.approx(0.2)
    assert swaps[1][0] == 0.5  # capped at the wallet balance
    assert sell.cost == pytest.approx(0.5 * 100 - 99.0)


def test_legacy_csv_positions_are_imported_once(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    pd.DataFrame(
        {
            "close": [150.0],
            "time": ["2026-08-10 12:00:00"],
            "position": [True],
            "position_size": [1.4e-8],
            "entry_price": [151.0],
            "stoploss": [145.0],
            "takeprofit": [160.0],
            "highest_price": [152.0],
        }
    ).to_csv(data / "SOL_data.csv", index=False)
    pd.DataFrame({"close": [1.0], "position": [False]}).to_csv(data / "JUP_data.csv", index=False)

    assert ledger.migrate_legacy_csv() == ["SOL"]

    position = ledger.open_position(ledger.LIVE, "SOL")
    assert position is not None
    assert position.size == pytest.approx(1.4e-8)
    assert position.stop == 145.0 and position.take_profit == 160.0
    assert position.entry_ts == int(datetime(2026, 8, 10, 12, tzinfo=UTC).timestamp())
    assert ledger.open_position(ledger.LIVE, "JUP") is None
    assert (data / "SOL_data.csv.migrated").exists()
    assert (data / "JUP_data.csv.migrated").exists()
    assert not (data / "SOL_data.csv").exists()
    assert ledger.migrate_legacy_csv() == []  # nothing left to import


def test_weight_cap(monkeypatch):
    monkeypatch.setattr(config(), "secondary_mint_symbols", ["SOL", "JUP"])
    monkeypatch.setattr(config(), "secondary_weights", [0.6, 0.4])
    monkeypatch.setattr(trading, "_current_total_capital", 200.0)
    assert trading._weight_cap("SOL") == pytest.approx(120.0)
    monkeypatch.setattr(config(), "secondary_weights", [])
    assert trading._weight_cap("SOL") is None


def test_pick_resolves_indicator_names():
    values = {"ema_fast": 1.5, "ema_mid": 1.8, "ema_s": float("nan")}
    assert trading._pick(values, "ema_s", "ema_fast") == 1.5
    assert trading._pick(values, "ema_m", "ema_mid") == 1.8
    assert trading._pick({}, "rsi") == 0.0


@pytest.fixture
def dry_run_market(monkeypatch, tmp_path):
    ticks = make_ticks(120, seed=11)
    exchange = FakeExchange(ticks, 60, lambda: 0.0)
    clock = {"now": exchange.t0 + 100 * 60 + 5}
    exchange.clock = lambda: clock["now"]
    monkeypatch.setattr(data_source, "get_exchange", lambda *_args, **_kwargs: exchange)
    monkeypatch.setattr(trading.time, "time", lambda: clock["now"])
    monkeypatch.setattr(costs, "model_for", lambda _s: FREE)
    monkeypatch.setattr(trading, "fetch_prices", lambda mints: dict.fromkeys(mints, 1.0))
    monkeypatch.setattr(trading, "_dry_run", True)
    folder = tmp_path / "strategies"
    folder.mkdir()
    (folder / "parity.json").write_text(json.dumps(FIXTURE))
    monkeypatch.setattr(config(), "token_strategies", {"SOL": "parity", "JUP": "missing_one"})
    return clock


def test_dry_run_cycle_uses_the_virtual_book(dry_run_market, monkeypatch):
    def _no_rpc(*args, **kwargs):
        raise AssertionError("a dry run must not read the wallet")

    monkeypatch.setattr(trading, "find_balance", _no_rpc)
    state = UIState()

    trading.perform_analysis(state)

    snapshot = state.snapshot()
    assert [t.symbol for t in snapshot.tokens] == ["SOL"]  # JUP's strategy is missing: skipped
    assert "missing_one" in trading._strategy_errors["JUP"]
    assert snapshot.primary_balance == pytest.approx(config().risk["max_live_capital"])
    assert ledger.load_state(ledger.DRY_RUN, "SOL").last_bar_ts is not None


def test_main_book_starts_from_virtual_capital(monkeypatch):
    monkeypatch.setattr(trading, "_dry_run", True)
    book = trading._main_book()
    assert book.name == ledger.DRY_RUN
    assert book.cash == pytest.approx(config().risk["max_live_capital"])
    assert isinstance(book.executor, SimExecutor)


def test_fills_are_stored_and_logged(caplog):
    book = trading._persist_and_log(Book(ledger.DRY_RUN, 100.0, SimExecutor(lambda _s: FREE), 100.0, 100.0))
    with caplog.at_level("INFO", logger="transaction_logger"):
        book.open_position("SOL", 60, 10.0, Risk(), "signal", "s", "h")
    assert len(ledger.fills(ledger.DRY_RUN)) == 1
    assert "dry_run: buy 10 SOL at 10" in caplog.text
