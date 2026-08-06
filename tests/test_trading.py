"""Tests for the dry-run (paper trading) path."""

import pandas as pd
import pytest

from sol_trade import trading
from sol_trade.config import config


def _reset_dry_run() -> None:
    trading._dry_run = False
    trading._balance_cache.set_paper_mode(False)
    trading._balance_cache._paper = {}


def test_paper_buy_updates_ledger():
    _reset_dry_run()
    trading._balance_cache.set_paper_mode(True)
    trading._balance_cache.set("USDC_MINT", 100.0)
    trading._balance_cache.set("SOL_MINT", 0.0)
    df = pd.DataFrame({"close": [2.0]})

    assert trading._paper_buy(50.0, df, "USDC_MINT", "SOL_MINT", "USDC", "SOL")

    assert trading._balance_cache.get("USDC_MINT") == pytest.approx(50.0)
    assert trading._balance_cache.get("SOL_MINT") == pytest.approx(25.0)
    _reset_dry_run()


def test_paper_sell_updates_ledger():
    _reset_dry_run()
    trading._balance_cache.set_paper_mode(True)
    trading._balance_cache.set("USDC_MINT", 50.0)
    trading._balance_cache.set("SOL_MINT", 25.0)
    df = pd.DataFrame({"close": [4.0]})

    assert trading._paper_sell(25.0, df, "USDC_MINT", "SOL_MINT", "USDC", "SOL")

    assert trading._balance_cache.get("SOL_MINT") == pytest.approx(0.0)
    assert trading._balance_cache.get("USDC_MINT") == pytest.approx(150.0)  # 50 + 25*4
    _reset_dry_run()


def test_paper_skips_on_zero_price():
    _reset_dry_run()
    trading._balance_cache.set_paper_mode(True)
    trading._balance_cache.set("USDC_MINT", 100.0)
    trading._balance_cache.set("SOL_MINT", 0.0)
    df = pd.DataFrame({"close": [0.0]})

    assert not trading._paper_buy(50.0, df, "USDC_MINT", "SOL_MINT", "USDC", "SOL")
    assert trading._balance_cache.get("USDC_MINT") == pytest.approx(100.0)
    _reset_dry_run()


def test_partial_close_keeps_position():
    df = pd.DataFrame(
        {
            "position": [True, True],
            "position_size": [2.0, 2.0],
            "entry_price": [1.0, 1.0],
            "close": [1.1, 1.1],
        }
    )
    out = trading._close_position_bookkeeping(df, 0.5)
    assert bool(out["position"].iat[-1]) is True
    assert out["position_size"].iat[-1] == pytest.approx(1.5)
    assert "entry_price" in out.columns  # weighted-average entry kept


def test_full_close_clears_position():
    df = pd.DataFrame(
        {
            "position": [True, True],
            "position_size": [2.0, 2.0],
            "entry_price": [1.0, 1.0],
            "stoploss": [0.9, 0.9],
            "takeprofit": [1.2, 1.2],
            "trailing_stoploss": [0.95, 0.95],
            "trailing_stoploss_target": [1.05, 1.05],
        }
    )
    out = trading._close_position_bookkeeping(df, 2.0)
    assert bool(out["position"].iat[-1]) is False
    for col in ("position_size", "entry_price", "stoploss", "takeprofit"):
        assert col not in out.columns


def test_close_without_size_treats_as_full_close():
    df = pd.DataFrame({"position": [True, True]})
    out = trading._close_position_bookkeeping(df, 1.0)
    assert bool(out["position"].iat[-1]) is False


def _trading_df(entry: int = 0, exit_: int = 0, size: float | None = 2.0) -> pd.DataFrame:
    """A 60-bar frame with strategy indicators and optional carried position."""
    close = pd.Series(range(100, 160), dtype=float)
    df = pd.DataFrame(
        {
            "open": close,
            "close": close,
            "high": close + 1,
            "low": close - 1,
            "symbol": "SOL",
            "entry": 0,
            "exit": 0,
        }
    )
    df = trading.strategy(df)  # attaches strategy_instance + indicators
    df.loc[df.index[-1], "entry"] = entry
    df.loc[df.index[-1], "exit"] = exit_
    if size is not None:
        df["position"] = True
        df["position_size"] = size
        df["entry_price"] = 105.0
        df["stoploss"] = 95.0
        df["takeprofit"] = 130.0
        df["trailing_stoploss"] = 100.0
        df["trailing_stoploss_target"] = 110.0
    return df


def test_buy_records_fill_price_and_position_size(tmp_path, monkeypatch):
    _reset_dry_run()
    trading._dry_run = True
    trading._balance_cache.set_paper_mode(True)
    trading._balance_cache.set(config().primary_mint, 100.0)
    trading._balance_cache.set(config().secondary_mints[0], 0.0)
    monkeypatch.setattr(config(), "confluence_enabled", False)
    df = _trading_df(entry=1)

    ok = trading.handle_buy_signal(df, config().secondary_mints[0], str(tmp_path / "sol.csv"), "SOL")

    assert ok
    assert bool(df["position"].iat[-1]) is True
    assert float(df["position_size"].iat[-1]) == pytest.approx(100.0 / 159.0)
    # entry_price = input / bought (the real fill price), not the signal close
    assert float(df["entry_price"].iat[-1]) == pytest.approx(159.0)
    sl = float(df.strategy_instance.stoploss)
    assert float(df["stoploss"].iat[-1]) == pytest.approx(159.0 * (1 - sl / 100))
    _reset_dry_run()


def test_partial_sell_keeps_position(tmp_path, monkeypatch):
    _reset_dry_run()
    trading._dry_run = True
    trading._balance_cache.set_paper_mode(True)
    trading._balance_cache.set(config().primary_mint, 0.0)
    trading._balance_cache.set(config().secondary_mints[0], 0.75)  # sell 0.75 of a 2.0 position
    monkeypatch.setattr(config(), "confluence_enabled", False)
    monkeypatch.setattr(config(), "sentiment_enabled", False)
    df = _trading_df(exit_=1, size=2.0)
    csv_path = tmp_path / "sol.csv"

    ok = trading.handle_sell_signal(df, config().secondary_mints[0], str(csv_path), "SOL")

    assert ok
    assert bool(df["position"].iat[-1]) is True  # position stays open
    # The persisted frame carries the reduced size and the kept entry price.
    saved = pd.read_csv(csv_path)
    assert float(saved["position_size"].iat[-1]) == pytest.approx(1.25)
    assert float(saved["entry_price"].iat[-1]) == pytest.approx(105.0)
    _reset_dry_run()


def test_full_close_clears_position_and_columns(tmp_path, monkeypatch):
    _reset_dry_run()
    trading._dry_run = True
    trading._balance_cache.set_paper_mode(True)
    trading._balance_cache.set(config().primary_mint, 0.0)
    trading._balance_cache.set(config().secondary_mints[0], 2.0)  # sell the entire position
    monkeypatch.setattr(config(), "confluence_enabled", False)
    monkeypatch.setattr(config(), "sentiment_enabled", False)
    df = _trading_df(exit_=1, size=2.0)
    csv_path = tmp_path / "sol.csv"

    ok = trading.handle_sell_signal(df, config().secondary_mints[0], str(csv_path), "SOL")

    assert ok
    assert bool(df["position"].iat[-1]) is False
    # The persisted frame has no position/risk columns left.
    saved = pd.read_csv(csv_path)
    for col in ("position_size", "entry_price", "stoploss", "takeprofit"):
        assert col not in saved.columns
    _reset_dry_run()
