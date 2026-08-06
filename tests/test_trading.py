"""Tests for the dry-run (paper trading) path."""

import pandas as pd
import pytest

from sol_trade import trading


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
