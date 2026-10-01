"""Tests for the dry-run (paper trading) path."""

import pandas as pd
import pytest

from sol_trade import trading
from sol_trade.config import config


def _reset_dry_run() -> None:
    trading._dry_run = False
    trading._balance_cache.set_paper_mode(False)
    trading._balance_cache._paper = {}
    trading._balance_cache._cache = {}
    trading._current_total_capital = 0.0


def test_candles_to_frame_includes_volume():
    candles = [
        {
            "close": 1.0,
            "high": 1.1,
            "low": 0.9,
            "open": 1.0,
            "volume": 100.0,
            "totalvolume": 200.0,
            "time": 1,
        }
    ]
    df = trading._candles_to_frame(candles)
    assert list(df.columns) == [
        "close", "high", "low", "open", "volume", "totalvolume", "time",
    ]
    assert float(df["volume"].iat[0]) == pytest.approx(100.0)
    assert float(df["totalvolume"].iat[0]) == pytest.approx(200.0)


def test_pick_indicator_resolves_strategy_column_names():
    import pandas as pd

    # Custom strategies may use their own column names — no ema_s/ema_m columns.
    custom = pd.Series({"ema_fast": 1.5, "ema_slow": 2.0, "ema_mid": 1.8})
    assert trading._pick_indicator(custom, "ema_s", "ema_fast") == 1.5
    assert trading._pick_indicator(custom, "ema_m", "ema_mid", "ema_slow") == 1.8

    # Default-strategy names win when present.
    default = pd.Series({"ema_s": 3.0, "ema_fast": 1.5})
    assert trading._pick_indicator(default, "ema_s", "ema_fast") == 3.0

    # NaN in the preferred column falls through to the next candidate.
    nan_first = pd.Series({"ema_s": float("nan"), "ema_fast": 1.5})
    assert trading._pick_indicator(nan_first, "ema_s", "ema_fast") == 1.5

    # Nothing present -> 0.0 (dashboard shows a dash-like zero, not a crash).
    assert trading._pick_indicator(pd.Series({"rsi": 10.0}), "ema_s", "ema_fast") == 0.0


def test_balance_cache_tolerates_rpc_errors(monkeypatch):
    _reset_dry_run()

    def _boom(*args, **kwargs):
        raise RuntimeError("rpc down")

    monkeypatch.setattr(trading, "find_balance", _boom)
    # A non-rate-limit RPC failure must degrade to 0.0, not abort the cycle.
    assert trading._balance_cache.get("SOME_MINT") == 0.0
    _reset_dry_run()


def test_capture_baseline_tolerates_rpc_errors(monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("rpc down")

    monkeypatch.setattr(trading, "find_balance", _boom)
    monkeypatch.setattr(trading, "fetch_prices", lambda mints: {})
    # Startup baseline must survive a transient RPC outage.
    trading._capture_baseline()
    assert trading.initial_primary_balance == 0.0
    assert len(trading.initial_secondary_balances) == len(config().secondary_mints)
    assert all(b == 0.0 for b in trading.initial_secondary_balances)


def test_live_buy_chain_mocked_swap(tmp_path, monkeypatch):
    _reset_dry_run()  # live path (_dry_run False), swap mocked
    monkeypatch.setattr(config(), "confluence_enabled", False)
    monkeypatch.setattr(config(), "sentiment_enabled", False)
    # Single-token view so the weighted budget is 100% (config has 6 tokens).
    monkeypatch.setattr(config(), "secondary_mint_symbols", ["SOL"])
    monkeypatch.setattr(config(), "secondary_mints", [config().secondary_mints[0]])
    monkeypatch.setattr(config(), "secondary_weights", [])

    async def fake_swap(*args, **kwargs):
        return {"out_amount": 1.0, "sent_amount": 100.0}

    monkeypatch.setattr(trading, "perform_swap", fake_swap)
    trading._balance_cache._cache[config().primary_mint] = 100.0
    trading._balance_cache._cache[config().secondary_mints[0]] = 0.0
    df = _trading_df(entry=1)
    csv_path = tmp_path / "sol.csv"

    ok = trading.handle_buy_signal(
        df, config().secondary_mints[0], str(csv_path), "SOL"
    )

    assert ok
    assert bool(df["position"].iat[-1]) is True
    assert float(df["position_size"].iat[-1]) == pytest.approx(1.0)
    assert float(df["entry_price"].iat[-1]) == pytest.approx(100.0)  # 100 USDC / 1 SOL
    saved = pd.read_csv(csv_path)
    assert bool(saved["position"].iat[-1]) is True
    assert "stoploss" in saved.columns  # risk levels persisted with the position
    # Both balances invalidated so the next cycle re-reads the wallet.
    assert config().primary_mint not in trading._balance_cache._cache
    assert config().secondary_mints[0] not in trading._balance_cache._cache
    _reset_dry_run()


def test_live_sell_chain_mocked_swap(tmp_path, monkeypatch):
    _reset_dry_run()  # live path, swap mocked
    monkeypatch.setattr(config(), "confluence_enabled", False)
    monkeypatch.setattr(config(), "sentiment_enabled", False)

    async def fake_swap(*args, **kwargs):
        return {"out_amount": 150.0, "sent_amount": 1.0}  # 1 SOL -> 150 USDC

    monkeypatch.setattr(trading, "perform_swap", fake_swap)
    trading._balance_cache._cache[config().secondary_mints[0]] = 1.0
    trading._balance_cache._cache[config().primary_mint] = 0.0
    df = _trading_df(exit_=1, size=1.0)
    csv_path = tmp_path / "sol.csv"

    ok = trading.handle_sell_signal(
        df, config().secondary_mints[0], str(csv_path), "SOL"
    )

    assert ok
    assert bool(df["position"].iat[-1]) is False  # full close
    saved = pd.read_csv(csv_path)
    for col in ("position_size", "entry_price", "stoploss", "takeprofit"):
        assert col not in saved.columns
    assert config().secondary_mints[0] not in trading._balance_cache._cache
    assert config().primary_mint not in trading._balance_cache._cache
    _reset_dry_run()


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


def test_weighted_buy_capped_at_weight_slice(tmp_path, monkeypatch):
    _reset_dry_run()
    trading._dry_run = True
    trading._balance_cache.set_paper_mode(True)
    trading._balance_cache.set(config().primary_mint, 100.0)
    trading._balance_cache.set(config().secondary_mints[0], 0.0)
    monkeypatch.setattr(config(), "confluence_enabled", False)
    monkeypatch.setattr(config(), "secondary_mint_symbols", ["SOL", "JUP"])
    monkeypatch.setattr(config(), "secondary_weights", [0.6, 0.4])
    df = _trading_df(entry=1)

    ok = trading.handle_buy_signal(df, config().secondary_mints[0], str(tmp_path / "sol.csv"), "SOL")

    assert ok
    # Budget = 0.6 x 100 capital = 60 USDC -> 60 / 159 SOL, not the full balance.
    assert float(df["position_size"].iat[-1]) == pytest.approx(60.0 / 159.0)
    assert float(df["entry_price"].iat[-1]) == pytest.approx(159.0)
    _reset_dry_run()


def test_zero_weight_token_never_buys(tmp_path, monkeypatch):
    _reset_dry_run()
    trading._dry_run = True
    trading._balance_cache.set_paper_mode(True)
    trading._balance_cache.set(config().primary_mint, 100.0)
    trading._balance_cache.set(config().secondary_mints[0], 0.0)
    monkeypatch.setattr(config(), "confluence_enabled", False)
    monkeypatch.setattr(config(), "secondary_mint_symbols", ["SOL", "JUP"])
    monkeypatch.setattr(config(), "secondary_weights", [0.0, 1.0])
    df = _trading_df(entry=1)

    ok = trading.handle_buy_signal(df, config().secondary_mints[0], str(tmp_path / "sol.csv"), "SOL")

    assert not ok
    # No trade: the carried position bookkeeping is untouched.
    assert float(df["position_size"].iat[-1]) == pytest.approx(2.0)
    assert float(df["entry_price"].iat[-1]) == pytest.approx(105.0)
    _reset_dry_run()


def test_token_change_guard_blocks_removal_of_open_position(tmp_path, monkeypatch):
    from sol_trade.config import config as get_config

    cfg = get_config()
    prev = {
        "secondary_mints": ["M1"],
        "secondary_mint_symbols": ["SOL"],
        "secondary_weights": [],
    }
    # The removed token has an open position persisted in its CSV.
    open_csv = tmp_path / "SOL_data.csv"
    open_csv.write_text("close,position\n1.0,True\n")
    monkeypatch.setattr(
        trading, "read_dataframe_from_csv", lambda p: pd.read_csv(str(open_csv))
    )
    monkeypatch.setattr(cfg, "secondary_mints", ["M2"])
    monkeypatch.setattr(cfg, "secondary_mint_symbols", ["JUP"])

    trading._enforce_token_change_guard(prev)

    assert cfg.secondary_mints == ["M1"]  # rolled back
    assert cfg.secondary_mint_symbols == ["SOL"]


def test_token_change_guard_allows_removal_without_position(monkeypatch):
    from sol_trade.config import config as get_config

    cfg = get_config()
    prev = {
        "secondary_mints": ["M1"],
        "secondary_mint_symbols": ["SOL"],
        "secondary_weights": [],
    }
    def _no_csv(_path: str) -> pd.DataFrame:
        raise FileNotFoundError()

    monkeypatch.setattr(trading, "read_dataframe_from_csv", _no_csv)
    monkeypatch.setattr(cfg, "secondary_mints", ["M2"])
    monkeypatch.setattr(cfg, "secondary_mint_symbols", ["JUP"])

    trading._enforce_token_change_guard(prev)

    assert cfg.secondary_mints == ["M2"]  # change stands
    assert cfg.secondary_mint_symbols == ["JUP"]


def test_buy_records_fill_price_and_position_size(tmp_path, monkeypatch):
    _reset_dry_run()
    trading._dry_run = True
    trading._balance_cache.set_paper_mode(True)
    trading._balance_cache.set(config().primary_mint, 100.0)
    trading._balance_cache.set(config().secondary_mints[0], 0.0)
    monkeypatch.setattr(config(), "confluence_enabled", False)
    monkeypatch.setattr(config(), "secondary_mint_symbols", ["SOL"])
    monkeypatch.setattr(config(), "secondary_weights", [])
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


def test_trailing_stop_fires_immediately_on_fresh_breach(tmp_path, monkeypatch):
    _reset_dry_run()
    trading._dry_run = True
    trading._balance_cache.set_paper_mode(True)
    trading._balance_cache.set(config().primary_mint, 0.0)
    trading._balance_cache.set(config().secondary_mints[0], 2.0)
    monkeypatch.setattr(config(), "confluence_enabled", False)
    monkeypatch.setattr(config(), "sentiment_enabled", False)
    # Position entered at 1.00 with a 1.20 carried peak, so the ratcheted stop
    # is 1.20 * 0.98 = 1.176. The last close (1.17) sits below the fresh stop
    # but above the 1.00 stop carried from the previous cycle — the strategy
    # layer never set exit, so handle_sell_signal must.
    close = pd.Series([1.02, 1.05, 1.10, 1.17])
    df = pd.DataFrame(
        {
            "open": close,
            "close": close,
            "high": close + 0.02,
            "low": close - 0.02,
            "symbol": "SOL",
            "entry": 0,
            "exit": 0,
            "position": True,
            "position_size": 2.0,
            "entry_price": 1.00,
            "stoploss": 0.90,
            "takeprofit": 1.30,
            "trailing_stoploss": 1.00,  # carried constant from last cycle
            "trailing_stoploss_target": 1.05,
            "highest_price": 1.20,  # carried peak -> ratchet lifts the stop
        }
    )
    df = trading.strategy(df)  # attaches strategy_instance with risk parameters
    csv_path = tmp_path / "sol.csv"

    ok = trading.handle_sell_signal(
        df, config().secondary_mints[0], str(csv_path), "SOL"
    )

    assert ok
    saved = pd.read_csv(csv_path)
    assert bool(saved["position"].iat[-1]) is False  # full close
    # The paper ledger received the proceeds of the full 2.0 position.
    assert trading._balance_cache.get(config().primary_mint) == pytest.approx(
        2.0 * 1.17
    )
    _reset_dry_run()


def test_sell_capped_at_tracked_position(tmp_path, monkeypatch):
    _reset_dry_run()
    trading._dry_run = True
    trading._balance_cache.set_paper_mode(True)
    trading._balance_cache.set(config().primary_mint, 0.0)
    trading._balance_cache.set(config().secondary_mints[0], 5.0)  # pre-existing extra
    monkeypatch.setattr(config(), "confluence_enabled", False)
    monkeypatch.setattr(config(), "sentiment_enabled", False)
    df = _trading_df(exit_=1, size=2.0)  # tracked position is 2.0
    csv_path = tmp_path / "sol.csv"

    ok = trading.handle_sell_signal(
        df, config().secondary_mints[0], str(csv_path), "SOL"
    )

    assert ok
    # Only the tracked 2.0 was sold; the pre-existing 3.0 stays in the ledger.
    assert trading._balance_cache.get(config().secondary_mints[0]) == pytest.approx(3.0)
    saved = pd.read_csv(csv_path)
    assert bool(saved["position"].iat[-1]) is False  # position fully closed
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
