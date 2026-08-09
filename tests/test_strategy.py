"""Tests for the strategy layer and signal generation."""

import numpy as np
import pandas as pd
import pytest

from sol_trade.strategy import (
    calc_entry_price,
    calc_stoploss,
    calc_takeprofit,
    calc_trailing_stoploss,
    load_strategy_class,
    resolve_strategy_name,
    strategy,
)
from strategies.default_strategy import DefaultStrategy


@pytest.fixture
def default_strategy(monkeypatch):
    """Force the default strategy so tests are independent of config.json."""
    from sol_trade.config import config

    monkeypatch.setattr(config(), "strategy", "default")
    return config()


def _ramp_df() -> pd.DataFrame:
    """80-bar gentle uptrend — RSI stays high but no TA exit triggers."""
    close = np.linspace(100.0, 110.0, 80)
    return pd.DataFrame(
        {"close": close, "high": close + 1, "low": close - 1, "open": close}
    )


def test_no_exit_without_risk_columns(default_strategy):
    out = DefaultStrategy(_ramp_df()).apply_strategy()
    assert pd.isna(out["exit"].iat[-1])


def test_protective_exit_fires_with_risk_columns(default_strategy):
    merged = _ramp_df()
    merged["position"] = True
    merged["entry_price"] = 105.0
    merged["trailing_stoploss"] = 110.5  # close (110.0) at/below the stop
    merged["stoploss"] = 90.0
    merged["takeprofit"] = 130.0
    out = DefaultStrategy(merged).apply_strategy()
    assert pd.notna(out["exit"].iat[-1])
    assert out["exit"].iat[-1] == 1


def test_strategy_instance_is_per_dataframe(default_strategy):
    s1 = strategy(_ramp_df())
    s2 = strategy(_ramp_df().assign(close=lambda d: d.close * 10))
    assert s1.strategy_instance is not s2.strategy_instance


def test_risk_calculation_uses_own_instance(default_strategy):
    s = strategy(_ramp_df())
    s = calc_entry_price(s)
    s = calc_stoploss(s)
    s = calc_takeprofit(s)
    s = calc_trailing_stoploss(s)
    assert float(s["stoploss"].iat[-1]) == pytest.approx(110.0 * 0.95)
    assert float(s["takeprofit"].iat[-1]) == pytest.approx(110.0 * 1.10)
    assert "trailing_stoploss" in s.columns


def test_resolve_strategy_name_per_token_override(default_strategy):
    from sol_trade.config import config

    config().token_strategies = {"SOL": "momentum"}
    try:
        assert resolve_strategy_name("SOL") == "momentum"  # per-token wins
        assert resolve_strategy_name("JUP") == "default"  # unlisted -> global
        assert resolve_strategy_name(None) == "default"
    finally:
        config().token_strategies = {}


def test_load_strategy_class_snake_case_name(default_strategy):
    # "jup_trend" must map to JupTrendStrategy, not the legacy Jup_trendStrategy.
    cls = load_strategy_class("jup_trend")
    assert cls.__name__ == "JupTrendStrategy"


def test_load_strategy_class_unknown_raises(default_strategy):
    import pytest

    # Missing module -> ModuleNotFoundError; missing class -> AttributeError.
    with pytest.raises((ModuleNotFoundError, AttributeError)):
        load_strategy_class("does_not_exist")


def test_trailing_stoploss_carries_peak_across_windows(default_strategy):
    # Window 1: last-bar high spikes to 1.20, far above entry (1.01) and the
    # 1.05x target, so the trailing stop activates at 1.20 * 0.98.
    df1 = pd.DataFrame(
        {"close": [1.00, 1.00, 1.00, 1.01], "high": [1.00, 1.05, 1.10, 1.20], "low": [0.99, 0.99, 0.99, 1.00]}
    )
    s1 = strategy(df1)
    s1 = calc_entry_price(s1)
    s1 = calc_trailing_stoploss(s1)
    assert float(s1["trailing_stoploss"].iat[-1]) == pytest.approx(1.20 * 0.98)

    df2 = pd.DataFrame(
        {"close": [1.04, 1.02, 1.01], "high": [1.07, 1.05, 1.04], "low": [1.03, 1.01, 1.00]}
    )
    # Without the carry, the same window would start from its own first high.
    # (Computed before s2 mutates df2 with the carried column.)
    s3 = strategy(df2.copy())
    s3 = calc_entry_price(s3)
    s3 = calc_trailing_stoploss(s3)
    assert float(s3["highest_price"].iat[-1]) < 1.20

    # Window 2: fresh window with LOWER highs — the carried peak (1.20) must
    # persist instead of resetting to the new window's first high.
    s2 = strategy(df2.copy())
    s2 = calc_entry_price(s2)
    s2["highest_price"] = float(s1["highest_price"].iat[-1])  # as the CSV carry-over
    s2 = calc_trailing_stoploss(s2)
    assert float(s2["highest_price"].iat[-1]) == pytest.approx(1.20)
    assert float(s2["trailing_stoploss"].iat[-1]) == pytest.approx(1.20 * 0.98)
