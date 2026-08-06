"""Tests for the strategy layer and signal generation."""

import numpy as np
import pandas as pd
import pytest

from sol_trade.strategy import (
    calc_entry_price,
    calc_stoploss,
    calc_takeprofit,
    calc_trailing_stoploss,
    strategy,
)
from strategies.default_strategy import DefaultStrategy


def _ramp_df() -> pd.DataFrame:
    """80-bar gentle uptrend — RSI stays high but no TA exit triggers."""
    close = np.linspace(100.0, 110.0, 80)
    return pd.DataFrame(
        {"close": close, "high": close + 1, "low": close - 1, "open": close}
    )


def test_no_exit_without_risk_columns():
    out = DefaultStrategy(_ramp_df()).apply_strategy()
    assert pd.isna(out["exit"].iat[-1])


def test_protective_exit_fires_with_risk_columns():
    merged = _ramp_df()
    merged["position"] = True
    merged["entry_price"] = 105.0
    merged["trailing_stoploss"] = 110.5  # close (110.0) at/below the stop
    merged["stoploss"] = 90.0
    merged["takeprofit"] = 130.0
    out = DefaultStrategy(merged).apply_strategy()
    assert pd.notna(out["exit"].iat[-1])
    assert out["exit"].iat[-1] == 1


def test_strategy_instance_is_per_dataframe():
    s1 = strategy(_ramp_df())
    s2 = strategy(_ramp_df().assign(close=lambda d: d.close * 10))
    assert s1.strategy_instance is not s2.strategy_instance


def test_risk_calculation_uses_own_instance():
    s = strategy(_ramp_df())
    s = calc_entry_price(s)
    s = calc_stoploss(s)
    s = calc_takeprofit(s)
    s = calc_trailing_stoploss(s)
    assert float(s["stoploss"].iat[-1]) == pytest.approx(110.0 * 0.95)
    assert float(s["takeprofit"].iat[-1]) == pytest.approx(110.0 * 1.10)
    assert "trailing_stoploss" in s.columns
