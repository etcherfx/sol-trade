"""JSON strategy specs: validation errors, hashing, loading and evaluation."""

import json

import numpy as np
import pandas as pd
import pytest

from sol_trade import spec
from sol_trade.config import config
from tests.test_indicators import _reference_ema, _reference_rsi


def _spec(**overrides):
    base = {
        "schema": 1,
        "name": "fixture",
        "timeframe": "1h",
        "lookback": 60,
        "risk": {"stoploss": 4, "takeprofit": 8},
        "indicators": {"fast": {"fn": "ema", "period": 9}, "slow": {"fn": "ema", "period": 21}},
        "entry": {"crosses_above": ["fast", "slow"]},
        "exit": {"crosses_below": ["fast", "slow"]},
    }
    base.update(overrides)
    return {k: v for k, v in base.items() if v is not None}


def _errors(data, name="fixture") -> list[str]:
    with pytest.raises(spec.SpecError) as raised:
        spec.compile_spec(data, name)
    return raised.value.errors


def test_valid_spec_compiles():
    strategy = spec.compile_spec(_spec(), "fixture")
    assert strategy.timeframe == "1h"
    assert strategy.period_seconds == 3600
    assert strategy.warmup == 22  # ema 21 plus one bar for the cross
    assert strategy.risk.stoploss == 4.0
    assert strategy.risk.trailing_stoploss == 0.0


def test_unknown_function_is_named_with_its_path():
    errors = _errors(_spec(indicators={"fast": {"fn": "emaa", "period": 9}, "slow": {"fn": "ema", "period": 21}}))
    assert len(errors) == 1
    assert errors[0].startswith("indicators.fast.fn: unknown fn 'emaa'")


def test_lookback_shorter_than_warmup():
    errors = _errors(_spec(lookback=21))
    assert errors == ["lookback: 21 is smaller than the required warm-up of 22 bars"]


def test_unknown_series_and_output():
    errors = _errors(_spec(entry=["close", ">", "bb.lower"], exit=["close", "<", "fast.upper"]))
    assert errors[0].startswith("entry[2]: unknown series 'bb.lower'")
    assert errors[1].startswith("exit[2]: 'fast' has no output 'upper'")


def test_multi_output_indicator_needs_an_output():
    errors = _errors(_spec(indicators={"bb": {"fn": "bbands"}}, entry=["close", "<", "bb"], exit=None))
    assert any("several outputs" in e for e in errors)


def test_circular_indicators():
    errors = _errors(
        _spec(
            indicators={"a": {"fn": "ema", "period": 3, "input": "b"}, "b": {"fn": "sma", "period": 3, "input": "a"}},
            entry=["a", ">", "b"],
        )
    )
    assert any("circular reference" in e for e in errors)


def test_bad_parameters_and_keys():
    errors = _errors(
        _spec(
            colour="red",
            risk={"stoploss": -1, "stop": 3},
            indicators={"fast": {"fn": "ema", "period": 0, "length": 3}, "slow": {"fn": "ema", "period": 21}},
        )
    )
    joined = "\n".join(errors)
    assert "colour: unknown key" in joined
    assert "risk.stop: unknown key" in joined
    assert "risk.stoploss: must be a percentage" in joined
    assert "indicators.fast.length: unknown parameter for ema" in joined
    assert "indicators.fast.period: must be an integer from 1 to 1000" in joined


def test_position_size_is_a_fraction_of_the_trade_size():
    assert spec.compile_spec(_spec(), "fixture").risk.position_size == 1.0
    half = spec.compile_spec(_spec(risk={"stoploss": 4, "position_size": 0.5}), "fixture")
    assert half.risk.position_size == 0.5
    assert half.hash != spec.compile_spec(_spec(), "fixture").hash  # part of the identity
    for bad in (0, -0.1, 1.5, "half", True):
        errors = _errors(_spec(risk={"position_size": bad}))
        assert errors == [
            "risk.position_size: must be a fraction of risk.max_trade_size above 0 and up to 1, or null"
        ]


def test_name_must_match_the_file():
    assert _errors(_spec(name="other")) == ["name: 'other' must equal the file name 'fixture'"]


def test_hash_is_canonical():
    a = spec.compile_spec(_spec(), "fixture")
    reordered = json.loads(json.dumps(_spec(), sort_keys=True))
    assert spec.compile_spec(reordered, "fixture").hash == a.hash
    assert spec.compile_spec(_spec(lookback=61), "fixture").hash != a.hash


def test_load_strategy_from_the_runtime_folder(tmp_path):
    folder = tmp_path / "strategies"
    folder.mkdir()
    (folder / "fixture.json").write_text(json.dumps(_spec()))
    assert spec.load_strategy("fixture").name == "fixture"


def test_python_strategy_gets_a_conversion_error(tmp_path):
    folder = tmp_path / "strategies"
    folder.mkdir()
    (folder / "momentum_strategy.py").write_text("class MomentumStrategy: ...\n")
    with pytest.raises(spec.SpecError, match="Python strategies are no longer supported; convert to strategies/momentum.json"):
        spec.load_strategy("momentum")


def test_missing_strategy(tmp_path):
    with pytest.raises(spec.SpecError, match="not found"):
        spec.load_strategy("nothing_here")


def test_builtin_default_is_used_without_a_file(monkeypatch):
    monkeypatch.setattr(config(), "strategies_path", "no_such_folder")
    strategy = spec.load_strategy("default")
    assert strategy.timeframe == "1m"
    assert strategy.lookback == 50


def test_resolve_strategy_name_per_token_override(monkeypatch):
    monkeypatch.setattr(config(), "strategy", "default")
    monkeypatch.setattr(config(), "token_strategies", {"SOL": "momentum"})
    assert spec.resolve_strategy_name("SOL") == "momentum"
    assert spec.resolve_strategy_name("JUP") == "default"
    assert spec.resolve_strategy_name(None) == "default"


def _random_bars(n: int, seed: int = 3) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 0.6, n))
    return {
        "open": close + rng.normal(0, 0.2, n),
        "high": close + rng.random(n),
        "low": close - rng.random(n),
        "close": close,
        "volume": rng.random(n) * 100,
    }


def test_builtin_default_reproduces_the_old_python_default():
    """The JSON default gives the same last-bar signals as the removed DefaultStrategy."""
    strategy = spec.load_strategy("default")
    bars = _random_bars(1500)
    batch = {k: spec.windows(v, 50) for k, v in bars.items()}
    result = strategy.evaluate(batch, spec.windows(np.arange(1500) * 60.0, 50))
    for j in range(batch["close"].shape[1]):
        close = pd.Series(batch["close"][:, j])
        ema_s, ema_m = _reference_ema(close, 5), _reference_ema(close, 21)
        sma, std = close.rolling(14).mean(), close.rolling(14).std()
        rsi = _reference_rsi(close, 14)
        entry = ((ema_s > ema_m) | (close < sma - std * 2)) & (rsi <= 30)
        exit_ = ((ema_s < ema_m) | (close > sma + std * 2)) & (rsi >= 70)
        assert bool(entry.iat[-1]) == result.entry[j]
        assert bool(exit_.iat[-1]) == result.exit[j]
    assert result.entry.any() and result.exit.any()


def test_operators_shift_arithmetic_and_not():
    data = _spec(
        lookback=5,
        indicators={},
        entry={"all": [["close", ">", {"shift": ["close", 1]}], {"not": ["close", ">", {"mul": [2, "open"]}]}]},
        exit=["close", "<", {"sub": ["open", 0.5]}],
    )
    strategy = spec.compile_spec(data, "fixture")
    closes = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    bars = {"open": closes - 0.1, "high": closes, "low": closes, "close": closes, "volume": closes}
    result = strategy.evaluate(bars, np.zeros_like(closes))
    assert result.entry.tolist() == [True]
    assert result.exit.tolist() == [False]


def test_nan_comparisons_are_false():
    data = _spec(lookback=30, indicators={"slow": {"fn": "sma", "period": 25}}, entry=["close", "<", "slow"], exit=None)
    strategy = spec.compile_spec(data, "fixture")
    bars = {k: v[:, None] for k, v in _random_bars(30).items()}
    bars["close"][-1, 0] = np.nan
    assert strategy.evaluate(bars, np.zeros((30, 1))).entry.tolist() == [False]
