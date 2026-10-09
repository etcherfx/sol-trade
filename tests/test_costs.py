"""The cost model: probes, interpolation and the default."""

import pytest

from sol_trade import costs, jupiter
from sol_trade.config import config


def _model(points, fee=0.0):
    return costs.CostModel("TKN", "measured", tuple(points), fee)


def test_interpolates_between_sizes():
    model = _model([(25, 10.0), (100, 20.0), (250, 50.0)])
    assert model.bps(25) == pytest.approx(10.0)
    assert model.bps(62.5) == pytest.approx(15.0)
    assert model.bps(175) == pytest.approx(35.0)


def test_flat_below_and_extrapolated_above():
    model = _model([(25, 10.0), (100, 20.0)])
    assert model.bps(5) == pytest.approx(10.0)
    assert model.bps(175) == pytest.approx(30.0)


def test_cost_adds_the_network_fee():
    model = _model([(25, 10.0)], fee=0.02)
    assert model.cost_usd(100) == pytest.approx(100 * 10 / 1e4 + 0.02)


def test_default_model_is_flagged():
    model = costs.model_for("NEVER_MEASURED")
    assert model.source == "default"
    assert model.bps(100) == config().costs["default_cost_bps"]


def _quote(in_usd, out_usd, gasless=True, lamports=0):
    return {
        "inUsdValue": in_usd,
        "outUsdValue": out_usd,
        "feeBps": 5,
        "priceImpact": -0.01,
        "router": "metis",
        "gasless": gasless,
        "signatureFeeLamports": lamports,
        "prioritizationFeeLamports": 0,
        "outAmount": "1",
    }


def test_measure_stores_probes_and_builds_the_model(monkeypatch):
    cfg = config()
    responses = {
        ("buy", 25): _quote(25, 24.95),  # 20 bps
        ("sell", 25): _quote(25, 24.94),  # 24 bps
        ("buy", 100): _quote(100, 99.7, gasless=False, lamports=10_000),  # 30 bps
        ("sell", 100): _quote(100, 99.66),  # 34 bps
    }
    calls = []

    def fake_quote(input_mint, output_mint, amount):
        side = "buy" if input_mint == cfg.primary_mint else "sell"
        size = 25 if len([c for c in calls if c[0] == side]) == 0 else 100
        calls.append((side, size, amount))
        return responses[(side, size)]

    monkeypatch.setattr(jupiter, "quote", fake_quote)
    monkeypatch.setattr(
        jupiter, "fetch_prices", lambda mints: {cfg.primary_mint: 1.0, "MINT": 2.0, cfg.sol_mint: 200.0}
    )
    monkeypatch.setattr(jupiter, "decimals", lambda mint: 6)

    record = costs.measure("TKN", "MINT", [25, 100])

    assert len(record["samples"]) == 4
    assert calls[0] == ("buy", 25, 25_000_000)
    assert calls[1] == ("sell", 25, 12_500_000)  # $25 of a $2 token
    model = costs.model_for("TKN")
    safety = config().costs["safety_multiplier"]
    assert model.source == "measured"
    assert model.bps(25) == pytest.approx(22.0 * safety)
    assert model.bps(100) == pytest.approx(32.0 * safety)
    assert costs.sample_age_days("TKN") == pytest.approx(0.0, abs=0.01)


def test_a_skewed_reference_price_cancels_between_legs():
    """Quotes valued at an off-mid USD price: the buy looks 20 bps cheap, the sell 40 bps dear."""
    from sol_trade.utils import write_json_atomic

    samples = [
        {"side": "buy", "size_usd": 100.0, "cost_bps": -20.0, "network_fee_usd": 0.0},
        {"side": "sell", "size_usd": 100.0, "cost_bps": 40.0, "network_fee_usd": 0.0},
    ]
    write_json_atomic(costs.COSTS_PATH, {"TKN": {"samples": samples, "measured_ts": 0}})
    safety = config().costs["safety_multiplier"]
    assert costs.model_for("TKN").bps(100) == pytest.approx(10.0 * safety)  # the round trip is 20 bps
