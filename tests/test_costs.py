"""The cost model: probes, interpolation and the default."""

import httpx
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
        jupiter, "fetch_prices", lambda mints, **_: {cfg.primary_mint: 1.0, "MINT": 2.0, cfg.sol_mint: 200.0}
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


def _too_many(retry_after=None):
    headers = {"retry-after": retry_after} if retry_after else {}
    response = httpx.Response(429, headers=headers, request=httpx.Request("GET", "https://api.jup.ag/swap/v2/order"))
    return httpx.HTTPStatusError("429 Too Many Requests", request=response.request, response=response)


@pytest.fixture
def jupiter_fakes(monkeypatch):
    """Quotes that answer 20 bps unless a test makes them fail; sleeps are recorded, not slept."""
    cfg = config()
    state = {"fail": {}, "calls": 0}
    slept = []

    def fake_quote(input_mint, output_mint, amount):
        state["calls"] += 1
        side = "buy" if input_mint == cfg.primary_mint else "sell"
        mint = output_mint if side == "buy" else input_mint
        failures = state["fail"].get(mint)
        if failures:
            error = failures.pop(0) if isinstance(failures, list) else failures
            if error is not None:
                raise error
        return _quote(100, 99.8)

    monkeypatch.setattr(jupiter, "quote", fake_quote)
    monkeypatch.setattr(jupiter, "fetch_prices", lambda mints, **_: dict.fromkeys(mints, 1.0))
    monkeypatch.setattr(jupiter, "decimals", lambda mint: 6)
    monkeypatch.setattr(costs.time, "sleep", slept.append)
    return state, slept


def test_rate_limited_probes_are_retried_with_backoff(jupiter_fakes):
    state, slept = jupiter_fakes
    state["fail"]["MINT"] = [_too_many(), _too_many("7")]  # the first two requests

    record = costs.measure("TKN", "MINT", [25, 100])

    assert record["complete"] and record["errors"] == []
    assert len(record["samples"]) == 4
    assert slept == [5.0, 7.0]  # doubling backoff from rate_limit_backoff_seconds, or Retry-After


def test_a_probe_that_stays_rate_limited_is_reported(jupiter_fakes):
    state, slept = jupiter_fakes
    retries = config().costs["rate_limit_retries"]
    state["fail"]["MINT"] = [None, None] + [_too_many()] * (retries + 1)  # $25 works, $100 buy never does

    record = costs.measure("TKN", "MINT", [25, 100])

    assert not record["complete"]
    assert [(s["side"], s["size_usd"]) for s in record["samples"]] == [("buy", 25), ("sell", 25), ("sell", 100)]
    assert record["errors"][0].startswith("buy $100: HTTPStatusError: 429")
    assert slept == [5.0 * 2**i for i in range(retries)]


def test_jupiter_requests_are_spaced_while_paced(monkeypatch):
    clock = {"now": 100.0}
    waits = []
    monkeypatch.setattr(jupiter.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(jupiter.time, "sleep", lambda s: waits.append(s) or clock.update(now=clock["now"] + s))
    monkeypatch.setitem(jupiter._pace, "last", 0.0)
    with jupiter.paced(2.0):
        jupiter._wait_turn()
        clock["now"] += 0.5
        jupiter._wait_turn()
        jupiter._wait_turn()
    jupiter._wait_turn()  # unpaced again: the bot never waits
    assert waits == [pytest.approx(1.5), pytest.approx(2.0)]
