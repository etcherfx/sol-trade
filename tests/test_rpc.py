"""Tests for the budgeted RPC client: token bucket, daily budget, usage file, 429s."""

import asyncio
import json

import httpx
import pytest

from sol_trade import rpc
from sol_trade.config import config


class _Clock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def test_rpc_method_name():
    assert rpc.rpc_method_name("get_balance") == "getBalance"
    assert (
        rpc.rpc_method_name("get_token_accounts_by_owner_json_parsed")
        == "getTokenAccountsByOwner"
    )
    assert rpc.rpc_method_name("get_minimum_balance_for_rent_exemption") == (
        "getMinimumBalanceForRentExemption"
    )


def test_token_bucket_waits_on_a_fake_clock():
    clock = _Clock()
    bucket = rpc.TokenBucket(30.0, clock=clock)
    # A full bucket (30 CU) covers three 10-CU calls at once.
    assert [bucket.reserve(10) for _ in range(3)] == [0.0, 0.0, 0.0]
    # The fourth is 10 CU in debt: one third of a second at 30 CU/s.
    assert bucket.reserve(10) == pytest.approx(10 / 30)
    # Callers queue: the next one waits for both debts.
    assert bucket.reserve(10) == pytest.approx(20 / 30)
    clock.now += 1.0  # refills 30 CU, clearing the 20 CU debt
    assert bucket.reserve(10) == 0.0


def test_token_bucket_refill_is_capped():
    clock = _Clock()
    bucket = rpc.TokenBucket(10.0, clock=clock)
    clock.now += 100.0  # a long idle period cannot bank more than one second
    assert bucket.reserve(10) == 0.0
    assert bucket.reserve(10) == pytest.approx(1.0)


def test_usage_counter_writes_daily_file(tmp_path):
    clock = _Clock(1_760_000_000.0)  # 2025-10-09 UTC
    counter = rpc.UsageCounter(str(tmp_path / "rpc_usage.json"), clock=clock)
    counter.record("getBalance", 10)
    counter.record("getBalance", 10)
    counter.record("getTokenAccountsByOwner", 10)

    data = json.loads((tmp_path / "rpc_usage.json").read_text())
    day = data["2025-10-09"]
    assert day["calls"] == 3
    assert day["cu"] == 30
    assert day["methods"]["getBalance"] == {"calls": 2, "cu": 20}
    assert counter.today()["cu"] == 30
    assert counter.snapshot() == (3, 30)


def _fake_client_factory(calls: list, fail_with: list | None = None):
    class _Fake:
        def __init__(self, endpoint: str) -> None:
            self.endpoint = endpoint

        async def get_balance(self, *args):
            calls.append(("get_balance", args))
            if fail_with:
                raise fail_with.pop(0)
            return "ok"

    return _Fake


@pytest.fixture
def isolated_usage(tmp_path, monkeypatch):
    counter = rpc.UsageCounter(str(tmp_path / "usage.json"))
    monkeypatch.setattr(rpc, "usage", counter)
    monkeypatch.setattr(rpc, "_bucket", rpc.TokenBucket(1_000_000.0))
    return counter


def test_budgeted_client_counts_calls(isolated_usage):
    calls: list = []
    client = rpc.BudgetedClient("http://unused", factory=_fake_client_factory(calls))

    assert asyncio.run(client.get_balance("addr")) == "ok"

    assert calls == [("get_balance", ("addr",))]
    assert isolated_usage.snapshot() == (1, 10)
    assert isolated_usage.today()["methods"]["getBalance"]["cu"] == 10


def test_daily_budget_blocks_non_essential_calls(isolated_usage, monkeypatch):
    monkeypatch.setitem(config().rpc, "daily_cu_budget", 15)
    calls: list = []
    client = rpc.BudgetedClient("http://unused", factory=_fake_client_factory(calls))

    asyncio.run(client.get_balance("a"))  # 10 of 15 CU
    with pytest.raises(rpc.RpcBudgetExceeded):
        client.get_balance("b")  # would reach 20 CU
    with rpc.essential():
        assert asyncio.run(client.get_balance("c")) == "ok"  # swap reads still run

    assert [args for _, args in calls] == [("a",), ("c",)]


def test_429_retries_honor_retry_after(isolated_usage, monkeypatch):
    request = httpx.Request("POST", "http://unused")
    response = httpx.Response(429, headers={"Retry-After": "2"}, request=request)
    error = httpx.HTTPStatusError("429", request=request, response=response)
    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr(rpc.asyncio, "sleep", fake_sleep)
    calls: list = []
    client = rpc.BudgetedClient(
        "http://unused", factory=_fake_client_factory(calls, fail_with=[error])
    )

    assert asyncio.run(client.get_balance("a")) == "ok"

    assert slept == [2.0]
    assert len(calls) == 2
    assert isolated_usage.snapshot() == (2, 20)  # every attempt is counted


def test_non_429_errors_are_not_retried(isolated_usage):
    calls: list = []
    client = rpc.BudgetedClient(
        "http://unused", factory=_fake_client_factory(calls, fail_with=[RuntimeError("boom")])
    )
    with pytest.raises(RuntimeError):
        asyncio.run(client.get_balance("a"))
    assert len(calls) == 1


def test_method_cu_overrides(monkeypatch):
    assert rpc.method_cu("getBalance") == 10
    assert rpc.method_cu("someNewMethod") == 40
    monkeypatch.setitem(config().rpc, "method_cu", {"getBalance": 3})
    assert rpc.method_cu("getBalance") == 3
