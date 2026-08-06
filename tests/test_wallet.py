"""Tests for the dynamic SOL reserve."""

import pytest

from sol_trade import wallet


def _rent_response(lamports: int):
    return type("R", (), {"value": lamports})()


def test_fallback_when_rpc_down(monkeypatch):
    wallet._sol_reserve_cache = None
    monkeypatch.setattr(wallet, "token_account_ui_amounts", lambda owner, mint: [1.0])

    def _boom(*args, **kwargs):
        raise RuntimeError("rpc down")

    monkeypatch.setattr(wallet, "run_async", _boom)
    assert wallet.minimum_sol_needed() == pytest.approx(0.02)


def test_only_signature_buffer_when_atas_exist(monkeypatch):
    wallet._sol_reserve_cache = None
    monkeypatch.setattr(wallet, "token_account_ui_amounts", lambda owner, mint: [1.0])
    monkeypatch.setattr(wallet, "run_async", lambda coro: _rent_response(2_039_280))
    # No missing ATA -> just the 2-signature fee buffer.
    assert wallet.minimum_sol_needed() == pytest.approx((2 * 5000) / 1e9)


def test_rent_included_when_ata_missing(monkeypatch):
    wallet._sol_reserve_cache = None
    monkeypatch.setattr(
        wallet,
        "token_account_ui_amounts",
        lambda owner, mint: [] if mint != wallet.config().sol_mint else [1.0],
    )
    monkeypatch.setattr(wallet, "run_async", lambda coro: _rent_response(2_039_280))
    # One missing ATA -> rent-exempt deposit + signature buffer.
    assert wallet.minimum_sol_needed() == pytest.approx((2_039_280 + 2 * 5000) / 1e9)


def test_get_all_token_holdings_aggregates_and_adds_sol(monkeypatch):
    import json as _json

    class _Response:
        def __init__(self, json_body: str | None = None, value: int | None = None):
            self._json = json_body
            self.value = value

        def to_json(self) -> str:
            return self._json or ""

    token_accounts = [
        {"account": {"data": {"parsed": {"info": {"mint": "M1", "tokenAmount": {"uiAmount": 1.5}}}}}},
        {"account": {"data": {"parsed": {"info": {"mint": "M1", "tokenAmount": {"uiAmount": 2.0}}}}}},
        {"account": {"data": {"parsed": {"info": {"isNative": True, "tokenAmount": {"uiAmount": 9.0}}}}}},
        {"account": {"data": {"parsed": {"info": {"mint": "M2", "tokenAmount": {"uiAmount": None}}}}}},
    ]
    calls: list = []

    def fake_run_async(coro):
        coro.close()  # never awaited in this test; discard the real coroutine
        calls.append(coro)
        if len(calls) == 1:
            return _Response(json_body=_json.dumps({"result": {"value": token_accounts}}))
        return _Response(value=5_000_000_000)  # 5 native SOL

    monkeypatch.setattr(wallet, "run_async", fake_run_async)

    holdings = wallet.get_all_token_holdings()

    assert holdings["M1"] == pytest.approx(3.5)  # two accounts aggregated
    assert holdings[wallet.config().sol_mint] == pytest.approx(5.0)  # native SOL added
    assert "M2" not in holdings  # null uiAmount skipped
