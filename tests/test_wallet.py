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
