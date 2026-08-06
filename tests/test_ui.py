"""Tests for the UI shared state."""

from sol_trade.ui import TokenStatus, UIState


def test_update_holds_lock_and_snapshot_copies():
    state = UIState()
    state.update(lambda s: setattr(s, "running", True))
    snap = state.snapshot()
    assert snap.running is True
    assert snap is not state

    snap.tokens.append(TokenStatus(symbol="SOL"))
    assert len(snap.tokens) == 1
    assert len(state.tokens) == 0  # snapshot is a copy, original untouched


def test_tokens_replaced_atomically():
    state = UIState()
    state.update(lambda s: setattr(s, "tokens", [TokenStatus(symbol="SOL")]))
    snap = state.snapshot()
    assert [t.symbol for t in snap.tokens] == ["SOL"]
