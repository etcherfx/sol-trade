"""Tests for the UI shared state."""

from types import SimpleNamespace
from typing import Any, cast

import pytest
from prompt_toolkit.formatted_text import to_formatted_text
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.layout import HSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.output import DummyOutput

from sol_trade import ui
from sol_trade.config import config
from sol_trade.ui import TokenStatus, UIState, _token_price


def test_run_ui_fills_initial_state(monkeypatch):
    def fake_run(coro):
        coro.close()

    monkeypatch.setattr(ui, "build_application", lambda state: None)
    monkeypatch.setattr(ui.asyncio, "run", fake_run)
    monkeypatch.setattr(ui, "_wallet_label", lambda: "abc123…wxyz")
    # Values that differ from the UIState defaults, so each assignment shows.
    monkeypatch.setattr(config(), "strategy", "momentum")
    monkeypatch.setattr(config(), "data_exchange", "kraken")
    monkeypatch.setattr(config(), "candles_path", "data/other.db")

    state = UIState()
    ui.run_ui(state)

    assert state.strategy == "momentum"
    assert state.exchange == "kraken"
    assert state.candles_path == "data/other.db"
    assert state.wallet_address == "abc123…wxyz"


def _digit_handler(app, digit):
    (binding,) = [b for b in app.key_bindings.bindings if b.keys == (digit,)]
    return binding.handler


def test_digit_keys_switch_to_each_screen():
    with create_pipe_input() as pipe:
        app = ui.build_application(UIState(), output=DummyOutput(), input=pipe)
        for digit, name in zip(("1", "2", "3"), ui.SCREENS, strict=True):
            _digit_handler(app, digit)(cast(Any, SimpleNamespace(app=app)))
            screen = app.layout.container
            assert isinstance(screen, HSplit)
            footer = screen.children[-1]
            assert isinstance(footer, Window)
            assert isinstance(footer.content, FormattedTextControl)
            text = to_formatted_text(footer.content.text)
            assert text == to_formatted_text(ui.footer_fragments(name))


def test_screen_without_digit_key_fails_loudly(monkeypatch):
    # A screen added without a digit binding must not be silently unreachable.
    monkeypatch.setattr(ui, "SCREENS", (*ui.SCREENS, "extra"))
    with create_pipe_input() as pipe, pytest.raises(ValueError):
        ui.build_application(UIState(), output=DummyOutput(), input=pipe)


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


def test_token_price_micro_cap_keeps_significant_decimals():
    # BONK-class tokens: 4 decimals would round $0.0000024 to 0.0000.
    assert _token_price(0.0000024) == "0.00000240"
    assert _token_price(76.32) == "76.3200"
    assert _token_price(0.0) == "0.0000"
