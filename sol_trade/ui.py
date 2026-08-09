"""Full-screen terminal UI for SolTrade — live dashboard, logs, and help.

The trading loop runs in a background thread and pushes data into a shared
``UIState``; this module renders that state with prompt_toolkit.

Keybindings:
    Tab / 1 / 2 / 3   switch between Dashboard, Logs, and Help
    Up / Down / PgUp / PgDn / Home / End   scroll the logs
    q / Ctrl-C        quit
"""

import asyncio
import logging
import threading
import time
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from prompt_toolkit.application import Application
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, VSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.styles import Style

from sol_trade.config import config
from sol_trade.log import get_recent_logs

SCREENS = ("dashboard", "logs", "help")


@dataclass
class Holding:
    """One wallet token holding for the dashboard's full-account view."""

    mint: str
    symbol: str | None  # None when the mint isn't one of the configured tokens
    balance: float
    value: float | None  # None when no price is available


@dataclass
class TokenStatus:
    """Per-token metrics for one analysis cycle."""

    symbol: str
    price: float = 0.0
    rsi: float = 0.0
    ema_short: float = 0.0
    ema_medium: float = 0.0
    entry_signal: bool = False
    exit_signal: bool = False
    position: bool = False
    stoploss: float | None = None
    takeprofit: float | None = None
    entry_price: float | None = None


@dataclass
class UIState:
    """Shared state between the trading thread and the UI."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    running: bool = False
    dry_run: bool = False
    primary_balance: float = 0.0
    reserved_fees: float = 0.02
    portfolio_value: float = 0.0
    total_profit: float = 0.0
    tokens: list[TokenStatus] = field(default_factory=list)
    full_holdings: list[Holding] = field(default_factory=list)
    total_account_value: float = 0.0
    countdown: int = 0
    last_refresh: str = "-"
    wallet_address: str = "-"
    error_count: int = 0
    strategy: str = "default"
    exchange: str = "okx"
    candles_path: str = "data/candles.db"

    def update(self, fn: Callable[["UIState"], None]) -> None:
        with self.lock:
            fn(self)

    def snapshot(self) -> "UIState":
        with self.lock:
            state = UIState(lock=self.lock)
            for name, value in self.__dict__.items():
                if name != "lock":
                    setattr(state, name, deepcopy(value))
            return state


def _level_style(levelno: int) -> str:
    if levelno >= logging.ERROR:
        return "bold red"
    if levelno >= logging.WARNING:
        return "bold yellow"
    if levelno >= logging.INFO:
        return "ansicyan"
    return "dim"


def _money(value: float) -> str:
    return f"${value:,.2f}"


def _token_price(value: float) -> str:
    """Format a token price/level with enough decimals to stay meaningful.

    Micro-cap tokens (e.g. BONK at $0.0000024) round to 0.0000 at the default
    four decimals, so sub-mill values get eight instead.
    """
    if value and value < 0.001:
        return f"{value:.8f}"
    return f"{value:.4f}"


# ---------------------------------------------------------------- dashboard


def header_fragments(state: UIState, screen_name: str) -> StyleAndTextTuples:
    state = state.snapshot()
    status = "● RUNNING" if state.running else "● STOPPED"
    status_style = "bold green" if state.running else "bold red"
    clock = time.strftime("%H:%M:%S")
    frags: StyleAndTextTuples = [
        ("reverse bold", "  SolTrade  "),
        ("", " "),
        (status_style, status),
        ("", " "),
    ]
    if state.dry_run:
        frags.append(("reverse bold yellow", " PAPER "))
        frags.append(("", " "))
    frags.append(("", f"[{screen_name}]"))
    frags.append(("dim", f"   {clock}"))
    frags.append(("", "\n"))
    return frags


def _wallet_fragments(state: UIState) -> StyleAndTextTuples:
    state = state.snapshot()
    frags: StyleAndTextTuples = [
        ("bold cyan", "── WALLET ───────────────────────────────\n")
    ]
    profit_style = "bold green" if state.total_profit >= 0 else "bold red"
    rows: list[tuple[str, str, str]] = [
        ("Primary Balance", f"{state.primary_balance:,.4f} USDC", ""),
        ("Reserved for Fees (SOL)", f"{state.reserved_fees:.4f} SOL", ""),
        ("Portfolio Value", _money(state.portfolio_value), ""),
        ("Total Account Value", _money(state.total_account_value), "bold"),
        ("Total Profit", _money(state.total_profit), profit_style),
        ("Wallet", state.wallet_address, "dim"),
    ]
    for label, value, style in rows:
        frags.append(("", f"  {label:<18} "))
        frags.append((style, f"{value}\n"))

    if state.full_holdings:
        frags.append(("", "\n"))
        frags.append(("bold cyan", "  HOLDINGS\n"))
        for holding in state.full_holdings:
            symbol = holding.symbol or f"{holding.mint[:6]}…"
            value = _money(holding.value) if holding.value is not None else "no price"
            frags.append(("", f"  {symbol:<10} {holding.balance:>12.6f} "))
            frags.append(("dim", f"{value:>12}\n"))
    return frags


def _status_fragments(state: UIState) -> StyleAndTextTuples:
    state = state.snapshot()
    features = []
    if config().whale_tracking_enabled:
        features.append("Whale")
    if config().confluence_enabled:
        features.append("Confluence")
    if config().market_regime_enabled:
        features.append("Regime")
    if config().sentiment_enabled:
        features.append("Sentiment")

    frags: StyleAndTextTuples = [
        ("bold cyan", "── STATUS ───────────────────────────────\n")
    ]
    rows: list[tuple[str, str]] = [
        ("Strategy", state.strategy),
        ("Exchange", state.exchange),
        ("Candle store", state.candles_path),
        ("Next update", f"in {state.countdown}s"),
        ("Last refresh", state.last_refresh),
        ("Features", ", ".join(features) if features else "—"),
        ("Errors", str(state.error_count)),
    ]
    for label, value in rows:
        frags.append(("", f"  {label:<18} "))
        frags.append(("dim" if label != "Errors" else "bold red" if state.error_count else "dim", f"{value}\n"))
    return frags


def _market_fragments(state: UIState) -> StyleAndTextTuples:
    state = state.snapshot()
    frags: StyleAndTextTuples = [
        ("bold cyan", "── MARKET ────────────────────────────────────────────────────────\n"),
        (
            "bold",
            (
                f"  {'TOKEN':<7}{'PRICE':>10} {'RSI':>6} {'EMA-S':>10} {'EMA-M':>10} "
                f"{'SIGNAL':>6} {'POS':>5} {'SL':>10} {'TP':>10}\n"
            ),
        ),
    ]
    if not state.tokens:
        frags.append(("dim", "  Waiting for the first analysis cycle…\n"))
        return frags
    for t in state.tokens:
        signal = "BUY" if t.entry_signal else ("SELL" if t.exit_signal else "—")
        sig_style = (
            "bold green"
            if signal == "BUY"
            else "bold red"
            if signal == "SELL"
            else "dim"
        )
        pos = "IN" if t.position else "OUT"
        pos_style = "bold green" if t.position else "dim"
        sl = _token_price(t.stoploss) if t.stoploss is not None else "—"
        tp = _token_price(t.takeprofit) if t.takeprofit is not None else "—"
        frags.append(
            (
                "",
                (
                    f"  {t.symbol:<7}{_token_price(t.price):>10} {t.rsi:>6.1f} "
                    f"{_token_price(t.ema_short):>10} {_token_price(t.ema_medium):>10} "
                ),
            )
        )
        frags.append((sig_style, f"{signal:>6}"))
        frags.append((pos_style, f" {pos:>5}"))
        frags.append(("", f" {sl:>10} {tp:>10}\n"))
    return frags


def _activity_fragments() -> StyleAndTextTuples:
    frags: StyleAndTextTuples = [
        ("bold cyan", "── ACTIVITY ─────────────────────────────────────────────────────\n")
    ]
    lines = get_recent_logs(8)
    if not lines:
        frags.append(("dim", "  No activity yet…\n"))
    for levelno, _created, text in lines:
        frags.append((_level_style(levelno), f"  {text[:110]}\n"))
    return frags


def footer_fragments(screen_name: str) -> StyleAndTextTuples:
    return [
        (
            "dim",
            "  [Tab] screens · [1] Dashboard [2] Logs [3] Help"
            + ("" if screen_name != "logs" else " · [↑/↓/PgUp/PgDn] scroll")
            + "  ·  [q] quit\n",
        )
    ]


# -------------------------------------------------------------------- logs


def logs_fragments() -> StyleAndTextTuples:
    frags: StyleAndTextTuples = [
        (
            "bold",
            "── LOGS ────────────────────────────────────────────────────────────────\n",
        )
    ]
    lines = get_recent_logs(500)
    if not lines:
        frags.append(("dim", "  No log entries yet…\n"))
    for levelno, _created, text in lines:
        frags.append((_level_style(levelno), f"  {text}\n"))
    return frags


def help_fragments() -> StyleAndTextTuples:
    rows = [
        ("Tab / 1 / 2 / 3", "switch between Dashboard, Logs, and Help"),
        ("Up / Down", "scroll the logs"),
        ("PgUp / PgDn", "scroll the logs faster"),
        ("Home / End", "jump to the start / end of the logs"),
        ("q / Ctrl-C", "quit SolTrade"),
    ]
    frags: StyleAndTextTuples = [
        ("bold", "── HELP ───────────────────────────────────────────────────────────────\n\n")
    ]
    for key, action in rows:
        frags.append(("bold cyan", f"  {key:<16}"))
        frags.append(("", f"{action}\n"))
    frags.append(
        (
            "dim",
            (
                "\n  SolTrade — automated Solana trading. Data: ccxt → local candle store.\n"
                "  Wallet operations are signed locally and never leave this machine.\n"
            ),
        )
    )
    return frags


# ------------------------------------------------------------- application


def build_application(
    state: UIState, output: Any | None = None, input: Any | None = None
) -> Application:
    """Build the prompt_toolkit application (does not run it)."""
    kb = KeyBindings()
    screen = {"name": "dashboard"}
    logs_scroll = {"offset": 0}

    def make_header(screen_name: str) -> Window:
        return Window(
            FormattedTextControl(lambda: header_fragments(state, screen_name)),
            height=1,
            style="class:header",
        )

    def make_footer(screen_name: str) -> Window:
        return Window(
            FormattedTextControl(lambda: footer_fragments(screen_name)),
            height=1,
            style="class:footer",
        )

    logs_window = Window(
        content=FormattedTextControl(logs_fragments),
        wrap_lines=False,
        always_hide_cursor=True,
    )

    dashboard = HSplit(
        [
            make_header("dashboard"),
            VSplit(
                [
                    HSplit(
                        [
                            Window(
                                FormattedTextControl(lambda: _wallet_fragments(state)),
                                always_hide_cursor=True,
                                wrap_lines=False,
                            ),
                            Window(
                                FormattedTextControl(lambda: _status_fragments(state)),
                                always_hide_cursor=True,
                                wrap_lines=False,
                            ),
                        ]
                    ),
                    Window(
                        FormattedTextControl(lambda: _market_fragments(state)),
                        always_hide_cursor=True,
                        wrap_lines=False,
                    ),
                ]
            ),
            Window(
                FormattedTextControl(_activity_fragments),
                height=9,
                always_hide_cursor=True,
                wrap_lines=False,
            ),
            make_footer("dashboard"),
        ]
    )

    logs_screen = HSplit(
        [make_header("logs"), logs_window, make_footer("logs")]
    )

    help_screen = HSplit(
        [
            make_header("help"),
            Window(
                FormattedTextControl(help_fragments),
                always_hide_cursor=True,
                wrap_lines=False,
            ),
            make_footer("help"),
        ]
    )

    containers: dict[str, Any] = {
        "dashboard": dashboard,
        "logs": logs_screen,
        "help": help_screen,
    }

    def switch(app: Application, name: str) -> None:
        screen["name"] = name
        app.layout.container = containers[name]
        app.invalidate()

    def clamp_scroll(offset: int) -> int:
        return max(0, min(offset, 10_000))

    @kb.add("tab")
    def _(event: Any) -> None:
        idx = (SCREENS.index(screen["name"]) + 1) % len(SCREENS)
        switch(event.app, SCREENS[idx])

    for digit, name in zip(("1", "2", "3"), SCREENS):

        @kb.add(digit)
        def _(event: Any, name: str = name) -> None:
            switch(event.app, name)

    @kb.add("q")
    def _(event: Any) -> None:
        event.app.exit()

    @kb.add("c-c")
    def _(event: Any) -> None:
        event.app.exit()

    def scroll(delta: int) -> Callable[[Any], None]:
        def _handle(event: Any) -> None:
            if screen["name"] != "logs":
                return
            logs_scroll["offset"] = clamp_scroll(logs_scroll["offset"] + delta)
            logs_window.vertical_scroll = logs_scroll["offset"]
            event.app.invalidate()

        return _handle

    @kb.add("up")
    def _(event: Any) -> None:
        scroll(-1)(event)

    @kb.add("down")
    def _(event: Any) -> None:
        scroll(1)(event)

    @kb.add("pageup")
    def _(event: Any) -> None:
        scroll(-20)(event)

    @kb.add("pagedown")
    def _(event: Any) -> None:
        scroll(20)(event)

    @kb.add("home")
    def _(event: Any) -> None:
        if screen["name"] == "logs":
            logs_scroll["offset"] = 0
            logs_window.vertical_scroll = 0
            event.app.invalidate()

    @kb.add("end")
    def _(event: Any) -> None:
        if screen["name"] == "logs":
            logs_scroll["offset"] = 10_000
            logs_window.vertical_scroll = 10_000
            event.app.invalidate()

    app = Application(
        layout=Layout(containers["dashboard"]),
        key_bindings=kb,
        style=Style.from_dict(
            {"header": "bg:#0d2b45 #ffffff", "footer": "bg:#0d2b45 #9adcff"}
        ),
        full_screen=True,
        output=output,
        input=input,
    )
    return app


async def _run_app(app: Application) -> None:
    """Run the app with a 0.5s repaint clock scheduled inside its event loop."""

    async def _clock() -> None:
        while True:
            app.invalidate()
            await asyncio.sleep(0.5)

    clock = asyncio.create_task(_clock())
    try:
        await app.run_async()
    finally:
        clock.cancel()


def run_ui(state: UIState) -> None:
    """Fill initial state and run the UI until the user quits."""
    state.update(
        lambda s: (
            setattr(s, "strategy", config().strategy),
            setattr(s, "exchange", config().data_exchange),
            setattr(s, "candles_path", config().candles_path),
            setattr(s, "wallet_address", _short_address(str(config().public_address))),
        )
    )
    asyncio.run(_run_app(build_application(state)))


def _short_address(address: str) -> str:
    return f"{address[:6]}…{address[-4:]}"
