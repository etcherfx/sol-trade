"""The trading loop: runs every configured token through the decision engine.

Each cycle, per token: refresh its candles, replay every closed bar since
the last one processed through the engine's open / low / close steps, then
apply the forming bar's open (queued fills) and running low (stops). The
book is ``live`` (Jupiter swaps from the wallet) or, with ``--dry-run``,
``dry_run`` (a virtual ledger with modeled costs). Positions persist in
``data/research.db``.
"""

import asyncio
import threading
import time
from datetime import UTC, datetime
from typing import Any

import numpy as np

from sol_trade import backtest, costs, history, ledger
from sol_trade.config import config
from sol_trade.engine import BUY, DUST, SELL, Book, Fill, SimExecutor
from sol_trade.jupiter import fetch_prices
from sol_trade.log import log_general, log_transaction
from sol_trade.market_regime import update_regime
from sol_trade.sentiment import is_market_crash, is_token_blocked, update_sentiment
from sol_trade.spec import SpecError, Strategy, load_strategy, resolve_strategy_name
from sol_trade.transactions import perform_swap
from sol_trade.ui import Holding, TokenStatus, UIState
from sol_trade.wallet import find_balance, get_all_token_holdings, minimum_sol_needed
from sol_trade.whale_tracker import update_whale_data

# Missed bars replayed after downtime, at most.
_MAX_REPLAY_BARS = 2000


class BalanceCache:
    """Wallet balance reads, reused for ``rpc.balance_refresh_seconds`` and dropped after trades."""

    def __init__(self, clock: Any = time.time) -> None:
        self._cache: dict[str, tuple[float, float]] = {}
        self._clock = clock

    def get(self, mint: str) -> float:
        ttl = float(config().rpc.get("balance_refresh_seconds", 900))
        cached = self._cache.get(mint)
        if cached is None or self._clock() - cached[1] >= ttl:
            # find_balance may return None on persistent rate limiting; treat
            # as zero so downstream arithmetic never sees None.
            cached = (_safe_balance(mint), self._clock())
            self._cache[mint] = cached
        return cached[0]

    def set(self, mint: str, value: float) -> None:
        self._cache[mint] = (value, self._clock())

    def invalidate(self, mint: str) -> None:
        self._cache.pop(mint, None)


def _safe_balance(mint: str) -> float:
    """Balance read that degrades to 0.0 on RPC errors.

    find_balance returns None on persistent rate limiting but RAISES on other
    RPC failures; an unhandled raise here would abort the entire analysis
    cycle (or startup) for a transient outage.
    """
    try:
        return find_balance(mint) or 0.0
    except Exception as e:  # noqa: BLE001 - balance outage must not kill the cycle
        log_general.warning(f"failed to fetch balance for {mint}: {e}")
        return 0.0


_balance_cache = BalanceCache()
_dry_run = False
_last_price_map: dict[str, float] = {}  # by mint, USD
_strategy_errors: dict[str, str] = {}  # symbol -> last load error logged
# Total capital (cash plus positions) of the main book, recomputed every cycle.
_current_total_capital: float = 0.0
# P&L baseline: value at run start; recaptured when the token set changes.
_baseline_value: float = 0.0
_baseline_key: tuple[Any, ...] = ()

# Hot-reloadable attributes that together define the traded token set.
_TOKEN_SET_ATTRS = ("secondary_mints", "secondary_mint_symbols", "secondary_weights")


def main_book_name() -> str:
    return ledger.DRY_RUN if _dry_run else ledger.LIVE


def _token_weight(token_symbol: str) -> float:
    """Portfolio weight for a token; equal split when weights are unset."""
    symbols = config().secondary_mint_symbols
    weights = config().secondary_weights
    if weights:
        if token_symbol in symbols:
            return float(weights[symbols.index(token_symbol)])
        return 0.0
    return 1.0 / max(len(symbols), 1)


def _weight_cap(symbol: str) -> float | None:
    """A token's slice of total capital when ``secondary_weights`` is set."""
    if not config().secondary_weights:
        return None
    return _token_weight(symbol) * _current_total_capital


def _token_set_snapshot() -> dict[str, Any]:
    """Snapshot the hot-reloadable token-set attributes."""
    cfg = config()
    return {attr: getattr(cfg, attr) for attr in _TOKEN_SET_ATTRS}


def _is_dust(size: float, price: float) -> bool:
    """A position worth less than ``risk.dust_threshold_usd`` (needs a known price)."""
    threshold = float(config().risk.get("dust_threshold_usd", 0) or 0)
    return price > 0 and size * price < threshold


def _has_open_position(token_symbol: str, mint: str | None = None) -> bool:
    """True when the main book holds a non-dust position in the token."""
    position = ledger.open_position(main_book_name(), token_symbol)
    if position is None:
        return False
    return not (mint and _is_dust(position.size, _last_price_map.get(mint, 0.0)))


def _enforce_token_change_guard(prev: dict[str, Any]) -> None:
    """Reject a token-set change that would orphan an open position.

    Removed tokens stop being managed (SL/TP/trailing stops stop evaluating),
    so a removal while a position is open is rolled back with an error.
    """
    cfg = config()
    removed = [
        symbol
        for mint, symbol in zip(prev["secondary_mints"], prev["secondary_mint_symbols"], strict=False)
        if symbol not in cfg.secondary_mint_symbols and _has_open_position(symbol, mint)
    ]
    if removed:
        log_general.error(
            f"token change rejected: open position in {removed}; "
            "close the position before removing the token"
        )
        for attr, value in prev.items():
            setattr(cfg, attr, value)


# ---------------------------------------------------------------- execution


class LiveExecutor:
    """Swaps through Jupiter from the wallet; records actual next to modeled costs."""

    def network_fee(self, symbol: str) -> float:
        return costs.model_for(symbol).network_fee_usd

    def buy(self, symbol: str, notional: float, price: float, ts: int, reason: str) -> Fill | None:
        cfg = config()
        mint = cfg.mint_for_symbol(symbol)
        if mint is None:
            return None
        log_transaction.info(f"buying {symbol} with {notional:.4f} {cfg.primary_mint_symbol} ({reason})")
        result = asyncio.run(
            perform_swap(notional, cfg.primary_mint, mint, cfg.primary_mint_symbol, symbol)
        )
        _balance_cache.invalidate(cfg.primary_mint)
        _balance_cache.invalidate(mint)
        if not result or float(result.get("out_amount") or 0) <= 0:
            return None
        size = float(result["out_amount"])
        market = _last_price_map.get(mint) or price
        modeled = costs.model_for(symbol).cost_usd(notional)
        return Fill(symbol, BUY, ts, notional / size, size, notional, notional - size * market, 0.0, reason, modeled)

    def sell(self, symbol: str, size: float, price: float, ts: int, reason: str) -> Fill | None:
        cfg = config()
        mint = cfg.mint_for_symbol(symbol)
        if mint is None:
            return None
        # Sell at most the tracked size: the wallet may hold more than the bot
        # bought, and it may hold less after an outside transfer.
        amount = min(size, _balance_cache.get(mint))
        if amount <= 0:
            log_general.warning(f"no sellable {symbol} balance; skipping the {reason} sell")
            return None
        log_transaction.info(f"selling {amount:g} {symbol} ({reason})")
        result = asyncio.run(perform_swap(amount, mint, cfg.primary_mint, symbol, cfg.primary_mint_symbol))
        _balance_cache.invalidate(cfg.primary_mint)
        _balance_cache.invalidate(mint)
        if not result:
            return None
        proceeds = float(result.get("out_amount") or 0)
        market = _last_price_map.get(mint) or price
        notional = amount * market
        modeled = costs.model_for(symbol).cost_usd(notional)
        return Fill(symbol, SELL, ts, market, amount, notional, notional - proceeds, 0.0, reason, modeled)


def _confluence_gate(symbol: str, side: str) -> float:
    """Size factor from the optional sentiment breaker and confluence filter."""
    from sol_trade.confluence import evaluate_buy_confluence, evaluate_sell_confluence

    cfg = config()
    if side == BUY and cfg.sentiment_enabled:
        if is_token_blocked(symbol):
            log_transaction.info(f"buy for {symbol} skipped: sentiment circuit breaker active")
            return 0.0
        if is_market_crash():
            log_transaction.info("buy skipped: market sentiment crash detected")
            return 0.0
    result = evaluate_buy_confluence(symbol) if side == BUY else evaluate_sell_confluence(symbol)
    factor = float(result["size_modifier"])  # type: ignore[arg-type]
    if result["action"] == "skip":
        log_transaction.info(f"{side} for {symbol} skipped: {result['reason']}")
    elif factor < 1.0:
        log_transaction.info(f"{side} for {symbol} sized to {factor * 100:.0f}%: {result['reason']}")
    return factor


def _main_book() -> Book:
    cfg = config()
    executor = SimExecutor(costs.model_for) if _dry_run else LiveExecutor()
    gated = cfg.confluence_enabled or cfg.sentiment_enabled
    book = Book(
        main_book_name(),
        0.0,
        executor,
        float(cfg.risk["max_trade_size"]),
        float(cfg.risk["max_live_capital"]),
        weight_cap=_weight_cap,
        gate=_confluence_gate if gated else None,
    )
    ledger.persist(book)
    if _dry_run:
        meta = ledger.load_meta(book.name)
        book.cash = float(meta.get("cash", cfg.risk["max_live_capital"]))
    else:
        book.cash = _balance_cache.get(cfg.primary_mint)
    return book


# ------------------------------------------------------------------ per bar


def _load(symbol: str) -> Strategy | None:
    name = resolve_strategy_name(symbol)
    try:
        strategy = load_strategy(name)
    except SpecError as e:
        message = f"{symbol}: strategy {name!r} unusable, token skipped: {e}"
        if _strategy_errors.get(symbol) != message:
            log_general.error(message)
            _strategy_errors[symbol] = message
        return None
    _strategy_errors.pop(symbol, None)
    return strategy


def run_symbol(
    book: Book, symbol: str, strategy: Strategy, now: float | None = None, allow_entry: bool = True
) -> dict[str, Any]:
    """Advance one symbol of a book to ``now``; returns what the dashboard shows.

    Closed bars since the last one processed are replayed in order (on first
    sight only the newest closed bar counts). A live book ignores entry
    signals of replayed bars: only the newest bar may open a position.
    """
    now = time.time() if now is None else now
    risk = strategy.risk
    period = strategy.period_seconds
    state = book.state(symbol)
    newest_closed = int(now // period) * period - period
    missed = 0
    if state.last_bar_ts is not None:
        missed = int(min(max((newest_closed - state.last_bar_ts) // period, 0), _MAX_REPLAY_BARS))
    frame, forming = history.window(symbol, strategy.timeframe, strategy.lookback + missed, now=now)
    entry, exit_ = backtest.compute_signals(strategy, frame, symbol)
    ts = frame["ts"].to_numpy()
    new = np.flatnonzero(ts > state.last_bar_ts) if state.last_bar_ts is not None else np.array([len(ts) - 1])
    is_live = book.name == ledger.LIVE
    for i in new:
        bar_ts = int(ts[i])
        row = frame.iloc[i]
        book.on_open(symbol, bar_ts, float(row.open), risk)
        book.on_low(symbol, bar_ts, float(row.open), float(row.low), risk)
        action = book.on_close(
            symbol, bar_ts, float(row.high), float(row.close), bool(entry[i]), bool(exit_[i]),
            risk, strategy.name, strategy.hash, allow_entry,
        )
        if is_live and action is not None and action.side == BUY and i != len(ts) - 1:
            log_general.info(f"{symbol}: stale entry signal from a replayed bar ignored")
            state.pending = None
    if forming is not None and len(ts) and forming.ts == int(ts[-1]) + period:
        book.on_open(symbol, forming.ts, forming.open, risk)
        book.on_low(symbol, forming.ts, forming.open, forming.low, risk)
    view = strategy.evaluate(
        {c: frame[c].to_numpy(dtype=float)[-strategy.lookback :, None] for c in ("open", "high", "low", "close", "volume")},
        frame["ts"].to_numpy(dtype=float)[-strategy.lookback :, None],
        symbol,
    )
    return {
        "close": float(frame["close"].iat[-1]) if len(frame) else 0.0,
        "entry": bool(entry[-1]) if len(entry) else False,
        "exit": bool(exit_[-1]) if len(exit_) else False,
        "values": {k: float(v[0]) for k, v in view.values.items()},
    }


# ----------------------------------------------------------------- the loop


def _sweep_live_dust(price_by_mint: dict[str, float]) -> None:
    """Write off dust positions of the live book in the database (no swap)."""
    for _, position in ledger.open_positions(ledger.LIVE):
        mint = config().mint_for_symbol(position.symbol)
        price = price_by_mint.get(mint, 0.0) if mint else 0.0
        if _is_dust(position.size, price):
            book = ledger.persist(Book(ledger.LIVE, 0.0, SimExecutor(costs.model_for), 0.0, 0.0))
            book.symbols[position.symbol] = ledger.load_state(ledger.LIVE, position.symbol)
            book.write_off(position.symbol, int(time.time()), DUST)
            ledger.save_state(ledger.LIVE, position.symbol, book.state(position.symbol))
            log_transaction.info(
                f"dust write-off: {position.symbol} position of {position.size:g} "
                f"(${position.size * price:.4f}) closed without a swap"
            )


def _pick(values: dict[str, float], *names: str) -> float:
    """First finite value among candidate indicator names, else 0.0.

    Strategies name their indicators differently (``ema_s``/``ema_m``,
    ``ema_fast``/``ema_mid``), so the dashboard tries them in order.
    """
    for name in names:
        value = values.get(name)
        if value is not None and np.isfinite(value):
            return float(value)
    return 0.0


def _book_value(book: Book, price_by_symbol: dict[str, float]) -> float:
    if _dry_run:
        return book.equity(price_by_symbol)
    cfg = config()
    value = _balance_cache.get(cfg.primary_mint) * (_last_price_map.get(cfg.primary_mint) or 1.0)
    for mint in cfg.secondary_mints:
        value += _balance_cache.get(mint) * _last_price_map.get(mint, 0.0)
    return value


def _capture_baseline() -> None:
    """Snapshot the account value at run start for P&L."""
    global _baseline_value
    cfg = config()
    prices = fetch_prices([cfg.primary_mint, *cfg.secondary_mints])
    for mint, price in prices.items():
        if price > 0:
            _last_price_map[mint] = price
    book = _main_book()
    ledger.load_book(book, cfg.secondary_mint_symbols)
    if _dry_run:
        book.cash = float(ledger.load_meta(book.name).get("cash", cfg.risk["max_live_capital"]))
    by_symbol = {s: _last_price_map.get(m, 0.0) for m, s in zip(cfg.secondary_mints, cfg.secondary_mint_symbols, strict=False)}
    _baseline_value = _book_value(book, by_symbol)
    if any(_last_price_map.get(m, 0.0) == 0 for m in cfg.secondary_mints):
        log_general.warning("P&L baseline captured with missing prices; profit figures may be unreliable.")


_last_holdings_refresh = 0.0
_full_holdings: list[Holding] = []
_full_holdings_total = 0.0


def _refresh_holdings(price_map: dict[str, float]) -> None:
    """Refresh the display-only wallet view every ``rpc.holdings_refresh_seconds``."""
    global _last_holdings_refresh, _full_holdings, _full_holdings_total
    cfg = config()
    interval = float(cfg.rpc.get("holdings_refresh_seconds", 0) or 0)
    if interval <= 0 or not cfg.private_key:
        _full_holdings, _full_holdings_total = [], 0.0
        return
    if time.time() - _last_holdings_refresh < interval:
        return
    _last_holdings_refresh = time.time()
    try:
        raw_holdings = get_all_token_holdings()
    except Exception:  # noqa: BLE001 - holdings are display-only; keep the previous view
        log_general.debug("failed to fetch full account holdings")
        return
    holdings: list[Holding] = []
    total = 0.0
    for mint, balance in raw_holdings.items():
        price = price_map.get(mint) or _last_price_map.get(mint)
        value = balance * price if price else None
        if value:
            total += value
        holdings.append(Holding(mint=mint, symbol=cfg.symbol_for_mint(mint), balance=balance, value=value))
    _full_holdings, _full_holdings_total = holdings, total


def _update_optional_layers() -> None:
    cfg = config()
    if cfg.whale_tracking_enabled:
        try:
            update_whale_data()
        except Exception as e:  # noqa: BLE001 - optional feature failure; log and continue
            log_general.warning(f"whale tracker update failed: {e}")
    if cfg.market_regime_enabled:
        try:
            update_regime()
        except Exception as e:  # noqa: BLE001 - optional feature failure; log and continue
            log_general.warning(f"market regime update failed: {e}")
    if cfg.sentiment_enabled:
        try:
            update_sentiment(cfg.secondary_mint_symbols)
        except Exception as e:  # noqa: BLE001 - optional feature failure; log and continue
            log_general.warning(f"sentiment update failed: {e}")


def perform_analysis(state: UIState) -> None:
    global _current_total_capital
    cfg = config()
    price_map = fetch_prices([cfg.primary_mint, *cfg.secondary_mints])
    for mint, price in price_map.items():
        if price > 0:
            _last_price_map[mint] = price
        else:
            # Fall back to the last known price so a price-API outage does not
            # produce a fake 0-value portfolio or profit swing.
            price_map[mint] = _last_price_map.get(mint, 0.0)
    _sweep_live_dust(_last_price_map)

    _update_optional_layers()

    book = _main_book()
    symbols = list(cfg.secondary_mint_symbols)
    ledger.load_book(book, symbols)
    if not _dry_run:
        book.cash = _balance_cache.get(cfg.primary_mint)
    price_by_symbol = {
        s: price_map.get(m, 0.0) for m, s in zip(cfg.secondary_mints, symbols, strict=False)
    }
    _current_total_capital = book.equity(price_by_symbol)

    tokens: list[TokenStatus] = []
    for mint, symbol in zip(cfg.secondary_mints, symbols, strict=False):
        position = book.position(symbol)
        if position is not None and _is_dust(position.size, price_map.get(mint, 0.0)):
            log_transaction.info(f"dust write-off: {symbol} position of {position.size:g} closed without a swap")
            book.write_off(symbol, int(time.time()), DUST)
        strategy = _load(symbol)
        if strategy is None:
            continue
        try:
            view = run_symbol(book, symbol, strategy)
        except Exception as e:  # noqa: BLE001 - one token must not abort the cycle
            log_general.warning(f"analysis failed for {symbol}: {e}")
            continue
        finally:
            ledger.save_state(book.name, symbol, book.state(symbol))
        position = book.position(symbol)
        values = view["values"]
        tokens.append(
            TokenStatus(
                symbol=symbol,
                price=view["close"],
                rsi=_pick(values, "rsi"),
                ema_short=_pick(values, "ema_s", "ema_fast"),
                ema_medium=_pick(values, "ema_m", "ema_mid", "ema_slow"),
                entry_signal=view["entry"],
                exit_signal=view["exit"],
                position=position is not None,
                stoploss=position.stop if position else None,
                takeprofit=position.take_profit if position else None,
                entry_price=position.entry_price if position else None,
            )
        )
    if _dry_run:
        meta = ledger.load_meta(book.name)
        meta["cash"] = book.cash
        ledger.save_meta(book.name, meta)

    current_value = _book_value(book, price_by_symbol)
    total_profit = current_value - _baseline_value
    _refresh_holdings(price_map)
    full_holdings = list(_full_holdings)
    total_account_value = _full_holdings_total or float(current_value or 0.0)
    reserved_fees = 0.0 if _dry_run else minimum_sol_needed()

    state.update(
        lambda s: (
            setattr(s, "primary_balance", float(book.cash or 0.0)),
            setattr(s, "portfolio_value", float(current_value or 0.0)),
            setattr(s, "total_profit", float(total_profit or 0.0)),
            setattr(s, "reserved_fees", reserved_fees),
            setattr(s, "tokens", tokens),
            setattr(s, "full_holdings", full_holdings),
            setattr(s, "total_account_value", total_account_value),
            setattr(s, "last_refresh", datetime.now(UTC).strftime("%H:%M:%S")),
        )
    )


def start_trading(state: UIState, dry_run: bool = False) -> None:
    """Run the trading loop in a background thread, feeding the UI state.

    With ``dry_run=True`` swaps are simulated in a virtual ledger and the
    wallet is never touched.
    """
    global _stop_event, _trading_thread, _dry_run, _baseline_key

    _dry_run = dry_run
    state.update(lambda s: setattr(s, "dry_run", dry_run))
    _stop_event = threading.Event()
    if dry_run:
        log_general.warning(
            "PAPER TRADING: swaps are simulated, the wallet is not touched."
        )
    try:
        ledger.migrate_legacy_csv()
    except Exception as e:  # noqa: BLE001 - migration problems must not stop the bot
        log_general.error(f"legacy position import failed: {e}")
    log_general.info("SolTrade has now initialized the trading algorithm.")
    _capture_baseline()
    _baseline_key = (config().primary_mint, tuple(config().secondary_mints))

    def _run() -> None:
        global _baseline_key
        state.update(lambda s: setattr(s, "running", True))
        try:
            while not _stop_event.is_set():
                prev_tokens = _token_set_snapshot()
                config().maybe_reload_config()
                if _token_set_snapshot() != prev_tokens:
                    # Reject removals that would orphan an open position.
                    _enforce_token_change_guard(prev_tokens)
                token_key = (config().primary_mint, tuple(config().secondary_mints))
                if token_key != _baseline_key:
                    log_general.info(
                        "trading tokens changed — recapturing P&L baseline"
                    )
                    _capture_baseline()
                    _baseline_key = token_key
                try:
                    perform_analysis(state)
                except Exception as e:  # noqa: BLE001 - keep the loop alive across errors
                    log_general.error(f"analysis cycle failed: {e}")
                    state.update(lambda s: setattr(s, "error_count", s.error_count + 1))
                for remaining in range(config().price_update_seconds, 0, -1):
                    if _stop_event.is_set():
                        return
                    state.update(lambda s, r=remaining: setattr(s, "countdown", r))
                    time.sleep(1)
        finally:
            state.update(lambda s: setattr(s, "running", False))

    _trading_thread = threading.Thread(
        target=_run, name="soltrade-trading", daemon=True
    )
    _trading_thread.start()


def stop_trading() -> None:
    """Signal the trading thread to stop and wait for it to finish."""
    if _stop_event is not None:
        _stop_event.set()
    if _trading_thread is not None:
        _trading_thread.join(timeout=10)
        if _trading_thread.is_alive():
            log_general.warning(
                "Trading thread did not stop within 10s; it will be terminated on exit."
            )
    log_general.info("SolTrade has been stopped.")


_stop_event: threading.Event | None = None
_trading_thread: threading.Thread | None = None
