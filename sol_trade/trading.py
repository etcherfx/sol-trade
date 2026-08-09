import asyncio
import os
import threading
import time
from datetime import UTC, datetime
from typing import Any, cast

import pandas as pd
import requests

from sol_trade import data_source
from sol_trade.config import config
from sol_trade.confluence import _is_protective_exit
from sol_trade.log import log_general, log_transaction
from sol_trade.strategy import (
    calc_stoploss,
    calc_takeprofit,
    calc_trailing_stoploss,
    set_position,
    strategy,
)
from sol_trade.transactions import perform_swap
from sol_trade.ui import Holding, TokenStatus, UIState
from sol_trade.wallet import find_balance, get_all_token_holdings, minimum_sol_needed

_http_session = requests.Session()


class BalanceCache:
    """Lazy balance fetcher; in dry-run mode it keeps a simulated paper ledger."""

    def __init__(self) -> None:
        self._cache: dict[str, float] = {}
        self._paper_mode = False
        self._paper: dict[str, float] = {}

    def set_paper_mode(self, enabled: bool) -> None:
        self._paper_mode = enabled
        self._paper = {}

    @property
    def paper_mode(self) -> bool:
        return self._paper_mode

    def get(self, mint: str) -> float:
        if self._paper_mode:
            if mint not in self._paper:
                # Seed the paper ledger from the real wallet once.
                self._paper[mint] = find_balance(mint) or 0.0
            return self._paper[mint]
        if mint not in self._cache:
            # find_balance may return None on persistent rate limiting; treat
            # as zero so downstream arithmetic never sees None.
            self._cache[mint] = find_balance(mint) or 0.0
        return self._cache[mint]

    def set(self, mint: str, value: float) -> None:
        """Record a simulated balance change (dry-run fills)."""
        if self._paper_mode:
            self._paper[mint] = value
        else:
            self._cache[mint] = value

    def invalidate(self, mint: str) -> None:
        if not self._paper_mode:
            self._cache.pop(mint, None)


_balance_cache = BalanceCache()

_dry_run = False

# Total capital (primary balance + all tracked positions), recomputed every
# analysis cycle. Used for weighted buy budgeting.
_current_total_capital: float = 0.0
# Token set the P&L baseline was captured against; recaptures on change.
_baseline_key: tuple[Any, ...] = ()

# Hot-reloadable attributes that together define the traded token set.
_TOKEN_SET_ATTRS = ("secondary_mints", "secondary_mint_symbols", "secondary_weights")


def _token_weight(token_symbol: str) -> float:
    """Portfolio weight for a token; equal split when weights are unset."""
    symbols = config().secondary_mint_symbols
    weights = config().secondary_weights
    if weights:
        if token_symbol in symbols:
            return float(weights[symbols.index(token_symbol)])
        return 0.0
    return 1.0 / max(len(symbols), 1)


def _token_set_snapshot() -> dict[str, Any]:
    """Snapshot the hot-reloadable token-set attributes."""
    cfg = config()
    return {attr: getattr(cfg, attr) for attr in _TOKEN_SET_ATTRS}


def _has_open_position(token_symbol: str) -> bool:
    """True when the persisted position CSV for a token has an open position."""
    try:
        df = read_dataframe_from_csv(f"data/{token_symbol}_data.csv")
        return len(df) > 0 and bool(df["position"].iat[-1])
    except (FileNotFoundError, KeyError, IndexError):
        return False


def _enforce_token_change_guard(prev: dict[str, Any]) -> None:
    """Reject a token-set change that would orphan an open position.

    Removed tokens stop being managed (SL/TP/trailing stops stop evaluating),
    so a removal while a position is open is rolled back with an error.
    """
    cfg = config()
    removed = [
        symbol
        for symbol in prev["secondary_mint_symbols"]
        if symbol not in cfg.secondary_mint_symbols and _has_open_position(symbol)
    ]
    if removed:
        log_general.error(
            f"token change rejected: open position in {removed}; "
            "close the position before removing the token"
        )
        for attr, value in prev.items():
            setattr(cfg, attr, value)


def _paper_buy(
    input_amount: float,
    df: pd.DataFrame,
    primary_mint: str,
    secondary_mint: str,
    primary_mint_symbol: str,
    secondary_mint_symbol: str,
) -> float | None:
    """Simulate a buy at the latest close; returns the amount bought or None."""
    price = float(df["close"].iat[-1])
    if price <= 0:
        log_general.warning("paper buy skipped: no usable price")
        return None
    bought = input_amount / price
    _balance_cache.set(secondary_mint, _balance_cache.get(secondary_mint) + bought)
    _balance_cache.set(primary_mint, _balance_cache.get(primary_mint) - input_amount)
    log_transaction.info(
        f"PAPER BUY {bought:.4f} {secondary_mint_symbol} for "
        f"{input_amount:.4f} {primary_mint_symbol} at {price:.6f}."
    )
    return bought


def _paper_sell(
    input_amount: float,
    df: pd.DataFrame,
    primary_mint: str,
    secondary_mint: str,
    primary_mint_symbol: str,
    secondary_mint_symbol: str,
) -> float | None:
    """Simulate a sell at the latest close; returns the proceeds or None."""
    price = float(df["close"].iat[-1])
    if price <= 0:
        log_general.warning("paper sell skipped: no usable price")
        return None
    proceeds = input_amount * price
    _balance_cache.set(primary_mint, _balance_cache.get(primary_mint) + proceeds)
    _balance_cache.set(secondary_mint, _balance_cache.get(secondary_mint) - input_amount)
    log_transaction.info(
        f"PAPER SELL {input_amount:.4f} {secondary_mint_symbol} for "
        f"{proceeds:.4f} {primary_mint_symbol} at {price:.6f}."
    )
    return proceeds


def fetch_prices(mints: list[str]) -> dict[str, float]:
    """Fetch multiple token prices with a single HTTP call."""
    if not mints:
        return {}

    unique_mints = list(dict.fromkeys(mints))  # preserve order
    params = {"ids": ",".join(unique_mints)}
    url = "https://lite-api.jup.ag/price/v3"

    try:
        response = _http_session.get(url, params=params, timeout=10)
        response.raise_for_status()
        response_json = cast(dict[str, Any], response.json())
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 401:
            log_general.error(
                "401 Unauthorized fetching prices from the lite API; "
                "check the Jupiter API key and plan"
            )
        else:
            log_general.error(f"HTTP error fetching prices for {unique_mints}: {e}")
        return {mint: 0.0 for mint in unique_mints}
    except Exception as e:  # noqa: BLE001 - network errors
        log_general.error(f"failed to fetch prices for {unique_mints}: {e}")
        return {mint: 0.0 for mint in unique_mints}

    prices: dict[str, float] = {}
    for mint in unique_mints:
        mint_data = cast(dict[str, Any], response_json.get(mint, {}) or {})
        price = float(mint_data.get("usdPrice") or 0)
        if price == 0:
            log_general.debug(f"price for {mint} missing from response; defaulting to 0")
        prices[mint] = price
    return prices


# P&L baseline — captured once per run by start_trading, not at import time.
initial_primary_balance: float = 0.0
initial_secondary_balances: list[float] = []
initial_primary_price: float = 0.0
initial_secondary_prices: list[float] = []
_last_price_map: dict[str, float] = {}


def _capture_baseline() -> None:
    """Snapshot wallet balances and prices at run start for P&L math."""
    global initial_primary_balance, initial_secondary_balances
    global initial_primary_price, initial_secondary_prices

    cfg = config()
    initial_primary_balance = find_balance(cfg.primary_mint) or 0.0
    initial_secondary_balances = [
        find_balance(mint) or 0.0 for mint in cfg.secondary_mints
    ]

    prices = fetch_prices([cfg.primary_mint, *cfg.secondary_mints])
    for mint, price in prices.items():
        if price > 0:
            _last_price_map[mint] = price
    initial_primary_price = _last_price_map.get(cfg.primary_mint, 0.0)
    initial_secondary_prices = [
        _last_price_map.get(mint, 0.0) for mint in cfg.secondary_mints
    ]

    if not initial_primary_price or any(p == 0 for p in initial_secondary_prices):
        log_general.warning(
            "P&L baseline captured with missing prices; profit figures may be unreliable."
        )







def _as_float_or_none(value: Any) -> float | None:
    try:
        return float(value) if value is not None and not pd.isna(value) else None
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float:
    return _as_float_or_none(value) or 0.0


# Canonical analysis-frame columns. Strategies may use volume for
# confirmation (e.g. volume-SMA filters), so it must reach the dataframe.
_CANDLE_COLUMNS = ["close", "high", "low", "open", "volume", "totalvolume", "time"]


def _candles_to_frame(candles: list[dict]) -> pd.DataFrame:
    """Build the analysis dataframe from candle dicts (OHLCV + time)."""
    df = pd.DataFrame(candles)
    for col in _CANDLE_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[_CANDLE_COLUMNS]


def perform_analysis(state: UIState) -> None:
    cfg = config()
    data_frames: list[pd.DataFrame] = []
    price_map = fetch_prices([cfg.primary_mint, *cfg.secondary_mints])
    for mint, price in price_map.items():
        if price > 0:
            _last_price_map[mint] = price
        else:
            # Fall back to the last known price so a price-API outage does not
            # produce a fake 0-value portfolio or profit swing.
            price_map[mint] = _last_price_map.get(mint, 0.0)

    for secondary_mint, secondary_mint_symbol in zip(
        cfg.secondary_mints, cfg.secondary_mint_symbols
    ):
        try:
            candles = data_source.fetch_candles(
                secondary_mint_symbol,
                cfg.primary_mint_symbol,
                "1m",
                50,
                exchange_id=cfg.token_exchanges.get(secondary_mint_symbol),
            )
            new_df = _candles_to_frame(candles)
            if new_df.empty:
                log_general.warning(
                    f"no candle data for {secondary_mint_symbol}; skipping this cycle"
                )
                data_frames.append(None)
                continue
            new_df["time"] = pd.to_datetime(new_df["time"], unit="s")
            new_df["symbol"] = secondary_mint_symbol
            new_df["position"] = False
            data_file_path = f"data/{secondary_mint_symbol}_data.csv"

            try:
                existing_df = read_dataframe_from_csv(data_file_path)
                if len(existing_df) > 0 and existing_df["position"].iat[-1]:
                    columns_to_merge = [
                        "position",
                        "entry_price",
                        "position_size",
                        "takeprofit",
                        "stoploss",
                        "trailing_stoploss",
                        "trailing_stoploss_target",
                        "highest_price",
                    ]

                    for col in columns_to_merge:
                        if col in existing_df.columns:
                            new_df[col] = existing_df.iloc[-1][col]
            except FileNotFoundError:
                pass

            # Evaluate entry/exit AFTER risk columns are attached so protective
            # exits (stoploss / takeprofit / trailing_stoploss) can fire.
            df = strategy(new_df, secondary_mint_symbol)
        except Exception as e:  # noqa: BLE001 - one token must not abort the cycle
            log_general.warning(f"analysis failed for {secondary_mint_symbol}: {e}")
            data_frames.append(None)
            continue

        data_frames.append(df)

    # Update whale tracking data
    if config().whale_tracking_enabled:
        try:
            from sol_trade.whale_tracker import update_whale_data

            update_whale_data()
        except Exception as e:  # noqa: BLE001 - optional feature failure; log and continue
            log_general.warning(f"whale tracker update failed: {e}")

    # Update market regime (only if stale)
    if config().market_regime_enabled:
        try:
            from sol_trade.market_regime import update_regime

            update_regime()
        except Exception as e:  # noqa: BLE001 - optional feature failure; log and continue
            log_general.warning(f"market regime update failed: {e}")

    # Update sentiment data (only if stale)
    if config().sentiment_enabled:
        try:
            from sol_trade.sentiment import update_sentiment

            update_sentiment(cfg.secondary_mint_symbols)
        except Exception as e:  # noqa: BLE001 - optional feature failure; log and continue
            log_general.warning(f"sentiment update failed: {e}")

    current_primary_balance = _balance_cache.get(cfg.primary_mint)
    current_secondary_balances = [
        _balance_cache.get(mint) for mint in cfg.secondary_mints
    ]
    initial_total_value = (initial_primary_balance * initial_primary_price) + sum(
        initial_secondary_balance * initial_secondary_price
        for initial_secondary_balance, initial_secondary_price in zip(
            initial_secondary_balances, initial_secondary_prices
        )
    )
    current_total_value = (current_primary_balance * price_map.get(cfg.primary_mint, 0.0)) + sum(
        current_secondary_balance * price_map.get(secondary_mint, 0.0)
        for current_secondary_balance, secondary_mint in zip(
            current_secondary_balances, cfg.secondary_mints
        )
    )
    total_profit = current_total_value - initial_total_value

    # Total capital for weighted buy budgeting: cash plus every tracked position.
    global _current_total_capital
    positions_value = 0.0
    for df, mint, symbol in zip(data_frames, cfg.secondary_mints, cfg.secondary_mint_symbols):
        if df is None:
            continue
        last = df.iloc[-1]
        size = _as_float(last.get("position_size"))
        if size > 0:
            price = price_map.get(mint, 0.0) or _as_float(last.get("close"))
            positions_value += size * price
    _current_total_capital = current_primary_balance + positions_value

    for df, secondary_mint, secondary_mint_symbol in zip(
        data_frames, cfg.secondary_mints, cfg.secondary_mint_symbols
    ):
        if df is None:
            continue
        data_file_path = f"data/{secondary_mint_symbol}_data.csv"
        if not df["position"].iat[-1]:
            handle_buy_signal(df, secondary_mint, data_file_path, secondary_mint_symbol)
        else:
            handle_sell_signal(df, secondary_mint, data_file_path, secondary_mint_symbol)

    # Push the analysis results to the UI
    tokens = []
    for df, symbol in zip(data_frames, cfg.secondary_mint_symbols):
        if df is None:
            continue
        last = df.iloc[-1]
        tokens.append(
            TokenStatus(
                symbol=symbol,
                price=_as_float(last.get("close")),
                rsi=_as_float(last.get("rsi")),
                ema_short=_as_float(last.get("ema_s")),
                ema_medium=_as_float(last.get("ema_m")),
                entry_signal=last.get("entry") == 1,
                exit_signal=last.get("exit") == 1,
                position=bool(last.get("position")),
                stoploss=_as_float_or_none(last.get("stoploss")),
                takeprofit=_as_float_or_none(last.get("takeprofit")),
                entry_price=_as_float_or_none(last.get("entry_price")),
            )
        )
    # Full-account view: every token the wallet holds, valued where a price exists.
    full_holdings: list[Holding] = []
    total_account_value = float(current_total_value or 0.0)
    try:
        raw_holdings = get_all_token_holdings()
        mint_to_symbol = {
            cfg.primary_mint: cfg.primary_mint_symbol,
            config().sol_mint: "SOL",
            **dict(zip(cfg.secondary_mints, cfg.secondary_mint_symbols)),
        }
        known_prices = {mint: price for mint, price in price_map.items() if price > 0}
        total = 0.0
        for mint, balance in raw_holdings.items():
            price = known_prices.get(mint)
            value = balance * price if price else None
            if value:
                total += value
            full_holdings.append(
                Holding(mint=mint, symbol=mint_to_symbol.get(mint), balance=balance, value=value)
            )
        if total > 0:
            total_account_value = total
    except Exception:  # noqa: BLE001 - holdings are display-only; keep the previous view
        log_general.debug("failed to fetch full account holdings")

    state.update(
        lambda s: (
            setattr(s, "primary_balance", float(current_primary_balance or 0.0)),
            setattr(s, "portfolio_value", float(current_total_value or 0.0)),
            setattr(s, "total_profit", float(total_profit or 0.0)),
            setattr(s, "reserved_fees", minimum_sol_needed()),
            setattr(s, "tokens", tokens),
            setattr(s, "full_holdings", full_holdings),
            setattr(s, "total_account_value", total_account_value),
            setattr(s, "last_refresh", datetime.now(UTC).strftime("%H:%M:%S")),
        )
    )


def handle_buy_signal(df: pd.DataFrame, secondary_mint: str, data_file_path: str, secondary_mint_symbol: str) -> bool:
    """Execute a buy when the last bar has an entry signal; returns success."""
    if df["entry"].iat[-1] == 1:
        mint_symbol = cast(str, df["symbol"].iat[0])
        cfg = config()

        # Check sentiment circuit breaker
        if config().sentiment_enabled:
            from sol_trade.sentiment import is_market_crash, is_token_blocked

            if is_token_blocked(secondary_mint_symbol):
                log_transaction.info(
                    f"trading paused for {secondary_mint_symbol}: sentiment circuit breaker active"
                )
                return False
            if is_market_crash():
                log_transaction.info(
                    "all new entries paused: market sentiment crash detected"
                )
                return False

        # Evaluate confluence
        from sol_trade.confluence import evaluate_buy_confluence

        result = evaluate_buy_confluence("BUY", secondary_mint_symbol)
        if result["action"] == "skip":
            log_transaction.info(
                f"buy signal for {secondary_mint_symbol} skipped: {result['reason']}"
            )
            return False

        # Apply position size modifier
        input_amount = _balance_cache.get(cfg.primary_mint)
        if input_amount <= 0:
            log_transaction.info(
                f"SolTrade has detected a buy signal, but does not have enough {cfg.primary_mint_symbol} to trade."
            )
            return False

        # Weighted portfolio budget: buy only up to this token's slice of total
        # capital (cash + all tracked positions). Equal split when weights unset.
        # The buy path is only reached with no open position (perform_analysis
        # dispatches position-holding tokens to the sell handler), so this
        # token contributes no position value to subtract.
        capital = _current_total_capital or input_amount
        budget = _token_weight(secondary_mint_symbol) * capital
        if budget <= 0:
            log_transaction.info(
                f"buy signal for {secondary_mint_symbol} skipped: "
                "portfolio weight is zero or the target slice is filled"
            )
            return False
        input_amount = min(input_amount, budget)

        size_modifier = result["size_modifier"]
        if size_modifier < 1.0:
            input_amount = input_amount * size_modifier
            log_transaction.info(
                f"position size reduced to {size_modifier*100:.0f}% for {secondary_mint_symbol}: {result['reason']}"
            )

        log_transaction.info(
            f"SolTrade has detected a buy signal for {mint_symbol} using {input_amount} {cfg.primary_mint_symbol}."
        )
        is_swapped = (
            _paper_buy(
                input_amount,
                df,
                cfg.primary_mint,
                secondary_mint,
                cfg.primary_mint_symbol,
                secondary_mint_symbol,
            )
            if _dry_run
            else asyncio.run(
                perform_swap(
                    input_amount,
                    cfg.primary_mint,
                    secondary_mint,
                    cfg.primary_mint_symbol,
                    secondary_mint_symbol,
                )
            )
        )
        if is_swapped:
            bought = (
                float(is_swapped["out_amount"])
                if isinstance(is_swapped, dict)
                else float(is_swapped)
            )
            fill_price = (
                input_amount / bought
                if bought > 0
                else float(df["close"].iat[-1])
            )
            df["entry_price"] = fill_price
            df["position_size"] = bought
            df = calc_stoploss(df)
            df = calc_takeprofit(df)
            df = calc_trailing_stoploss(df)
            df = set_position(df, True)
            save_dataframe_to_csv(df, data_file_path)
            _balance_cache.invalidate(cfg.primary_mint)
            _balance_cache.invalidate(secondary_mint)
            return True
        return False
    return False


def _close_position_bookkeeping(df: pd.DataFrame, sold_amount: float) -> pd.DataFrame:
    """Update position columns after a sell; partial closes keep the position.

    A sell of less than ``position_size`` keeps the position open at the
    reduced size with the same (weighted-average) entry price and risk levels.
    """
    remaining = None
    if "position_size" in df.columns:
        remaining = _as_float(df["position_size"].iat[-1]) - sold_amount
    if remaining is None or remaining <= 1e-9:
        # Full close: clear the position and all risk columns.
        df = set_position(df, False)
        drop_cols = [
            c
            for c in (
                "stoploss",
                "entry_price",
                "trailing_stoploss",
                "trailing_stoploss_target",
                "takeprofit",
                "position_size",
                "highest_price",
            )
            if c in df.columns
        ]
        df = df.drop(columns=drop_cols)
    else:
        df["position"] = True
        df["position_size"] = remaining
    return df


def handle_sell_signal(df: pd.DataFrame, secondary_mint: str, data_file_path: str, secondary_mint_symbol: str) -> bool:
    """Execute a sell when the last bar has an exit signal; returns success."""
    input_amount = _balance_cache.get(secondary_mint)
    df = calc_trailing_stoploss(df)

    # The strategy layer evaluated exits against the trailing stop carried from
    # the previous cycle, but the ratchet may have lifted the stop above price
    # since. Re-evaluate so a fresh breach exits immediately instead of lagging
    # a cycle. Pre-tracking bars are None -> NaN and never match.
    trailing_stop = pd.to_numeric(df["trailing_stoploss"], errors="coerce")
    df.loc[df["close"] <= trailing_stop, "exit"] = 1

    if df["exit"].iat[-1] == 1:
        mint_symbol = cast(str, df["symbol"].iat[0])

        # Sell at most the tracked position: the wallet may hold more than the
        # bot bought (pre-existing balance), and selling beyond the position
        # would make bookkeeping and the real balance diverge.
        if "position_size" in df.columns:
            position_size = _as_float(df["position_size"].iat[-1])
            if position_size > 0:
                input_amount = min(input_amount, position_size)

        # Nothing to sell — mirrors the buy-side balance guard.
        if input_amount <= 0:
            log_general.warning(
                f"no sellable {secondary_mint_symbol} balance; skipping sell signal"
            )
            return False

        # Protective exits (stop-loss / take-profit / trailing stop) always execute at 100%
        if not _is_protective_exit(df):
            # Evaluate confluence for sells
            from sol_trade.confluence import evaluate_sell_confluence

            result = evaluate_sell_confluence("SELL", secondary_mint_symbol)
            if result["action"] == "skip":
                log_transaction.info(
                    f"sell signal for {secondary_mint_symbol} skipped: {result['reason']}"
                )
                return False

            # Apply position size modifier
            size_modifier = result["size_modifier"]
            if size_modifier < 1.0:
                input_amount = input_amount * size_modifier
                log_transaction.info(
                    f"sell position size reduced to {size_modifier*100:.0f}% for {secondary_mint_symbol}: {result['reason']}"
                )

        log_transaction.info(
            f"SolTrade has detected a sell signal for {input_amount} {mint_symbol}."
        )
        is_swapped = (
            _paper_sell(
                input_amount,
                df,
                config().primary_mint,
                secondary_mint,
                config().primary_mint_symbol,
                secondary_mint_symbol,
            )
            if _dry_run
            else asyncio.run(
                perform_swap(
                    input_amount,
                    secondary_mint,
                    config().primary_mint,
                    secondary_mint_symbol,
                    config().primary_mint_symbol,
                )
            )
        )
        if is_swapped:
            df = _close_position_bookkeeping(df, input_amount)
            save_dataframe_to_csv(df, data_file_path)
            _balance_cache.invalidate(secondary_mint)
            _balance_cache.invalidate(config().primary_mint)
            return True
        return False
    return False


def start_trading(state: UIState, dry_run: bool = False) -> None:
    """Run the trading loop in a background thread, feeding the UI state.

    With ``dry_run=True`` swaps are simulated against the latest close and the
    real wallet is never touched.
    """
    global _stop_event, _trading_thread, _dry_run, _baseline_key

    _dry_run = dry_run
    _balance_cache.set_paper_mode(dry_run)
    state.update(lambda s: setattr(s, "dry_run", dry_run))
    _stop_event = threading.Event()
    if dry_run:
        log_general.warning(
            "PAPER TRADING: swaps are simulated, the wallet is not touched."
        )
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


def save_dataframe_to_csv(df: pd.DataFrame, file_path: str) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        log_general.info(f"data saved to {file_path}")
    except Exception as e:  # noqa: BLE001 - data save failure; log and continue
        log_general.error(f"failed to save data to {file_path}: {e}")


def read_dataframe_from_csv(file_path: str) -> pd.DataFrame:
    """Load the position CSV (raises FileNotFoundError when absent)."""
    return pd.read_csv(file_path)  # type: ignore[reportGeneralTypeIssues]
