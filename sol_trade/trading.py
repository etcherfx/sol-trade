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
    calc_entry_price,
    calc_stoploss,
    calc_takeprofit,
    calc_trailing_stoploss,
    set_position,
    strategy,
)
from sol_trade.transactions import perform_swap
from sol_trade.ui import TokenStatus, UIState
from sol_trade.wallet import find_balance, minimum_sol_needed

config_instance = config()
primary_mint: str = config_instance.primary_mint
primary_mint_symbol: str = config_instance.primary_mint_symbol
secondary_mints: list[str] = config_instance.secondary_mints
secondary_mint_symbols: list[str] = config_instance.secondary_mint_symbols
price_update_seconds: int = config_instance.price_update_seconds
whale_tracking_enabled: bool = config_instance.whale_tracking_enabled
confluence_enabled: bool = config_instance.confluence_enabled
market_regime_enabled: bool = config_instance.market_regime_enabled
sentiment_enabled: bool = config_instance.sentiment_enabled

if not primary_mint or not primary_mint_symbol:
    raise ValueError("Primary mint configuration is missing.")
if not secondary_mints or not secondary_mint_symbols:
    raise ValueError("At least one secondary mint must be configured.")

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


def _paper_buy(
    input_amount: float,
    df: pd.DataFrame,
    primary_mint: str,
    secondary_mint: str,
    primary_mint_symbol: str,
    secondary_mint_symbol: str,
) -> bool:
    """Simulate a buy at the latest close, updating the paper ledger."""
    price = float(df["close"].iat[-1])
    if price <= 0:
        log_general.warning("paper buy skipped: no usable price")
        return False
    bought = input_amount / price
    _balance_cache.set(secondary_mint, _balance_cache.get(secondary_mint) + bought)
    _balance_cache.set(primary_mint, _balance_cache.get(primary_mint) - input_amount)
    log_transaction.info(
        f"PAPER BUY {bought:.4f} {secondary_mint_symbol} for "
        f"{input_amount:.4f} {primary_mint_symbol} at {price:.6f}."
    )
    return True


def _paper_sell(
    input_amount: float,
    df: pd.DataFrame,
    primary_mint: str,
    secondary_mint: str,
    primary_mint_symbol: str,
    secondary_mint_symbol: str,
) -> bool:
    """Simulate a sell at the latest close, updating the paper ledger."""
    price = float(df["close"].iat[-1])
    if price <= 0:
        log_general.warning("paper sell skipped: no usable price")
        return False
    proceeds = input_amount * price
    _balance_cache.set(primary_mint, _balance_cache.get(primary_mint) + proceeds)
    _balance_cache.set(secondary_mint, 0.0)
    log_transaction.info(
        f"PAPER SELL {input_amount:.4f} {secondary_mint_symbol} for "
        f"{proceeds:.4f} {primary_mint_symbol} at {price:.6f}."
    )
    return True


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
                "401 Unauthorized: Endpoint requires Pro plan, falling back to lite-api"
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

    initial_primary_balance = find_balance(primary_mint) or 0.0
    initial_secondary_balances = [find_balance(mint) or 0.0 for mint in secondary_mints]

    prices = fetch_prices([primary_mint, *secondary_mints])
    for mint, price in prices.items():
        if price > 0:
            _last_price_map[mint] = price
    initial_primary_price = _last_price_map.get(primary_mint, 0.0)
    initial_secondary_prices = [_last_price_map.get(mint, 0.0) for mint in secondary_mints]

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


def perform_analysis(state: UIState) -> None:
    data_frames: list[pd.DataFrame] = []
    price_map = fetch_prices([primary_mint, *secondary_mints])
    for mint, price in price_map.items():
        if price > 0:
            _last_price_map[mint] = price
        else:
            # Fall back to the last known price so a price-API outage does not
            # produce a fake 0-value portfolio or profit swing.
            price_map[mint] = _last_price_map.get(mint, 0.0)

    for secondary_mint, secondary_mint_symbol in zip(
        secondary_mints, secondary_mint_symbols
    ):
        try:
            candles = data_source.fetch_candles(
                secondary_mint_symbol, primary_mint_symbol, "1m", 50
            )
            new_df = pd.DataFrame(
                candles, columns=["close", "high", "low", "open", "time"]
            )
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
                        "takeprofit",
                        "stoploss",
                        "trailing_stoploss",
                        "trailing_stoploss_target",
                    ]

                    for col in columns_to_merge:
                        if col in existing_df.columns:
                            new_df[col] = existing_df.iloc[-1][col]
            except FileNotFoundError:
                pass

            # Evaluate entry/exit AFTER risk columns are attached so protective
            # exits (stoploss / takeprofit / trailing_stoploss) can fire.
            df = strategy(new_df)
        except Exception as e:  # noqa: BLE001 - one token must not abort the cycle
            log_general.warning(f"analysis failed for {secondary_mint_symbol}: {e}")
            data_frames.append(None)
            continue

        data_frames.append(df)

    # Update whale tracking data
    if whale_tracking_enabled:
        try:
            from sol_trade.whale_tracker import update_whale_data

            update_whale_data()
        except Exception as e:  # noqa: BLE001 - optional feature failure; log and continue
            log_general.warning(f"whale tracker update failed: {e}")

    # Update market regime (only if stale)
    if market_regime_enabled:
        try:
            from sol_trade.market_regime import update_regime

            update_regime()
        except Exception as e:  # noqa: BLE001 - optional feature failure; log and continue
            log_general.warning(f"market regime update failed: {e}")

    # Update sentiment data (only if stale)
    if sentiment_enabled:
        try:
            from sol_trade.sentiment import update_sentiment

            update_sentiment(secondary_mint_symbols)
        except Exception as e:  # noqa: BLE001 - optional feature failure; log and continue
            log_general.warning(f"sentiment update failed: {e}")

    current_primary_balance = _balance_cache.get(primary_mint)
    current_secondary_balances = [_balance_cache.get(mint) for mint in secondary_mints]
    initial_total_value = (initial_primary_balance * initial_primary_price) + sum(
        initial_secondary_balance * initial_secondary_price
        for initial_secondary_balance, initial_secondary_price in zip(
            initial_secondary_balances, initial_secondary_prices
        )
    )
    current_total_value = (current_primary_balance * price_map.get(primary_mint, 0.0)) + sum(
        current_secondary_balance * price_map.get(secondary_mint, 0.0)
        for current_secondary_balance, secondary_mint in zip(
            current_secondary_balances, secondary_mints
        )
    )
    total_profit = current_total_value - initial_total_value

    for df, secondary_mint, secondary_mint_symbol in zip(
        data_frames, secondary_mints, secondary_mint_symbols
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
    for df, symbol in zip(data_frames, secondary_mint_symbols):
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
    state.update(
        lambda s: (
            setattr(s, "primary_balance", float(current_primary_balance or 0.0)),
            setattr(s, "portfolio_value", float(current_total_value or 0.0)),
            setattr(s, "total_profit", float(total_profit or 0.0)),
            setattr(s, "reserved_fees", minimum_sol_needed()),
            setattr(s, "tokens", tokens),
            setattr(s, "last_refresh", datetime.now(UTC).strftime("%H:%M:%S")),
        )
    )


def handle_buy_signal(df: pd.DataFrame, secondary_mint: str, data_file_path: str, secondary_mint_symbol: str) -> bool:
    """Execute a buy when the last bar has an entry signal; returns success."""
    if df["entry"].iat[-1] == 1:
        mint_symbol = cast(str, df["symbol"].iat[0])

        # Check sentiment circuit breaker
        if sentiment_enabled:
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
        input_amount = _balance_cache.get(primary_mint)
        if input_amount <= 0:
            log_transaction.info(
                f"SolTrade has detected a buy signal, but does not have enough {primary_mint_symbol} to trade."
            )
            return False

        size_modifier = result["size_modifier"]
        if size_modifier < 1.0:
            input_amount = input_amount * size_modifier
            log_transaction.info(
                f"position size reduced to {size_modifier*100:.0f}% for {secondary_mint_symbol}: {result['reason']}"
            )

        log_transaction.info(
            f"SolTrade has detected a buy signal for {mint_symbol} using {input_amount} {primary_mint_symbol}."
        )
        is_swapped = (
            _paper_buy(
                input_amount,
                df,
                primary_mint,
                secondary_mint,
                primary_mint_symbol,
                secondary_mint_symbol,
            )
            if _dry_run
            else asyncio.run(
                perform_swap(
                    input_amount,
                    primary_mint,
                    secondary_mint,
                    primary_mint_symbol,
                    secondary_mint_symbol,
                )
            )
        )
        if is_swapped:
            df = calc_entry_price(df)
            df = calc_stoploss(df)
            df = calc_takeprofit(df)
            df = calc_trailing_stoploss(df)
            df = set_position(df, True)
            save_dataframe_to_csv(df, data_file_path)
            _balance_cache.invalidate(primary_mint)
            _balance_cache.invalidate(secondary_mint)
            return True
        return False
    return False


def handle_sell_signal(df: pd.DataFrame, secondary_mint: str, data_file_path: str, secondary_mint_symbol: str) -> bool:
    """Execute a sell when the last bar has an exit signal; returns success."""
    input_amount = _balance_cache.get(secondary_mint)
    df = calc_trailing_stoploss(df)

    if df["exit"].iat[-1] == 1:
        mint_symbol = cast(str, df["symbol"].iat[0])

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
                primary_mint,
                secondary_mint,
                primary_mint_symbol,
                secondary_mint_symbol,
            )
            if _dry_run
            else asyncio.run(
                perform_swap(
                    input_amount,
                    secondary_mint,
                    primary_mint,
                    secondary_mint_symbol,
                    primary_mint_symbol,
                )
            )
        )
        if is_swapped:
            df = set_position(df, False)
            df = df.drop(
                columns=[
                    "stoploss",
                    "entry_price",
                    "trailing_stoploss",
                    "trailing_stoploss_target",
                    "takeprofit",
                ]
            )
            save_dataframe_to_csv(df, data_file_path)
            _balance_cache.invalidate(secondary_mint)
            _balance_cache.invalidate(primary_mint)
            return True
        return False
    return False


def start_trading(state: UIState, dry_run: bool = False) -> None:
    """Run the trading loop in a background thread, feeding the UI state.

    With ``dry_run=True`` swaps are simulated against the latest close and the
    real wallet is never touched.
    """
    global _stop_event, _trading_thread, _dry_run

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

    def _run() -> None:
        state.update(lambda s: setattr(s, "running", True))
        try:
            while not _stop_event.is_set():
                try:
                    perform_analysis(state)
                except Exception as e:  # noqa: BLE001 - keep the loop alive across errors
                    log_general.error(f"analysis cycle failed: {e}")
                    state.update(lambda s: setattr(s, "error_count", s.error_count + 1))
                for remaining in range(price_update_seconds, 0, -1):
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
