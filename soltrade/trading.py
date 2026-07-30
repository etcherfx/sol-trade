import asyncio
import os
import pandas as pd
import requests
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, cast
from rich.console import Console, Group, RenderableType
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.text import Text
from rich import box

from soltrade.config import config
from soltrade.log import log_general, log_transaction, silence_console_logging
from soltrade.strategy import (
    strategy,
    calc_stoploss,
    calc_trailing_stoploss,
    calc_entry_price,
    calc_takeprofit,
    set_position,
)
from soltrade.transactions import perform_swap
from soltrade.wallet import find_balance

config_instance = config()
primary_mint: str = config_instance.primary_mint
primary_mint_symbol: str = config_instance.primary_mint_symbol
secondary_mints: List[str] = config_instance.secondary_mints
secondary_mint_symbols: List[str] = config_instance.secondary_mint_symbols
api_key: str = config_instance.api_key
trading_interval_minutes: int = config_instance.trading_interval_minutes
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
    """Lazy balance fetcher that caches until explicitly invalidated."""

    def __init__(self) -> None:
        self._cache: Dict[str, float] = {}

    def get(self, mint: str) -> float:
        if mint not in self._cache:
            self._cache[mint] = find_balance(mint)
        return self._cache[mint]

    def invalidate(self, mint: str) -> None:
        self._cache.pop(mint, None)


_balance_cache = BalanceCache()


def fetch_prices(mints: List[str]) -> Dict[str, float]:
    """Fetch multiple token prices with a single HTTP call."""
    if not mints:
        return {}

    unique_mints = list(dict.fromkeys(mints))  # preserve order
    params = {"ids": ",".join(unique_mints)}
    url = "https://lite-api.jup.ag/price/v3"

    try:
        response = _http_session.get(url, params=params, timeout=10)
        response.raise_for_status()
        response_json = cast(Dict[str, Any], response.json())
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 401:
            log_general.error(
                "401 Unauthorized: Endpoint requires Pro plan, falling back to lite-api"
            )
        else:
            log_general.error(f"HTTP error fetching prices for {unique_mints}: {e}")
        return {mint: 0.0 for mint in unique_mints}
    except Exception as e:  # pragma: no cover - network errors
        log_general.error(f"Failed to fetch prices for {unique_mints}: {e}")
        return {mint: 0.0 for mint in unique_mints}

    prices: dict[str, float] = {}
    for mint in unique_mints:
        mint_data = cast(Dict[str, Any], response_json.get(mint, {}) or {})
        price = float(mint_data.get("usdPrice") or 0)
        if price == 0:
            log_general.debug(f"Price for {mint} missing from response; defaulting to 0")
        prices[mint] = price
    return prices


initial_primary_balance = find_balance(primary_mint)
initial_secondary_balances = [find_balance(mint) for mint in secondary_mints]
initial_price_map = fetch_prices([primary_mint, *secondary_mints])
initial_primary_price = initial_price_map.get(primary_mint, 0.0)
initial_secondary_prices = [initial_price_map.get(mint, 0.0) for mint in secondary_mints]

console = Console()
live_display: Optional[Live] = None


def fetch_candlestick(primary_mint_symbol: str, secondary_mint_symbol: str) -> Dict[str, Any]:
    """Fetch candlestick data from CryptoCompare API."""
    url = "https://min-api.cryptocompare.com/data/v2/histominute"
    headers = {"authorization": api_key}
    params: Dict[str, str | int] = {
        "tsym": primary_mint_symbol,
        "fsym": secondary_mint_symbol,
        "limit": 50,
        "aggregate": trading_interval_minutes,
    }
    try:
        response = _http_session.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        response_json = cast(Dict[str, Any], response.json())
        if response_json.get("Response") == "Error":
            log_general.error(response_json.get("Message"))
            exit()
        return response_json
    except Exception as e:
        log_general.error(f"Failed to fetch candlestick data: {e}")
        exit()


def format_as_money(value: float) -> str:
    return "${:,.2f}".format(value)


def _render_dashboard(wallet_panel: Panel, market_table: Table, countdown_text: str) -> Group:
    """Combine dashboard sections into a single Live-friendly renderable."""
    countdown = Text(countdown_text, style="dim")
    return Group(wallet_panel, Text(""), market_table, Text(""), countdown)


def _update_live(renderable: RenderableType) -> None:
    """Safely update the Live display or fall back to standard printing."""
    if live_display and live_display.is_started:
        live_display.update(renderable)
    else:
        console.print(renderable)


def perform_analysis() -> None:
    data_frames: List[pd.DataFrame] = []
    price_map = fetch_prices([primary_mint, *secondary_mints])

    for secondary_mint, secondary_mint_symbol in zip(
        secondary_mints, secondary_mint_symbols
    ):
        candle_json = fetch_candlestick(primary_mint_symbol, secondary_mint_symbol)
        candle_dict = candle_json["Data"]["Data"]
        columns = ["close", "high", "low", "open", "time"]
        new_df = pd.DataFrame(candle_dict, columns=columns)
        new_df["time"] = pd.to_datetime(new_df["time"], unit="s")
        new_df = strategy(new_df)
        new_df["total_profit"] = 0
        new_df["mint"] = secondary_mint_symbol
        new_df["position"] = False
        data_file_path = f"data/{secondary_mint_symbol}_data.csv"

        try:
            existing_df = read_dataframe_from_csv(data_file_path)
            if existing_df["position"].iat[-1]:
                columns_to_merge = [
                    "position",
                    "entry_price",
                    "takeprofit",
                    "stoploss",
                    "trailing_stoploss",
                    "trailing_stoploss_target",
                ]

                for col in columns_to_merge:
                    new_df[col] = existing_df.iloc[-1][col]

            df = new_df
        except FileNotFoundError:
            df = new_df

        data_frames.append(df)

    # Update whale tracking data
    if whale_tracking_enabled:
        try:
            from soltrade.whale_tracker import update_whale_data

            update_whale_data()
        except Exception as e:
            log_general.warning(f"Whale tracker update failed: {e}")

    # Update market regime (only if stale)
    if market_regime_enabled:
        try:
            from soltrade.market_regime import update_regime

            update_regime()
        except Exception as e:
            log_general.warning(f"Market regime update failed: {e}")

    # Update sentiment data (only if stale)
    if sentiment_enabled:
        try:
            from soltrade.sentiment import update_sentiment

            update_sentiment(secondary_mint_symbols)
        except Exception as e:
            log_general.warning(f"Sentiment update failed: {e}")

    combined_df: pd.DataFrame = pd.concat(data_frames, axis=0)
    combined_df.drop_duplicates(subset=["time", "mint"], keep="last", inplace=True)

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
    
    profit_color = "green" if total_profit >= 0 else "red"
    profit_symbol = "📈" if total_profit >= 0 else "📉"
    
    wallet_info = Table.grid(padding=(0, 2))
    wallet_info.add_column(style="bold cyan", justify="right", no_wrap=True)
    wallet_info.add_column(style="white", no_wrap=True)
    
    wallet_info.add_row("💰 Primary Balance:", f"{current_primary_balance:.4f} {primary_mint_symbol}")
    wallet_info.add_row("📌 Reserved for Fees:", f"0.02 {primary_mint_symbol}")
    wallet_info.add_row("💵 Portfolio Value:", format_as_money(current_total_value))
    wallet_info.add_row(f"{profit_symbol} Total Profit:", f"[{profit_color}]{format_as_money(total_profit)}[/{profit_color}]")
    
    last_rows = combined_df.groupby("mint").tail(1)

    pivot_columns = [
        "close",
        "ema_s",
        "ema_m",
        "upper_bband",
        "lower_bband",
        "rsi",
        "entry",
        "exit",
        "entry_price",
        "stoploss",
        "takeprofit",
        "trailing_stoploss",
        "trailing_stoploss_target",
        "position",
    ]

    pivot_columns = [col for col in pivot_columns if col in last_rows.columns]
    last_rows_pivoted = last_rows.pivot(
        index="mint", columns="time", values=pivot_columns
    )
    last_rows_pivoted.columns = [f"{col[0]}" for col in last_rows_pivoted.columns]
    last_rows_pivoted = last_rows_pivoted.T

    custom_headers = {
        "close": "Price",
        "ema_s": "EMA Short",
        "ema_m": "EMA Medium",
        "upper_bband": "Upper Bollinger Band",
        "lower_bband": "Lower Bollinger Band",
        "rsi": "RSI",
        "entry": "Entry Signal",
        "exit": "Exit Signal",
        "entry_price": "Entry Price",
        "stoploss": "Stoploss",
        "takeprofit": "Take Profit",
        "trailing_stoploss": "Trailing Stoploss",
        "trailing_stoploss_target": "Trailing Stoploss Target",
        "position": "Position",
    }
    last_rows_pivoted.rename(index=custom_headers, inplace=True)

    price_row = cast(pd.Series, last_rows_pivoted.loc["Price"])
    last_rows_pivoted.loc["Price"] = price_row.apply(format_as_money)  # pyright: ignore[reportGeneralTypeIssues]

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    market_table = Table(
        title=f"📊 [bold cyan]Market Analysis[/bold cyan] [dim]({timestamp})[/dim]", 
        box=box.ROUNDED, 
        show_header=True,
        header_style="bold magenta",
        border_style="cyan",
        row_styles=["", "dim"]
    )
    
    market_table.add_column("📈 Metric", style="bold yellow", no_wrap=True, width=25)
    for col in last_rows_pivoted.columns:
        market_table.add_column(str(col), style="cyan", justify="right", width=15)
    
    for idx in last_rows_pivoted.index:
        row_data = [str(idx)]
        for col in last_rows_pivoted.columns:
            value = last_rows_pivoted.loc[idx, col]
            
            if idx == "Entry Signal" and value:
                value_str = "[bold green]✓ BUY[/bold green]"
            elif idx == "Exit Signal" and value:
                value_str = "[bold red]✗ SELL[/bold red]"
            elif idx in ["Entry Signal", "Exit Signal"]:
                value_str = "[dim]-[/dim]"
            else:
                value_str = str(value)
            
            row_data.append(value_str)
        market_table.add_row(*row_data)
    
    wallet_panel = Panel(wallet_info, title="💼 Wallet Overview", border_style="cyan", padding=(1, 2), expand=False)

    dashboard = _render_dashboard(wallet_panel, market_table, "⏳ Refreshing data...")
    _update_live(dashboard)

    for df, secondary_mint, secondary_mint_symbol in zip(
        data_frames, secondary_mints, secondary_mint_symbols
    ):
        data_file_path = f"data/{secondary_mint_symbol}_data.csv"
        if not df["position"].iat[-1]:
            handle_buy_signal(df, secondary_mint, data_file_path, secondary_mint_symbol)
        else:
            handle_sell_signal(df, secondary_mint, data_file_path, secondary_mint_symbol)

    try:
        for remaining in range(price_update_seconds, 0, -1):
            countdown_text = f"⏱️  Next update in {remaining} seconds | Press Ctrl+C to stop"
            _update_live(_render_dashboard(wallet_panel, market_table, countdown_text))
            time.sleep(1)
    except KeyboardInterrupt:
        _update_live(_render_dashboard(wallet_panel, market_table, "⏹️  Stopping..."))
        raise


def handle_buy_signal(df: pd.DataFrame, secondary_mint: str, data_file_path: str, secondary_mint_symbol: str) -> bool:
    if df["entry"].iat[-1] == 1:
        mint_symbol = cast(str, df["mint"].iat[0])

        # Check sentiment circuit breaker
        if sentiment_enabled:
            from soltrade.sentiment import is_token_blocked, is_market_crash

            if is_token_blocked(secondary_mint_symbol):
                log_transaction.info(
                    f"Trading paused for {secondary_mint_symbol}: sentiment circuit breaker active"
                )
                return False
            if is_market_crash():
                log_transaction.info(
                    f"All new entries paused: market sentiment crash detected"
                )
                return False

        # Evaluate confluence
        from soltrade.confluence import evaluate_buy_confluence

        result = evaluate_buy_confluence("BUY", secondary_mint_symbol)
        if result["action"] == "skip":
            log_transaction.info(
                f"Buy signal for {secondary_mint_symbol} skipped: {result['reason']}"
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
                f"Position size reduced to {size_modifier*100:.0f}% for {secondary_mint_symbol}: {result['reason']}"
            )

        log_transaction.info(
            f"SolTrade has detected a buy signal for {mint_symbol} using {input_amount} {primary_mint_symbol}."
        )
        is_swapped = asyncio.run(
            perform_swap(
                input_amount,
                primary_mint,
                secondary_mint,
                primary_mint_symbol,
                secondary_mint_symbol,
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
    input_amount = _balance_cache.get(secondary_mint)
    df = calc_trailing_stoploss(df)

    if df["exit"].iat[-1] == 1:
        mint_symbol = cast(str, df["mint"].iat[0])

        # Evaluate confluence for sells
        from soltrade.confluence import evaluate_sell_confluence

        result = evaluate_sell_confluence("SELL", secondary_mint_symbol)
        if result["action"] == "skip":
            log_transaction.info(
                f"Sell signal for {secondary_mint_symbol} skipped: {result['reason']}"
            )
            return False

        # Apply position size modifier
        size_modifier = result["size_modifier"]
        if size_modifier < 1.0:
            input_amount = input_amount * size_modifier
            log_transaction.info(
                f"Sell position size reduced to {size_modifier*100:.0f}% for {secondary_mint_symbol}: {result['reason']}"
            )

        log_transaction.info(
            f"SolTrade has detected a sell signal for {input_amount} {mint_symbol}."
        )
        is_swapped = asyncio.run(
            perform_swap(
                input_amount,
                secondary_mint,
                primary_mint,
                secondary_mint_symbol,
                primary_mint_symbol,
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


def start_trading():
    global live_display

    silence_console_logging()
    log_general.info("Soltrade has now initialized the trading algorithm.")

    with Live(console=console, refresh_per_second=4, transient=False) as live:
        live_display = live
        _update_live(Panel.fit("🔍 Loading market data...", border_style="yellow"))

        try:
            while True:
                perform_analysis()
        except KeyboardInterrupt:
            log_general.info("SolTrade has been stopped by user.")
        finally:
            live_display = None

    console.print("\n[yellow]⏹️  Shutting down SolTrade...[/yellow]")


def save_dataframe_to_csv(df: pd.DataFrame, file_path: str) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        log_general.info(f"Data successfully saved to {file_path}")
    except Exception as e:
        log_general.error(f"Failed to save data to {file_path}: {e}")


def read_dataframe_from_csv(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)  # type: ignore[reportGeneralTypeIssues]
