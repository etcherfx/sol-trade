import importlib

import numpy as np
import pandas as pd

from sol_trade.config import config
from sol_trade.log import log_general


def ema(close: pd.Series, period: int) -> pd.Series:
    """Exponential moving average, TA-Lib equivalent.

    Reproduces TA-Lib's EMA exactly: NaN for the first ``period - 1`` bars,
    seeded at bar ``period - 1`` with the SMA of the first ``period`` closes,
    then Wilder-style recursion (alpha = 2 / (period + 1)).
    """
    out = np.full(len(close), np.nan)
    if len(close) < period:
        return pd.Series(out, index=close.index)
    alpha = 2.0 / (period + 1)
    out[period - 1] = close.iloc[:period].mean()
    for i in range(period, len(close)):
        out[i] = alpha * close.iloc[i] + (1 - alpha) * out[i - 1]
    return pd.Series(out, index=close.index)


def sma(close: pd.Series, period: int) -> pd.Series:
    """Simple moving average, TA-Lib equivalent (identical to a rolling mean)."""
    return close.rolling(period).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index, TA-Lib equivalent (Wilder smoothing).

    Reproduces TA-Lib's RSI exactly: NaN for the first ``period`` bars, the
    smoothed averages seeded with the SMA of the first ``period`` gains/losses,
    then Wilder recursion. Uses the 100 * gain / (gain + loss) form so a flat
    series yields 0 (TA-Lib's behavior), not NaN or 100.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    n = len(close)
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    out = np.full(n, np.nan)
    if n <= period:
        return pd.Series(out, index=close.index)
    avg_gain[period] = gain.iloc[1 : period + 1].mean()
    avg_loss[period] = loss.iloc[1 : period + 1].mean()
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain.iloc[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss.iloc[i]) / period
    total = avg_gain + avg_loss
    for i in range(period, n):
        out[i] = 0.0 if total[i] == 0.0 else 100.0 * avg_gain[i] / total[i]
    return pd.Series(out, index=close.index)


def load_strategy_class(strategy_name: str) -> type:
    """Load the strategy class for ``strategies/{name}_strategy.py``.

    The class is expected as ``{PascalCase}Strategy`` — ``jup_trend`` maps to
    ``JupTrendStrategy``. A fallback to the legacy single-word capitalization
    (``Jup_trendStrategy``) is tried for names that predate the convention.
    """
    strategy_module = importlib.import_module(f"strategies.{strategy_name}_strategy")
    pascal = "".join(part.capitalize() for part in strategy_name.split("_"))
    candidates = (f"{pascal}Strategy", f"{strategy_name.capitalize()}Strategy")
    for candidate in candidates:
        if hasattr(strategy_module, candidate):
            return getattr(strategy_module, candidate)
    raise AttributeError(
        f"module 'strategies.{strategy_name}_strategy' has no class "
        f"named {candidates[0]}"
    )


def resolve_strategy_name(symbol: str | None = None) -> str:
    """Pick the strategy for a token: per-token override, else the global one."""
    name = config().strategy or "default"
    if symbol:
        name = config().token_strategies.get(symbol) or name
    return name


def strategy(df: pd.DataFrame, symbol: str | None = None) -> pd.DataFrame:
    """Apply the configured strategy to the dataframe and return it.

    ``symbol`` selects a per-token strategy override (``token_strategies`` in
    config.json); when absent or unlisted the global ``strategy`` is used.

    The strategy instance is attached to the dataframe so risk parameters stay
    bound to the token they were computed for (a plain attribute, since pandas
    deep-copies ``df.attrs`` which would recurse through the instance's df).
    """
    strategy_name = resolve_strategy_name(symbol)
    try:
        StrategyClass = load_strategy_class(strategy_name)
        instance = StrategyClass(df)
        df = instance.apply_strategy()
        df.strategy_instance = instance
    except (ModuleNotFoundError, AttributeError) as e:
        log_general.error(f"Strategy {strategy_name} not found: {e}")
        raise

    return df


def set_position(df: pd.DataFrame, position: bool) -> pd.DataFrame:
    """Set the position flag on the dataframe."""
    df["position"] = position
    return df


def calc_stoploss(df: pd.DataFrame) -> pd.DataFrame:
    """Set a stop-loss level below the fill entry price."""
    sl = float(df.strategy_instance.stoploss)
    df["stoploss"] = df["entry_price"].iat[-1] * (1 - (sl / 100))
    return df


def calc_takeprofit(df: pd.DataFrame) -> pd.DataFrame:
    """Set a take-profit level above the fill entry price."""
    tp = float(df.strategy_instance.takeprofit)
    df["takeprofit"] = df["entry_price"].iat[-1] * (1 + (tp / 100))
    return df


def calc_trailing_stoploss(df: pd.DataFrame) -> pd.DataFrame:
    """Set a trailing stop that ratchets up after a target gain.

    The running peak is carried across cycles via the ``highest_price``
    column, so the stop never resets when a new data window starts.
    """
    tsl = float(df.strategy_instance.trailing_stoploss)
    tslt = float(df.strategy_instance.trailing_stoploss_target)

    entry_price = float(df["entry_price"].iat[0])
    # Resume the carried peak if present; otherwise start from the window.
    if "highest_price" in df.columns and not pd.isna(df["highest_price"].iat[0]):
        highest_price = float(df["highest_price"].iat[0])
    else:
        highest_price = float(df["high"].iat[0])

    trailing_stop = []
    tracking_started = False
    for price in df["high"]:
        if not tracking_started and price >= entry_price * (1 + tslt / 100):
            tracking_started = True
            highest_price = max(highest_price, price)
        if tracking_started:
            highest_price = max(highest_price, price)
            trailing_stop.append(highest_price * (1 - tsl / 100))
        else:
            trailing_stop.append(None)

    df["trailing_stoploss"] = trailing_stop
    df["trailing_stoploss_target"] = df["entry_price"] * (1 + tslt / 100)
    df["highest_price"] = highest_price  # persist the peak for the next cycle

    return df
