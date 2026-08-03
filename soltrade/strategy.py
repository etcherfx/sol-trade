import importlib

import numpy as np
import pandas as pd

from soltrade.config import config
from soltrade.log import log_general

strategy_instance = None


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


def load_strategy_class(strategy_name):
    strategy_module = importlib.import_module(f"strategies.{strategy_name}_strategy")
    strategy_class = getattr(strategy_module, f"{strategy_name.capitalize()}Strategy")
    return strategy_class


def strategy(df: pd.DataFrame):
    global strategy_instance
    strategy_name = config().strategy or "default"
    try:
        StrategyClass = load_strategy_class(strategy_name)
        strategy_instance = StrategyClass(df)
        df = strategy_instance.apply_strategy()
    except (ModuleNotFoundError, AttributeError) as e:
        log_general.error(f"Strategy {strategy_name} not found: {e}")
        raise

    return df


def set_position(df, position):
    df["position"] = position
    return df


def calc_entry_price(df):
    entry_price = df["close"].iat[-1]
    df["entry_price"] = entry_price
    return df


def calc_stoploss(df):
    global strategy_instance
    sl = float(strategy_instance.stoploss)
    df["stoploss"] = df["close"].iat[-1] * (1 - (sl / 100))
    return df


def calc_takeprofit(df):
    global strategy_instance
    tp = float(strategy_instance.takeprofit)
    df["takeprofit"] = df["close"].iat[-1] * (1 + (tp / 100))
    return df


def calc_trailing_stoploss(df):
    global strategy_instance
    tsl = float(strategy_instance.trailing_stoploss)
    tslt = float(strategy_instance.trailing_stoploss_target)

    high_prices = df["high"]
    trailing_stop = []
    tracking_started = False
    highest_price = df["high"].iat[0]

    for price in high_prices:
        if not tracking_started and price >= df["entry_price"].iat[0] * (
            1 + tslt / 100
        ):
            tracking_started = True
            highest_price = price
        if tracking_started:
            if price > highest_price:
                highest_price = price
            stop_price = highest_price * (1 - tsl / 100)
            trailing_stop.append(stop_price)
        else:
            trailing_stop.append(None)

    df["trailing_stoploss"] = trailing_stop
    df["trailing_stoploss_target"] = df["entry_price"] * (1 + tslt / 100)

    return df
