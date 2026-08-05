"""Market regime detector — determines Solana market direction via SOL/USDC trend."""

import json
import os
from datetime import UTC, datetime, timedelta
from typing import Any

import requests

from sol_trade.config import config
from sol_trade.log import log_general


def _load_regime() -> dict[str, Any]:
    """Load persisted regime data from disk."""
    path = config().regime_data_path
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, KeyError) as e:
        log_general.warning(f"Failed to load regime data from {path}: {e}")
        return {}


def _save_regime(regime: str, modifier: float) -> None:
    """Persist regime data to disk."""
    path = config().regime_data_path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {
        "regime": regime,
        "modifier": modifier,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _should_refresh() -> bool:
    """Check if regime data is stale (older than 1 hour)."""
    data = _load_regime()
    if not data or "timestamp" not in data:
        return True
    try:
        last_update = datetime.fromisoformat(data["timestamp"])
        return (datetime.now(UTC) - last_update) > timedelta(hours=1)
    except (ValueError, TypeError):
        return True


def _fetch_sol_usdc_daily() -> list:
    """Fetch SOL/USDC daily candlestick data from CryptoCompare."""
    url = "https://min-api.cryptocompare.com/data/v2/histoday"
    params = {"fsym": "SOL", "tsym": "USDC", "limit": 30}
    if config().api_key:
        params["api_key"] = config().api_key
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    if data.get("Response") != "Success":
        raise ValueError(f"CryptoCompare error: {data.get('Message', 'Unknown')}")
    return data["Data"]


def _compute_sma(prices: list, period: int) -> list:
    """Compute simple moving average."""
    sma = []
    for i in range(len(prices)):
        if i < period - 1:
            sma.append(None)
        else:
            window = prices[i - period + 1 : i + 1]
            sma.append(sum(window) / period)
    return sma


def update_regime() -> None:
    """Fetch daily data and compute regime."""
    cfg = config()
    if not cfg.market_regime_enabled:
        return

    if not _should_refresh():
        return

    try:
        candles = _fetch_sol_usdc_daily()
        if not candles:
            log_general.warning("Market regime: no daily data received")
            _save_regime("NEUTRAL", 1.0)
            return

        prices = [c["close"] for c in candles]
        volumes = [c.get("totalvolume", 0) for c in candles]

        # 20-day SMA trend
        sma20 = _compute_sma(prices, 20)
        current_price = prices[-1]
        current_sma = sma20[-1] if sma20[-1] is not None else current_price

        bullish_trend = current_price > current_sma
        bearish_trend = current_price < current_sma

        # Volume trend: compare last 5-day avg vs prior 10-day avg
        if len(volumes) >= 15:
            recent_vol = sum(volumes[-5:]) / 5
            prior_vol = sum(volumes[-15:-5]) / 10
            rising_volume = recent_vol > prior_vol
        else:
            rising_volume = True  # default if insufficient data

        # Determine regime
        if bullish_trend and rising_volume:
            regime = "BULLISH"
            modifier = 1.0
        elif bearish_trend and not rising_volume:
            regime = "BEARISH"
            modifier = 0.5
        else:
            regime = "NEUTRAL"
            modifier = 1.0

        _save_regime(regime, modifier)
        log_general.info(f"Market regime updated: {regime} (modifier {modifier})")

    except Exception as e:  # noqa: BLE001 - RPC failure; fall back to NEUTRAL
        log_general.warning(f"Market regime: failed to update, falling back to NEUTRAL: {e}")
        _save_regime("NEUTRAL", 1.0)


def get_regime() -> str:
    """Return 'BULLISH', 'BEARISH', or 'NEUTRAL'.

    If regime detection is disabled, returns NEUTRAL.
    """
    cfg = config()
    if not cfg.market_regime_enabled:
        return "NEUTRAL"

    data = _load_regime()
    return data.get("regime", "NEUTRAL")


def get_position_modifier() -> float:
    """Return position size multiplier: 0.5 for BEARISH, 1.0 otherwise.

    If regime detection is disabled, returns 1.0.
    """
    cfg = config()
    if not cfg.market_regime_enabled:
        return 1.0

    data = _load_regime()
    return data.get("modifier", 1.0)
