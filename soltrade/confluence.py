"""Confluence filter — sits between TA signals and trade execution.

Adjusts position sizing based on whale signals, market regime,
and sentiment circuit breaker state.
"""

from typing import Dict

from soltrade.config import config


def evaluate_buy_confluence(ta_signal: str, token_symbol: str) -> Dict[str, object]:
    """Evaluate confluence for a buy signal.

    Returns:
        {
            "action": "full" | "half" | "skip",
            "reason": str,
            "size_modifier": float,
        }
    """
    cfg = config()

    # If confluence is disabled, always allow full position
    if not cfg.confluence_enabled:
        return {"action": "full", "reason": "confluence_disabled", "size_modifier": 1.0}

    # Check sentiment circuit breaker first
    if cfg.sentiment_enabled:
        from soltrade.sentiment import is_token_blocked, is_market_crash

        if is_token_blocked(token_symbol):
            return {
                "action": "skip",
                "reason": f"sentiment_blocked_{token_symbol}",
                "size_modifier": 0.0,
            }
        if is_market_crash():
            return {
                "action": "skip",
                "reason": "market_sentiment_crash",
                "size_modifier": 0.0,
            }

    # Get whale signal
    whale_signal = _get_whale_signal(token_symbol)

    # Decision matrix for buys
    decision = _buy_decision_matrix(ta_signal, whale_signal)

    # Apply regime modifier
    regime_modifier = _get_regime_modifier()
    effective_modifier = decision["size_modifier"] * regime_modifier

    reason = decision["reason"]
    if regime_modifier < 1.0:
        reason = f"{reason}; bearish_regime"

    # Determine action based on final modifier
    if effective_modifier == 0.0:
        action = "skip"
    elif effective_modifier < 0.75:
        action = "half"
    else:
        action = "full"

    return {
        "action": action,
        "reason": reason,
        "size_modifier": effective_modifier,
    }


def evaluate_sell_confluence(ta_signal: str, token_symbol: str) -> Dict[str, object]:
    """Evaluate confluence for a sell signal.

    Returns:
        {
            "action": "full" | "half" | "partial" | "skip",
            "reason": str,
            "size_modifier": float,
        }
    """
    cfg = config()

    # If confluence is disabled, always allow full position
    if not cfg.confluence_enabled:
        return {"action": "full", "reason": "confluence_disabled", "size_modifier": 1.0}

    # Get whale signal
    whale_signal = _get_whale_signal(token_symbol)

    # Decision matrix for sells
    decision = _sell_decision_matrix(ta_signal, whale_signal)

    # Sells are not affected by regime modifier — we want to exit in bearish markets
    # Determine action based on modifier
    if decision["size_modifier"] == 0.0:
        action = "skip"
    elif decision["size_modifier"] < 0.75:
        action = "partial"
    else:
        action = "full"

    return {
        "action": action,
        "reason": decision["reason"],
        "size_modifier": decision["size_modifier"],
    }


def _get_whale_signal(token_symbol: str) -> str:
    """Get whale signal for a token, with graceful fallback."""
    cfg = config()
    if not cfg.whale_tracking_enabled:
        return "NEUTRAL"

    try:
        from soltrade.whale_tracker import get_whale_signal

        return get_whale_signal(token_symbol)
    except ImportError:
        return "NEUTRAL"


def _get_regime_modifier() -> float:
    """Get regime position modifier, with graceful fallback."""
    cfg = config()
    if not cfg.market_regime_enabled:
        return 1.0

    try:
        from soltrade.market_regime import get_position_modifier

        return get_position_modifier()
    except ImportError:
        return 1.0


def _buy_decision_matrix(ta_signal: str, whale_signal: str) -> Dict[str, object]:
    """Buy decision matrix based on TA signal and whale activity.

    | TA    | Whale          | Action | Size |
    |-------|----------------|--------|------|
    | BUY   | ACCUMULATING   | full   | 1.0  |
    | BUY   | NEUTRAL        | half   | 0.5  |
    | BUY   | DUMPING        | skip   | 0.0  |
    """
    if whale_signal == "ACCUMULATING":
        return {"size_modifier": 1.0, "reason": "whales_accumulating"}
    elif whale_signal == "DUMPING":
        return {"size_modifier": 0.0, "reason": "whales_dumping"}
    else:
        return {"size_modifier": 0.5, "reason": "whales_neutral"}


def _sell_decision_matrix(ta_signal: str, whale_signal: str) -> Dict[str, object]:
    """Sell decision matrix based on TA signal and whale activity.

    | TA    | Whale          | Action  | Size |
    |-------|----------------|---------|------|
    | SELL  | DUMPING        | full    | 1.0  |
    | SELL  | NEUTRAL        | half    | 0.5  |
    | SELL  | ACCUMULATING   | partial | 0.5  |
    """
    if whale_signal == "DUMPING":
        return {"size_modifier": 1.0, "reason": "whales_dumping"}
    elif whale_signal == "ACCUMULATING":
        return {"size_modifier": 0.5, "reason": "whales_accumulating_partial_sell"}
    else:
        return {"size_modifier": 0.5, "reason": "whales_neutral"}
