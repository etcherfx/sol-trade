"""Whale wallet tracker — monitors token balance changes for configured whale wallets."""

import json
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from solana.rpc.types import TokenAccountOpts
from solders.pubkey import Pubkey

from soltrade.config import config
from soltrade.log import log_general
from soltrade.utils import handle_rate_limiting


@handle_rate_limiting(retry_attempts=3, retry_delay=10)
def _get_wallet_token_balance(wallet: str, token_mint: str) -> float:
    """Query token balance for a single wallet and token via Solana RPC."""
    cfg = config()
    # Handle native SOL
    if token_mint == cfg.sol_mint:
        response = cfg.client.get_balance(Pubkey.from_string(wallet))
        balance = response.value / (10**9)
        return balance

    response = (
        cfg.client.get_token_accounts_by_owner_json_parsed(
            Pubkey.from_string(wallet),
            TokenAccountOpts(mint=Pubkey.from_string(token_mint)),
        )
        .to_json()
    )
    json_response = json.loads(response)
    accounts = json_response.get("result", {}).get("value", [])
    if not accounts:
        return 0.0

    # Sum across all token accounts for this mint (handles multiple ATA accounts)
    total = 0.0
    for account in accounts:
        ui_amount = (
            account["account"]["data"]["parsed"]["info"]["tokenAmount"]["uiAmount"]
        )
        if ui_amount is not None:
            total += ui_amount
    return total


def _load_data() -> List[Dict[str, Any]]:
    """Load whale snapshot data from disk."""
    path = config().whale_data_path
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except (json.JSONDecodeError, KeyError) as e:
        log_general.warning(f"Failed to load whale data from {path}: {e}")
        return []


def _save_data(data: List[Dict[str, Any]]) -> None:
    """Persist whale snapshot data to disk."""
    path = config().whale_data_path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def update_whale_data() -> None:
    """Poll all tracked wallets and persist snapshots."""
    cfg = config()
    if not cfg.whale_tracking_enabled:
        return

    whale_wallets = cfg.whale_wallets
    if not whale_wallets:
        return

    existing_data = _load_data()

    latest_ts = max((e.get("ts") for e in existing_data), default=None)
    if latest_ts:
        try:
            if (datetime.now(timezone.utc) - datetime.fromisoformat(latest_ts)) < timedelta(
                minutes=cfg.whale_poll_interval_minutes
            ):
                return
        except (ValueError, TypeError):
            pass  # unparseable timestamp -> poll anyway

    timestamp = datetime.now(timezone.utc).isoformat()

    for token_symbol, wallets in whale_wallets.items():
        # Resolve token mint from config
        token_mint = None
        if token_symbol in cfg.secondary_mint_symbols:
            idx = cfg.secondary_mint_symbols.index(token_symbol)
            token_mint = cfg.secondary_mints[idx]
        elif token_symbol == cfg.primary_mint_symbol:
            token_mint = cfg.primary_mint
        elif token_symbol == "SOL":
            token_mint = cfg.sol_mint

        if token_mint is None:
            log_general.warning(
                f"Whale tracker: could not resolve mint for symbol '{token_symbol}'"
            )
            continue

        for wallet in wallets:
            try:
                balance = _get_wallet_token_balance(wallet, token_mint)
                if balance is None:
                    # RPC failure after retries
                    continue
                snapshot = {
                    "ts": timestamp,
                    "wallet": wallet,
                    "token": token_symbol,
                    "balance": round(balance, 6),
                }
                existing_data.append(snapshot)
            except Exception as e:
                log_general.warning(
                    f"Whale tracker: failed to fetch balance for wallet "
                    f"{wallet[:8]}... token {token_symbol}: {e}"
                )

    # Prune: keep only last 24 hours of data to prevent unbounded growth
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
    existing_data = [entry for entry in existing_data if entry["ts"] >= cutoff]
    _save_data(existing_data)


def get_whale_signal(token_symbol: str) -> str:
    """Return 'ACCUMULATING', 'DUMPING', 'NEUTRAL', or 'NO_DATA' for a token.

    Signal is based on net balance delta across all tracked whale wallets
    over rolling time windows. Returns NEUTRAL if there is no significant
    movement. Returns NO_DATA when no wallets are configured or there are
    insufficient snapshots to compute a signal.
    """
    cfg = config()
    if not cfg.whale_tracking_enabled:
        return "NO_DATA"

    whale_wallets = cfg.whale_wallets.get(token_symbol, [])
    if len(whale_wallets) < 2:
        return "NO_DATA"

    data = _load_data()
    if not data:
        return "NO_DATA"

    # Filter to this token's data
    token_data = [entry for entry in data if entry.get("token") == token_symbol]
    if len(token_data) < 2:
        return "NO_DATA"

    now = datetime.now(timezone.utc)

    # Evaluate multiple time windows: 1h, 4h, 24h
    windows = [
        timedelta(hours=1),
        timedelta(hours=4),
        timedelta(hours=24),
    ]

    accumulating_score = 0
    dumping_score = 0

    for window in windows:
        cutoff = (now - window).isoformat()
        window_entries = [e for e in token_data if e["ts"] >= cutoff]

        if len(window_entries) < 2:
            continue

        # Group by wallet, find earliest and latest balance per wallet
        wallet_snapshots: Dict[str, List[Dict[str, Any]]] = {}
        for entry in window_entries:
            wallet_snapshots.setdefault(entry["wallet"], []).append(entry)

        total_start_balance = 0.0
        total_end_balance = 0.0

        for wallet, snapshots in wallet_snapshots.items():
            sorted_snapshots = sorted(snapshots, key=lambda s: s["ts"])
            total_start_balance += sorted_snapshots[0]["balance"]
            total_end_balance += sorted_snapshots[-1]["balance"]

        if total_start_balance == 0:
            continue

        pct_change = (total_end_balance - total_start_balance) / total_start_balance

        # Threshold: 10% change triggers a signal
        if pct_change > 0.10:
            accumulating_score += 1
        elif pct_change < -0.10:
            dumping_score += 1

    # Majority vote across windows
    if accumulating_score > dumping_score:
        return "ACCUMULATING"
    elif dumping_score > accumulating_score:
        return "DUMPING"
    return "NEUTRAL"
