"""Whale discovery helper — find top token holders to track."""

import sys
from typing import Any

from solders.pubkey import Pubkey

from sol_trade.config import config
from sol_trade.log import enable_console_logging, log_general
from sol_trade.utils import handle_rate_limiting, run_async


@handle_rate_limiting(retry_attempts=3, retry_delay=10)
def find_top_holders(token_mint: str, limit: int = 20) -> list[dict[str, Any]]:
    """Query RPC for top token holders.

    Uses get_token_largest_accounts which is more efficient than
    getProgramAccounts for finding top holders of a specific token.

    Args:
        token_mint: Token mint address
        limit: Number of top holders to return

    Returns:
        List of dicts with 'address' and 'balance' keys.
    """
    cfg = config()
    mint_pubkey = Pubkey.from_string(token_mint)

    response = run_async(cfg.client.get_token_largest_accounts(mint_pubkey))
    accounts = response.value

    holders = []
    for account in accounts[:limit]:
        # Get UI amount by querying token account info
        ui_amount = account.amount.ui_amount
        if ui_amount is not None:
            holders.append({
                "address": str(account.address),
                "balance": round(ui_amount, 6),
            })

    return holders


def main() -> None:
    """CLI entry point for whale discovery."""
    enable_console_logging()
    if len(sys.argv) < 2:
        print("Usage: uv run -m sol_trade.whale_discovery TOKEN_MINT [LIMIT]")
        print()
        print("Example:")
        print('  uv run -m sol_trade.whale_discovery So11111111111111111111111111111111111111112 10')
        sys.exit(1)

    token_mint = sys.argv[1]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    try:
        holders = find_top_holders(token_mint, limit)
        if not holders:
            print(f"No holders found for mint {token_mint}")
            return

        print(f"Top {len(holders)} holders for {token_mint}:")
        print(f"{'#':<5} {'Address':<44} {'Balance':>15}")
        print("-" * 65)
        for i, holder in enumerate(holders, 1):
            addr = holder["address"]
            print(f"{i:<5} {addr:<44} {holder['balance']:>15,.6f}")

        print()
        print("To track these wallets, add their addresses to config.json:")
        print('  "whale_wallets": {')
        print('    "YOUR_TOKEN_SYMBOL": [')
        addrs = ", ".join(f'"{h["address"]}"' for h in holders[:5])
        print(f"      {addrs}")
        print("    ]")
        print("  }")

    except Exception as e:  # noqa: BLE001 - CLI failure; log and exit
        log_general.error(f"Failed to fetch holders: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
