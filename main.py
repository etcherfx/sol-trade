"""SolTrade — automated Solana trading bot."""

import argparse
import sys

from sol_trade import trading, ui
from sol_trade.config import config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SolTrade — automated Solana trading bot."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="paper trade: simulate swaps, never touch the wallet",
    )
    args = parser.parse_args()

    config()
    if not config().keypair or not config().secondary_mints:
        print(
            "Configuration incomplete: set SOLTRADE_PRIVATE_KEY in .env and "
            "secondary_mints in config.json. See the README.",
            file=sys.stderr,
        )
        sys.exit(1)

    state = ui.UIState()
    trading.start_trading(state, dry_run=args.dry_run)
    try:
        ui.run_ui(state)
    finally:
        trading.stop_trading()


if __name__ == "__main__":
    main()
