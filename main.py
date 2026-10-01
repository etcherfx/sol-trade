"""SolTrade — automated Solana trading bot."""

import argparse
import sys

from sol_trade import trading, ui
from sol_trade.config import config
from sol_trade.log import setup_file_logging


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
    setup_file_logging()  # before config(), so its warnings reach the log files

    try:
        config().keypair  # noqa: B018 - raises ValueError on a missing/invalid key
        problem = "" if config().secondary_mints else "no secondary_mints"
    except (OSError, ValueError) as e:
        problem = str(e)
    if problem:
        print(
            "Configuration incomplete: set SOLTRADE_PRIVATE_KEY in .env and "
            f"secondary_mints in config.json. See the README. ({problem})",
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
