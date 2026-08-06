"""SolTrade — automated Solana trading bot."""

import sys

from sol_trade import trading, ui
from sol_trade.config import config


def main() -> None:
    config()
    if not config().keypair or not config().secondary_mints:
        print(
            "Configuration incomplete: set SOLTRADE_PRIVATE_KEY in .env and "
            "secondary_mints in config.json. See the README.",
            file=sys.stderr,
        )
        sys.exit(1)

    state = ui.UIState()
    trading.start_trading(state)
    try:
        ui.run_ui(state)
    finally:
        trading.stop_trading()


if __name__ == "__main__":
    main()
