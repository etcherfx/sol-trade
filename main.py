"""SolTrade — automated Solana trading bot."""

import argparse
import signal
import sys
import threading

from sol_trade import trading, ui
from sol_trade.config import config
from sol_trade.log import enable_json_console, log_general, setup_file_logging


def run_headless(dry_run: bool) -> int:
    """Trade without the terminal UI until SIGTERM or SIGINT; returns the exit code."""
    stop = threading.Event()

    def _request_stop(signum: int, _frame: object) -> None:
        log_general.info(f"received {signal.Signals(signum).name}; stopping")
        stop.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _request_stop)
    trading.start_trading(ui.UIState(), dry_run=dry_run)
    try:
        while not stop.wait(1.0):
            if not trading.is_running():
                log_general.error("trading thread exited unexpectedly")
                return 1
    finally:
        trading.stop_trading()
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SolTrade — automated Solana trading bot."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="paper trade: simulate swaps, never touch the wallet",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="no terminal UI: JSON-lines logs on stdout, stop with SIGTERM or Ctrl-C",
    )
    args = parser.parse_args()
    setup_file_logging()  # before config(), so its warnings reach the log files
    if args.headless:
        enable_json_console()

    try:
        if not args.dry_run:
            config().keypair  # noqa: B018 - raises ValueError on a missing/invalid key
        else:
            config()  # paper trading needs the config but no wallet key
        problem = ""
    except (OSError, ValueError) as e:
        problem = str(e)
    if problem:
        print(
            "Configuration incomplete: set SOLTRADE_PRIVATE_KEY in .env (not needed "
            f"with --dry-run) and the primary token in config.json. See the README. ({problem})",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.headless:
        sys.exit(run_headless(args.dry_run))
    state = ui.UIState()
    trading.start_trading(state, dry_run=args.dry_run)
    try:
        ui.run_ui(state)
    finally:
        trading.stop_trading()


if __name__ == "__main__":
    main()
