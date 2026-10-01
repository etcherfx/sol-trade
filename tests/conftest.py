"""Hermetic test environment.

Tests never read the machine's config.json or .env, never sign with the real
wallet key, and never write to the repo's logs/ or data/. Runs before any test
module imports sol_trade, so the config singleton below is the only one.
"""

import json
import logging
import os
import shutil
import tempfile
from unittest import mock

from solders.keypair import Keypair

_REPO_CWD = os.getcwd()
_SANDBOX = tempfile.mkdtemp(prefix="sol-trade-tests-")
# sol_trade.log opens logs/ and the default data paths are relative to the cwd.
os.chdir(_SANDBOX)
for _var in ("SOLTRADE_PRIVATE_KEY", "SOLTRADE_JUPITER_API_KEY"):
    os.environ.pop(_var, None)

_TEST_CONFIG = {
    "private_key": str(Keypair()),  # throwaway key, never funded
    "rpc_https": "http://127.0.0.1:9",  # accidental RPC calls fail fast, offline
    "primary_mint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
    "primary_mint_symbol": "USDC",
    "secondary_mints": [
        "So11111111111111111111111111111111111111112",
        "JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN",
    ],
    "secondary_mint_symbols": ["SOL", "JUP"],
    "strategy": "default",
}

def _install_test_config() -> None:
    # Imported only now: sol_trade.log opens its files relative to the sandbox cwd.
    from sol_trade import config as config_module

    path = os.path.join(_SANDBOX, "config.json")
    with open(path, "w") as file:
        json.dump(_TEST_CONFIG, file)
    # Config() hardcodes the repo's config.json and .env; skip both, then load ours.
    with (
        mock.patch.object(config_module, "load_dotenv"),
        mock.patch.object(config_module.Config, "load_config"),
    ):
        cfg = config_module.Config()
    cfg.path = path
    cfg.dotenv_path = os.devnull
    cfg.load_config()
    config_module._config_instance = cfg


_install_test_config()


def pytest_sessionfinish(session, exitstatus):
    logging.shutdown()  # release the sandbox's log files before deleting them
    os.chdir(_REPO_CWD)
    shutil.rmtree(_SANDBOX, ignore_errors=True)
