"""Hermetic test environment.

Tests never read the machine's config.json or .env, never sign with the real
wallet key, and never write to the repo's data/. Runs before any test module
imports sol_trade, so the config singleton below is the only one.
"""

import json
import os
import shutil
import tempfile

from solders.keypair import Keypair

from sol_trade import config as config_module

_REPO_CWD = os.getcwd()
_SANDBOX = tempfile.mkdtemp(prefix="sol-trade-tests-")
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
    path = os.path.join(_SANDBOX, "config.json")
    with open(path, "w") as file:
        json.dump(_TEST_CONFIG, file)
    config_module._config_instance = config_module.Config(
        config_path=path, dotenv_path=os.devnull
    )


_install_test_config()


def pytest_sessionstart(session):
    # The default data/ paths are relative to the cwd; keep any stray writes here.
    # Not at import: pytest resolves testpaths against the cwd after loading this.
    os.chdir(_SANDBOX)


def pytest_sessionfinish(session, exitstatus):
    os.chdir(_REPO_CWD)
    shutil.rmtree(_SANDBOX, ignore_errors=True)
