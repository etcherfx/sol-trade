import json
import os
from typing import Any

from dotenv import load_dotenv
from solana.rpc.async_api import AsyncClient
from solders.keypair import Keypair
from solders.pubkey import Pubkey

from sol_trade.log import log_general
from sol_trade.utils import run_async

# Keys that are NOT hot-reloaded — changing them requires a restart.
# Secondary tokens ARE hot-reloaded (guarded by the trading loop against
# orphaning open positions); primary mint stays structural because candle
# symbols and P&L baselines are quoted in it.
_STRUCTURAL_KEYS = (
    "primary_mint",
    "primary_mint_symbol",
    "sol_mint",
    "rpc_https",
    "jup_api",
    "data_exchange",
    "candles_path",
    "private_key",
    "jupiter_api_key",
)


def _file_mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


class Config:
    def __init__(self) -> None:
        self.jupiter_api_key: str = ""
        self.private_key: str = ""
        self.rpc_https: str = "https://api.mainnet-beta.solana.com"
        self.jup_api: str = "https://api.jup.ag/swap/v2"
        self.primary_mint: str = ""
        self.primary_mint_symbol: str = ""
        self.sol_mint: str = "So11111111111111111111111111111111111111112"
        self.secondary_mints: list[str] = []
        self.secondary_mint_symbols: list[str] = []
        # Portfolio weights per secondary token (parallel to secondary_mints);
        # empty list means equal split. Normalized to sum to 1 on load.
        self.secondary_weights: list[float] = []
        # Per-token strategy overrides, keyed by symbol; missing -> global strategy.
        self.token_strategies: dict[str, str] = {}
        # Per-token candle-data exchange overrides, keyed by symbol; missing ->
        # the global data_exchange.
        self.token_exchanges: dict[str, str] = {}
        self.price_update_seconds: int = 60
        self.max_slippage: int = 50
        self.strategy: str = "default"
        self.data_exchange: str = "okx"
        self.candles_path: str = "data/candles.db"
        self.path = os.path.join(os.path.dirname(__file__), "..", "config.json")
        self.dotenv_path = os.path.join(os.path.dirname(__file__), "..", ".env")
        load_dotenv(self.dotenv_path)
        self._client: AsyncClient | None = None
        self._decimals_cache: dict[str, int] = {}

        # Whale Tracker (ENABLED by default)
        self.whale_tracking_enabled: bool = True
        self.whale_wallets: dict[str, list[str]] = {}
        self.whale_poll_interval_minutes: int = 5
        self.whale_data_path: str = "data/whale_data.json"

        # Confluence Filter (ENABLED by default)
        self.confluence_enabled: bool = True

        # Market Regime (disabled — opt-in addon)
        self.market_regime_enabled: bool = False
        self.regime_data_path: str = "data/regime_data.json"

        # Sentiment Circuit Breaker (disabled — opt-in addon)
        self.sentiment_enabled: bool = False
        self.sentiment_pause_hours: int = 4
        self.sentiment_threshold: float = -0.5
        self.sentiment_crash_threshold: float = -0.7
        self.sentiment_data_path: str = "data/sentiment_data.json"
        self.load_config()
        self._config_mtime: float = _file_mtime(self.path)

    def _apply_config_file(self) -> None:
        default_config: dict[str, Any] = {
            "jupiter_api_key": "",
            "private_key": "",
            "rpc_https": "https://api.mainnet-beta.solana.com",
            "jup_api": "https://api.jup.ag/swap/v2",
            "primary_mint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
            "primary_mint_symbol": "USDC",
            "secondary_mints": ["So11111111111111111111111111111111111111112"],
            "secondary_mint_symbols": ["SOL"],
            "secondary_weights": [],
            "token_strategies": {},
            "token_exchanges": {},
            "price_update_seconds": 60,
            "max_slippage": 50,
            "strategy": "default",
            "data_exchange": "okx",
            "candles_path": "data/candles.db",

            # Whale Tracker
            "whale_tracking_enabled": True,
            "whale_wallets": {},
            "whale_poll_interval_minutes": 5,
            "whale_data_path": "data/whale_data.json",

            # Confluence Filter
            "confluence_enabled": True,

            # Market Regime
            "market_regime_enabled": False,
            "regime_data_path": "data/regime_data.json",

            # Sentiment Circuit Breaker
            "sentiment_enabled": False,
            "sentiment_pause_hours": 4,
            "sentiment_threshold": -0.5,
            "sentiment_crash_threshold": -0.7,
            "sentiment_data_path": "data/sentiment_data.json",
        }

        with open(self.path) as file:
            try:
                config_data: dict[str, Any] = json.load(file)
            except json.JSONDecodeError as e:
                raise ValueError(f"Error loading config: {e}") from e

        for key, fallback in default_config.items():
            value = config_data.get(key, fallback)
            if value in ("", None):
                value = fallback
            setattr(self, key, value)

        # Normalize portfolio weights to sum to 1 (empty list = equal split).
        weights = self.secondary_weights
        if isinstance(weights, list):
            try:
                weights = [float(w) for w in weights]
            except (TypeError, ValueError):
                weights = []
            total = sum(weights)
            weights = [w / total for w in weights] if total > 0 else []
        else:
            weights = []
        self.secondary_weights = weights

        if not isinstance(self.token_strategies, dict):
            self.token_strategies = {}
        if not isinstance(self.token_exchanges, dict):
            self.token_exchanges = {}

        self._config_mtime = _file_mtime(self.path)

    def load_config(self) -> None:
        self._apply_config_file()

        # Credentials: environment variables (.env) take precedence over config.json
        env_overrides = {
            "private_key": "SOLTRADE_PRIVATE_KEY",
            "jupiter_api_key": "SOLTRADE_JUPITER_API_KEY",
        }
        for attr, env_var in env_overrides.items():
            env_value = os.getenv(env_var)
            if env_value:
                setattr(self, attr, env_value)

        self._validate_config()

    def reload_config(self) -> None:
        """Re-read config.json for lightweight settings (hot-reload).

        Structural keys (tokens, RPC, exchange) and credentials are pinned —
        changing those still requires a restart. A malformed or invalid token
        set keeps the previous configuration.
        """
        pinned = {key: getattr(self, key) for key in _STRUCTURAL_KEYS}
        token_set = {
            attr: getattr(self, attr)
            for attr in (
                "secondary_mints",
                "secondary_mint_symbols",
                "secondary_weights",
                "token_strategies",
                "token_exchanges",
            )
        }
        self._apply_config_file()
        for key, value in pinned.items():
            setattr(self, key, value)
        try:
            self._validate_config()
        except ValueError:
            for key, value in token_set.items():
                setattr(self, key, value)
            raise

    def maybe_reload_config(self) -> None:
        """Reload config.json when it changed on disk (malformed edits are kept out)."""
        mtime = _file_mtime(self.path)
        if mtime != self._config_mtime and mtime != 0.0:
            try:
                log_general.info("config.json changed — reloading configuration.")
                self.reload_config()
            except (ValueError, OSError) as e:
                log_general.error(f"failed to reload config.json: {e}")

    def _validate_config(self) -> None:
        """Validate that critical configuration fields are properly set."""
        # Token-set shape errors raise so a malformed hot-reload keeps the
        # previous configuration (reload_config is wrapped in maybe_reload_config).
        if not self.secondary_mints or not self.secondary_mint_symbols:
            raise ValueError("secondary_mints and secondary_mint_symbols must not be empty")
        if len(self.secondary_mints) != len(self.secondary_mint_symbols):
            raise ValueError(
                "secondary_mints and secondary_mint_symbols must have equal length"
            )
        if self.secondary_weights and len(self.secondary_weights) != len(self.secondary_mint_symbols):
            raise ValueError(
                "secondary_weights must be empty or match the number of secondary tokens"
            )
        unknown_strategies = [
            symbol for symbol in self.token_strategies
            if symbol not in self.secondary_mint_symbols
        ]
        if unknown_strategies:
            log_general.warning(
                f"token_strategies references unknown token(s): {unknown_strategies}"
            )
        unknown_exchanges = [
            symbol for symbol in self.token_exchanges
            if symbol not in self.secondary_mint_symbols
        ]
        if unknown_exchanges:
            log_general.warning(
                f"token_exchanges references unknown token(s): {unknown_exchanges}"
            )

        if not self.private_key or self.private_key == "":
            log_general.warning("Private key is not set in .env or config.json. Bot cannot trade.")

        if not self.jupiter_api_key or self.jupiter_api_key == "":
            log_general.warning("Jupiter API key is not set. Optional unless required by your api.jup.ag endpoint.")

        if not self.rpc_https:
            log_general.error("RPC endpoint is not set in config.json.")

        if not self.jup_api:
            log_general.error("Jupiter API endpoint is not set in config.json.")

    def decimals(self, mint_address: str) -> int:
        """Return the smallest-unit multiplier for a mint, cached.

        Despite the name this is ``10 ** decimals`` (e.g. 1_000_000_000 for a
        9-decimal token), the value callers multiply token amounts by to get
        lamports/smallest units — not the decimals count itself.
        """
        if mint_address in self._decimals_cache:
            return self._decimals_cache[mint_address]

        response = run_async(
            self.client.get_account_info_json_parsed(
                Pubkey.from_string(mint_address)
            )
        ).to_json()
        json_response = json.loads(response)
        value = (
            10 ** json_response["result"]["value"]["data"]["parsed"]["info"]["decimals"]
        )

        self._decimals_cache[mint_address] = value
        return value

    @property
    def keypair(self) -> Keypair:
        """Wallet keypair; raises ValueError when the private key is missing or invalid."""
        try:
            return Keypair.from_base58_string(self.private_key)
        except (ValueError, TypeError) as e:
            raise ValueError("invalid or missing private key") from e

    @property
    def public_address(self) -> Pubkey:
        return self.keypair.pubkey()

    @property
    def client(self) -> AsyncClient:
        """Cached RPC client to avoid creating new connections."""
        if self._client is None:
            self._client = AsyncClient(self.rpc_https)
        return self._client


_config_instance = None


def config() -> Config:
    """Singleton pattern to ensure only one Config instance exists."""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance
