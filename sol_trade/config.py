import json
import os
from typing import Any, Dict, List

from solana.rpc.async_api import AsyncClient
from solders.keypair import Keypair
from solders.pubkey import Pubkey

from sol_trade.log import log_general
from sol_trade.utils import run_async


class Config:
    def __init__(self):
        self.api_key: str = ""
        self.jupiter_api_key: str = ""
        self.private_key: str = ""
        self.rpc_https: str = "https://api.mainnet-beta.solana.com"
        self.jup_api: str = "https://api.jup.ag/swap/v2"
        self.primary_mint: str = ""
        self.primary_mint_symbol: str = ""
        self.sol_mint: str = "So11111111111111111111111111111111111111112"
        self.secondary_mints: List[str] = []
        self.secondary_mint_symbols: List[str] = []
        self.price_update_seconds: int = 60
        self.trading_interval_minutes: int = 1
        self.max_slippage: int = 50
        self.strategy: str = "default"
        self.path = os.path.join(os.path.dirname(__file__), "..", "config.json")
        self._client: AsyncClient | None = None
        self._decimals_cache: Dict[str, int] = {}

        # Whale Tracker (ENABLED by default)
        self.whale_tracking_enabled: bool = True
        self.whale_wallets: Dict[str, List[str]] = {}
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

    def load_config(self):
        default_config: Dict[str, Any] = {
            "api_key": "",
            "jupiter_api_key": "",
            "private_key": "",
            "rpc_https": "https://api.mainnet-beta.solana.com",
            "jup_api": "https://api.jup.ag/swap/v2",
            "primary_mint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
            "primary_mint_symbol": "USDC",
            "secondary_mints": ["So11111111111111111111111111111111111111112"],
            "secondary_mint_symbols": ["SOL"],
            "price_update_seconds": 60,
            "trading_interval_minutes": 1,
            "max_slippage": 50,
            "strategy": "default",

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

        with open(self.path, "r") as file:
            try:
                config_data: Dict[str, Any] = json.load(file)
            except json.JSONDecodeError as e:
                raise ValueError(f"Error loading config: {e}") from e

        for key, fallback in default_config.items():
            value = config_data.get(key, fallback)
            if value in ("", None):
                value = fallback
            setattr(self, key, value)
        
        self._validate_config()
    
    def _validate_config(self):
        """Validate that critical configuration fields are properly set."""
        if not self.private_key or self.private_key == "":
            log_general.warning("Private key is not set in config.json. Bot cannot trade.")
        
        if not self.api_key or self.api_key == "":
            log_general.warning("CryptoCompare API key is not set in config.json. Price data unavailable.")
        
        if not self.jupiter_api_key or self.jupiter_api_key == "":
            log_general.warning("Jupiter API key is not set in config.json. Required for api.jup.ag endpoint.")
        
        if not self.rpc_https:
            log_general.error("RPC endpoint is not set in config.json.")
            
        if not self.jup_api:
            log_general.error("Jupiter API endpoint is not set in config.json.")

    def decimals(self, mint_address: str) -> int:
        """Get token decimals with caching to avoid repeated RPC calls."""
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
        try:
            b58_string = self.private_key
            keypair = Keypair.from_base58_string(b58_string)
            # print(f"Using Wallet: {keypair.pubkey()}")

            return keypair
        except Exception as e:
            log_general.error(f"Error decoding private key: {e}")
            exit(1)

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
