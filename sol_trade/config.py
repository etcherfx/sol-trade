import copy
import json
import os
from typing import Any

from dotenv import load_dotenv
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

# Nested settings sections: deep-merged over these defaults, so a config.json
# that sets one key of a section keeps the defaults for the rest.
SECTION_DEFAULTS: dict[str, dict[str, Any]] = {
    "risk": {
        # Total primary-token capital the live book may deploy, and per trade.
        "max_live_capital": 500.0,
        "max_trade_size": 100.0,
        # Drawdown (percent from the high-water mark of max_live_capital) that trips the kill switch;
        # the backtest, held-out and paper gates measure on the same capital.
        "max_drawdown_pct": 15.0,
        # Positions worth less than this (USD) are written off instead of sold.
        "dust_threshold_usd": 1.0,
    },
    "rpc": {
        # Ceiling for the shared Solana RPC limiter, in compute units per second.
        "max_cu_per_second": 30.0,
        # Past this many CU in a UTC day only the balance reads swaps need still run.
        "daily_cu_budget": 600000,
        # Per-method CU overrides, e.g. {"getBalance": 10}; merged over the built-in table.
        "method_cu": {},
        # Display-only wallet holdings view; 0 turns it off.
        "holdings_refresh_seconds": 600,
        # How long a wallet balance read is reused before it's read again.
        "balance_refresh_seconds": 900,
        # How long the computed SOL fee reserve is reused.
        "sol_reserve_ttl_seconds": 3600,
    },
    "sentiment": {
        # Subreddits the recorder reads (new posts and comments); empty means nothing is read.
        "subreddits": [],
        "record": True,
        # Reddit allows anonymous feed readers about one request a minute.
        "request_interval_seconds": 65,
        "user_agent": "sol-trade sentiment recorder (github.com/etcherfx/sol-trade; one request a minute)",
        # Words that count as a mention, per symbol; the symbol itself by default.
        "keywords": {},
        "db_path": "data/sentiment.db",
        # Daily Crypto Fear & Greed index (the fear_greed indicator); empty turns the live refresh off.
        "fear_greed_url": "https://api.alternative.me/fng/?limit=0&format=json",
        "fear_greed_refresh_hours": 4,
    },
    "costs": {
        # Per-side cost assumed for tokens without measured quotes.
        "default_cost_bps": 50.0,
        # Measured costs are multiplied by this before use.
        "safety_multiplier": 1.25,
        # Trade sizes (USD) probed with Jupiter quotes.
        "probe_sizes_usd": [25, 100, 250],
        # Probes older than this don't count as fresh evidence.
        "max_sample_age_days": 14,
        # Gap between Jupiter requests while measuring: keyless access allows 0.5 a second
        # (1.0 suits a free API key). HTTP 429s are retried with doubling waits.
        "request_interval_seconds": 2.1,
        "rate_limit_retries": 4,
        "rate_limit_backoff_seconds": 5.0,
    },
    "backfill": {
        "default_timeframes": ["1h"],
        "default_days": 730,
        # Bars requested per exchange call.
        "page_limit": 100,
    },
    "gates": {
        # The newest days of history, held out until a strategy is promoted.
        "holdout_days": 60,
        # Sequential folds the selection period's equity curve is split into.
        "folds": 4,
        "min_positive_folds": 3,
        # Fewer trades than this is insufficient evidence, not a pass.
        "min_trades": 30,
        # Probability that the Sharpe ratio beats the best of N random trials.
        "min_deflated_sharpe": 0.95,
        # Closed held-out trades needed to judge the held-out period; slower strategies need
        # what their selection-period trade rate predicts for holdout_days (at least 1).
        "holdout_min_trades": 8,
        # Share of missing bars above which the data is flagged DATA_GAPS.
        "max_gap_fraction": 0.02,
    },
    "promotion": {
        # Paper -> live waits for a human `approve` when true.
        "require_approval": False,
        "max_paper_candidates": 10,
        # Paper evidence needed before going live.
        "min_paper_days": 14,
        "min_paper_coverage": 0.9,
        "min_paper_trades": 2,
        # Paper return must stay within max(points, relative share) of a backtest replay.
        "replay_tolerance_points": 1.0,
        "replay_tolerance_relative": 0.25,
        # Paper candidates that don't qualify within this many days are retired.
        "max_paper_days": 60,
        # Live demotion: the strategy's own drawdown, a losing recent window, or costs off model.
        "demote_drawdown_pct": 10.0,
        "demote_window_days": 60,
        "demote_min_trades": 10,
        "max_cost_divergence_bps": 50.0,
        "divergence_fills": 5,
    },
}


def _merge_section(default: dict[str, Any], value: Any) -> dict[str, Any]:
    """Overlay a config.json section on its defaults (unknown keys are kept)."""
    merged = copy.deepcopy(default)
    if isinstance(value, dict):
        merged.update(value)
    return merged


def _file_mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


class Config:
    def __init__(
        self, config_path: str | None = None, dotenv_path: str | None = None
    ) -> None:
        """Load settings.

        Paths default to ``SOLTRADE_CONFIG_PATH`` / ``SOLTRADE_DOTENV_PATH`` when
        set, else config.json and .env in the project root.
        """
        root = os.path.join(os.path.dirname(__file__), "..")
        self.jupiter_api_key: str = ""
        self.private_key: str = ""
        self.rpc_https: str = "https://api.mainnet-beta.solana.com"
        self.jup_api: str = "https://api.jup.ag/swap/v2"
        self.primary_mint: str = ""
        self.primary_mint_symbol: str = ""
        # Native SOL mint, a protocol constant (wrapped SOL uses the same address).
        self.sol_mint: str = "So11111111111111111111111111111111111111112"
        self.secondary_mints: list[str] = []
        self.secondary_mint_symbols: list[str] = []
        # Portfolio weights per secondary token (parallel to secondary_mints);
        # empty list means equal split. Normalized to sum to 1 on load.
        self.secondary_weights: list[float] = []
        # Per-token strategy overrides, keyed by symbol; missing -> global strategy.
        self.token_strategies: dict[str, str] = {}
        # Per-token candle sources, keyed by symbol: an exchange id, or
        # {"exchange": ..., "quote": ...}; missing -> data_exchange and the primary token.
        self.token_exchanges: dict[str, str | dict[str, str]] = {}
        # Known mints by symbol beyond the traded set, for lookups and candidates.
        self.token_mints: dict[str, str] = {}
        self.price_update_seconds: int = 60
        self.max_slippage: int = 50
        self.strategy: str = "default"
        self.data_exchange: str = "okx"
        self.candles_path: str = "data/candles.db"
        # Runtime folder of JSON strategies, and the research/positions database.
        self.strategies_path: str = "strategies"
        self.research_db_path: str = "data/research.db"
        for section, defaults in SECTION_DEFAULTS.items():
            setattr(self, section, copy.deepcopy(defaults))
        self.path = config_path or os.getenv("SOLTRADE_CONFIG_PATH") or os.path.join(
            root, "config.json"
        )
        self.dotenv_path = dotenv_path or os.getenv("SOLTRADE_DOTENV_PATH") or os.path.join(
            root, ".env"
        )
        load_dotenv(self.dotenv_path)
        self._client: Any = None
        self._decimals_cache: dict[str, int] = {}

        # Whale Tracker (disabled — opt-in addon)
        self.whale_tracking_enabled: bool = False
        self.whale_wallets: dict[str, list[str]] = {}
        self.whale_poll_interval_minutes: int = 5
        self.whale_data_path: str = "data/whale_data.json"

        # Confluence Filter (disabled — opt-in addon)
        self.confluence_enabled: bool = False

        # Market Regime (disabled — opt-in addon); the pair is "BASE/QUOTE"
        self.market_regime_enabled: bool = False
        self.market_regime_pair: str = ""
        self.regime_data_path: str = "data/regime_data.json"

        # Sentiment Circuit Breaker (disabled — opt-in addon)
        self.sentiment_enabled: bool = False
        self.sentiment_pause_hours: int = 4
        self.sentiment_threshold: float = -0.5
        self.sentiment_crash_threshold: float = -0.7
        self.sentiment_data_path: str = "data/sentiment_data.json"
        self.load_config()
        self._config_mtime: float = _file_mtime(self.path)

    # Section attributes, typed for readers (values come from SECTION_DEFAULTS).
    risk: dict[str, Any]
    rpc: dict[str, Any]
    sentiment: dict[str, Any]
    costs: dict[str, Any]
    backfill: dict[str, Any]
    gates: dict[str, Any]
    promotion: dict[str, Any]

    def _apply_config_file(self) -> None:
        default_config: dict[str, Any] = {
            "jupiter_api_key": "",
            "private_key": "",
            "rpc_https": "https://api.mainnet-beta.solana.com",
            "jup_api": "https://api.jup.ag/swap/v2",
            "primary_mint": "",
            "primary_mint_symbol": "",
            "secondary_mints": [],
            "secondary_mint_symbols": [],
            "secondary_weights": [],
            "token_strategies": {},
            "token_exchanges": {},
            "token_mints": {},
            "price_update_seconds": 60,
            "max_slippage": 50,
            "strategy": "default",
            "data_exchange": "okx",
            "candles_path": "data/candles.db",
            "strategies_path": "strategies",
            "research_db_path": "data/research.db",

            # Whale Tracker
            "whale_tracking_enabled": False,
            "whale_wallets": {},
            "whale_poll_interval_minutes": 5,
            "whale_data_path": "data/whale_data.json",

            # Confluence Filter
            "confluence_enabled": False,

            # Market Regime
            "market_regime_enabled": False,
            "market_regime_pair": "",
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
        if not isinstance(config_data, dict):
            raise ValueError("Error loading config: the top level must be a JSON object")  # noqa: TRY004 - callers handle ValueError

        for key, fallback in default_config.items():
            value = config_data.get(key, fallback)
            if value in ("", None):
                value = fallback
            setattr(self, key, value)
        for section, defaults in SECTION_DEFAULTS.items():
            setattr(self, section, _merge_section(defaults, config_data.get(section)))

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

        for attr in ("token_strategies", "token_exchanges", "token_mints"):
            if not isinstance(getattr(self, attr), dict):
                setattr(self, attr, {})

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

        Structural keys (primary token, RPC, exchange) and credentials are
        pinned — changing those still requires a restart. A malformed or
        invalid token set keeps the previous configuration.
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
        if not self.primary_mint or not self.primary_mint_symbol:
            raise ValueError(
                "primary_mint and primary_mint_symbol must be set in config.json "
                "(the token you pay with; see config.json.sample)"
            )
        # Token-set shape errors raise so a malformed hot-reload keeps the
        # previous configuration (reload_config is wrapped in maybe_reload_config).
        # An empty token set is valid: the bot then stays flat.
        if not isinstance(self.secondary_mints, list) or not isinstance(
            self.secondary_mint_symbols, list
        ):
            raise ValueError("secondary_mints and secondary_mint_symbols must be lists")  # noqa: TRY004 - reload rolls back on ValueError
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
        for symbol, entry in self.token_exchanges.items():
            if isinstance(entry, str) and entry:
                continue
            if (
                not isinstance(entry, dict)
                or set(entry) - {"exchange", "quote"}
                or not all(isinstance(v, str) and v for v in entry.values())
            ):
                raise ValueError(
                    f"token_exchanges.{symbol} must be an exchange id or "
                    '{"exchange": "<ccxt id>", "quote": "<symbol>"} with non-empty strings'
                )
        unknown_exchanges = [
            symbol for symbol in self.token_exchanges
            if symbol not in self.secondary_mint_symbols and symbol not in self.token_mints
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

    def mint_for_symbol(self, symbol: str) -> str | None:
        """Mint address for a configured symbol: traded tokens, primary, then token_mints."""
        if symbol in self.secondary_mint_symbols:
            return self.secondary_mints[self.secondary_mint_symbols.index(symbol)]
        if symbol == self.primary_mint_symbol:
            return self.primary_mint
        return self.token_mints.get(symbol)

    def symbol_for_mint(self, mint: str) -> str | None:
        """Configured symbol for a mint, or None when no setting names it."""
        if mint in self.secondary_mints:
            return self.secondary_mint_symbols[self.secondary_mints.index(mint)]
        if mint == self.primary_mint:
            return self.primary_mint_symbol
        for symbol, known in self.token_mints.items():
            if known == mint:
                return symbol
        return None

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
    def client(self) -> Any:
        """Cached RPC client; every call goes through the shared CU budget."""
        if self._client is None:
            from sol_trade.rpc import BudgetedClient

            self._client = BudgetedClient(self.rpc_https)
        return self._client


_config_instance = None


def config() -> Config:
    """Singleton pattern to ensure only one Config instance exists."""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance
