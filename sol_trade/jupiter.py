"""Jupiter HTTP helpers: USD prices, token metadata and quote-only orders.

None of these sign anything or touch the Solana RPC.
"""

import time
from typing import Any, cast

import httpx

from sol_trade.config import config
from sol_trade.log import log_general
from sol_trade.utils import load_json_data, write_json_atomic

PRICE_API = "https://lite-api.jup.ag/price/v3"
TOKENS_API = "https://lite-api.jup.ag/tokens/v2/search"
META_CACHE_PATH = "data/token_meta.json"
_META_MAX_AGE_SECONDS = 7 * 86400

_http_session = httpx.Client(timeout=10.0)


def _headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if config().jupiter_api_key:
        headers["x-api-key"] = config().jupiter_api_key
    return headers


def fetch_prices(mints: list[str]) -> dict[str, float]:
    """USD prices for mints with one HTTP call; unknown or failed prices are 0."""
    if not mints:
        return {}

    unique_mints = list(dict.fromkeys(mints))  # preserve order
    params = {"ids": ",".join(unique_mints)}

    try:
        response = _http_session.get(PRICE_API, params=params)
        response.raise_for_status()
        response_json = cast(dict[str, Any], response.json())
    except httpx.HTTPStatusError as e:
        if e.response is not None and e.response.status_code == 401:
            log_general.error(
                "401 Unauthorized fetching prices from the lite API; "
                "check the Jupiter API key and plan"
            )
        else:
            log_general.error(f"HTTP error fetching prices for {unique_mints}: {e}")
        return dict.fromkeys(unique_mints, 0.0)
    except Exception as e:  # noqa: BLE001 - network errors
        log_general.error(f"failed to fetch prices for {unique_mints}: {e}")
        return dict.fromkeys(unique_mints, 0.0)

    prices: dict[str, float] = {}
    for mint in unique_mints:
        mint_data = cast(dict[str, Any], response_json.get(mint, {}) or {})
        price = float(mint_data.get("usdPrice") or 0)
        if price == 0:
            log_general.debug(f"price for {mint} missing from response; defaulting to 0")
        prices[mint] = price
    return prices


def token_meta(mint: str, refresh: bool = False) -> dict[str, Any] | None:
    """``{"symbol", "name", "decimals", "fetched_at"}`` for a mint from Jupiter's token API.

    Cached in ``data/token_meta.json`` for a week. Returns None when Jupiter
    doesn't know the mint or can't be reached (a stale cached entry is
    returned in that case).
    """
    cache = load_json_data(META_CACHE_PATH, {})
    cached = cache.get(mint) if isinstance(cache, dict) else None
    if cached and not refresh and time.time() - cached.get("fetched_at", 0) < _META_MAX_AGE_SECONDS:
        return cached
    try:
        response = _http_session.get(TOKENS_API, params={"query": mint}, headers=_headers())
        response.raise_for_status()
        results = response.json()
    except Exception as e:  # noqa: BLE001 - network errors; fall back to the cache
        log_general.warning(f"token metadata lookup failed for {mint}: {e}")
        return cached
    match = next((r for r in results or [] if isinstance(r, dict) and r.get("id") == mint), None)
    if match is None or match.get("decimals") is None:
        return None
    meta = {
        "symbol": match.get("symbol"),
        "name": match.get("name"),
        "decimals": int(match["decimals"]),
        "fetched_at": time.time(),
    }
    if not isinstance(cache, dict):
        cache = {}
    cache[mint] = meta
    try:
        write_json_atomic(META_CACHE_PATH, cache)
    except OSError as e:
        log_general.debug(f"token metadata cache not written: {e}")
    return meta


def decimals(mint: str) -> int:
    """Decimals count for a mint: Jupiter's token API first, the RPC as fallback."""
    meta = token_meta(mint)
    if meta is not None:
        return int(meta["decimals"])
    multiplier = config().decimals(mint)  # 10 ** decimals, via the budgeted RPC
    return len(str(multiplier)) - 1


def quote(input_mint: str, output_mint: str, amount: int) -> dict[str, Any]:
    """A quote-only ``/order`` (no ``taker``): nothing is built, signed or sent."""
    response = _http_session.get(
        f"{config().jup_api}/order",
        params={"inputMint": input_mint, "outputMint": output_mint, "amount": amount},
        headers=_headers(),
        timeout=30.0,
    )
    response.raise_for_status()
    return cast(dict[str, Any], response.json())
