"""Measured trading costs and the cost model shared by backtests and paper books.

``measure`` asks Jupiter for quote-only orders (no taker, nothing signed, no
RPC) at a few trade sizes in both directions and stores, per sample, the
all-in cost ``(inUsdValue - outUsdValue) / inUsdValue`` (fee, price impact
and spread together), Jupiter's ``feeBps`` and ``priceImpact``, and the
network fees in USD. Results go to ``data/costs.json``.

The model interpolates the all-in basis points linearly between the measured
sizes (flat below the smallest, extrapolated above the largest), multiplies
by ``costs.safety_multiplier`` and adds the network fee per swap. Tokens
without measurements cost ``costs.default_cost_bps`` per side and are
flagged ``COSTS_DEFAULT``.
"""

import itertools
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sol_trade import jupiter
from sol_trade.config import config
from sol_trade.log import log_general
from sol_trade.utils import load_json_data, write_json_atomic

COSTS_PATH = "data/costs.json"
_LAMPORTS_PER_SOL = 1e9


def _sample(response: dict[str, Any], side: str, size_usd: float, sol_usd: float) -> dict[str, Any]:
    in_usd = float(response.get("inUsdValue") or 0)
    out_usd = float(response.get("outUsdValue") or 0)
    lamports = sum(
        int(response.get(key) or 0)
        for key in ("signatureFeeLamports", "prioritizationFeeLamports")
    )
    gasless = bool(response.get("gasless"))
    return {
        "side": side,
        "size_usd": size_usd,
        "in_usd": in_usd,
        "out_usd": out_usd,
        "cost_bps": (in_usd - out_usd) / in_usd * 1e4 if in_usd > 0 else None,
        "fee_bps": response.get("feeBps"),
        "price_impact_pct": response.get("priceImpact"),
        "network_fee_usd": 0.0 if gasless else lamports / _LAMPORTS_PER_SOL * sol_usd,
        "router": response.get("router"),
        "gasless": gasless,
    }


def measure(symbol: str, mint: str, sizes_usd: list[float] | None = None) -> dict[str, Any]:
    """Probe Jupiter quotes for ``symbol`` and store them; returns the stored record."""
    cfg = config()
    sizes = [float(s) for s in (sizes_usd or cfg.costs["probe_sizes_usd"])]
    prices = jupiter.fetch_prices([cfg.primary_mint, mint, cfg.sol_mint])
    primary_usd, token_usd, sol_usd = (prices.get(m, 0.0) for m in (cfg.primary_mint, mint, cfg.sol_mint))
    if primary_usd <= 0 or token_usd <= 0:
        raise ValueError(f"no USD price for {symbol} or {cfg.primary_mint_symbol}; cannot size probes")
    primary_decimals = jupiter.decimals(cfg.primary_mint)
    token_decimals = jupiter.decimals(mint)
    samples = []
    errors = []
    for size in sizes:
        legs = (
            ("buy", cfg.primary_mint, mint, int(size / primary_usd * 10**primary_decimals)),
            ("sell", mint, cfg.primary_mint, int(size / token_usd * 10**token_decimals)),
        )
        for side, input_mint, output_mint, amount in legs:
            try:
                response = jupiter.quote(input_mint, output_mint, amount)
            except Exception as e:  # noqa: BLE001 - record and continue with the other probes
                errors.append(f"{side} ${size:g}: {type(e).__name__}: {e}")
                continue
            if response.get("errorCode") and not response.get("outAmount"):
                errors.append(f"{side} ${size:g}: {response.get('errorMessage') or response.get('errorCode')}")
                continue
            samples.append(_sample(response, side, size, sol_usd))
    record = {
        "symbol": symbol,
        "mint": mint,
        "measured_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "measured_ts": time.time(),
        "samples": samples,
        "errors": errors,
    }
    if samples:
        data = load_json_data(COSTS_PATH, {})
        if not isinstance(data, dict):
            data = {}
        data[symbol] = record
        write_json_atomic(COSTS_PATH, data)
    else:
        log_general.warning(f"cost probes for {symbol} all failed: {errors}")
    return record


def sample_age_days(symbol: str, now: float | None = None) -> float | None:
    """Days since ``symbol``'s last successful probe, or None when never measured."""
    data = load_json_data(COSTS_PATH, {})
    record = data.get(symbol) if isinstance(data, dict) else None
    if not record or not record.get("samples"):
        return None
    return ((time.time() if now is None else now) - float(record.get("measured_ts", 0))) / 86400


@dataclass(frozen=True)
class CostModel:
    """Per-side trading cost for one token."""

    symbol: str
    source: str  # "measured" or "default"
    points: tuple[tuple[float, float], ...]  # (size_usd, bps) after the safety multiplier
    network_fee_usd: float
    measured_at: str | None = None

    def bps(self, notional_usd: float) -> float:
        """Basis points charged on a swap of ``notional_usd``."""
        points = self.points
        if len(points) == 1 or notional_usd <= points[0][0]:
            return points[0][1]
        for (x0, y0), (x1, y1) in itertools.pairwise(points):
            if notional_usd <= x1:
                return y0 + (y1 - y0) * (notional_usd - x0) / (x1 - x0)
        (x0, y0), (x1, y1) = points[-2], points[-1]
        return max(y1 + (y1 - y0) * (notional_usd - x1) / (x1 - x0), 0.0)

    def cost_usd(self, notional_usd: float) -> float:
        """Total cost of one swap: the bps share plus the network fee."""
        return notional_usd * self.bps(notional_usd) / 1e4 + self.network_fee_usd

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "source": self.source,
            "points": [list(p) for p in self.points],
            "network_fee_usd": self.network_fee_usd,
            "measured_at": self.measured_at,
        }


def default_model(symbol: str) -> CostModel:
    bps = float(config().costs["default_cost_bps"])
    return CostModel(symbol, "default", ((0.0, bps),), 0.0)


def model_for(symbol: str) -> CostModel:
    """The cost model for ``symbol`` from stored probes, else the default.

    Each size's cost is the mean of its buy and sell legs, so an offset in the
    USD reference price (which makes one leg look cheaper and the other dearer
    by the same amount) cancels out; a negative mean counts as zero.
    """
    data = load_json_data(COSTS_PATH, {})
    record = (data.get(symbol) if isinstance(data, dict) else None) or {}
    samples = [s for s in record.get("samples", []) if s.get("cost_bps") is not None]
    if not samples:
        return default_model(symbol)
    safety = float(config().costs["safety_multiplier"])
    by_size: dict[float, list[float]] = {}
    for s in samples:
        by_size.setdefault(float(s["size_usd"]), []).append(float(s["cost_bps"]))
    points = tuple(
        (size, max(sum(values) / len(values), 0.0) * safety) for size, values in sorted(by_size.items())
    )
    fees = sorted(float(s.get("network_fee_usd") or 0.0) for s in samples)
    return CostModel(symbol, "measured", points, fees[len(fees) // 2], record.get("measured_at"))
