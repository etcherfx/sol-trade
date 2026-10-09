"""Budgeted Solana RPC access.

Every Solana RPC call goes through :class:`BudgetedClient`, a proxy around the
solana-py ``AsyncClient`` that ``config().client`` returns. Each call:

- waits on one shared token bucket sized in compute units per second
  (``rpc.max_cu_per_second``), using per-method CU weights;
- is refused with :class:`RpcBudgetExceeded` once the UTC day's CU passes
  ``rpc.daily_cu_budget``, unless it runs inside :func:`essential` (the balance
  reads a swap needs);
- retries HTTP 429 with exponential backoff that honors ``Retry-After``;
- is counted per method in ``data/rpc_usage.json`` and in process counters.
"""

import asyncio
import contextlib
import json
import threading
import time
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from typing import Any

from sol_trade.log import log_general
from sol_trade.utils import write_json_atomic

USAGE_PATH = "data/rpc_usage.json"
_USAGE_DAYS_KEPT = 31

# Alchemy's published CU per Solana method (alchemy.com/docs/reference/compute-unit-costs).
DEFAULT_METHOD_CU: dict[str, int] = {
    "getAccountInfo": 10,
    "getBalance": 10,
    "getMinimumBalanceForRentExemption": 10,
    "getTokenAccountsByOwner": 10,
    "getLatestBlockhash": 20,
    "getMultipleAccounts": 20,
    "getProgramAccounts": 20,
    "getSignatureStatuses": 20,
    "getTokenAccountBalance": 20,
    "getTokenLargestAccounts": 20,
    "getTokenSupply": 20,
    "sendTransaction": 20,
    "simulateTransaction": 20,
    "getSignaturesForAddress": 40,
    "getTransaction": 40,
}
# Unlisted methods are charged at Alchemy's heavier common tier.
_FALLBACK_CU = 40
_MAX_ATTEMPTS = 4
_MAX_BACKOFF_SECONDS = 30.0


class RpcBudgetExceeded(RuntimeError):
    """The daily CU budget is spent and the call is not essential."""


def rpc_method_name(attr: str) -> str:
    """solana-py method name to the RPC method name (``get_balance`` -> ``getBalance``)."""
    name = attr.removesuffix("_json_parsed")
    head, *rest = name.split("_")
    return head + "".join(part.capitalize() for part in rest)


def method_cu(method: str) -> int:
    from sol_trade.config import config

    overrides = config().rpc.get("method_cu") or {}
    if method in overrides:
        return int(overrides[method])
    return DEFAULT_METHOD_CU.get(method, _FALLBACK_CU)


class TokenBucket:
    """CU/s token bucket. ``reserve`` returns how long the caller must wait.

    Reservations may push the balance negative, so concurrent callers queue
    in order instead of racing for the next refill.
    """

    def __init__(
        self, rate: float, capacity: float | None = None, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.rate = float(rate)
        self.capacity = float(capacity if capacity is not None else rate)
        self._clock = clock
        self._tokens = self.capacity
        self._updated = clock()
        self._lock = threading.Lock()

    def set_rate(self, rate: float) -> None:
        with self._lock:
            self.rate = self.capacity = max(float(rate), 0.001)
            self._tokens = min(self._tokens, self.capacity)

    def reserve(self, cost: float) -> float:
        with self._lock:
            now = self._clock()
            self._tokens = min(
                self.capacity, self._tokens + (now - self._updated) * self.rate
            )
            self._updated = now
            self._tokens -= cost
            return 0.0 if self._tokens >= 0 else -self._tokens / self.rate


class UsageCounter:
    """Per-method daily CU counters persisted as JSON, plus process totals."""

    def __init__(self, path: str = USAGE_PATH, clock: Callable[[], float] = time.time) -> None:
        self.path = path
        self._clock = clock
        self._lock = threading.Lock()
        self.process_calls = 0
        self.process_cu = 0

    def _day(self) -> str:
        return datetime.fromtimestamp(self._clock(), UTC).strftime("%Y-%m-%d")

    def load(self) -> dict[str, Any]:
        try:
            with open(self.path) as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def today(self) -> dict[str, Any]:
        """Today's counters: ``{"calls", "cu", "methods": {name: {"calls", "cu"}}}``."""
        return self.load().get(self._day(), {"calls": 0, "cu": 0, "methods": {}})

    def record(self, method: str, cu: int) -> None:
        with self._lock:
            self.process_calls += 1
            self.process_cu += cu
            data = self.load()
            day = data.setdefault(self._day(), {"calls": 0, "cu": 0, "methods": {}})
            day["calls"] += 1
            day["cu"] += cu
            entry = day["methods"].setdefault(method, {"calls": 0, "cu": 0})
            entry["calls"] += 1
            entry["cu"] += cu
            for old in sorted(data)[:-_USAGE_DAYS_KEPT]:
                del data[old]
            try:
                write_json_atomic(self.path, data)
            except OSError as e:
                log_general.debug(f"rpc usage file not written: {e}")

    def snapshot(self) -> tuple[int, int]:
        """Process-lifetime ``(calls, cu)``; diff two snapshots to measure a span."""
        with self._lock:
            return self.process_calls, self.process_cu


_bucket = TokenBucket(30.0)
usage = UsageCounter()
_local = threading.local()
_budget_warned_day = ""


@contextlib.contextmanager
def essential() -> Iterator[None]:
    """Mark RPC calls in this block as needed for a swap (exempt from the daily budget)."""
    previous = getattr(_local, "essential", False)
    _local.essential = True
    try:
        yield
    finally:
        _local.essential = previous


def _check_budget(method: str, cu: int) -> None:
    global _budget_warned_day
    from sol_trade.config import config

    if getattr(_local, "essential", False):
        return
    budget = int(config().rpc.get("daily_cu_budget") or 0)
    if budget <= 0:
        return
    spent = int(usage.today().get("cu", 0))
    if spent + cu > budget:
        day = usage._day()
        if _budget_warned_day != day:
            _budget_warned_day = day
            log_general.warning(
                f"RPC daily budget reached ({spent}/{budget} CU); "
                "only balance reads for swaps run until UTC midnight"
            )
        raise RpcBudgetExceeded(f"daily RPC budget spent; {method} skipped")


def _retry_after(error: BaseException) -> float | None:
    """Seconds to wait when ``error`` is an HTTP 429, else None (0 = unknown)."""
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        response = getattr(current, "response", None)
        status = getattr(response, "status_code", None)
        if status == 429:
            header = response.headers.get("Retry-After") if response is not None else None
            try:
                return max(float(header), 0.0) if header else 0.0
            except ValueError:
                return 0.0
        current = current.__cause__ or current.__context__
    text = str(error).lower()
    if "429" in text or "too many requests" in text:
        return 0.0
    return None


async def _guarded_call(
    fn: Callable[..., Any], method: str, cu: int, args: tuple, kwargs: dict
) -> Any:
    from sol_trade.config import config

    for attempt in range(_MAX_ATTEMPTS):
        _bucket.set_rate(float(config().rpc.get("max_cu_per_second") or 30.0))
        delay = _bucket.reserve(cu)
        if delay > 0:
            await asyncio.sleep(delay)
        usage.record(method, cu)
        try:
            return await fn(*args, **kwargs)
        except Exception as e:
            wait = _retry_after(e)
            if wait is None or attempt == _MAX_ATTEMPTS - 1:
                raise
            wait = wait or min(2.0**attempt, _MAX_BACKOFF_SECONDS)
            log_general.warning(f"RPC {method} rate-limited (429); retrying in {wait:.1f}s")
            await asyncio.sleep(wait)
    raise RuntimeError("unreachable")


class BudgetedClient:
    """Proxy around ``AsyncClient`` that routes every RPC method through the budget."""

    _PASSTHROUGH = frozenset({"close", "is_connected"})

    def __init__(self, endpoint: str, factory: Callable[[str], Any] | None = None) -> None:
        self._endpoint = endpoint
        self._factory = factory
        self._inner: Any = None

    def _client(self) -> Any:
        if self._inner is None:
            if self._factory is None:
                from solana.rpc.async_api import AsyncClient

                self._factory = AsyncClient
            self._inner = self._factory(self._endpoint)
        return self._inner

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._client(), name)
        if name.startswith("_") or name in self._PASSTHROUGH or not callable(attr):
            return attr
        method = rpc_method_name(name)

        def call(*args: Any, **kwargs: Any) -> Any:
            cu = method_cu(method)
            _check_budget(method, cu)
            return _guarded_call(attr, method, cu, args, kwargs)

        return call
