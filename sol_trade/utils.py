import asyncio
import threading
import time
from concurrent.futures import Future
from functools import wraps
from typing import Any

from solana.exceptions import SolanaRpcException

from sol_trade.log import log_general

# Shared background event loop that runs the async Solana RPC client.
# The application itself is synchronous, and asyncio.run() cannot be used
# because RPC calls also happen from inside an already-running event loop
# (e.g. trading.py's asyncio.run(perform_swap(...)) -> config().decimals()).
_async_loop_holder: dict[str, asyncio.AbstractEventLoop] = {}
_async_loop_ready = threading.Event()


def _run_event_loop() -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    _async_loop_holder["loop"] = loop
    _async_loop_ready.set()
    loop.run_forever()


threading.Thread(
    target=_run_event_loop, daemon=True, name="solana-async-loop"
).start()
_async_loop_ready.wait()


def run_async(coro: Any) -> Any:
    """Run a coroutine on the shared background loop and block for its result.

    Safe to call from synchronous code and from inside another running event
    loop. The AsyncClient stays bound to this single loop.
    """
    loop = _async_loop_holder["loop"]
    future: Future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result()


def handle_rate_limiting(retry_attempts=3, retry_delay=10):
    def decorator(client_function):
        @wraps(client_function)
        def wrapper(*args, **kwargs):
            for _ in range(retry_attempts):
                try:
                    return client_function(*args, **kwargs)
                except SolanaRpcException as e:
                    if 'HTTPStatusError' in e.error_msg:
                        log_general.warning(
                            f"Rate limit exceeded in {client_function.__name__}, retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        raise
            log_general.warning("Rate limit error persisting, skipping this iteration.")
            return None

        return wrapper

    return decorator
