"""The bot's heartbeat: ``data/status.json``, rewritten every trading cycle.

``status`` and the container healthcheck read it to tell whether the bot is
running and which mode it runs in.
"""

import os
import time
from datetime import UTC, datetime
from typing import Any

from sol_trade.utils import load_json_data, write_json_atomic

HEARTBEAT_PATH = "data/status.json"


def write(mode: str, interval_seconds: float, **fields: Any) -> None:
    now = time.time()
    write_json_atomic(
        HEARTBEAT_PATH,
        {
            "pid": os.getpid(),
            "mode": mode,
            "at": now,
            "at_iso": datetime.fromtimestamp(now, UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "interval_seconds": interval_seconds,
            **fields,
        },
    )


def read(now: float | None = None) -> dict[str, Any] | None:
    """The last heartbeat plus its age in seconds and whether it is fresh."""
    beat = load_json_data(HEARTBEAT_PATH, None)
    if not isinstance(beat, dict) or "at" not in beat:
        return None
    now = time.time() if now is None else now
    age = now - float(beat["at"])
    # A cycle may also wait out network timeouts; three intervals is "stopped".
    fresh = age <= 3 * float(beat.get("interval_seconds") or 60) + 60
    return {**beat, "age_seconds": round(age, 1), "fresh": fresh}
