"""The Fear & Greed index: storage, no look-ahead, and the indicator in backtests."""

import json
from datetime import datetime
from pathlib import Path

import httpx
import numpy as np
import pytest

from sol_trade import backtest, cli, fear_greed, spec
from sol_trade.config import config
from tests.test_evaluation import _market

# Real API entries for 2024-10-23..28; the API has no value for 2024-10-26.
SAMPLE = json.loads((Path(__file__).parent / "fixtures" / "fear_greed.json").read_text(encoding="utf-8"))
URL = "https://fng.example/fng/?limit=0&format=json"


def _day(month_day: str) -> int:
    return int(datetime.fromisoformat(f"2024-{month_day}T00:00:00+00:00").timestamp())


@pytest.fixture(autouse=True)
def _index(monkeypatch):
    monkeypatch.setattr(fear_greed, "_next_refresh", 0.0)
    monkeypatch.setitem(config().sentiment, "fear_greed_url", URL)
    monkeypatch.setitem(config().sentiment, "fear_greed_refresh_hours", 4)


def _client(payload=SAMPLE, status=200, seen=None):
    def answer(request):
        if seen is not None:
            seen.append(str(request.url))
        return httpx.Response(status, json=payload)

    return httpx.Client(transport=httpx.MockTransport(answer))


def test_backfill_is_idempotent_and_reports_missing_days():
    first = fear_greed.backfill(_client(), now=1.0)
    assert first["url"] == URL
    assert (first["received"], first["inserted"], first["updated"]) == (5, 5, 0)
    assert (first["days"], first["first"], first["last"], first["missing_days"]) == (5, "2024-10-23", "2024-10-28", 1)
    again = fear_greed.backfill(_client(), now=2.0)
    assert (again["inserted"], again["updated"]) == (0, 0)
    revised = json.loads(json.dumps(SAMPLE))
    revised["data"][0]["value"] = "50"
    assert fear_greed.backfill(_client(revised), now=3.0)["updated"] == 1


def test_a_day_is_known_only_after_it_ends():
    fear_greed.store(fear_greed.parse(SAMPLE))
    d = _day("10-25")  # 72 on the 25th, 69 on the 24th
    closes = np.array([
        _day("10-23") + 43200,  # the 22nd is before the first value
        _day("10-24"),  # the 23rd just ended
        d,  # the 25th just began: the 24th's value
        d + 86400 - 1,  # the 25th's last second: still the 24th's
        d + 86400,  # the 25th just ended
        _day("10-27") + 3600,  # the 26th is missing: not the 25th's value
        _day("10-28"),
        _day("10-30"),  # after the last value
        np.nan,
    ])
    out = fear_greed.values(closes)
    expected = [np.nan, 71, 69, 69, 72, np.nan, 74, np.nan, np.nan]
    np.testing.assert_array_equal(out, np.array(expected, dtype=float))
    assert fear_greed.values(closes.reshape(3, 3)).shape == (3, 3)


def test_refresh_is_paced_and_can_be_turned_off():
    seen: list[str] = []
    client = _client(seen=seen)
    assert fear_greed.refresh(now=1000.0, client=client) is not None
    assert fear_greed.refresh(now=1000.0 + 3 * 3600, client=client) is None
    assert fear_greed.refresh(now=1000.0 + 4 * 3600, client=client) is not None
    assert seen == [URL, URL]
    failing = _client(status=503)
    with pytest.raises(httpx.HTTPStatusError):
        fear_greed.refresh(now=100_000.0, client=failing)
    assert fear_greed.refresh(now=100_000.0 + 600, client=failing) is None  # retried after 15 minutes
    config().sentiment["fear_greed_url"] = ""
    assert fear_greed.refresh(now=1e9, client=client) is None


def test_bad_responses_raise():
    with pytest.raises(ValueError, match="no data"):
        fear_greed.parse({"metadata": {"error": None}})
    with pytest.raises(ValueError, match="API error"):
        fear_greed.parse({"data": [], "metadata": {"error": "down"}})
    assert fear_greed.parse({"data": [{"value": "x", "timestamp": "1"}, {"value": "101", "timestamp": "86400"}]}) == []


def test_the_indicator_reads_the_index_in_backtests():
    fear_greed.store(fear_greed.parse(SAMPLE))
    frame = _market(24 * 7, t0=_day("10-23"))
    strategy = spec.compile_spec({
        "schema": 1, "name": "crowd", "timeframe": "1h", "lookback": 2,
        "indicators": {"fg": {"fn": "fear_greed"}},
        "entry": ["fg", ">=", 72],
    }, "crowd")
    assert strategy.uses_fear_greed and not strategy.uses_sentiment
    entry, _ = backtest.compute_signals(strategy, frame)
    known = fear_greed.values(frame["ts"].to_numpy(dtype=float) + 3600)
    np.testing.assert_array_equal(entry[1:], (known >= 72)[1:])
    assert entry.any() and not entry[: 24 * 3 - 1].any()  # nothing reaches 72 before the 25th has ended
    unwired = strategy.evaluate({c: frame[c].to_numpy()[-2:, None] for c in spec.COLUMNS},
                                frame["ts"].to_numpy(dtype=float)[-2:, None])
    assert np.isnan(unwired.values["fg"][0])


def test_the_indicator_takes_no_parameters():
    with pytest.raises(spec.SpecError, match="indicators.fg.period"):
        spec.compile_spec({
            "schema": 1, "name": "x", "timeframe": "1h", "lookback": 2,
            "indicators": {"fg": {"fn": "fear_greed", "period": 3}}, "entry": ["fg", ">", 50],
        })


def test_cli_backfill_fear_greed(monkeypatch):
    monkeypatch.setattr(cli, "setup_file_logging", lambda: None)
    monkeypatch.setattr(fear_greed.httpx, "get", lambda url, **_: _client().get(url))
    lines = []
    assert cli.run(["backfill", "--fear-greed"], out=lines.append) == 0
    payload = json.loads(lines[0])
    assert payload["fear_greed"]["inserted"] == 5 and payload["fear_greed"]["missing_days"] == 1
    config().sentiment["fear_greed_url"] = ""
    assert cli.run(["backfill", "--fear-greed"], out=lines.append) == 1
    assert json.loads(lines[1])["error"] == "fear_greed_unavailable"
