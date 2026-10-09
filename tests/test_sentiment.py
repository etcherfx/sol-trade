"""The Reddit recorder, point-in-time features and the breaker that reads them."""

import json
from datetime import UTC, datetime

import httpx
import numpy as np
import pytest

from sol_trade import backtest, sentiment, spec
from sol_trade.config import config
from tests.test_evaluation import _market


def _atom(entries: list[dict]) -> str:
    body = []
    for e in entries:
        published = f"<published>{e['at']}</published>" if e.get("published", True) else ""
        body.append(
            f"""<entry><author><name>/u/{e.get('author', 'someone')}</name></author>
            <content type="html">{e['html']}</content><id>{e['id']}</id>
            <updated>{e['at']}</updated>{published}<title>{e.get('title', '')}</title></entry>"""
        )
    return f'<?xml version="1.0" encoding="UTF-8"?><feed xmlns="http://www.w3.org/2005/Atom">{"".join(body)}</feed>'


POST = {
    "id": "t3_a",
    "at": "2026-10-08T12:00:00+00:00",
    "title": "SOL breaks out, great news",
    "html": "&lt;table&gt;&lt;tr&gt;&lt;td&gt;Huge &lt;b&gt;win&lt;/b&gt;&lt;/td&gt;&lt;/tr&gt;&lt;/table&gt; submitted by /u/x [link] [comments]",
}
COMMENT = {
    "id": "t1_b",
    "at": "2026-10-08T12:05:00+00:00",
    "title": "/u/y on SOL breaks out",
    "html": "&lt;div class=&quot;md&quot;&gt;&lt;p&gt;this is a terrible rug&lt;/p&gt;&lt;/div&gt;",
    "published": False,
}


@pytest.fixture(autouse=True)
def _recorder(monkeypatch):
    monkeypatch.setattr(sentiment, "_next_allowed", 0.0)
    monkeypatch.setattr(sentiment, "_backoff", 0.0)
    monkeypatch.setattr(sentiment, "_cursor", 0)
    monkeypatch.setitem(config().sentiment, "subreddits", ["sub"])
    monkeypatch.setitem(config().sentiment, "keywords", {})


def test_parse_strips_markup_and_skips_comment_titles():
    post = sentiment.parse_feed(_atom([POST]), "sub", "post")[0]
    assert post["text"] == "Huge win"
    assert post["created"] == int(datetime(2026, 10, 8, 12, tzinfo=UTC).timestamp())
    comment = sentiment.parse_feed(_atom([COMMENT]), "sub", "comment")[0]
    assert comment["title"] == ""  # "/u/y on <post title>" would repeat the post's mention
    assert comment["text"] == "this is a terrible rug"
    assert comment["compound"] < 0 < post["compound"]


class _Client(httpx.Client):
    """An httpx client that answers from a script and remembers the URLs it was asked for."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.urls: list[str] = []
        super().__init__(transport=httpx.MockTransport(self._answer))

    def _answer(self, request: httpx.Request) -> httpx.Response:
        self.urls.append(str(request.url))
        status, text, extra = self.responses.pop(0)
        return httpx.Response(status, text=text, headers=extra)


def test_one_request_per_window_round_robin():
    client = _Client([(200, _atom([POST]), {}), (200, _atom([COMMENT]), {})])
    first = sentiment.record(now=1000.0, client=client)
    assert first is not None and first["feed"] == "sub/post" and first["new"] == 1
    assert sentiment.record(now=1010.0, client=client) is None  # within the interval
    second = sentiment.record(now=1070.0, client=client)
    assert second is not None and second["feed"] == "sub/comment"
    assert client.urls[0].endswith("/r/sub/new/.rss?limit=100")
    assert client.urls[1].endswith("/r/sub/comments/.rss?limit=100")


def test_rate_limit_headers_and_429_backoff():
    client = _Client([
        (200, _atom([POST]), {"x-ratelimit-remaining": "0", "x-ratelimit-reset": "300"}),
        (429, "", {"x-ratelimit-reset": "30"}),
        (200, _atom([COMMENT]), {}),
    ])
    sentiment.record(now=0.0, client=client)
    assert sentiment.record(now=200.0, client=client) is None  # waits out the 300 s reset
    limited = sentiment.record(now=302.0, client=client)
    assert limited is not None and limited["status"] == 429
    assert sentiment.record(now=330.0, client=client) is None  # backing off at least 65 s
    retried = sentiment.record(now=400.0, client=client)
    assert retried is not None and retried["feed"] == "sub/comment"  # the same feed again


def test_no_overlap_is_recorded_as_a_gap():
    later = {**POST, "id": "t3_z", "at": "2026-10-08T14:00:00+00:00"}
    client = _Client([(200, _atom([POST]), {}), (200, "", {}), (200, _atom([later]), {})])
    sentiment.record(now=0.0, client=client)
    sentiment.record(now=100.0, client=client)  # the comment feed (empty)
    gap = sentiment.record(now=200.0, client=client)
    assert gap is not None and gap["gap"] is True
    assert sentiment.stats(now=300.0)["last_24h"]["gaps"] == 1


def _seed(items, fetched_at=0.0):
    with sentiment._connect() as conn:
        conn.execute("INSERT INTO fetches (feed, at, status) VALUES ('sub/post', ?, 200)", (fetched_at,))
        for i, (created, text) in enumerate(items):
            conn.execute(
                "INSERT INTO items VALUES (?, 'sub', 'post', 'a', ?, '', ?, ?, ?)",
                (f"t3_{i}", created, text, sentiment.score_text(text), fetched_at),
            )


def test_features_are_point_in_time_and_word_bounded():
    _seed([
        (1000, "SOL is great"),
        (2000, "$sol looks terrible"),
        (3000, "console output"),  # not a mention
        (9000, "SOL later"),
    ])
    closes = np.array([[500.0, 3600.0], [3600.0, 8000.0]])
    out = sentiment.features("SOL", closes, 1.0)
    assert np.isnan(out["mentions"][0, 0])  # before recording started (t - 1h < 0)
    assert out["mentions"][0, 1] == 2 and out["mentions"][1, 0] == 2
    assert out["mentions"][1, 1] == 0 and np.isnan(out["score"][1, 1])
    expected = (sentiment.score_text("SOL is great") + sentiment.score_text("$sol looks terrible")) / 2
    assert out["score"][0, 1] == pytest.approx(expected)


def test_configured_keywords():
    _seed([(1000, "Solana validators"), (1100, "SOL")])
    config().sentiment["keywords"] = {"SOL": ["Solana"]}
    assert sentiment.features("SOL", np.array([4000.0]), 1.0)["mentions"][0] == 1


def test_strategies_read_recorded_features_in_backtests():
    frame = _market(300)
    t0 = float(frame["ts"].iat[0])
    _seed([(t0 + 3600 * k + 10, "TKN rally") for k in range(0, 300, 5)], fetched_at=t0)
    strategy = spec.compile_spec({
        "schema": 1, "name": "buzz", "timeframe": "1h", "lookback": 10,
        "indicators": {"crowd": {"fn": "sentiment", "window_hours": 6}},
        "entry": ["crowd.mentions", ">=", 2],
        "exit": ["crowd.mentions", "<", 1],
    }, "buzz")
    entry, _ = backtest.compute_signals(strategy, frame, "TKN")
    assert entry.any()
    assert not entry[: 6].any()  # the first six hours aren't fully recorded


def test_breaker_reads_recorded_items(monkeypatch, tmp_path):
    import time

    now = time.time()
    _seed([(now - 600, "SOL total rug scam crash")], fetched_at=now - 2 * 86400)
    monkeypatch.setattr(config(), "sentiment_enabled", True)
    monkeypatch.setattr(config(), "sentiment_data_path", str(tmp_path / "breaker.json"))
    sentiment.update_sentiment(["SOL"])
    stored = json.loads((tmp_path / "breaker.json").read_text())
    assert stored["SOL"]["score"] < -0.5
    assert sentiment.is_token_blocked("SOL")
