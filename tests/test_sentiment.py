"""The Reddit recorder, point-in-time features and the breaker that reads them."""

import json
from datetime import UTC, datetime
from pathlib import Path

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


FIXTURES = Path(__file__).parent / "fixtures"
ARCHIVE = FIXTURES / "reddit_archive.ndjson"


def _feed_items() -> dict[str, dict]:
    items = {}
    for kind in ("post", "comment"):
        xml = (FIXTURES / f"reddit_feed_{kind}s.xml").read_text(encoding="utf-8")
        items.update({i["id"]: i for i in sentiment.parse_feed(xml, "solana", kind)})
    return items


def _archived() -> list[dict]:
    rows = []
    for line in ARCHIVE.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def test_archived_items_match_what_the_feeds_record():
    """Real feed entries (authors renamed) next to the same items in archive form: raw markdown, bare ids."""
    live = _feed_items()
    assert set(live) == {"t3_1x1ope1", "t3_1x1kblg", "t1_pewp3ap", "t1_pewmkl9"}
    archived = {}
    for obj in _archived():
        item = sentiment.archive_item(obj, "solana")
        if item is not None and item["id"] in live:
            archived[item["id"]] = item
    assert set(archived) == set(live)
    for item_id, item in archived.items():
        for key in ("subreddit", "kind", "author", "created", "title", "text"):
            assert item[key] == live[item_id][key], (item_id, key)
        assert item["compound"] == pytest.approx(live[item_id]["compound"], abs=1e-6)


@pytest.mark.parametrize(
    ("markdown", "rendered"),
    [
        ("> quoted **bold** and *em*\n\n# Heading",
         "<blockquote><p>quoted <strong>bold</strong> and <em>em</em></p></blockquote><h1>Heading</h1>"),
        ("- one\n- two\n\n1. first\n2. second",
         "<ul><li>one</li><li>two</li></ul><ol><li>first</li><li>second</li></ol>"),
        ("a | b\n---|---\nc | d",
         "<table><thead><tr><th>a</th><th>b</th></tr></thead><tbody><tr><td>c</td><td>d</td></tr></tbody></table>"),
        ("~~gone~~ >!hidden!< ^(tiny words) ^up `code` snake_case 2 * 3",
         ('<p><del>gone</del> <span class="md-spoiler-text">hidden</span> <sup>tiny words</sup> <sup>up</sup> '
          "<code>code</code> snake_case 2 * 3</p>")),
        (r'\*not em\* see [the docs](https://example.com "title") or https://example.com',
         ('<p>*not em* see <a href="https://example.com" title="title">the docs</a> or '
          '<a href="https://example.com">https://example.com</a></p>')),
        ("```\nprint(1)\n```\n\n***\n\nAT&amp;T &lt;3", "<pre><code>print(1)\n</code></pre><hr/><p>AT&amp;T &lt;3</p>"),
        ("[removed]", ""),
    ],
)
def test_markdown_reduces_to_the_text_of_its_html(markdown, rendered):
    assert sentiment.markdown_plain(markdown) == sentiment._plain(rendered)


def test_archive_ids_kinds_and_times():
    post = sentiment.archive_item({"id": "t3_abc", "created_utc": "1700000000.0", "title": "A &amp; B"}, "sub")
    assert post is not None and post["id"] == "t3_abc" and post["kind"] == "post"
    assert post["created"] == 1700000000 and post["title"] == "A & B" and post["author"] == ""
    comment = sentiment.archive_item({"id": "xyz", "created_utc": 1700000000, "body": "[deleted]"}, "sub")
    assert comment is not None and comment["id"] == "t1_xyz" and comment["text"] == ""
    assert sentiment.archive_item({"id": "q", "created_utc": "soon"}, "sub") is None
    assert sentiment.archive_item({"created_utc": 1}, "sub") is None


def test_import_is_idempotent_and_live_fetches_duplicate_it(monkeypatch):
    monkeypatch.setattr(sentiment, "_IMPORT_BATCH", 2)
    first = sentiment.import_archive(ARCHIVE, ["Solana"], now=50.0)
    assert {k: v for k, v in first.items() if k != "path"} == {
        "read": 9, "kept": 6, "inserted": 6, "duplicates": 0,
        "skipped_unknown_subreddit": 1, "skipped_out_of_range": 0, "bad_lines": 2,
    }
    again = sentiment.import_archive(ARCHIVE, ["solana"], now=60.0)
    assert again["inserted"] == 0 and again["duplicates"] == 6
    live = [i for i in _feed_items().values() if i["kind"] == "post"]
    assert sentiment._store("solana/post", live, 200, None, 70.0)["new"] == 0
    with sentiment._connect() as conn:
        rows = conn.execute("SELECT id, subreddit, title, text, fetched_at FROM items ORDER BY created").fetchall()
    assert len(rows) == 6
    assert rows[0] == ("t3_1x0aaaa", "Solana", "SOL to the moon", "", 50.0)  # [removed] body, title kept


def test_import_keeps_the_requested_days():
    after = int(datetime(2026, 10, 9, 12, tzinfo=UTC).timestamp())
    before = int(datetime(2026, 10, 9, 19, 5, tzinfo=UTC).timestamp())
    report = sentiment.import_archive(ARCHIVE, ["solana"], after=after, before=before)
    assert report["kept"] == 3 and report["skipped_out_of_range"] == 3
    with sentiment._connect() as conn:
        ids = {r[0] for r in conn.execute("SELECT id FROM items")}
    assert ids == {"t3_1x1kblg", "t3_1x1ope1", "t1_pewmkl9"}


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
