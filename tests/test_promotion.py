"""The promotion pipeline: states, evidence, paper books, config edits, governor."""

import json
import time
from pathlib import Path

import pytest

from sol_trade import costs, data_source, evaluation, ledger, promotion, spec, trading
from sol_trade.config import config
from sol_trade.costs import CostModel
from sol_trade.engine import Book, Position, SimExecutor, SymbolState, Trade
from tests.fakes import FakeExchange, make_ticks
from tests.test_evaluation import SOL_6H, TREND_EMA_6H
from tests.test_parity import FIXTURE

MEASURED = CostModel("TKN", "measured", ((0.0, 10.0),), 0.0)
TKN_MINT = "TKNmint1111111111111111111111111111111111111"


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    """A writable copy of the test config with a known TKN mint; the token set is restored afterwards."""
    c = config()
    data = json.loads(Path(c.path).read_text())
    data["token_mints"] = {"TKN": TKN_MINT}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data))
    for attr in ("path", "_config_mtime", "secondary_mints", "secondary_mint_symbols", "secondary_weights",
                 "token_strategies", "token_mints"):
        monkeypatch.setattr(c, attr, getattr(c, attr))
    c.path = str(path)
    c.token_mints = {"TKN": TKN_MINT}
    monkeypatch.setattr(costs, "model_for", lambda _s: MEASURED)
    monkeypatch.setattr(costs, "sample_age_days", lambda _s, now=None: 1.0)
    monkeypatch.setattr(promotion.costs, "model_for", lambda _s: MEASURED)
    return c


@pytest.fixture
def strategy():
    return spec.compile_spec(FIXTURE)


def _candidate(strategy) -> promotion.Candidate:
    found = promotion.get(strategy.hash, "TKN")
    assert found is not None
    return found


def _passed(strategy, symbol="TKN"):
    promotion.record_backtest(strategy, symbol, {"verdict": evaluation.PASS, "metrics": {"trades": 40}})


def _holdout(monkeypatch, passed=True, verdict=None):
    calls = []
    verdict = verdict or (evaluation.PASS if passed else evaluation.FAIL)

    def fake(strategy, symbol, frame=None, model=None, start_ts=None, end_ts=None):
        calls.append(strategy.hash)
        reasons = {evaluation.PASS: [], evaluation.FAIL: ["holdout_return_pct -3 is not above 0"],
                   evaluation.INSUFFICIENT: ["holdout_trades 0 is below 2"]}[verdict]
        return {"verdict": verdict, "passed": verdict == evaluation.PASS, "reasons": reasons, "start_ts": 1000}

    monkeypatch.setattr(evaluation, "holdout", fake)
    return calls


def _to_paper(strategy, monkeypatch):
    _passed(strategy)
    _holdout(monkeypatch)
    return promotion.promote(strategy, "TKN")


def _qualify(candidate, days=20):
    """Make a paper candidate's evidence sufficient."""
    with ledger.connect() as conn:
        conn.execute(
            "UPDATE candidates SET entered_at = ? WHERE strategy_hash = ?",
            (time.time() - days * 86400, candidate.hash),
        )
    meta = ledger.load_meta(candidate.book_name)
    meta["observed_bars"] = days * 24 * 60
    ledger.save_meta(candidate.book_name, meta)
    book = Book(candidate.book_name, 100.0, SimExecutor(costs.model_for), 100.0, 100.0)
    for i in range(2):
        ledger.record_trade(book, Trade("TKN", candidate.strategy, candidate.hash, i, i + 60, 1, 1.1, 1, 1, 1.1, "signal"))


def test_backtest_verdicts_move_between_candidate_and_passed(cfg, strategy):
    promotion.record_backtest(strategy, "TKN", {"verdict": evaluation.FAIL})
    assert _candidate(strategy).state == promotion.CANDIDATE
    _passed(strategy)
    assert _candidate(strategy).state == promotion.BACKTEST_PASSED
    promotion.record_backtest(strategy, "TKN", {"verdict": evaluation.INSUFFICIENT})
    assert _candidate(strategy).state == promotion.CANDIDATE
    history = promotion.transitions(strategy.hash)
    assert [t["to"] for t in history] == ["candidate", "backtest_passed"]
    assert history[1]["evidence"]["gate_config"]["min_trades"] == 30


def test_promote_needs_a_pass(cfg, strategy):
    with pytest.raises(promotion.Refused) as refused:
        promotion.promote(strategy, "TKN")
    assert refused.value.code == "not_passed"


def test_promote_needs_measured_costs(cfg, strategy, monkeypatch):
    _passed(strategy)
    monkeypatch.setattr(promotion.costs, "model_for", lambda _s: CostModel("TKN", "default", ((0.0, 50.0),), 0.0))
    with pytest.raises(promotion.Refused) as refused:
        promotion.promote(strategy, "TKN")
    assert refused.value.code == "costs_unmeasured"


def test_failed_holdout_rejects_the_hash_for_good(cfg, strategy, monkeypatch):
    _passed(strategy)
    calls = _holdout(monkeypatch, passed=False)
    result = promotion.promote(strategy, "TKN")
    assert result["to"] == promotion.REJECTED
    promotion.record_backtest(strategy, "TKN", {"verdict": evaluation.PASS})  # a rerun can't revive it
    with pytest.raises(promotion.Refused) as refused:
        promotion.promote(strategy, "TKN")
    assert refused.value.code == promotion.REJECTED
    assert calls == [strategy.hash]


def test_an_inconclusive_holdout_keeps_the_hash_eligible(cfg, strategy, monkeypatch):
    _passed(strategy)
    fake_calls = _holdout(monkeypatch, verdict=evaluation.INSUFFICIENT)
    fake = evaluation.holdout
    pinned = []
    monkeypatch.setattr(evaluation, "holdout", lambda *a, **k: pinned.append(k.get("start_ts")) or fake(*a, **k))

    for _ in range(2):
        with pytest.raises(promotion.Refused) as refused:
            promotion.promote(strategy, "TKN")
        assert refused.value.code == "holdout_insufficient"
        assert _candidate(strategy).state == promotion.BACKTEST_PASSED

    assert pinned == [None, 1000]  # the second attempt judges the same period, grown
    assert len(fake_calls) == 2
    assert _candidate(strategy).holdout["start_ts"] == 1000
    _holdout(monkeypatch)  # enough trades now, and a profit
    assert promotion.promote(strategy, "TKN")["to"] == promotion.PAPER


def _legacy_rejection(strategy, symbol, first, last, bars):
    """A hash rejected before held-out verdicts existed, stored as that version stored it."""
    promotion.record_backtest(strategy, symbol, {"verdict": evaluation.PASS})
    candidate = promotion.get(strategy.hash, symbol)
    old = {
        "reasons": ["holdout_trades 0 is below 8", "holdout_return_pct 0.0 is not above 0"],
        "gates": [{"gate": "holdout_trades", "value": 0, "threshold": 8, "passed": False}],
        "data": {"first": first, "last": last, "bars": bars},
        "passed": False,
    }
    promotion._store_holdout(candidate, old)
    promotion._transition(candidate, promotion.REJECTED, "held-out period failed", old)


@pytest.fixture
def sol_store(monkeypatch):
    import pandas as pd

    frame = pd.read_csv(SOL_6H)
    monkeypatch.setattr(evaluation, "load_frame", lambda _strategy, _symbol: frame)
    return frame


def test_reopen_a_hash_rejected_under_the_old_trade_minimum(cfg, sol_store):
    """trend_ema_6h on SOL: rejected for 0 of 8 held-out trades; under the scaled rule that is inconclusive."""
    strategy = spec.compile_spec(TREND_EMA_6H)
    _legacy_rejection(strategy, "SOL", "2026-08-10T12:00:00Z", "2026-10-09T06:00:00Z", 240)

    result = promotion.reopen(strategy, "SOL")

    assert result["from"] == promotion.REJECTED and result["to"] == promotion.BACKTEST_PASSED
    candidate = promotion.get(strategy.hash, "SOL")
    assert candidate.holdout["verdict"] == evaluation.INSUFFICIENT
    assert candidate.holdout["data"]["first"] == "2026-08-10T12:00:00Z"  # the same period, pinned
    logged = promotion.transitions(strategy.hash)[0]
    assert logged["reason"].startswith("reopened")
    assert logged["evidence"]["old_holdout"]["passed"] is False
    with pytest.raises(promotion.Refused) as refused:
        promotion.promote(strategy, "SOL")
    assert refused.value.code == "holdout_insufficient"  # still too few trades, but no longer rejected

    promotion._transition(promotion.get(strategy.hash, "SOL"), promotion.REJECTED, "held-out period failed", {})
    with pytest.raises(promotion.Refused) as refused:
        promotion.reopen(strategy, "SOL")
    assert refused.value.code == "already_reopened"  # a one-off


def test_reopen_refuses_a_window_that_fails_under_the_current_rule(cfg, sol_store):
    strategy = spec.compile_spec(TREND_EMA_6H)
    _legacy_rejection(strategy, "SOL", "2026-04-02T00:00:00Z", "2026-05-31T18:00:00Z", 240)  # two losing trades
    with pytest.raises(promotion.Refused) as refused:
        promotion.reopen(strategy, "SOL")
    assert refused.value.code == "not_reopenable"
    assert "fails under the current rule" in refused.value.message
    assert any(r.startswith("holdout_return_pct") for r in refused.value.details)
    assert promotion.get(strategy.hash, "SOL").state == promotion.REJECTED


def test_reopen_refuses_hashes_judged_under_the_current_rule(cfg, strategy, monkeypatch):
    with pytest.raises(promotion.Refused) as refused:
        promotion.reopen(strategy, "TKN")
    assert refused.value.code == "not_rejected"
    _passed(strategy)
    _holdout(monkeypatch, passed=False)
    promotion.promote(strategy, "TKN")
    with pytest.raises(promotion.Refused) as refused:
        promotion.reopen(strategy, "TKN")
    assert refused.value.code == "not_reopenable"


def test_promote_starts_a_paper_book(cfg, strategy, monkeypatch):
    result = _to_paper(strategy, monkeypatch)
    assert result["to"] == promotion.PAPER
    candidate = _candidate(strategy)
    assert ledger.load_meta(candidate.book_name)["cash"] == config().risk["max_live_capital"]


def test_paper_slots_are_limited(cfg, strategy, monkeypatch):
    monkeypatch.setitem(config().promotion, "max_paper_candidates", 1)
    _to_paper(strategy, monkeypatch)
    other = spec.compile_spec({**FIXTURE, "lookback": 31})
    _passed(other)
    with pytest.raises(promotion.Refused) as refused:
        promotion.promote(other, "TKN")
    assert refused.value.code == "paper_full"


def test_fresh_paper_is_not_ready(cfg, strategy, monkeypatch):
    _to_paper(strategy, monkeypatch)
    with pytest.raises(promotion.Refused) as refused:
        promotion.promote(strategy, "TKN")
    assert refused.value.code == "not_ready"
    joined = " ".join(refused.value.details)
    assert "paper days" in joined and "coverage" in joined and "round trips" in joined


def test_paper_goes_live_and_edits_the_config_atomically(cfg, strategy, monkeypatch):
    _to_paper(strategy, monkeypatch)
    _qualify(_candidate(strategy))

    result = promotion.promote(strategy, "TKN")

    assert result["to"] == promotion.LIVE
    written = json.loads(Path(cfg.path).read_text())
    assert written["secondary_mint_symbols"][-1] == "TKN"
    assert written["secondary_mints"][-1] == TKN_MINT
    assert written["token_strategies"]["TKN"] == "parity"
    assert written["private_key"] == json.loads(Path(cfg.path).read_text())["private_key"]  # other keys kept
    assert cfg.secondary_mint_symbols[-1] == "TKN"  # reloaded in-process
    backups = list(Path(promotion.BACKUP_DIR).glob("config-*.json"))
    assert len(backups) == 1
    assert "TKN" not in json.loads(backups[0].read_text())["secondary_mint_symbols"]
    assert "TKN" in promotion.managed_symbols()


def test_approval_gate(cfg, strategy, monkeypatch):
    monkeypatch.setitem(config().promotion, "require_approval", True)
    _to_paper(strategy, monkeypatch)
    _qualify(_candidate(strategy))
    assert promotion.live_readiness(_candidate(strategy))[0] == ["awaiting approve (promotion.require_approval is on)"]
    promotion.approve(strategy, "TKN")
    assert promotion.live_readiness(_candidate(strategy))[0] == []


def test_a_token_traded_by_another_strategy_is_refused(cfg, strategy, monkeypatch):
    _to_paper(strategy, monkeypatch)
    _qualify(_candidate(strategy))
    cfg.secondary_mints = [*cfg.secondary_mints, TKN_MINT]
    cfg.secondary_mint_symbols = [*cfg.secondary_mint_symbols, "TKN"]
    assert promotion.live_readiness(_candidate(strategy))[0] == ["TKN is already traded by another strategy"]


def test_stale_paper_candidates_retire(cfg, strategy, monkeypatch):
    _to_paper(strategy, monkeypatch)
    with ledger.connect() as conn:
        conn.execute("UPDATE candidates SET entered_at = ?", (time.time() - 61 * 86400,))
    done = promotion.advance_paper()
    assert [d["to"] for d in done] == [promotion.RETIRED]


def _live(strategy, monkeypatch):
    _to_paper(strategy, monkeypatch)
    _qualify(_candidate(strategy))
    promotion.promote(strategy, "TKN")
    return _candidate(strategy)


def test_demoting_a_flat_live_strategy_removes_the_token(cfg, strategy, monkeypatch):
    _live(strategy, monkeypatch)
    result = promotion.demote("parity", "TKN", "agent request")
    assert result["to"] == promotion.DEMOTED_STATE
    assert "TKN" not in json.loads(Path(cfg.path).read_text())["secondary_mint_symbols"]
    assert "TKN" not in json.loads(Path(cfg.path).read_text())["token_strategies"]


def test_demoting_a_holding_strategy_waits_for_the_bot(cfg, strategy, monkeypatch):
    candidate = _live(strategy, monkeypatch)
    position = Position("TKN", "parity", candidate.hash, 1.0, 50.0, 50.0, 0, None, None)
    ledger.save_state(ledger.DRY_RUN, "TKN", SymbolState(position=position))

    assert promotion.demote("parity", "TKN")["to"] == promotion.DEMOTE_PENDING

    book = ledger.persist(Book(ledger.DRY_RUN, 0.0, SimExecutor(costs.model_for), 100.0, 500.0))
    ledger.load_book(book, ["TKN"])
    done = promotion.govern(book, {"TKN": 1.1})
    assert [d["to"] for d in done] == [promotion.DEMOTED_STATE]
    assert book.position("TKN") is None
    assert ledger.trades(ledger.DRY_RUN, "TKN")[0]["reason"] == "demoted"


def test_governor_demotes_on_strategy_drawdown(cfg, strategy, monkeypatch):
    """Drawdown is measured on the capital live risks (max_live_capital), like every drawdown gate."""
    candidate = _live(strategy, monkeypatch)
    book = ledger.persist(Book(ledger.DRY_RUN, 500.0, SimExecutor(costs.model_for), 100.0, 500.0))
    ledger.record_trade(book, Trade("TKN", "parity", candidate.hash, 0, 60, 1.0, 0.88, 100, 100, 88, "stoploss"))
    assert promotion.govern(book, {"TKN": 1.0}) == []  # $12 of $500 is 2.4%
    ledger.record_trade(book, Trade("TKN", "parity", candidate.hash, 60, 120, 1.0, 0.52, 100, 100, 52, "stoploss"))
    done = promotion.govern(book, {"TKN": 1.0})
    assert done[0]["to"] == promotion.DEMOTED_STATE
    assert "strategy drawdown 12.00%" in done[0]["reason"]  # $60 of $500


def test_governor_demotes_when_fills_cost_more_than_modeled(cfg, strategy, monkeypatch):
    candidate = _live(strategy, monkeypatch)
    book = ledger.persist(Book(ledger.LIVE, 500.0, SimExecutor(costs.model_for), 100.0, 500.0))
    with ledger.connect() as conn:
        for i in range(5):
            conn.execute(
                """INSERT INTO fills (book, symbol, ts, side, price, size, notional, cost, fee, modeled_cost,
                   reason, strategy, strategy_hash, created_at) VALUES ('live','TKN',?, 'buy', 1, 100, 100, 1.0,
                   0, 0.2, 'signal', 'parity', ?, 0)""",
                (i, candidate.hash),
            )
    done = promotion.govern(book, {"TKN": 1.0})
    assert "bps more than modeled" in done[0]["reason"]


def test_kill_switch_flattens_and_pauses_until_resume(cfg):
    book = ledger.persist(Book(ledger.DRY_RUN, 400.0, SimExecutor(costs.model_for), 100.0, 500.0))
    book.state("SOL").position = Position("SOL", "s", "h", 100.0, 1.0, 100.0, 0, None, None)
    assert promotion.check_kill_switch(book, {"SOL": 100.0}) is None  # equity 500, at the peak

    record = promotion.check_kill_switch(book, {"SOL": 20.0})  # equity 420: 16% down

    assert record is not None
    assert record["flattened"] == ["SOL"]
    assert promotion.paused()["drawdown_pct"] == pytest.approx(16.0)
    assert book.position("SOL") is None
    assert promotion.check_kill_switch(book, {}) is None  # already paused
    promotion.resume()
    assert promotion.paused() is None
    with pytest.raises(promotion.Refused):
        promotion.resume()


def test_paused_trading_takes_no_entries(cfg, monkeypatch):
    promotion._set_setting("paused", {"reason": "test"})
    seen = []
    view = {"bars": 0, "close": 1.0, "entry": False, "exit": False, "values": {}}
    monkeypatch.setattr(trading, "run_symbol", lambda *a, **k: seen.append(k["allow_entry"]) or view)
    monkeypatch.setattr(trading, "fetch_prices", lambda mints: dict.fromkeys(mints, 1.0))
    monkeypatch.setattr(trading, "_dry_run", True)
    from sol_trade.ui import UIState

    trading.perform_analysis(UIState())
    assert seen and not any(seen)


def test_managed_symbols_skip_confluence_and_use_the_stored_spec(cfg, strategy, monkeypatch):
    _live(strategy, monkeypatch)
    monkeypatch.setattr(config(), "sentiment_enabled", True)
    monkeypatch.setattr(trading, "is_token_blocked", lambda _s: True)
    assert trading._confluence_gate("TKN", "buy") == 1.0
    assert trading._load("TKN").hash == strategy.hash  # no file needed


def test_paper_books_match_a_backtest_replay(cfg, strategy, monkeypatch):
    """The paper book is the engine on live candles, so its return equals the replay's."""
    ticks = make_ticks(260, seed=11)
    clock = {"now": 0.0}
    exchange = FakeExchange(ticks, 60, lambda: clock["now"])
    monkeypatch.setattr(data_source, "get_exchange", lambda *_args, **_kwargs: exchange)
    _to_paper(strategy, monkeypatch)
    for k in range(40, 250):
        clock["now"] = exchange.t0 + k * 60 + 30
        trading._run_paper_books(now=clock["now"])

    candidate = _candidate(strategy)
    meta = ledger.load_meta(candidate.book_name)
    assert meta["observed_bars"] == 210
    assert ledger.trades(candidate.book_name)  # it traded
    _, evidence = promotion.live_readiness(candidate)
    assert evidence["paper_return_pct"] == evidence["replay_return_pct"]
