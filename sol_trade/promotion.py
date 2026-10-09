"""The promotion pipeline: candidate → backtest_passed → paper → live, and back out.

A candidate is one strategy hash on one symbol. ``backtest`` records it and
moves it to ``backtest_passed`` on a PASS verdict. ``promote`` runs the
held-out check once per hash (a failure rejects the hash for good) and
starts a paper book. The bot advances paper candidates to ``live`` when
their paper evidence qualifies, retires those that don't, and demotes live
ones that misbehave. A live strategy trades from the spec stored here, so
editing its file can't change what was promoted.

Every transition stores its evidence. The kill switch pauses all entries
until a human runs ``resume``. Config edits are atomic, with a backup.
"""

import json
import os
import shutil
import statistics
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sol_trade import backtest, costs, evaluation, history, ledger
from sol_trade.config import config
from sol_trade.engine import DEMOTED, KILL_SWITCH, Book
from sol_trade.log import log_general
from sol_trade.spec import Strategy, canonical_json, compile_spec

CANDIDATE = "candidate"
BACKTEST_PASSED = "backtest_passed"
PAPER = "paper"
LIVE = "live"
DEMOTE_PENDING = "demote_pending"
REJECTED = "rejected"
DEMOTED_STATE = "demoted"
RETIRED = "retired"
ACTIVE = (CANDIDATE, BACKTEST_PASSED, PAPER, LIVE, DEMOTE_PENDING)
MANAGED = (LIVE, DEMOTE_PENDING)

BACKUP_DIR = "data/config_backups"
_PAUSED = "paused"


class Refused(Exception):
    """A request the pipeline won't carry out; ``code`` is machine-readable."""

    def __init__(self, code: str, message: str, details: list[str] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or []


@dataclass
class Candidate:
    strategy: str
    hash: str
    symbol: str
    timeframe: str
    state: str
    spec: dict[str, Any]
    approved: bool
    holdout: dict[str, Any] | None
    created_at: float
    entered_at: float

    @property
    def book_name(self) -> str:
        return f"paper:{self.hash[:12]}:{self.symbol}"

    def compile(self) -> Strategy:
        return compile_spec(self.spec, self.strategy)

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "hash": self.hash,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "state": self.state,
            "approved": self.approved,
            "since": _iso(self.entered_at),
            "created": _iso(self.created_at),
        }


def _iso(ts: float | None) -> str | None:
    return datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%dT%H:%M:%SZ") if ts else None


def _row(row: Any) -> Candidate:
    holdout = ledger.decode(row[7], "candidate holdout") if row[7] else None
    return Candidate(
        row[2], row[0], row[1], row[3], row[4], ledger.decode(row[5], "candidate spec"),
        bool(row[6]), holdout, row[8], row[9],
    )


def get(strategy_hash: str, symbol: str) -> Candidate | None:
    with ledger.connect() as conn:
        row = conn.execute(
            """
            SELECT strategy_hash, symbol, strategy, timeframe, state, spec, approved, holdout, created_at, entered_at
            FROM candidates WHERE strategy_hash = ? AND symbol = ?
            """,
            (strategy_hash, symbol),
        ).fetchone()
    return _row(row) if row else None


def candidates(symbol: str | None = None, states: tuple[str, ...] | None = None) -> list[Candidate]:
    with ledger.connect() as conn:
        rows = conn.execute(
            """
            SELECT strategy_hash, symbol, strategy, timeframe, state, spec, approved, holdout, created_at, entered_at
            FROM candidates WHERE ? IS NULL OR symbol = ? ORDER BY created_at
            """,
            (symbol, symbol),
        ).fetchall()
    found = [_row(r) for r in rows]
    return [c for c in found if states is None or c.state in states]


def find_active(strategy: str, symbol: str) -> Candidate | None:
    """The newest active candidate of a strategy name on a symbol, whatever its hash."""
    matches = [c for c in candidates(symbol, ACTIVE) if c.strategy == strategy]
    return max(matches, key=lambda c: c.entered_at) if matches else None


def managed_symbols() -> dict[str, Candidate]:
    """Symbols the pipeline trades live, with their candidate."""
    return {c.symbol: c for c in candidates(states=MANAGED)}


def transitions(strategy_hash: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    with ledger.connect() as conn:
        rows = conn.execute(
            """
            SELECT strategy, strategy_hash, symbol, from_state, to_state, reason, evidence, created_at
            FROM transitions WHERE ? IS NULL OR strategy_hash = ? ORDER BY id DESC LIMIT ?
            """,
            (strategy_hash, strategy_hash, limit),
        ).fetchall()
    return [
        {
            "strategy": r[0], "hash": r[1], "symbol": r[2], "from": r[3], "to": r[4], "reason": r[5],
            "evidence": ledger.decode(r[6], "transition evidence"), "at": _iso(r[7]),
        }
        for r in rows
    ]


def _transition(candidate: Candidate, to_state: str, reason: str, evidence: dict[str, Any]) -> dict[str, Any]:
    now = time.time()
    with ledger.connect() as conn:
        conn.execute(
            "UPDATE candidates SET state = ?, entered_at = ? WHERE strategy_hash = ? AND symbol = ?",
            (to_state, now, candidate.hash, candidate.symbol),
        )
        conn.execute(
            """
            INSERT INTO transitions (strategy_hash, symbol, strategy, from_state, to_state, reason, evidence, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (candidate.hash, candidate.symbol, candidate.strategy, candidate.state, to_state, reason,
             json.dumps(evidence, default=str), now),
        )
    log_general.info(f"pipeline: {candidate.strategy} on {candidate.symbol} {candidate.state} -> {to_state} ({reason})")
    previous, candidate.state, candidate.entered_at = candidate.state, to_state, now
    return {**candidate.as_dict(), "from": previous, "to": to_state, "reason": reason}


# ----------------------------------------------------------------- settings


def _setting(key: str) -> Any:
    with ledger.connect() as conn:
        row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    return ledger.decode(row[0], f"setting {key}") if row else None


def _set_setting(key: str, value: Any) -> None:
    with ledger.connect() as conn:
        if value is None:
            conn.execute("DELETE FROM settings WHERE key = ?", (key,))
        else:
            conn.execute(
                "INSERT OR REPLACE INTO settings (key, value, updated_at) VALUES (?, ?, ?)",
                (key, json.dumps(value, default=str), time.time()),
            )


def paused() -> dict[str, Any] | None:
    """The kill switch record while trading is paused, else None."""
    return _setting(_PAUSED)


def resume() -> dict[str, Any]:
    """Clear the kill switch (human-only; the CLI checks for a terminal)."""
    record = paused()
    if record is None:
        raise Refused("not_paused", "trading is not paused")
    _set_setting(_PAUSED, None)
    for name in (ledger.LIVE, ledger.DRY_RUN):
        meta = ledger.load_meta(name)
        meta.pop("risk", None)  # the drawdown peak restarts from here
        ledger.save_meta(name, meta)
    log_general.warning("kill switch cleared by resume")
    return {"resumed": True, "was": record}


# ------------------------------------------------------------------- config


def _edit_config(change: Any) -> str:
    """Apply ``change(dict)`` to config.json atomically; returns the backup path."""
    cfg = config()
    path = Path(cfg.path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise Refused("config_invalid", f"can't read {path}: {e}") from e
    change(data)
    backups = Path(BACKUP_DIR)
    backups.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    backup = backups / f"config-{stamp}.json"
    shutil.copy2(path, backup)
    fd, tmp = tempfile.mkstemp(prefix=".config-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    cfg.maybe_reload_config()
    return str(backup)


def _add_token(symbol: str, mint: str, strategy: str) -> str:
    def change(data: dict[str, Any]) -> None:
        if symbol in data.get("secondary_mint_symbols", []):
            raise Refused("token_in_use", f"{symbol} is already in the token set")
        data.setdefault("secondary_mints", []).append(mint)
        data.setdefault("secondary_mint_symbols", []).append(symbol)
        data.setdefault("token_strategies", {})[symbol] = strategy

    return _edit_config(change)


def _remove_token(symbol: str) -> str:
    def change(data: dict[str, Any]) -> None:
        symbols = data.get("secondary_mint_symbols", [])
        if symbol in symbols:
            i = symbols.index(symbol)
            for key in ("secondary_mints", "secondary_mint_symbols", "secondary_weights"):
                values = data.get(key) or []
                if i < len(values):
                    values.pop(i)
        data.get("token_strategies", {}).pop(symbol, None)

    return _edit_config(change)


# --------------------------------------------------------------- transitions


def record_backtest(strategy: Strategy, symbol: str, result: dict[str, Any]) -> dict[str, Any]:
    """Register the evaluated hash and move it between candidate and backtest_passed."""
    now = time.time()
    with ledger.connect() as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO candidates
                (strategy_hash, symbol, strategy, timeframe, state, spec, approved, holdout, created_at, entered_at)
            VALUES (?, ?, ?, ?, ?, ?, 0, NULL, ?, ?)
            """,
            (strategy.hash, symbol, strategy.name, strategy.timeframe, CANDIDATE, canonical_json(strategy.spec),
             now, now),
        )
    candidate = get(strategy.hash, symbol)
    if candidate is None:
        raise RuntimeError(f"candidate {strategy.hash[:12]} on {symbol} was not stored")
    if candidate.state in (CANDIDATE, BACKTEST_PASSED):
        target = BACKTEST_PASSED if result.get("verdict") == evaluation.PASS else CANDIDATE
        if target != candidate.state:
            _transition(candidate, target, f"backtest {result.get('verdict')}", _evidence(result))
    return candidate.as_dict()


def _evidence(result: dict[str, Any]) -> dict[str, Any]:
    keep = ("verdict", "reasons", "metrics", "folds", "benchmarks", "gates", "deflated_sharpe", "trial", "n_trials",
            "data", "costs", "flags", "passed")
    return {
        **{k: result[k] for k in keep if k in result},
        "gate_config": dict(config().gates),
        "risk": dict(config().risk),
    }


def promote(strategy: Strategy, symbol: str, now: float | None = None) -> dict[str, Any]:
    """Advance a candidate one step: backtest_passed → paper, or paper → live."""
    now = time.time() if now is None else now
    candidate = get(strategy.hash, symbol)
    if candidate is None or candidate.state == CANDIDATE:
        raise Refused(
            "not_passed",
            f"{strategy.name} ({strategy.hash[:12]}) has no PASS verdict on {symbol}; run backtest first",
        )
    if candidate.state == BACKTEST_PASSED:
        return _to_paper(candidate, strategy, symbol)
    if candidate.state == PAPER:
        reasons, evidence = live_readiness(candidate, now)
        if reasons:
            raise Refused("not_ready", "paper evidence does not qualify for live yet", reasons)
        return _to_live(candidate, evidence)
    raise Refused(candidate.state, f"{strategy.name} on {symbol} is {candidate.state}")


def _to_paper(candidate: Candidate, strategy: Strategy, symbol: str) -> dict[str, Any]:
    model = costs.model_for(symbol)
    if model.source == "default":
        raise Refused("costs_unmeasured", f"no measured costs for {symbol}; run measure-costs first")
    age = costs.sample_age_days(symbol)
    if age is not None and age > config().costs["max_sample_age_days"]:
        raise Refused("costs_stale", f"cost samples for {symbol} are {age:.0f} days old; run measure-costs")
    limit = int(config().promotion["max_paper_candidates"])
    if len(candidates(states=(PAPER,))) >= limit:
        raise Refused("paper_full", f"{limit} candidates are already in paper")
    if candidate.holdout is None:
        result = evaluation.holdout(strategy, symbol, model=model)
        with ledger.connect() as conn:
            conn.execute(
                "UPDATE candidates SET holdout = ? WHERE strategy_hash = ? AND symbol = ?",
                (json.dumps(_evidence(result), default=str), candidate.hash, symbol),
            )
        if not result["passed"]:
            return _transition(candidate, REJECTED, "held-out period failed", _evidence(result))
    ledger.delete_book(candidate.book_name)
    capital = float(config().risk["max_trade_size"])
    ledger.save_meta(candidate.book_name, {"cash": capital, "capital": capital, "observed_bars": 0})
    return _transition(candidate, PAPER, "held-out period passed", {"holdout": candidate.holdout or "passed"})


def approve(strategy: Strategy, symbol: str) -> dict[str, Any]:
    """Allow paper → live while ``promotion.require_approval`` is on (human-only)."""
    candidate = get(strategy.hash, symbol)
    if candidate is None or candidate.state not in (BACKTEST_PASSED, PAPER):
        raise Refused("not_approvable", f"{strategy.name} on {symbol} is not in backtest_passed or paper")
    with ledger.connect() as conn:
        conn.execute(
            "UPDATE candidates SET approved = 1 WHERE strategy_hash = ? AND symbol = ?", (candidate.hash, symbol)
        )
    candidate.approved = True
    return candidate.as_dict()


def demote(strategy_name: str, symbol: str, reason: str = "requested") -> dict[str, Any]:
    """Take a candidate out of the pipeline.

    A live strategy holding a position is queued as ``demote_pending``; the
    bot flattens it and finishes. Anything else is demoted at once. The CLI
    never trades.
    """
    candidate = find_active(strategy_name, symbol)
    if candidate is None:
        raise Refused("not_found", f"no active candidate of {strategy_name} on {symbol}")
    if candidate.state in MANAGED:
        if _holding(symbol):
            if candidate.state == DEMOTE_PENDING:
                return {**candidate.as_dict(), "from": DEMOTE_PENDING, "to": DEMOTE_PENDING, "reason": reason}
            return _transition(candidate, DEMOTE_PENDING, reason, {"requested": reason})
        return finish_demotion(candidate, reason, {"requested": reason})
    return _transition(candidate, DEMOTED_STATE, reason, {"requested": reason})


def finish_demotion(candidate: Candidate, reason: str, evidence: dict[str, Any]) -> dict[str, Any]:
    backup = _remove_token(candidate.symbol)
    return _transition(candidate, DEMOTED_STATE, reason, {**evidence, "config_backup": backup})


def _holding(symbol: str) -> bool:
    return any(ledger.open_position(book, symbol) is not None for book in (ledger.LIVE, ledger.DRY_RUN))


def _to_live(candidate: Candidate, evidence: dict[str, Any]) -> dict[str, Any]:
    mint = config().mint_for_symbol(candidate.symbol)
    if mint is None:
        raise Refused("unknown_mint", f"no mint known for {candidate.symbol}; add it to token_mints")
    backup = _add_token(candidate.symbol, mint, candidate.strategy)
    return _transition(candidate, LIVE, "paper evidence qualified", {**evidence, "config_backup": backup})


# -------------------------------------------------------------- paper books


def paper_book_state(candidate: Candidate) -> dict[str, Any]:
    meta = ledger.load_meta(candidate.book_name)
    state = ledger.load_state(candidate.book_name, candidate.symbol)
    return {"meta": meta, "state": state}


def _last_close(candidate: Candidate) -> float | None:
    frame = history.load_bars(history.pair_for(candidate.symbol), candidate.timeframe, limit=1)
    return float(frame["close"].iat[-1]) if len(frame) else None


def record_paper_cycle(candidate: Candidate, book: Book, bars_processed: int) -> None:
    """Update a paper book's bookkeeping after the bot advanced it."""
    meta = ledger.load_meta(candidate.book_name)
    meta["cash"] = book.cash
    state = book.state(candidate.symbol)
    if meta.get("first_bar_ts") is None and state.last_bar_ts is not None:
        meta["first_bar_ts"] = state.last_bar_ts
    if bars_processed:
        meta["observed_bars"] = int(meta.get("observed_bars", 0)) + 1
    price = _last_close(candidate)
    equity = book.equity({candidate.symbol: price} if price else {})
    peak = max(float(meta.get("peak", meta.get("capital", equity))), equity)
    meta["peak"] = peak
    meta["equity"] = equity
    meta["max_drawdown_pct"] = max(float(meta.get("max_drawdown_pct", 0.0)), (1 - equity / peak) * 100 if peak else 0.0)
    ledger.save_meta(candidate.book_name, meta)


def live_readiness(candidate: Candidate, now: float | None = None) -> tuple[list[str], dict[str, Any]]:
    """Reasons a paper candidate can't go live yet (empty when it can), plus the evidence."""
    now = time.time() if now is None else now
    p = config().promotion
    cfg = config()
    reasons: list[str] = []
    meta = ledger.load_meta(candidate.book_name)
    strategy = candidate.compile()
    capital = float(meta.get("capital", cfg.risk["max_trade_size"]))

    days = (now - candidate.entered_at) / 86400
    if days < p["min_paper_days"]:
        reasons.append(f"{days:.1f} of {p['min_paper_days']} paper days")
    expected = max(int((now - candidate.entered_at) // strategy.period_seconds), 1)
    coverage = min(int(meta.get("observed_bars", 0)) / expected, 1.0)
    if coverage < p["min_paper_coverage"]:
        reasons.append(f"bar coverage {coverage:.2f} is below {p['min_paper_coverage']}")
    paper_trades = ledger.trades(candidate.book_name, candidate.symbol)
    if len(paper_trades) < p["min_paper_trades"]:
        reasons.append(f"{len(paper_trades)} of {p['min_paper_trades']} paper round trips")

    state = ledger.load_state(candidate.book_name, candidate.symbol)
    paper_pnl, replay_pnl = _replay_comparison(candidate, strategy, paper_trades, meta, state.last_bar_ts, capital)
    if paper_pnl is not None and replay_pnl is not None:
        tolerance = max(p["replay_tolerance_points"], p["replay_tolerance_relative"] * abs(replay_pnl))
        if abs(paper_pnl - replay_pnl) > tolerance:
            reasons.append(f"paper return {paper_pnl:.2f}% is off the replay's {replay_pnl:.2f}% by more than {tolerance:.2f}")
    drawdown = float(meta.get("max_drawdown_pct", 0.0))
    if drawdown > cfg.risk["max_drawdown_pct"]:
        reasons.append(f"paper drawdown {drawdown:.2f}% is above {cfg.risk['max_drawdown_pct']}%")

    reasons.extend(_live_slot_problems(candidate))
    if p["require_approval"] and not candidate.approved:
        reasons.append("awaiting approve (promotion.require_approval is on)")
    evidence = {
        "paper_days": round(days, 2),
        "coverage": round(coverage, 4),
        "paper_trades": len(paper_trades),
        "paper_return_pct": paper_pnl,
        "replay_return_pct": replay_pnl,
        "paper_max_drawdown_pct": round(drawdown, 4),
        "promotion_config": dict(p),
    }
    return reasons, evidence


def _live_slot_problems(candidate: Candidate) -> list[str]:
    cfg = config()
    problems = []
    if paused():
        problems.append("trading is paused by the kill switch")
    if cfg.mint_for_symbol(candidate.symbol) is None:
        problems.append(f"no mint known for {candidate.symbol}; add it to token_mints")
    if candidate.symbol in cfg.secondary_mint_symbols:
        problems.append(f"{candidate.symbol} is already traded by another strategy")
    if cfg.secondary_weights:
        problems.append("secondary_weights is set; the pipeline doesn't change weights")
    slots = int(float(cfg.risk["max_live_capital"]) // float(cfg.risk["max_trade_size"]))
    if len(managed_symbols()) >= slots:
        problems.append(f"all {slots} live slots (max_live_capital / max_trade_size) are taken")
    return problems


def _replay_comparison(
    candidate: Candidate,
    strategy: Strategy,
    paper_trades: list[dict[str, Any]],
    meta: dict[str, Any],
    last_bar_ts: int | None,
    capital: float,
) -> tuple[float | None, float | None]:
    """Realized return (percent) of the paper book and of a backtest over the same closed bars."""
    first = meta.get("first_bar_ts")
    if first is None or last_bar_ts is None:
        return None, None
    frame = history.load_bars(history.pair_for(candidate.symbol), candidate.timeframe)
    ts = frame["ts"].to_numpy()
    start, end = int(ts.searchsorted(first)), int(ts.searchsorted(last_bar_ts, side="right"))
    if start < strategy.lookback - 1 or end <= start:
        return None, None
    signals = backtest.compute_signals(strategy, frame, candidate.symbol)
    sim = backtest.simulate(
        strategy, candidate.symbol, frame, *signals, start, end, costs.model_for(candidate.symbol), capital,
        force_close=False,
    )
    paper = sum(t["pnl"] for t in paper_trades if t["exit_ts"] <= last_bar_ts)
    replay = sum(t.pnl for t in sim.trades if t.exit_ts <= last_bar_ts)
    return round(paper / capital * 100, 4), round(replay / capital * 100, 4)


def advance_paper(now: float | None = None) -> list[dict[str, Any]]:
    """Promote qualifying paper candidates and retire stale ones (the bot calls this)."""
    now = time.time() if now is None else now
    done = []
    for candidate in candidates(states=(PAPER,)):
        reasons, evidence = live_readiness(candidate, now)
        if not reasons:
            try:
                done.append(_to_live(candidate, evidence))
            except Refused as e:
                log_general.warning(f"pipeline: {candidate.strategy} on {candidate.symbol} not promoted: {e.message}")
        elif (now - candidate.entered_at) / 86400 > config().promotion["max_paper_days"]:
            done.append(_transition(candidate, RETIRED, "paper period ran out", {**evidence, "reasons": reasons}))
    return done


# ----------------------------------------------------------------- governor


def _risk_meta(book: Book) -> dict[str, Any]:
    meta = ledger.load_meta(book.name)
    return meta.get("risk") or {"epoch": int(time.time()), "peak": float(config().risk["max_live_capital"])}


def book_equity(book: Book, prices: dict[str, float], since: int) -> float:
    """The bot's capital: ``max_live_capital`` plus realized and unrealized P&L since ``since``."""
    realized = sum(t["pnl"] for t in ledger.trades(book.name, since=since))
    unrealized = 0.0
    for symbol, state in book.symbols.items():
        if state.position and prices.get(symbol):
            unrealized += state.position.size * prices[symbol] - state.position.cost_basis
    return float(config().risk["max_live_capital"]) + realized + unrealized


def check_kill_switch(book: Book, prices: dict[str, float], now: float | None = None) -> dict[str, Any] | None:
    """Flatten everything and pause when the book's drawdown reaches ``risk.max_drawdown_pct``."""
    now = time.time() if now is None else now
    risk = _risk_meta(book)
    equity = book_equity(book, prices, int(risk["epoch"]))
    risk["peak"] = max(float(risk["peak"]), equity)
    drawdown = (1 - equity / risk["peak"]) * 100 if risk["peak"] > 0 else 0.0
    risk["drawdown_pct"] = round(drawdown, 4)
    meta = ledger.load_meta(book.name)
    meta["risk"] = risk
    ledger.save_meta(book.name, meta)
    limit = float(config().risk["max_drawdown_pct"])
    if drawdown < limit or paused():
        return None
    closed = []
    for symbol, state in list(book.symbols.items()):
        if state.position and prices.get(symbol) and book.close_position(symbol, int(now), prices[symbol], KILL_SWITCH):
            closed.append(symbol)
    record = {
        "at": _iso(now),
        "book": book.name,
        "equity": round(equity, 4),
        "peak": round(risk["peak"], 4),
        "drawdown_pct": round(drawdown, 4),
        "limit_pct": limit,
        "flattened": closed,
    }
    _set_setting(_PAUSED, record)
    log_general.error(f"KILL SWITCH: drawdown {drawdown:.2f}% >= {limit}%; flattened {closed}; trading paused until resume")
    return record


def _strategy_health(book: Book, candidate: Candidate, price: float | None, now: float) -> list[str]:
    p = config().promotion
    problems = []
    trades = [t for t in ledger.trades(book.name, candidate.symbol) if t["strategy_hash"] == candidate.hash]
    capital = float(config().risk["max_trade_size"])
    equity, peak, worst = capital, capital, 0.0
    for t in trades:
        equity += t["pnl"]
        peak = max(peak, equity)
        worst = max(worst, (1 - equity / peak) * 100)
    position = book.position(candidate.symbol)
    if position and price:
        current = equity + position.size * price - position.cost_basis
        worst = max(worst, (1 - current / peak) * 100)
    if worst >= p["demote_drawdown_pct"]:
        problems.append(f"strategy drawdown {worst:.2f}% reached {p['demote_drawdown_pct']}%")
    window = [t for t in trades if t["exit_ts"] >= now - p["demote_window_days"] * 86400]
    if len(window) >= p["demote_min_trades"] and sum(t["pnl"] for t in window) < 0:
        problems.append(f"{len(window)} trades in {p['demote_window_days']} days lost money after costs")
    fills = [
        f for f in ledger.fills(book.name, candidate.symbol)
        if f["strategy_hash"] == candidate.hash and f["modeled_cost"] is not None and f["notional"]
    ][-int(p["divergence_fills"]):]
    if len(fills) >= p["divergence_fills"]:
        gap = statistics.median((f["cost"] - f["modeled_cost"]) / f["notional"] * 1e4 for f in fills)
        if gap > p["max_cost_divergence_bps"]:
            problems.append(f"fills cost {gap:.1f} bps more than modeled (limit {p['max_cost_divergence_bps']})")
    return problems


def govern(book: Book, prices: dict[str, float], now: float | None = None) -> list[dict[str, Any]]:
    """The bot's per-cycle checks on its main book: kill switch, then demotions."""
    now = time.time() if now is None else now
    done: list[dict[str, Any]] = []
    tripped = check_kill_switch(book, prices, now)
    if tripped:
        done.append({"kill_switch": tripped})
    for candidate in candidates(states=MANAGED):
        price = prices.get(candidate.symbol)
        if candidate.state == DEMOTE_PENDING:
            problems = ["demotion requested"]
        else:
            problems = _strategy_health(book, candidate, price, now)
        if not problems:
            continue
        holding = book.position(candidate.symbol) is not None
        if holding and (not price or book.close_position(candidate.symbol, int(now), price, DEMOTED) is None):
            log_general.warning(f"pipeline: could not flatten {candidate.symbol} for demotion; retrying next cycle")
            if candidate.state == LIVE:
                _transition(candidate, DEMOTE_PENDING, "; ".join(problems), {"problems": problems})
            continue
        done.append(finish_demotion(candidate, "; ".join(problems), {"problems": problems}))
    return done
