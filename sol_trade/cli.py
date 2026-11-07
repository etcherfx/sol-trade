"""Agent CLI: ``uv run -m sol_trade.cli <command>``.

Every command prints exactly one JSON object on stdout:
``{"ok": true, "command": ..., ...}`` or
``{"ok": false, "command": ..., "error": "<code>", "message": ..., "details": [...]}``.
Exit codes: 0 ok, 1 refused or failed, 2 usage error. Logs go to the log
files only. ``approve``, ``reopen``, ``resume`` and ``swap-smoke`` are for
humans and refuse to run without an interactive terminal. Only ``swap-smoke``
signs and sends transactions, after a typed ``yes``.
"""

import argparse
import asyncio
import json
import math
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NoReturn

import httpx
import numpy as np

from sol_trade import (
    costs,
    evaluation,
    fear_greed,
    heartbeat,
    history,
    jupiter,
    ledger,
    promotion,
    rpc,
    sentiment,
    transactions,
    wallet,
)
from sol_trade.config import config
from sol_trade.log import log_general, log_transaction, setup_file_logging
from sol_trade.spec import SpecError, Strategy, load_strategy, resolve_strategy_name

REPORT_LIMIT = 1900
SMOKE_DEFAULT_USD = 5.0
SMOKE_MAX_USD = 10.0
_LAMPORTS_PER_SOL = 1e9


class CliError(Exception):
    def __init__(self, code: str, message: str, details: list[Any] | None = None, exit_code: int = 1) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or []
        self.exit_code = exit_code


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise CliError("usage", message, [self.format_usage().strip()], exit_code=2)

    def exit(self, status: int = 0, message: str | None = None) -> NoReturn:
        # --help: the usage goes into the envelope instead of stdout.
        raise CliError("usage", message or "help requested", [self.format_help()], exit_code=2)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _clean(value: Any) -> Any:
    """NaN and infinity become null so the output is strict JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_clean(v) for v in value]
    if isinstance(value, np.generic):
        return _clean(value.item())
    return value


def _symbols(value: str | None, default: list[str]) -> list[str]:
    return [s.strip() for s in value.split(",") if s.strip()] if value else default


def _strategy(name: str) -> Strategy:
    try:
        return load_strategy(name)
    except SpecError as e:
        raise CliError("invalid_strategy", f"strategy {name!r} can't be used", e.errors) from e


def _require_tty(command: str) -> None:
    if not sys.stdin.isatty():
        raise CliError("tty_required", f"{command} is for humans: run it in an interactive terminal")


def _refused(e: promotion.Refused) -> CliError:
    return CliError(e.code, e.message, e.details)


# ----------------------------------------------------------------- commands


def cmd_backfill(args: argparse.Namespace) -> dict[str, Any]:
    if args.fear_greed:
        try:
            return {"fear_greed": fear_greed.backfill()}
        except (httpx.HTTPError, ValueError) as e:
            raise CliError("fear_greed_unavailable", f"{type(e).__name__}: {e}") from e
    cfg = config()
    symbols = _symbols(args.symbols, list(cfg.secondary_mint_symbols))
    if not symbols:
        raise CliError("no_symbols", "no symbols given and none configured")
    timeframes = _symbols(args.timeframes, list(cfg.backfill["default_timeframes"]))
    reports = history.backfill(symbols, timeframes, days=args.days)
    failed = [r for r in reports if r["errors"]]
    if failed:
        raise CliError(
            "backfill_incomplete",
            f"{len(failed)} of {len(reports)} series had errors",
            [{"series": f"{r['pair']} {r['timeframe']}", "errors": r["errors"]} for r in failed] + [{"reports": reports}],
        )
    return {"series": reports}


def _day(value: str | None, flag: str) -> int | None:
    if value is None:
        return None
    try:
        return int(datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=UTC).timestamp())
    except ValueError as e:
        raise CliError("usage", f"{flag} must be a date like 2024-01-31", exit_code=2) from e


def cmd_import_sentiment(args: argparse.Namespace) -> dict[str, Any]:
    subreddits = _symbols(args.subreddits, [s for s in config().sentiment.get("subreddits") or [] if s])
    if not subreddits:
        raise CliError("no_subreddits", "no subreddits given and none configured in sentiment.subreddits")
    after, before = _day(args.after, "--after"), _day(args.before, "--before")
    if after is not None and before is not None and after >= before:
        raise CliError("usage", "--after must be earlier than --before", exit_code=2)
    missing = [p for p in args.paths if not Path(p).is_file()]
    if missing:
        raise CliError("file_not_found", f"{len(missing)} file(s) not found", missing)
    files = [sentiment.import_archive(p, subreddits, after, before) for p in args.paths]
    return {"subreddits": subreddits, "files": files}


def cmd_backtest(args: argparse.Namespace) -> dict[str, Any]:
    strategy = _strategy(args.strategy)
    result = evaluation.evaluate(strategy, args.symbol)
    if result.get("trial") is not None:
        result["candidate"] = promotion.record_backtest(strategy, args.symbol, result)
    warnings = []
    cfg = config()
    if cfg.confluence_enabled or cfg.market_regime_enabled or cfg.sentiment_enabled:
        warnings.append("live_sizing_differs_from_backtest")
    if strategy.uses_sentiment:
        warnings.append("sentiment_inputs_only_cover_the_recorded_period")
    return {**result, "warnings": warnings}


def cmd_candidates(args: argparse.Namespace) -> dict[str, Any]:
    rows = []
    for candidate in promotion.candidates(args.symbol):
        row = candidate.as_dict()
        if candidate.state == promotion.PAPER:
            meta = ledger.load_meta(candidate.book_name)
            row["paper"] = {
                "equity": meta.get("equity"),
                "max_drawdown_pct": meta.get("max_drawdown_pct"),
                "trades": len(ledger.trades(candidate.book_name)),
                "not_ready": promotion.live_readiness(candidate)[0],
            }
        rows.append(row)
    return {"candidates": rows, "trials": evaluation.trials(args.symbol), "paused": promotion.paused()}


def cmd_promote(args: argparse.Namespace) -> dict[str, Any]:
    strategy = _strategy(args.strategy)
    try:
        result = promotion.promote(strategy, args.symbol)
    except promotion.Refused as e:
        raise _refused(e) from e
    if result["to"] == promotion.REJECTED:
        candidate = promotion.get(strategy.hash, args.symbol)
        reasons = (candidate.holdout or {}).get("reasons", []) if candidate else []
        raise CliError("holdout_failed", "the held-out period failed; this hash is rejected for good",
                       [result, *reasons])
    return {"transition": result}


def cmd_demote(args: argparse.Namespace) -> dict[str, Any]:
    try:
        result = promotion.demote(args.strategy, args.symbol, args.reason)
    except promotion.Refused as e:
        raise _refused(e) from e
    return {"transition": result}


def cmd_measure_costs(args: argparse.Namespace) -> dict[str, Any]:
    cfg = config()
    symbols = _symbols(args.symbols, list(cfg.secondary_mint_symbols))
    measured, problems = [], []
    for symbol in symbols:
        mint = cfg.mint_for_symbol(symbol)
        if mint is None:
            problems.append({"symbol": symbol, "error": "no mint known; add it to token_mints"})
            continue
        try:
            record = costs.measure(symbol, mint)
        except Exception as e:  # noqa: BLE001 - reported per symbol
            problems.append({"symbol": symbol, "error": f"{type(e).__name__}: {e}"})
            continue
        model = costs.model_for(symbol)
        if record["samples"]:
            measured.append({"symbol": symbol, "model": model.as_dict(), "samples": record["samples"]})
        if record["errors"] or model.source != "measured":
            probed = sorted({s["size_usd"] for s in record["samples"]})
            problems.append({
                "symbol": symbol,
                "error": "not every probe succeeded" if record["samples"] else "every probe failed; nothing stored",
                "measured_sizes_usd": probed,
                "model_source": model.source,
                "errors": record["errors"],
            })
    if problems:
        raise CliError(
            "measure_incomplete",
            f"{len(problems)} of {len(symbols)} symbols were not fully measured: "
            + ", ".join(p["symbol"] for p in problems),
            [*problems, {"measured": measured}],
        )
    return {"costs": measured}


def _book_summary(name: str, capital: float | None = None) -> dict[str, Any]:
    positions = []
    unrealized = 0.0
    for _, position in ledger.open_positions(name):
        timeframe = _position_timeframe(position.symbol)
        last = history.load_bars(history.source_for(position.symbol), timeframe, limit=1) if timeframe else None
        price = float(last["close"].iat[-1]) if last is not None and len(last) else None
        value = position.size * price if price else None
        if value is not None:
            unrealized += value - position.cost_basis
        positions.append({
            "symbol": position.symbol,
            "strategy": position.strategy,
            "entry_price": position.entry_price,
            "size": position.size,
            "cost_basis": round(position.cost_basis, 4),
            "last_close": price,
            "unrealized_pnl": round(value - position.cost_basis, 4) if value is not None else None,
            "stop": position.stop,
            "take_profit": position.take_profit,
        })
    trades = ledger.trades(name)
    realized = sum(t["pnl"] for t in trades)
    summary = {
        "positions": positions,
        "trades": len(trades),
        "realized_pnl": round(realized, 4),
        "unrealized_pnl": round(unrealized, 4),
    }
    if capital:
        summary["return_pct"] = round((realized + unrealized) / capital * 100, 4)
    return summary


def _position_timeframe(symbol: str) -> str | None:
    managed = promotion.managed_symbols().get(symbol)
    if managed:
        return managed.timeframe
    try:
        return load_strategy(resolve_strategy_name(symbol)).timeframe
    except SpecError:
        return None


def _status() -> dict[str, Any]:
    cfg = config()
    beat = heartbeat.read()
    mode = beat["mode"] if beat else None
    books = {}
    for name in (ledger.LIVE, ledger.DRY_RUN):
        summary = _book_summary(name, float(cfg.risk["max_live_capital"]))
        if summary["positions"] or summary["trades"]:
            books[name] = summary
    paper = {}
    for candidate in promotion.candidates(states=(promotion.PAPER,)):
        meta = ledger.load_meta(candidate.book_name)
        paper[f"{candidate.strategy}/{candidate.symbol}"] = {
            "equity": meta.get("equity"),
            "max_drawdown_pct": meta.get("max_drawdown_pct"),
            "trades": len(ledger.trades(candidate.book_name)),
            "days": round((time.time() - candidate.entered_at) / 86400, 2),
        }
    warnings = []
    if beat is None:
        warnings.append("bot_not_running: no heartbeat")
    elif not beat["fresh"]:
        warnings.append(f"bot_stalled: last heartbeat {beat['age_seconds']:.0f} s ago")
    if cfg.confluence_enabled or cfg.market_regime_enabled:
        warnings.append("live_sizing_differs_from_backtest")
    for symbol in cfg.secondary_mint_symbols:
        if costs.model_for(symbol).source == "default":
            warnings.append(f"costs_default: {symbol}")
    usage = rpc.usage.today()
    return {
        "mode": mode,
        "heartbeat": beat,
        "paused": promotion.paused(),
        "tokens": list(cfg.secondary_mint_symbols),
        "books": books,
        "paper": paper,
        "live_candidates": [c.as_dict() for c in promotion.candidates(states=promotion.MANAGED)],
        "rpc_today": usage,
        "sentiment": sentiment.stats(),
        "fear_greed": fear_greed.stats(),
        "warnings": warnings,
    }


def cmd_status(args: argparse.Namespace) -> dict[str, Any]:
    if args.healthcheck:
        beat = heartbeat.read()
        if beat is None or not beat["fresh"]:
            raise CliError("unhealthy", "no fresh heartbeat", [beat] if beat else [])
        return {"healthy": True, "heartbeat": beat}
    return _status()


def cmd_report(args: argparse.Namespace) -> dict[str, Any]:
    del args
    status = _status()
    lines = ["**SolTrade report**"]
    beat = status["heartbeat"]
    if beat:
        state = "running" if beat["fresh"] else f"stalled {beat['age_seconds'] / 60:.0f} min"
        lines.append(f"Bot: {state}, mode {beat['mode']}")
    else:
        lines.append("Bot: not running")
    if status["paused"]:
        lines.append(f"KILL SWITCH: paused since {status['paused'].get('at')} "
                     f"(drawdown {status['paused'].get('drawdown_pct')}%)")
    for name, book in status["books"].items():
        ret = book.get("return_pct")
        lines.append(
            f"{name}: {book['trades']} trades, realized {book['realized_pnl']:+.2f}, "
            f"open {book['unrealized_pnl']:+.2f}" + (f" ({ret:+.2f}%)" if ret is not None else "")
        )
        for p in book["positions"]:
            pnl = p["unrealized_pnl"]
            lines.append(f"  {p['symbol']} ({p['strategy']}): {pnl:+.2f}" if pnl is not None else f"  {p['symbol']}")
    for c in status["live_candidates"]:
        lines.append(f"Live: {c['strategy']} on {c['symbol']} ({c['state']}) since {c['since']}")
    for key, p in status["paper"].items():
        equity = p["equity"]
        lines.append(f"Paper: {key} day {p['days']:.0f}, {p['trades']} trades"
                     + (f", equity {equity:.2f}" if equity is not None else ""))
    recent = promotion.transitions(limit=5)
    if recent:
        lines.append("Recent:")
        lines.extend(f"  {t['at']} {t['strategy']}/{t['symbol']} {t['from']}→{t['to']}: {t['reason'][:80]}" for t in recent)
    usage = status["rpc_today"]
    lines.append(f"RPC today: {usage.get('calls', 0)} calls, {usage.get('cu', 0)} CU")
    feed = status["sentiment"]
    if feed.get("feeds"):
        day = feed.get("last_24h", {})
        lines.append(f"Sentiment: {feed['items']} items, {day.get('new_items', 0)} new today, "
                     f"{day.get('gaps', 0)} gaps, {day.get('errors', 0)} errors")
    if status["warnings"]:
        lines.append("Warnings: " + "; ".join(status["warnings"]))
    text = "\n".join(lines)
    if len(text) > REPORT_LIMIT:
        text = text[: REPORT_LIMIT - 1] + "…"
    return {"text": text, "chars": len(text)}


def cmd_approve(args: argparse.Namespace) -> dict[str, Any]:
    _require_tty("approve")
    strategy = _strategy(args.strategy)
    try:
        return {"candidate": promotion.approve(strategy, args.symbol)}
    except promotion.Refused as e:
        raise _refused(e) from e


def cmd_reopen(args: argparse.Namespace) -> dict[str, Any]:
    _require_tty("reopen")
    strategy = _strategy(args.strategy)
    try:
        return {"transition": promotion.reopen(strategy, args.symbol)}
    except promotion.Refused as e:
        raise _refused(e) from e


def cmd_resume(args: argparse.Namespace) -> dict[str, Any]:
    del args
    _require_tty("resume")
    try:
        return promotion.resume()
    except promotion.Refused as e:
        raise _refused(e) from e


def _smoke_leg(result: dict[str, Any], sent_symbol: str, out_symbol: str, sol_usd: float) -> dict[str, Any]:
    """One swap's audit: quoted vs filled, slippage against the quote, fees."""
    order = result.get("order") or {}
    quoted, filled = float(result.get("quoted_out_amount") or 0), float(result["out_amount"])
    signature_fee = int(order.get("signatureFeeLamports") or 0)
    priority_fee = int(order.get("prioritizationFeeLamports") or 0)
    gasless = bool(order.get("gasless"))
    return {
        "signature": result.get("signature"),
        "request_id": order.get("requestId"),
        "sent": result["sent_amount"],
        "sent_symbol": sent_symbol,
        "quoted_out": quoted,
        "filled_out": filled,
        "out_symbol": out_symbol,
        "slippage_bps": (quoted - filled) / quoted * 1e4 if quoted > 0 else None,
        "jupiter_fee_bps": order.get("feeBps"),
        "router": order.get("router"),
        "fees": {
            "gasless": gasless,
            "signature_fee_lamports": signature_fee,
            "prioritization_fee_lamports": priority_fee,
            "rent_fee_lamports": int(order.get("rentFeeLamports") or 0),
            # Gasless orders are paid by Jupiter, as in the cost model's probes.
            "network_fee_usd": 0.0 if gasless else (signature_fee + priority_fee) / _LAMPORTS_PER_SOL * sol_usd,
        },
    }


def _confirm(lines: list[str]) -> bool:
    """Ask on stderr (stdout carries only the JSON envelope); True on a typed ``yes``."""
    sys.stderr.write("\n".join(lines) + "\nType yes to send both swaps: ")
    sys.stderr.flush()
    return sys.stdin.readline().strip() == "yes"


def cmd_swap_smoke(args: argparse.Namespace) -> dict[str, Any]:
    """Buy SOL with the primary token, then sell exactly what arrived, through the live swap path.

    A supervised plumbing check: it never touches config.json, the books,
    promotion state or the kill switch.
    """
    _require_tty("swap-smoke")
    usd = float(args.usd)
    if not math.isfinite(usd) or usd <= 0:
        raise CliError("usage", "--usd must be a positive amount", exit_code=2)
    if usd > SMOKE_MAX_USD:
        raise CliError("over_cap", f"--usd {usd:g} is above the {SMOKE_MAX_USD:g} USD cap")
    cfg = config()
    primary, sol = cfg.primary_mint_symbol, cfg.symbol_for_mint(cfg.sol_mint) or "SOL"
    if cfg.primary_mint == cfg.sol_mint:
        raise CliError("primary_is_sol", "swap-smoke buys SOL with the primary token, so the primary can't be SOL")
    paused = promotion.paused()
    if paused is not None:
        raise CliError("paused", "the kill switch is on; clear it with resume first", [paused])
    held = [
        {"book": book, "symbol": position.symbol, "size": position.size}
        for book, position in ledger.open_positions()
        if book in (ledger.LIVE, ledger.DRY_RUN)
    ]
    if held:
        raise CliError("open_positions", f"the bot holds {len(held)} open position(s); run this when it holds none",
                       held)

    prices = jupiter.fetch_prices([cfg.primary_mint, cfg.sol_mint])
    primary_usd, sol_usd = prices.get(cfg.primary_mint, 0.0), prices.get(cfg.sol_mint, 0.0)
    if primary_usd <= 0 or sol_usd <= 0:
        raise CliError("no_price", f"no USD price for {primary} or {sol}", [prices])
    amount = usd / primary_usd
    primary_balance = wallet.find_balance(cfg.primary_mint)
    reserve = wallet.minimum_sol_needed()
    sol_spendable = wallet.find_balance(cfg.sol_mint)
    if primary_balance is None or sol_spendable is None:
        raise CliError("balance_unavailable", "the RPC did not return the wallet's balances")
    if primary_balance < amount:
        raise CliError("insufficient_funds", f"the wallet holds {primary_balance:g} {primary}; the smoke needs {amount:g}")
    if sol_spendable <= 0:
        raise CliError("insufficient_sol", f"the wallet holds less SOL than the {reserve:g} SOL fee reserve")

    if not _confirm([
        f"swap-smoke sends two REAL mainnet swaps from {cfg.public_address}:",
        f"  1. buy {sol} with {amount:g} {primary} (about {usd:g} USD)",
        f"  2. sell exactly the {sol} received back to {primary}",
        f"Wallet: {primary_balance:g} {primary}, {sol_spendable:g} {sol} above the {reserve:g} SOL reserve.",
    ]):
        raise CliError("not_confirmed", "nothing was sent: the answer was not yes")

    log_transaction.info(f"swap-smoke: buying {sol} with {amount:g} {primary} (supervised plumbing check)")
    buy = asyncio.run(transactions.perform_swap(amount, cfg.primary_mint, cfg.sol_mint, primary, sol))
    if not buy or float(buy["out_amount"]) <= 0:
        raise CliError("buy_failed", "the buy did not complete; nothing else was sent")
    buy_leg = _smoke_leg(buy, primary, sol, sol_usd)
    log_transaction.info(f"swap-smoke: selling {buy['out_amount']:g} {sol} back to {primary}")
    sell = asyncio.run(transactions.perform_swap(float(buy["out_amount"]), cfg.sol_mint, cfg.primary_mint, sol, primary))
    if not sell:
        raise CliError("sell_failed", f"the sell did not complete; the wallet keeps the {buy['out_amount']:g} {sol} bought",
                       [{"buy": buy_leg}])
    sell_leg = _smoke_leg(sell, sol, primary, sol_usd)

    spent_usd = amount * primary_usd
    back_usd = float(sell["out_amount"]) * primary_usd
    fees_usd = buy_leg["fees"]["network_fee_usd"] + sell_leg["fees"]["network_fee_usd"]
    cost_usd = spent_usd - back_usd + fees_usd
    model = costs.model_for(sol)
    model_usd = 2 * model.cost_usd(spent_usd)  # one buy and one sell of the same size
    cost_bps = cost_usd / spent_usd * 1e4
    model_bps = model_usd / spent_usd * 1e4
    log_transaction.info(
        f"swap-smoke: round trip cost {cost_usd:.4f} USD ({cost_bps:.1f} bps); model {model_bps:.1f} bps"
    )
    return {
        "usd": usd,
        "prices_usd": {primary: primary_usd, sol: sol_usd},
        "buy": buy_leg,
        "sell": sell_leg,
        "round_trip": {
            "spent_usd": spent_usd,
            "returned_usd": back_usd,
            "network_fees_usd": fees_usd,
            "cost_usd": cost_usd,
            "cost_bps": cost_bps,
        },
        "model": {
            "source": model.source,
            "measured_at": model.measured_at,
            "per_side_bps": model.bps(spent_usd),
            "round_trip_usd": model_usd,
            "round_trip_bps": model_bps,
            "actual_minus_model_bps": cost_bps - model_bps,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = _Parser(prog="sol_trade.cli", description="SolTrade research and pipeline commands (JSON output).")
    sub = parser.add_subparsers(dest="command", required=True, parser_class=_Parser)

    p = sub.add_parser("backfill", help="download and validate candle history")
    p.add_argument("--symbols", help="comma-separated symbols (default: traded tokens)")
    p.add_argument("--timeframes", help="comma-separated timeframes (default: backfill.default_timeframes)")
    p.add_argument("--days", type=int, help="days of history (default: backfill.default_days)")
    p.add_argument("--fear-greed", action="store_true",
                   help="fetch the Fear & Greed index history (sentiment.fear_greed_url) instead of candles")
    p.set_defaults(run=cmd_backfill)

    p = sub.add_parser("import-sentiment", help="import archived Reddit posts and comments (NDJSON)")
    p.add_argument("paths", nargs="+", help="NDJSON files, one Pushshift or Arctic Shift object per line")
    p.add_argument("--subreddits", help="comma-separated subreddits to keep (default: sentiment.subreddits)")
    p.add_argument("--after", help="keep items created on or after this UTC day (YYYY-MM-DD)")
    p.add_argument("--before", help="keep items created before this UTC day (YYYY-MM-DD)")
    p.set_defaults(run=cmd_import_sentiment)

    p = sub.add_parser("backtest", help="evaluate a strategy against the gates")
    p.add_argument("strategy")
    p.add_argument("symbol")
    p.set_defaults(run=cmd_backtest)

    p = sub.add_parser("candidates", help="the promotion pipeline")
    p.add_argument("--symbol")
    p.set_defaults(run=cmd_candidates)

    p = sub.add_parser("promote", help="advance a candidate: to paper, or paper to live")
    p.add_argument("strategy")
    p.add_argument("symbol")
    p.set_defaults(run=cmd_promote)

    p = sub.add_parser("demote", help="take a candidate out of the pipeline")
    p.add_argument("strategy")
    p.add_argument("symbol")
    p.add_argument("--reason", default="requested")
    p.set_defaults(run=cmd_demote)

    p = sub.add_parser("measure-costs", help="probe Jupiter quotes for the cost model")
    p.add_argument("--symbols", help="comma-separated symbols (default: traded tokens)")
    p.set_defaults(run=cmd_measure_costs)

    p = sub.add_parser("status", help="bot, books, pipeline and RPC usage")
    p.add_argument("--healthcheck", action="store_true", help="exit 1 unless the bot's heartbeat is fresh")
    p.set_defaults(run=cmd_status)

    p = sub.add_parser("report", help="a Discord-ready summary")
    p.set_defaults(run=cmd_report)

    p = sub.add_parser("approve", help="allow paper -> live (human, interactive terminal)")
    p.add_argument("strategy")
    p.add_argument("symbol")
    p.set_defaults(run=cmd_approve)

    p = sub.add_parser(
        "reopen", help="re-open a hash rejected under the old fixed held-out trade minimum (human, interactive terminal)"
    )
    p.add_argument("strategy")
    p.add_argument("symbol")
    p.set_defaults(run=cmd_reopen)

    p = sub.add_parser("resume", help="clear the kill switch (human, interactive terminal)")
    p.set_defaults(run=cmd_resume)

    p = sub.add_parser(
        "swap-smoke", help="send one small real buy and sell to check the live swap path (human, interactive terminal)"
    )
    p.add_argument("--usd", type=float, default=SMOKE_DEFAULT_USD,
                   help=f"size in USD (default {SMOKE_DEFAULT_USD:g}, at most {SMOKE_MAX_USD:g})")
    p.set_defaults(run=cmd_swap_smoke)
    return parser


def run(argv: list[str] | None = None, out: Callable[[str], Any] | None = None) -> int:
    """Run one command; prints the envelope and returns the exit code."""
    out = out or (lambda text: print(text, flush=True))
    command = None
    try:
        args = build_parser().parse_args(argv)
        command = args.command
        setup_file_logging()
        payload = {"ok": True, "command": command, **args.run(args)}
        code = 0
    except CliError as e:
        payload = {"ok": False, "command": command, "error": e.code, "message": e.message, "details": e.details}
        code = e.exit_code
    except (OSError, ValueError) as e:
        payload = {"ok": False, "command": command, "error": "failed", "message": f"{type(e).__name__}: {e}",
                   "details": []}
        code = 1
    except Exception as e:  # noqa: BLE001 - the contract is one JSON object, whatever happens
        log_general.exception(f"cli {command} failed")
        payload = {"ok": False, "command": command, "error": "internal", "message": f"{type(e).__name__}: {e}",
                   "details": []}
        code = 1
    out(json.dumps(_clean(payload), default=_jsonable))
    return code


def main() -> None:
    sys.exit(run())


if __name__ == "__main__":
    main()
