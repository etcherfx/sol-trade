"""JSON strategy specs: validation, hashing, loading and evaluation.

A strategy is ``strategies/{name}.json`` in the runtime folder (see the README
for the grammar). The engine compiles it once and evaluates it on windows of
closed bars: :meth:`Strategy.evaluate` takes ``(L, N)`` arrays (``lookback``
bars by ``N`` windows) and returns the entry and exit signals of each window's
last bar. Live trading passes one window, the backtester many.

Protective exits (stop-loss, take-profit, trailing stop) are not part of the
rules: the engine applies them from the ``risk`` block.
"""

import hashlib
import json
import math
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from sol_trade import indicators as ind
from sol_trade.config import config

SCHEMA = 1
COLUMNS = ("open", "high", "low", "close", "volume")
NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
TIMEFRAME_RE = re.compile(r"^([1-9][0-9]*)([mhdw])$")
_UNIT_SECONDS = {"m": 60, "h": 3600, "d": 86400, "w": 604800}
MAX_LOOKBACK = 5000
MAX_PERIOD = 1000
_BUILTIN_DIR = Path(__file__).with_name("builtin_strategies")

_TOP_KEYS = {"schema", "name", "description", "timeframe", "lookback", "risk", "indicators", "entry", "exit"}
_RISK_KEYS = ("stoploss", "takeprofit", "trailing_stoploss", "trailing_stoploss_target")
_POSITION_SIZE = "position_size"
_COMPARE: dict[str, Callable[[Any, Any], Any]] = {
    ">": np.greater,
    ">=": np.greater_equal,
    "<": np.less,
    "<=": np.less_equal,
    "==": np.equal,
    "!=": np.not_equal,
}
_ARITH: dict[str, Callable[[Any, Any], Any]] = {
    "add": np.add,
    "sub": np.subtract,
    "mul": np.multiply,
    "div": np.divide,
}


class SpecError(ValueError):
    """A strategy spec is invalid; ``errors`` holds one ``path: message`` per problem."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("; ".join(errors))


def timeframe_seconds(timeframe: str) -> int:
    match = TIMEFRAME_RE.match(timeframe or "")
    if not match:
        raise ValueError(f"invalid timeframe {timeframe!r}; use forms like 15m, 1h, 4h, 1d")
    return int(match.group(1)) * _UNIT_SECONDS[match.group(2)]


# --------------------------------------------------------------- functions


@dataclass(frozen=True)
class _Param:
    kind: str  # "int", "number" or "choice"
    default: Any = None
    low: float | None = None
    high: float | None = None
    choices: tuple[str, ...] = ()


def _period(default: int | None = None, low: int = 1) -> _Param:
    return _Param("int", default, low, MAX_PERIOD)


@dataclass(frozen=True)
class _Fn:
    params: dict[str, _Param]
    # Bars each output needs, given its params and the bars its input needs.
    need: Callable[[dict[str, Any], int], dict[str, int]]
    outputs: tuple[str, ...] = ()  # empty: a single unnamed output
    takes_input: bool = True


def _same(own: Callable[[dict[str, Any]], int]) -> Callable[[dict[str, Any], int], dict[str, int]]:
    return lambda p, n: {"": n + own(p) - 1}


_FUNCTIONS: dict[str, _Fn] = {
    "sma": _Fn({"period": _period()}, _same(lambda p: p["period"])),
    "ema": _Fn({"period": _period()}, _same(lambda p: p["period"])),
    "rsi": _Fn({"period": _period(14)}, _same(lambda p: p["period"] + 1)),
    "stddev": _Fn(
        {"period": _period(low=2), "ddof": _Param("int", 1, 0, 1)}, _same(lambda p: p["period"])
    ),
    "bbands": _Fn(
        {"period": _period(20, low=2), "mult": _Param("number", 2.0, 0.0, 100.0), "ddof": _Param("int", 1, 0, 1)},
        lambda p, n: dict.fromkeys(("upper", "middle", "lower", "width"), n + p["period"] - 1),
        ("upper", "middle", "lower", "width"),
    ),
    "atr": _Fn(
        {"period": _period(14), "method": _Param("choice", "wilder", choices=("wilder", "sma"))},
        lambda p, n: {"": p["period"] + (1 if p["method"] == "wilder" else 0)},
        takes_input=False,
    ),
    "macd": _Fn(
        {"fast": _period(12), "slow": _period(26), "signal": _period(9)},
        lambda p, n: {
            "macd": n + max(p["fast"], p["slow"]) - 1,
            "signal": n + max(p["fast"], p["slow"]) + p["signal"] - 2,
            "hist": n + max(p["fast"], p["slow"]) + p["signal"] - 2,
        },
        ("macd", "signal", "hist"),
    ),
    "stoch": _Fn(
        {"k": _period(14), "d": _period(3), "smooth": _period(3)},
        lambda p, n: {
            "k": p["k"] + p["smooth"] - 1,
            "d": p["k"] + p["smooth"] + p["d"] - 2,
        },
        ("k", "d"),
        takes_input=False,
    ),
    "roc": _Fn({"period": _period()}, _same(lambda p: p["period"] + 1)),
    "highest": _Fn({"period": _period()}, _same(lambda p: p["period"])),
    "lowest": _Fn({"period": _period()}, _same(lambda p: p["period"])),
    "zscore": _Fn(
        {"period": _period(low=2), "ddof": _Param("int", 1, 0, 1)}, _same(lambda p: p["period"])
    ),
    "sentiment": _Fn(
        {"window_hours": _Param("number", 24.0, 0.01, 24.0 * 90)},
        lambda p, n: {"score": 1, "mentions": 1},
        ("score", "mentions"),
        takes_input=False,
    ),
}


# -------------------------------------------------------------- evaluation


@dataclass
class EvalContext:
    """Bar windows for one evaluation; arrays are ``(L, N)``."""

    columns: dict[str, np.ndarray]
    times: np.ndarray  # bar open times in seconds, (L, N)
    period_seconds: int
    symbol: str | None = None
    # (symbol, bar close times (L, N), window_hours) -> {"score", "mentions"}
    sentiment: Callable[[str | None, np.ndarray, float], dict[str, np.ndarray]] | None = None
    values: dict[str, Any] = field(default_factory=dict)


Operand = Callable[[EvalContext], Any]
Rule = Callable[[EvalContext], np.ndarray]


def _run_indicator(fn: str, params: dict[str, Any], source: Operand | None, ctx: EvalContext) -> Any:
    cols = ctx.columns
    if fn == "atr":
        return ind.atr(cols["high"], cols["low"], cols["close"], params["period"], params["method"])
    if fn == "stoch":
        return ind.stoch(cols["high"], cols["low"], cols["close"], params["k"], params["d"], params["smooth"])
    if fn == "sentiment":
        closes = ctx.times + ctx.period_seconds
        if ctx.sentiment is None:
            nan = np.full(closes.shape, np.nan)
            return {"score": nan, "mentions": nan}
        return ctx.sentiment(ctx.symbol, closes, float(params["window_hours"]))
    assert source is not None
    x = np.broadcast_to(np.asarray(source(ctx), dtype=float), ctx.columns["close"].shape).astype(float)
    if fn == "sma":
        return ind.sma(x, params["period"])
    if fn == "ema":
        return ind.ema(x, params["period"])
    if fn == "rsi":
        return ind.rsi(x, params["period"])
    if fn == "stddev":
        return ind.stddev(x, params["period"], params["ddof"])
    if fn == "bbands":
        return ind.bbands(x, params["period"], params["mult"], params["ddof"])
    if fn == "macd":
        return ind.macd(x, params["fast"], params["slow"], params["signal"])
    if fn == "roc":
        return ind.roc(x, params["period"])
    if fn == "highest":
        return ind.highest(x, params["period"])
    if fn == "lowest":
        return ind.lowest(x, params["period"])
    if fn == "zscore":
        return ind.zscore(x, params["period"], params["ddof"])
    raise AssertionError(fn)


# --------------------------------------------------------------- compiler


class _Compiler:
    """Validates a spec while compiling it; collects every error with its path."""

    def __init__(self, spec: Any, expected_name: str | None) -> None:
        self.spec = spec
        self.expected_name = expected_name
        self.errors: list[str] = []
        self.indicator_specs: dict[str, Any] = {}
        self.needs: dict[str, dict[str, int]] = {}  # indicator -> output -> bars needed
        self.compiled: dict[str, Callable[[EvalContext], None]] = {}
        self.order: list[str] = []
        self._visiting: list[str] = []
        self._failed: set[str] = set()
        self.uses_sentiment = False

    def error(self, path: str, message: str) -> None:
        self.errors.append(f"{path}: {message}")

    # -- top level

    def compile(self) -> "Strategy":
        spec = self.spec
        if not isinstance(spec, dict):
            raise SpecError(["$: a strategy must be a JSON object"])
        for key in sorted(set(spec) - _TOP_KEYS):
            self.error(key, f"unknown key; allowed: {', '.join(sorted(_TOP_KEYS))}")
        if spec.get("schema") != SCHEMA:
            self.error("schema", f"must be {SCHEMA}")
        name = spec.get("name")
        if not isinstance(name, str) or not NAME_RE.match(name):
            self.error("name", "must be lowercase letters, digits and underscores, starting with a letter")
        elif self.expected_name is not None and name != self.expected_name:
            self.error("name", f"{name!r} must equal the file name {self.expected_name!r}")
        if "description" in spec and not isinstance(spec["description"], str):
            self.error("description", "must be a string")
        timeframe = spec.get("timeframe")
        period_seconds = 0
        try:
            period_seconds = timeframe_seconds(timeframe if isinstance(timeframe, str) else "")
        except ValueError as e:
            self.error("timeframe", str(e))
        lookback = spec.get("lookback")
        if not (isinstance(lookback, int) and not isinstance(lookback, bool) and 2 <= lookback <= MAX_LOOKBACK):
            self.error("lookback", f"must be an integer from 2 to {MAX_LOOKBACK}")
            lookback = None
        risk = self._risk(spec.get("risk"))

        indicators = spec.get("indicators", {})
        if not isinstance(indicators, dict):
            self.error("indicators", "must be an object of name: {fn, ...}")
            indicators = {}
        self.indicator_specs = indicators
        for ind_name in indicators:
            if not isinstance(ind_name, str) or not NAME_RE.match(ind_name):
                self.error(f"indicators.{ind_name}", "names are lowercase letters, digits and underscores")
            elif ind_name in COLUMNS:
                self.error(f"indicators.{ind_name}", "name clashes with a bar column")
            else:
                self._indicator(ind_name)

        if "entry" not in spec:
            self.error("entry", "required")
            entry, entry_need = None, 0
        else:
            entry, entry_need = self._rule(spec["entry"], "entry")
        if "exit" in spec:
            exit_rule, exit_need = self._rule(spec["exit"], "exit")
        else:
            exit_rule, exit_need = None, 0
        warmup = max(entry_need, exit_need, 1)
        if lookback is not None and lookback < warmup:
            self.error("lookback", f"{lookback} is smaller than the required warm-up of {warmup} bars")
        if self.errors:
            raise SpecError(self.errors)

        canonical = canonical_json(spec)
        assert entry is not None and lookback is not None and isinstance(name, str)
        return Strategy(
            name=name,
            description=spec.get("description", ""),
            timeframe=str(timeframe),
            period_seconds=period_seconds,
            lookback=int(lookback),
            warmup=warmup,
            risk=risk,
            spec=spec,
            hash=hashlib.sha256(canonical.encode()).hexdigest(),
            uses_sentiment=self.uses_sentiment,
            _order=[self.compiled[n] for n in self.order],
            _entry=entry,
            _exit=exit_rule,
        )

    def _risk(self, risk: Any) -> "Risk":
        values: dict[str, float] = {}
        if risk is None:
            risk = {}
        if not isinstance(risk, dict):
            self.error("risk", "must be an object")
            risk = {}
        allowed = (*_RISK_KEYS, _POSITION_SIZE)
        for key in sorted(set(risk) - set(allowed)):
            self.error(f"risk.{key}", f"unknown key; allowed: {', '.join(allowed)}")
        for key in _RISK_KEYS:
            value = risk.get(key)
            if value is None:
                values[key] = 0.0
            elif not _is_number(value) or value < 0 or value >= 1000:
                self.error(f"risk.{key}", "must be a percentage from 0 to 1000, or null")
            else:
                values[key] = float(value)
        if 0 < values.get("stoploss", 0) >= 100:
            self.error("risk.stoploss", "must be below 100 percent")
        if 0 < values.get("trailing_stoploss", 0) >= 100:
            self.error("risk.trailing_stoploss", "must be below 100 percent")
        size = risk.get(_POSITION_SIZE)
        if size is None:
            size = 1.0
        elif not _is_number(size) or not 0 < size <= 1:
            self.error(f"risk.{_POSITION_SIZE}", "must be a fraction of risk.max_trade_size above 0 and up to 1, or null")
            size = 1.0
        return Risk(**{k: values.get(k, 0.0) for k in _RISK_KEYS}, position_size=float(size))

    # -- indicators

    def _indicator(self, name: str) -> dict[str, int] | None:
        if name in self.needs:
            return self.needs[name]
        if name in self._failed:
            return None
        needs = self._build_indicator(name)
        if needs is None and name not in self._visiting:
            self._failed.add(name)
        return needs

    def _build_indicator(self, name: str) -> dict[str, int] | None:
        if name in self._visiting:
            cycle = " -> ".join([*self._visiting[self._visiting.index(name) :], name])
            self.error(f"indicators.{name}", f"circular reference: {cycle}")
            return None
        path = f"indicators.{name}"
        spec = self.indicator_specs.get(name)
        if not isinstance(spec, dict):
            self.error(path, "must be an object like {\"fn\": \"ema\", \"period\": 21}")
            return None
        fn_name = spec.get("fn")
        fn = _FUNCTIONS.get(fn_name) if isinstance(fn_name, str) else None
        if fn is None:
            self.error(f"{path}.fn", f"unknown fn {fn_name!r}; known: {', '.join(sorted(_FUNCTIONS))}")
            return None
        allowed = {"fn", *fn.params, *(("input",) if fn.takes_input else ())}
        for key in sorted(set(spec) - allowed):
            self.error(f"{path}.{key}", f"unknown parameter for {fn_name}; allowed: {', '.join(sorted(allowed - {'fn'}))}")
        params: dict[str, Any] = {}
        for key, param in fn.params.items():
            params[key] = self._param(f"{path}.{key}", spec.get(key), param)
        if fn_name == "macd" and _is_int(params["fast"]) and _is_int(params["slow"]) and params["fast"] >= params["slow"]:
            self.error(f"{path}.fast", "must be smaller than slow")
        source: Operand | None = None
        input_need = 1
        self._visiting.append(name)
        try:
            if fn.takes_input:
                compiled_input = self._operand(spec.get("input", "close"), f"{path}.input")
                if compiled_input is not None:
                    source, input_need = compiled_input
        finally:
            self._visiting.pop()
        if any(v is None for v in params.values()) or (fn.takes_input and source is None):
            return None
        if fn_name == "sentiment":
            self.uses_sentiment = True
        needs = fn.need(params, input_need)
        self.needs[name] = needs

        fn_key = str(fn_name)

        def run(ctx: EvalContext) -> None:
            ctx.values[name] = _run_indicator(fn_key, params, source, ctx)

        self.compiled[name] = run
        self.order.append(name)
        return needs

    def _param(self, path: str, value: Any, param: _Param) -> Any:
        if value is None:
            if param.default is None:
                self.error(path, "required")
            return param.default
        if param.kind == "choice":
            if value not in param.choices:
                self.error(path, f"must be one of {', '.join(param.choices)}")
                return None
            return value
        ok = _is_int(value) if param.kind == "int" else _is_number(value)
        if not ok or (param.low is not None and value < param.low) or (param.high is not None and value > param.high):
            kind = "an integer" if param.kind == "int" else "a number"
            self.error(path, f"must be {kind} from {param.low:g} to {param.high:g}")
            return None
        return value

    # -- operands

    def _operand(self, value: Any, path: str) -> tuple[Operand, int] | None:
        if _is_number(value):
            number = float(value)
            return (lambda ctx: number), 1
        if isinstance(value, str):
            return self._reference(value, path)
        if isinstance(value, dict) and len(value) == 1:
            (op, args), = value.items()
            if op == "shift":
                if not (isinstance(args, list) and len(args) == 2):
                    self.error(path, "shift takes [operand, bars]")
                    return None
                bars = args[1]
                if not _is_int(bars) or not 0 <= bars <= MAX_PERIOD:
                    self.error(f"{path}.shift[1]", f"bars must be an integer from 0 to {MAX_PERIOD}")
                    return None
                inner = self._operand(args[0], f"{path}.shift[0]")
                if inner is None:
                    return None
                source, need = inner
                return (lambda ctx: _shift(source(ctx), bars)), need + bars
            if op in _ARITH:
                if not (isinstance(args, list) and len(args) == 2):
                    self.error(path, f"{op} takes [a, b]")
                    return None
                left = self._operand(args[0], f"{path}.{op}[0]")
                right = self._operand(args[1], f"{path}.{op}[1]")
                if left is None or right is None:
                    return None
                func = _ARITH[op]
                a, b = left[0], right[0]

                def arith(ctx: EvalContext) -> Any:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        return func(a(ctx), b(ctx))

                return arith, max(left[1], right[1])
        self.error(path, "expected a number, a column or indicator name, or {shift|add|sub|mul|div: [...]}")
        return None

    def _reference(self, ref: str, path: str) -> tuple[Operand, int] | None:
        if ref in COLUMNS:
            return (lambda ctx: ctx.columns[ref]), 1
        base, _, output = ref.partition(".")
        if base not in self.indicator_specs:
            known = [*COLUMNS, *self.indicator_specs]
            self.error(path, f"unknown series {ref!r}; known: {', '.join(known)}")
            return None
        needs = self._indicator(base)
        if needs is None:
            return None
        if output:
            if output not in needs or output == "":
                outputs = [o for o in needs if o]
                hint = f"; outputs: {', '.join(outputs)}" if outputs else " (it has a single output)"
                self.error(path, f"{base!r} has no output {output!r}{hint}")
                return None
            return (lambda ctx: ctx.values[base][output]), needs[output]
        if "" not in needs:
            self.error(path, f"{base!r} has several outputs; use one of {', '.join(f'{base}.{o}' for o in needs)}")
            return None
        return (lambda ctx: ctx.values[base]), needs[""]

    # -- rules

    def _rule(self, rule: Any, path: str) -> tuple[Rule | None, int]:
        if isinstance(rule, list):
            if len(rule) != 3 or rule[1] not in _COMPARE:
                self.error(path, f"a comparison is [a, op, b] with op one of {' '.join(_COMPARE)}")
                return None, 0
            left = self._operand(rule[0], f"{path}[0]")
            right = self._operand(rule[2], f"{path}[2]")
            if left is None or right is None:
                return None, 0
            compare = _COMPARE[rule[1]]
            a, b = left[0], right[0]

            def comparison(ctx: EvalContext) -> np.ndarray:
                with np.errstate(invalid="ignore"):
                    return np.broadcast_to(compare(a(ctx), b(ctx)), ctx.columns["close"].shape)

            return comparison, max(left[1], right[1])
        if isinstance(rule, dict) and len(rule) == 1:
            (key, args), = rule.items()
            if key in ("all", "any"):
                if not isinstance(args, list) or not args:
                    self.error(f"{path}.{key}", "must be a non-empty list of rules")
                    return None, 0
                parts = [self._rule(item, f"{path}.{key}[{i}]") for i, item in enumerate(args)]
                if any(part is None for part, _ in parts):
                    return None, 0
                rules = [part for part, _ in parts]
                need = max(n for _, n in parts)
                combine = np.logical_and if key == "all" else np.logical_or

                def group(ctx: EvalContext) -> np.ndarray:
                    result = rules[0](ctx)  # type: ignore[misc]
                    for part in rules[1:]:
                        result = combine(result, part(ctx))  # type: ignore[misc]
                    return result

                return group, need
            if key == "not":
                inner, need = self._rule(args, f"{path}.not")
                if inner is None:
                    return None, 0
                return (lambda ctx: np.logical_not(inner(ctx))), need
            if key in ("crosses_above", "crosses_below"):
                if not (isinstance(args, list) and len(args) == 2):
                    self.error(f"{path}.{key}", "takes [a, b]")
                    return None, 0
                left = self._operand(args[0], f"{path}.{key}[0]")
                right = self._operand(args[1], f"{path}.{key}[1]")
                if left is None or right is None:
                    return None, 0
                a, b = left[0], right[0]
                above = key == "crosses_above"

                def cross(ctx: EvalContext) -> np.ndarray:
                    shape = ctx.columns["close"].shape
                    x = np.broadcast_to(np.asarray(a(ctx), dtype=float), shape)
                    y = np.broadcast_to(np.asarray(b(ctx), dtype=float), shape)
                    with np.errstate(invalid="ignore"):
                        if above:
                            now, before = x > y, _shift(x, 1) <= _shift(y, 1)
                        else:
                            now, before = x < y, _shift(x, 1) >= _shift(y, 1)
                    return now & before

                return cross, max(left[1], right[1]) + 1
        self.error(
            path,
            "expected [a, op, b], {all: [...]}, {any: [...]}, {not: rule}, "
            "{crosses_above: [a, b]} or {crosses_below: [a, b]}",
        )
        return None, 0


def _shift(x: Any, bars: int) -> Any:
    array = np.asarray(x, dtype=float)
    if array.ndim < 2:
        return array  # constants don't move
    return ind.shift(array, bars)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def canonical_json(spec: Any) -> str:
    return json.dumps(spec, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


# --------------------------------------------------------------- strategy


@dataclass(frozen=True)
class Risk:
    """Protective exits in percent; 0 disables one.

    The trailing stop sits ``trailing_stoploss`` percent under the highest
    closed-bar high once a high reaches ``trailing_stoploss_target`` percent
    above entry (0: from entry on). ``position_size`` is the share of
    ``risk.max_trade_size`` an entry spends.
    """

    stoploss: float = 0.0
    takeprofit: float = 0.0
    trailing_stoploss: float = 0.0
    trailing_stoploss_target: float = 0.0
    position_size: float = 1.0


@dataclass
class Evaluation:
    entry: np.ndarray  # bool, one per window
    exit: np.ndarray
    values: dict[str, np.ndarray]  # last-bar indicator values, one per window


@dataclass
class Strategy:
    name: str
    description: str
    timeframe: str
    period_seconds: int
    lookback: int
    warmup: int
    risk: Risk
    spec: dict[str, Any]
    hash: str
    uses_sentiment: bool
    _order: list[Callable[[EvalContext], None]]
    _entry: Rule
    _exit: Rule | None

    def evaluate(
        self,
        bars: dict[str, np.ndarray],
        times: np.ndarray,
        symbol: str | None = None,
        sentiment: Callable[[str | None, np.ndarray, float], dict[str, np.ndarray]] | None = None,
    ) -> Evaluation:
        """Signals for the last bar of each window; arrays are ``(L, N)``."""
        columns = {c: np.asarray(bars[c], dtype=float) for c in COLUMNS}
        ctx = EvalContext(columns, times, self.period_seconds, symbol, sentiment)
        for step in self._order:
            step(ctx)
        entry = np.asarray(self._entry(ctx))[-1].astype(bool)
        exit_ = (
            np.asarray(self._exit(ctx))[-1].astype(bool)
            if self._exit is not None
            else np.zeros(entry.shape, dtype=bool)
        )
        values: dict[str, np.ndarray] = {}
        for name, value in ctx.values.items():
            if isinstance(value, dict):
                for output, array in value.items():
                    values[f"{name}.{output}"] = np.asarray(array)[-1]
            else:
                values[name] = np.asarray(value)[-1]
        return Evaluation(entry, exit_, values)


def compile_spec(spec: Any, expected_name: str | None = None) -> Strategy:
    """Validate and compile a parsed spec; raises :class:`SpecError`."""
    return _Compiler(spec, expected_name).compile()


def parse_spec(text: str, expected_name: str | None = None) -> Strategy:
    try:
        spec = json.loads(text)
    except json.JSONDecodeError as e:
        raise SpecError([f"$: invalid JSON at line {e.lineno} column {e.colno}: {e.msg}"]) from e
    return compile_spec(spec, expected_name)


# ----------------------------------------------------------------- loading

_cache: dict[str, tuple[tuple[int, int], Strategy]] = {}


def strategies_dir() -> Path:
    return Path(config().strategies_path)


def strategy_path(name: str) -> Path:
    return strategies_dir() / f"{name}.json"


def load_strategy(name: str) -> Strategy:
    """Load ``strategies/{name}.json``; ``default`` falls back to the built-in spec.

    Raises :class:`SpecError` for a missing, Python-only or invalid strategy.
    """
    if not isinstance(name, str) or not NAME_RE.match(name):
        raise SpecError([f"$: invalid strategy name {name!r}"])
    path = strategy_path(name)
    if not path.is_file():
        if name == "default":
            path = _BUILTIN_DIR / "default.json"
        elif (strategies_dir() / f"{name}_strategy.py").is_file():
            raise SpecError(
                [f"$: Python strategies are no longer supported; convert to strategies/{name}.json"]
            )
        else:
            raise SpecError([f"$: strategy {name!r} not found: expected {path}"])
    try:
        stat = os.stat(path)
        key = (stat.st_mtime_ns, stat.st_size)
        cached = _cache.get(str(path.resolve()))
        if cached and cached[0] == key:
            return cached[1]
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        raise SpecError([f"$: cannot read {path}: {e}"]) from e
    strategy = parse_spec(text, expected_name=name)
    _cache[str(path.resolve())] = (key, strategy)
    return strategy


def resolve_strategy_name(symbol: str | None = None) -> str:
    """Pick the strategy for a token: per-token override, else the global one."""
    name = config().strategy or "default"
    if symbol:
        name = config().token_strategies.get(symbol) or name
    return name


def windows(values: np.ndarray, length: int) -> np.ndarray:
    """All ``length``-bar windows of a 1-D series as an ``(length, N)`` array."""
    view = np.lib.stride_tricks.sliding_window_view(np.asarray(values, dtype=float), length)
    return np.ascontiguousarray(view.T)
