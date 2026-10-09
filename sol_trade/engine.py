"""The decision core shared by live trading, paper books and the backtester.

A book holds cash in the primary token and at most one position per symbol.
For every closed bar of a symbol the caller runs the same three steps, in
this order, whether the bar comes from history or from the exchange:

1. :meth:`Book.on_open` fills the action decided at the previous close
   (a buy or a signal/take-profit sell) at this bar's open.
2. :meth:`Book.on_low` checks the stop-loss and trailing stop against the
   bar's low using the levels frozen at the previous close; a hit fills at
   ``min(level, open)``. Live trading also runs this on the forming bar's
   running low every cycle, so wicks between polls are caught.
3. :meth:`Book.on_close` ratchets the trailing stop with the bar's high,
   checks take-profit (``close >= level``) and the strategy's signals, and
   queues the resulting action for the next open.

Fills go through an executor: :class:`SimExecutor` fills at the reference
price minus modeled costs (backtests, paper books, ``--dry-run``); the live
executor swaps through Jupiter.
"""

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from sol_trade.costs import CostModel
from sol_trade.spec import Risk

BUY = "buy"
SELL = "sell"

# Exit reasons
STOPLOSS = "stoploss"
TRAILING = "trailing_stop"
TAKEPROFIT = "takeprofit"
SIGNAL = "signal"
FORCED = "forced"
DUST = "dust"
DEMOTED = "demoted"
KILL_SWITCH = "kill_switch"

_MIN_ORDER = 1.0  # smallest notional (primary units) worth sending


@dataclass
class Position:
    symbol: str
    strategy: str
    strategy_hash: str
    entry_price: float
    size: float
    cost_basis: float  # primary spent, costs included
    entry_ts: int  # open time of the bar the entry filled in
    stop: float | None = None
    take_profit: float | None = None
    trailing_active: bool = False
    highest_price: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Position":
        return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})


@dataclass
class Action:
    side: str  # BUY or SELL
    reason: str
    signal_ts: int  # close of this bar decided it
    strategy: str = ""
    strategy_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Action":
        return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})


@dataclass
class Fill:
    symbol: str
    side: str
    ts: int  # open time of the bar it filled in
    price: float  # effective price per token, before costs
    size: float  # tokens bought or sold
    notional: float  # primary value at ``price``
    cost: float  # all costs (primary): the in-swap share plus ``fee``
    fee: float  # network fee paid outside the swap amounts
    reason: str
    modeled_cost: float  # what the cost model expected
    quoted_cost: float | None = None  # from a Jupiter quote, when known
    strategy: str = ""
    strategy_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Trade:
    symbol: str
    strategy: str
    strategy_hash: str
    entry_ts: int
    exit_ts: int
    entry_price: float
    exit_price: float
    size: float
    cost_basis: float
    proceeds: float
    reason: str

    @property
    def pnl(self) -> float:
        return self.proceeds - self.cost_basis

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "pnl": self.pnl}


# ---------------------------------------------------------- pure functions


def entry_levels(entry_price: float, risk: Risk) -> tuple[float | None, float | None]:
    """Fixed stop-loss and take-profit levels for an entry (None when disabled)."""
    stop = entry_price * (1 - risk.stoploss / 100) if risk.stoploss > 0 else None
    take_profit = entry_price * (1 + risk.takeprofit / 100) if risk.takeprofit > 0 else None
    return stop, take_profit


def trailing_level(position: Position, risk: Risk) -> float | None:
    if risk.trailing_stoploss <= 0 or not position.trailing_active:
        return None
    return position.highest_price * (1 - risk.trailing_stoploss / 100)


def stop_exit(position: Position, risk: Risk, bar_open: float, low: float) -> tuple[float, str] | None:
    """``(fill price, reason)`` when the bar's low reaches the stop, else None."""
    trailing = trailing_level(position, risk)
    candidates = [(level, reason) for level, reason in ((position.stop, STOPLOSS), (trailing, TRAILING)) if level]
    if not candidates:
        return None
    level, reason = max(candidates, key=lambda c: c[0])  # the higher stop binds
    if low <= level:
        return min(level, bar_open), reason
    return None


def update_trailing(position: Position, risk: Risk, high: float) -> None:
    """Activate the trailing stop and ratchet its peak with a closed bar's high."""
    if risk.trailing_stoploss <= 0:
        return
    if not position.trailing_active:
        target = position.entry_price * (1 + risk.trailing_stoploss_target / 100)
        if high >= target:
            position.trailing_active = True
    if position.trailing_active:
        position.highest_price = max(position.highest_price, high)


def decide_close(
    position: Position | None, close: float, entry: bool, exit_: bool
) -> tuple[str, str] | None:
    """The action a closed bar asks for: ``(side, reason)`` or None to hold."""
    if position is not None:
        if position.take_profit is not None and close >= position.take_profit:
            return SELL, TAKEPROFIT
        if exit_:
            return SELL, SIGNAL
        return None
    if entry:
        return BUY, SIGNAL
    return None


def size_order(
    cash: float,
    deployed: float,
    max_trade_size: float,
    max_capital: float,
    weight_cap: float | None = None,
) -> float:
    """Notional for a new entry: ``min(max trade, cash, capital left)``, capped by a weight slice."""
    notional = min(max_trade_size, cash, max_capital - deployed)
    if weight_cap is not None:
        notional = min(notional, weight_cap)
    return max(notional, 0.0)


# ---------------------------------------------------------------- execution


class Executor(Protocol):
    """Turns decisions into fills. ``buy`` spends ``notional``; ``sell`` sells ``size`` tokens."""

    def network_fee(self, symbol: str) -> float:
        """Expected fee per swap outside the swap amounts (reserved before sizing)."""
        raise NotImplementedError

    def buy(self, symbol: str, notional: float, price: float, ts: int, reason: str) -> Fill | None:
        raise NotImplementedError

    def sell(self, symbol: str, size: float, price: float, ts: int, reason: str) -> Fill | None:
        raise NotImplementedError


class SimExecutor:
    """Fills at the reference price; costs come from the cost model."""

    def __init__(self, model_for: Callable[[str], CostModel]) -> None:
        self.model_for = model_for

    def network_fee(self, symbol: str) -> float:
        return self.model_for(symbol).network_fee_usd

    def buy(self, symbol: str, notional: float, price: float, ts: int, reason: str) -> Fill | None:
        if price <= 0:
            return None
        model = self.model_for(symbol)
        cost = model.cost_usd(notional)
        size = notional * (1 - model.bps(notional) / 1e4) / price
        return Fill(symbol, BUY, ts, price, size, notional, cost, model.network_fee_usd, reason, cost)

    def sell(self, symbol: str, size: float, price: float, ts: int, reason: str) -> Fill | None:
        if price <= 0:
            return None
        model = self.model_for(symbol)
        notional = size * price
        cost = model.cost_usd(notional)
        return Fill(symbol, SELL, ts, price, size, notional, cost, model.network_fee_usd, reason, cost)


# --------------------------------------------------------------------- book


@dataclass
class SymbolState:
    position: Position | None = None
    pending: Action | None = None
    last_bar_ts: int | None = None  # newest closed bar processed


@dataclass
class Book:
    """Cash plus one position per symbol, advanced bar by bar."""

    name: str
    cash: float
    executor: Executor
    max_trade_size: float
    max_capital: float
    symbols: dict[str, SymbolState] = field(default_factory=dict)
    fills: list[Fill] = field(default_factory=list)
    trades: list[Trade] = field(default_factory=list)
    # Optional per-symbol notional cap (e.g. a portfolio weight slice).
    weight_cap: Callable[[str], float | None] | None = None
    # Optional size multiplier (0..1) for a buy or a signal sell: (symbol, side) -> factor.
    # Only the live loop sets it, for confluence sizing of unmanaged tokens.
    gate: Callable[[str, str], float] | None = None
    on_fill: Callable[["Book", Fill], None] | None = None
    on_trade: Callable[["Book", Trade], None] | None = None

    def state(self, symbol: str) -> SymbolState:
        return self.symbols.setdefault(symbol, SymbolState())

    def position(self, symbol: str) -> Position | None:
        return self.state(symbol).position

    @property
    def deployed(self) -> float:
        return sum(s.position.cost_basis for s in self.symbols.values() if s.position)

    def equity(self, prices: dict[str, float]) -> float:
        """Cash plus positions marked at ``prices`` (cost basis when a price is missing)."""
        value = self.cash
        for symbol, state in self.symbols.items():
            if state.position:
                price = prices.get(symbol)
                value += state.position.size * price if price else state.position.cost_basis
        return value

    # -- the three steps

    def on_open(self, symbol: str, ts: int, bar_open: float, risk: Risk) -> Fill | None:
        """Fill the action queued at the previous close at this bar's open."""
        state = self.state(symbol)
        action = state.pending
        if action is None:
            return None
        state.pending = None
        if action.side == BUY:
            if state.position is not None:
                return None
            return self.open_position(symbol, ts, bar_open, risk, action.reason, action.strategy, action.strategy_hash)
        if state.position is None:
            return None
        if action.reason == SIGNAL and self.gate is not None:
            factor = self.gate(symbol, SELL)
            if factor <= 0:
                return None
            if factor < 1:
                return self.reduce_position(symbol, ts, bar_open, factor, action.reason)
        return self.close_position(symbol, ts, bar_open, action.reason)

    def on_low(self, symbol: str, ts: int, bar_open: float, low: float, risk: Risk) -> Fill | None:
        """Stop-loss / trailing stop against a bar's (running) low."""
        position = self.position(symbol)
        if position is None:
            return None
        hit = stop_exit(position, risk, bar_open, low)
        if hit is None:
            return None
        price, reason = hit
        return self.close_position(symbol, ts, price, reason)

    def on_close(
        self,
        symbol: str,
        ts: int,
        high: float,
        close: float,
        entry: bool,
        exit_: bool,
        risk: Risk,
        strategy: str = "",
        strategy_hash: str = "",
        allow_entry: bool = True,
    ) -> Action | None:
        """Ratchet the trailing stop, then queue take-profit, a signal exit or an entry."""
        state = self.state(symbol)
        state.last_bar_ts = ts
        if state.position is not None:
            update_trailing(state.position, risk, high)
        decision = decide_close(state.position, close, entry, exit_)
        if decision is None or (decision[0] == BUY and not allow_entry):
            state.pending = None
            return None
        state.pending = Action(decision[0], decision[1], ts, strategy, strategy_hash)
        return state.pending

    # -- fills

    def open_position(
        self, symbol: str, ts: int, price: float, risk: Risk, reason: str, strategy: str, strategy_hash: str
    ) -> Fill | None:
        cap = self.weight_cap(symbol) if self.weight_cap else None
        fee = self.executor.network_fee(symbol)
        notional = size_order(self.cash - fee, self.deployed, self.max_trade_size, self.max_capital, cap)
        if self.gate is not None:
            notional *= max(min(self.gate(symbol, BUY), 1.0), 0.0)
        if notional < _MIN_ORDER:
            return None
        fill = self.executor.buy(symbol, notional, price, ts, reason)
        if fill is None or fill.size <= 0:
            return None
        fill.strategy, fill.strategy_hash = strategy, strategy_hash
        self.cash -= fill.notional + fill.fee
        stop, take_profit = entry_levels(fill.price, risk)
        self.state(symbol).position = Position(
            symbol,
            strategy,
            strategy_hash,
            fill.price,
            fill.size,
            fill.notional + fill.fee,
            ts,
            stop,
            take_profit,
            highest_price=fill.price,
        )
        self._record_fill(fill)
        return fill

    def close_position(self, symbol: str, ts: int, price: float, reason: str) -> Fill | None:
        state = self.state(symbol)
        position = state.position
        if position is None:
            return None
        fill = self.executor.sell(symbol, position.size, price, ts, reason)
        if fill is None:
            return None
        fill.strategy, fill.strategy_hash = position.strategy, position.strategy_hash
        proceeds = fill.notional - fill.cost
        self.cash += proceeds
        state.position = None
        state.pending = None
        self._record_fill(fill)
        trade = Trade(
            symbol,
            position.strategy,
            position.strategy_hash,
            position.entry_ts,
            ts,
            position.entry_price,
            fill.price,
            position.size,
            position.cost_basis,
            proceeds,
            reason,
        )
        self.trades.append(trade)
        if self.on_trade:
            self.on_trade(self, trade)
        return fill

    def reduce_position(self, symbol: str, ts: int, price: float, fraction: float, reason: str) -> Fill | None:
        """Sell part of a position (live confluence sizing); the rest stays open."""
        position = self.position(symbol)
        if position is None or not 0 < fraction < 1:
            return None
        fill = self.executor.sell(symbol, position.size * fraction, price, ts, reason)
        if fill is None:
            return None
        fill.strategy, fill.strategy_hash = position.strategy, position.strategy_hash
        sold = min(fill.size / position.size, 1.0) if position.size > 0 else 1.0
        self.cash += fill.notional - fill.cost
        position.size -= fill.size
        position.cost_basis *= 1 - sold
        self._record_fill(fill)
        return fill

    def write_off(self, symbol: str, ts: int, reason: str = DUST) -> Trade | None:
        """Close a position without a swap (dust); its value is booked as lost."""
        state = self.state(symbol)
        position = state.position
        if position is None:
            return None
        state.position = None
        state.pending = None
        trade = Trade(
            symbol, position.strategy, position.strategy_hash, position.entry_ts, ts,
            position.entry_price, 0.0, position.size, position.cost_basis, 0.0, reason,
        )
        self.trades.append(trade)
        if self.on_trade:
            self.on_trade(self, trade)
        return trade

    def _record_fill(self, fill: Fill) -> None:
        self.fills.append(fill)
        if self.on_fill:
            self.on_fill(self, fill)
