"""swap-smoke: the supervised real-swap check, against fake Jupiter and RPC (no network)."""

import base64
import io
import json
import logging
from pathlib import Path

import httpx
import pytest
from solders.message import to_bytes_versioned
from solders.transaction import VersionedTransaction

from sol_trade import cli, jupiter, ledger, promotion, transactions, wallet
from sol_trade.config import config
from sol_trade.engine import Position
from sol_trade.log import log_transaction
from tests.test_transactions import _txn_with_wallet_as_second_signer

USDC = config().primary_mint
SOL = config().sol_mint


class _Tty(io.StringIO):
    def isatty(self) -> bool:
        return True


class _Response:
    def __init__(self, payload: dict) -> None:
        self.status_code = 200
        self.text = json.dumps(payload)
        self._payload = payload

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        pass


class _FakeJupiter:
    """Answers /order and /execute like Jupiter's swap API; records every request."""

    def __init__(self) -> None:
        self.balances: dict[str, float] = {}
        self.orders: list[dict] = []
        self.executions: list[dict] = []
        self.transaction = _txn_with_wallet_as_second_signer()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def get(self, url, params, headers):
        assert url.endswith("/order")
        self.orders.append(dict(params))
        if params["inputMint"] == USDC:  # buy leg: 5 USDC -> SOL
            payload = {"outAmount": "33333333", "gasless": True, "signatureFeeLamports": 5000,
                       "prioritizationFeeLamports": 0, "feeBps": 5, "router": "rfq"}
        else:  # sell leg: SOL -> USDC
            payload = {"outAmount": "4990000", "gasless": False, "signatureFeeLamports": 5000,
                       "prioritizationFeeLamports": 10000, "feeBps": 5, "router": "metis"}
        leg = len(self.orders)
        return _Response({**payload, "requestId": f"req{leg}", "inAmount": str(params["amount"]),
                          "transaction": self.transaction})

    async def post(self, url, json, headers):
        assert url.endswith("/execute")
        self.executions.append(json)
        leg = len(self.executions)
        filled = "33300000" if leg == 1 else "4985000"
        return _Response({"status": "Success", "signature": f"sig{leg}", "totalOutputAmount": filled})


@pytest.fixture(autouse=True)
def _no_log_files(monkeypatch):
    monkeypatch.setattr(cli, "setup_file_logging", lambda: None)


@pytest.fixture
def jup(monkeypatch):
    fake = _FakeJupiter()
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: fake)
    monkeypatch.setattr(transactions, "decimals", lambda mint: 9 if mint == SOL else 6)
    monkeypatch.setattr(jupiter, "fetch_prices", lambda mints, **_: {USDC: 1.0, SOL: 150.0})
    balances = fake.balances
    balances.update({USDC: 20.0, SOL: 0.5})
    monkeypatch.setattr(wallet, "find_balance", lambda mint: balances[mint])
    monkeypatch.setattr(transactions, "find_balance", lambda mint: balances[mint])
    monkeypatch.setattr(wallet, "minimum_sol_needed", lambda: 0.004)
    return fake


@pytest.fixture
def answer(monkeypatch):
    def type_answer(text: str) -> None:
        monkeypatch.setattr("sys.stdin", _Tty(text + "\n"))

    type_answer("yes")
    return type_answer


def _run(*argv):
    lines = []
    code = cli.run(["swap-smoke", *argv], out=lines.append)
    assert len(lines) == 1
    return code, json.loads(lines[0])


def test_needs_a_terminal(jup, monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO("yes\n"))
    code, payload = _run()
    assert code == 1 and payload["error"] == "tty_required"
    assert jup.orders == [] and jup.executions == []


@pytest.mark.parametrize("usd", ["10.01", "50"])
def test_refuses_above_the_cap(jup, answer, usd):
    code, payload = _run("--usd", usd)
    assert code == 1 and payload["error"] == "over_cap"
    assert jup.orders == []


@pytest.mark.parametrize("usd", ["0", "-1", "nan"])
def test_size_must_be_positive(jup, answer, usd):
    code, payload = _run("--usd", usd)
    assert code == 2 and payload["error"] == "usage"


def test_refuses_while_the_kill_switch_is_on(jup, answer):
    promotion._set_setting(promotion._PAUSED, {"at": "now", "drawdown_pct": 16})
    code, payload = _run()
    assert code == 1 and payload["error"] == "paused"
    assert jup.orders == []


@pytest.mark.parametrize("book", [ledger.LIVE, ledger.DRY_RUN])
def test_refuses_while_the_bot_holds_a_position(jup, answer, monkeypatch, book):
    position = Position("SOL", "default", "h", 150.0, 0.1, 15.0, 0)
    monkeypatch.setattr(ledger, "open_positions", lambda name=None: [(book, position)])
    code, payload = _run()
    assert code == 1 and payload["error"] == "open_positions"
    assert payload["details"] == [{"book": book, "symbol": "SOL", "size": 0.1}]
    assert jup.orders == []


def test_refuses_without_enough_primary(jup, answer):
    jup.balances[USDC] = 4.99
    code, payload = _run()
    assert code == 1 and payload["error"] == "insufficient_funds"
    assert jup.orders == []


def test_refuses_below_the_sol_reserve(jup, answer):
    jup.balances[SOL] = 0.0  # find_balance reports 0 below minimum_sol_needed
    code, payload = _run()
    assert code == 1 and payload["error"] == "insufficient_sol"
    assert jup.orders == []


@pytest.mark.parametrize("text", ["", "y", "YES please", "no"])
def test_sends_nothing_without_a_typed_yes(jup, answer, text):
    answer(text)
    code, payload = _run()
    assert code == 1 and payload["error"] == "not_confirmed"
    assert jup.orders == [] and jup.executions == []


def test_round_trip_goes_through_the_live_swap_path(jup, answer, monkeypatch):
    sent = []
    real_perform_swap = transactions.perform_swap

    async def spy(*args):
        sent.append(args)
        return await real_perform_swap(*args)

    monkeypatch.setattr(transactions, "perform_swap", spy)
    records = []
    handler = logging.Handler()
    handler.emit = lambda record: records.append(record.getMessage())
    log_transaction.addHandler(handler)
    try:
        code, payload = _run()
    finally:
        log_transaction.removeHandler(handler)

    assert code == 0 and payload["ok"], payload
    # Buy 5 USDC of SOL, then sell exactly the SOL that arrived.
    assert sent == [(5.0, USDC, SOL, "USDC", "SOL"), (0.0333, SOL, USDC, "SOL", "USDC")]
    assert [o["inputMint"] for o in jup.orders] == [USDC, SOL]
    assert jup.orders[0]["amount"] == 5_000_000
    assert jup.orders[1]["amount"] == int(0.0333 * 10**9)
    assert all(o["taker"] == str(config().public_address) for o in jup.orders)
    # Both legs were signed by the wallet and sent to /execute.
    assert [e["requestId"] for e in jup.executions] == ["req1", "req2"]
    for execution in jup.executions:
        txn = VersionedTransaction.from_bytes(base64.b64decode(execution["signedTransaction"]))
        assert txn.signatures[1].verify(config().public_address, to_bytes_versioned(txn.message))

    buy, sell = payload["buy"], payload["sell"]
    assert (buy["signature"], sell["signature"]) == ("sig1", "sig2")
    assert buy["quoted_out"] == pytest.approx(0.033333333) and buy["filled_out"] == pytest.approx(0.0333)
    assert buy["slippage_bps"] == pytest.approx((33333333 - 33300000) / 33333333 * 1e4)
    assert sell["quoted_out"] == pytest.approx(4.99) and sell["filled_out"] == pytest.approx(4.985)
    assert sell["slippage_bps"] == pytest.approx((4.99 - 4.985) / 4.99 * 1e4)
    assert buy["fees"]["network_fee_usd"] == 0.0  # gasless: Jupiter pays
    assert sell["fees"]["prioritization_fee_lamports"] == 10000
    assert sell["fees"]["network_fee_usd"] == pytest.approx(15000 / 1e9 * 150)
    trip = payload["round_trip"]
    assert trip["cost_usd"] == pytest.approx(5 - 4.985 + 0.00225)
    assert trip["cost_bps"] == pytest.approx((5 - 4.985 + 0.00225) / 5 * 1e4)
    # No measured costs here: the default model charges default_cost_bps per side.
    model = payload["model"]
    assert model["source"] == "default"
    assert model["round_trip_bps"] == pytest.approx(2 * config().costs["default_cost_bps"])
    assert model["actual_minus_model_bps"] == pytest.approx(trip["cost_bps"] - model["round_trip_bps"])
    # Both swaps reach the transaction log, as the bot's own swaps do.
    assert any("SolTrade TxID: sig1" in r for r in records)
    assert any("SolTrade TxID: sig2" in r for r in records)
    assert any(r.startswith("swap-smoke: round trip cost") for r in records)


def test_leaves_books_promotion_and_config_alone(jup, answer):
    config_file = Path(config().path)
    config_text = config_file.read_text()
    before = (ledger.open_positions(), ledger.fills(ledger.LIVE), ledger.fills(ledger.DRY_RUN), promotion.paused())
    code, _ = _run()
    assert code == 0
    assert (ledger.open_positions(), ledger.fills(ledger.LIVE), ledger.fills(ledger.DRY_RUN),
            promotion.paused()) == before
    assert config_file.read_text() == config_text


def test_failed_buy_sends_no_sell(jup, answer, monkeypatch):
    calls = []

    async def failing(*args):
        calls.append(args)

    monkeypatch.setattr(transactions, "perform_swap", failing)
    code, payload = _run()
    assert code == 1 and payload["error"] == "buy_failed"
    assert len(calls) == 1
