"""swap-smoke: the human-supervised real-swap check, against fake Jupiter and RPC (no network)."""

import base64
import io
import json
import logging
from pathlib import Path

import httpx
import pytest
from solders.message import to_bytes_versioned
from solders.transaction import VersionedTransaction

from sol_trade import cli, jupiter, ledger, promotion, rpc, transactions, wallet
from sol_trade.config import config
from sol_trade.engine import Position
from sol_trade.log import log_transaction
from tests.test_transactions import _txn_with_wallet_as_second_signer

SOL_PRICE = 150.0


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
    """Answers /order and /execute like Jupiter's swap API; records every request.

    ``executions`` holds the /execute replies to give, in order; an Exception
    entry is raised instead, like a response lost after submission.
    """

    def __init__(self, primary: str, sol: str) -> None:
        self.primary, self.sol = primary, sol
        self.balances: dict[str, float] = {}
        self.orders: list[dict] = []
        self.sent: list[dict] = []
        self.replies: list = [
            {"status": "Success", "signature": "sig1", "totalOutputAmount": "33300000"},
            {"status": "Success", "signature": "sig2", "totalOutputAmount": "4985000"},
        ]
        self.transaction = _txn_with_wallet_as_second_signer()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def get(self, url, params, headers):
        assert url.endswith("/order")
        self.orders.append(dict(params))
        if params["inputMint"] == self.primary:  # buy leg: primary -> SOL
            payload = {"outAmount": "33333333", "gasless": True, "signatureFeeLamports": 5000,
                       "prioritizationFeeLamports": 0, "feeBps": 5, "router": "rfq"}
        else:  # sell leg: SOL -> primary
            payload = {"outAmount": "4990000", "gasless": False, "signatureFeeLamports": 5000,
                       "prioritizationFeeLamports": 10000, "feeBps": 5, "router": "metis"}
        return _Response({**payload, "requestId": f"req{len(self.orders)}", "inAmount": str(params["amount"]),
                          "transaction": self.transaction})

    async def post(self, url, json, headers):
        assert url.endswith("/execute")
        self.sent.append(json)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return _Response(reply)


class _RpcResponse:
    def __init__(self, result: dict | None) -> None:
        self.result = result

    def to_json(self) -> str:
        return json.dumps({"result": self.result})


class _FakeRpc:
    """A solana-py client whose get_transaction serves ``chain`` (signature -> result or None)."""

    def __init__(self) -> None:
        self.chain: dict[str, dict | None] = {}
        self.reads: list[str] = []

    async def get_transaction(self, signature, **kwargs):
        self.reads.append(str(signature))
        return _RpcResponse(self.chain.get(str(signature)))


@pytest.fixture
def mints():
    cfg = config()
    return {"primary": cfg.primary_mint, "sol": cfg.sol_mint, "primary_symbol": cfg.primary_mint_symbol,
            "sol_symbol": cfg.symbol_for_mint(cfg.sol_mint)}


def _sig(n: int) -> str:
    """A valid base58 signature string for fake transaction ``n``."""
    from solders.signature import Signature

    return str(Signature(bytes([n]) * 64))


def _chain_tx(mints, *, lamports: int, primary_raw: int, fee: int, payer_is_wallet: bool, err=None) -> dict:
    """A jsonParsed getTransaction result: the wallet's SOL and primary-token deltas."""
    wallet_key = str(config().public_address)
    other = "11111111111111111111111111111111"
    keys = [wallet_key, other] if payer_is_wallet else [other, wallet_key]
    wallet_index = keys.index(wallet_key)
    pre, post = [10**9, 10**9], [10**9, 10**9]
    post[wallet_index] += lamports - (fee if payer_is_wallet else 0)
    if not payer_is_wallet:
        post[0] -= fee

    def token(raw):
        return [{"accountIndex": 2, "mint": mints["primary"], "owner": wallet_key,
                 "uiTokenAmount": {"amount": str(raw), "decimals": 6}}]

    return {
        "slot": 1,
        "meta": {"err": err, "fee": fee, "preBalances": pre, "postBalances": post,
                 "preTokenBalances": token(20_000_000), "postTokenBalances": token(20_000_000 + primary_raw)},
        "transaction": {"signatures": ["a", "b"] if not payer_is_wallet else ["a"],
                        "message": {"accountKeys": [{"pubkey": k} for k in keys]}},
    }


@pytest.fixture(autouse=True)
def _no_log_files(monkeypatch):
    monkeypatch.setattr(cli, "setup_file_logging", lambda: None)


@pytest.fixture
def chain(monkeypatch):
    """Route config().client through the real budgeted proxy to a fake solana-py client."""
    fake = _FakeRpc()
    monkeypatch.setattr(config(), "_client", rpc.BudgetedClient("http://fake", factory=lambda _url: fake))
    monkeypatch.setattr(rpc, "_bucket", rpc.TokenBucket(1_000_000.0))
    monkeypatch.setitem(config().rpc, "max_cu_per_second", 1_000_000)  # still metered, just no waiting
    monkeypatch.setattr(transactions.time, "sleep", lambda _s: None)
    return fake


@pytest.fixture
def jup(monkeypatch, mints, chain):
    fake = _FakeJupiter(mints["primary"], mints["sol"])
    fake.replies = [
        {"status": "Success", "signature": _sig(1), "totalOutputAmount": "33300000"},
        {"status": "Success", "signature": _sig(2), "totalOutputAmount": "4985000"},
    ]
    # On chain the buy delivered 0.0333 SOL (gasless) and the sell 4.985 primary, paying a 15,000-lamport fee.
    chain.chain[_sig(1)] = _chain_tx(mints, lamports=33_300_000, primary_raw=-5_000_000, fee=10_000,
                                     payer_is_wallet=False)
    chain.chain[_sig(2)] = _chain_tx(mints, lamports=-33_300_000, primary_raw=4_985_000, fee=15_000,
                                     payer_is_wallet=True)
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: fake)
    monkeypatch.setattr(transactions, "decimals", lambda mint: 9 if mint == mints["sol"] else 6)
    monkeypatch.setattr(jupiter, "fetch_prices", lambda ms, **_: {mints["primary"]: 1.0, mints["sol"]: SOL_PRICE})
    fake.balances.update({mints["primary"]: 20.0, mints["sol"]: 0.5})
    monkeypatch.setattr(wallet, "find_balance", lambda mint: fake.balances[mint])
    monkeypatch.setattr(transactions, "find_balance", lambda mint: fake.balances[mint])
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


# ------------------------------------------------------------------- guards


def test_needs_a_terminal(jup, monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO("yes\n"))
    code, payload = _run()
    assert code == 1 and payload["error"] == "tty_required"
    assert jup.orders == [] and jup.sent == []


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


@pytest.mark.parametrize("book", [ledger.LIVE, ledger.DRY_RUN, "paper:swing/SOL"])
def test_refuses_while_any_bot_book_holds_a_position(jup, answer, monkeypatch, mints, book):
    position = Position(mints["sol_symbol"], "default", "h", SOL_PRICE, 0.1, 15.0, 0)
    monkeypatch.setattr(ledger, "open_positions", lambda name=None: [(book, position)])
    code, payload = _run()
    assert code == 1 and payload["error"] == "open_positions"
    assert payload["details"] == [{"book": book, "symbol": mints["sol_symbol"], "size": 0.1}]
    assert jup.orders == []


def test_refuses_without_enough_primary(jup, answer, mints):
    jup.balances[mints["primary"]] = 4.99
    code, payload = _run()
    assert code == 1 and payload["error"] == "insufficient_funds"
    assert jup.orders == []


def test_refuses_below_the_sol_reserve(jup, answer, mints):
    jup.balances[mints["sol"]] = 0.0  # find_balance reports 0 below minimum_sol_needed
    code, payload = _run()
    assert code == 1 and payload["error"] == "insufficient_sol"
    assert jup.orders == []


@pytest.mark.parametrize("text", ["", "y", "YES please", "no"])
def test_sends_nothing_without_a_typed_yes(jup, answer, text):
    answer(text)
    code, payload = _run()
    assert code == 1 and payload["error"] == "not_confirmed"
    assert jup.orders == [] and jup.sent == []


# ------------------------------------------------------------- round trip


def test_round_trip_goes_through_the_live_swap_path(jup, answer, chain, monkeypatch, mints):
    sent = []
    real_perform_swap = transactions.perform_swap

    async def spy(*args, **kwargs):
        sent.append(args)
        return await real_perform_swap(*args, **kwargs)

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
    primary, sol, p_sym, s_sym = mints["primary"], mints["sol"], mints["primary_symbol"], mints["sol_symbol"]
    # Buy 5 primary of SOL, then sell exactly the SOL the chain says arrived.
    assert sent == [(5.0, primary, sol, p_sym, s_sym), (pytest.approx(0.0333), sol, primary, s_sym, p_sym)]
    assert [o["inputMint"] for o in jup.orders] == [primary, sol]
    assert jup.orders[1]["amount"] == 33_300_000
    assert all(o["taker"] == str(config().public_address) for o in jup.orders)
    # Both legs were signed by the wallet, sent to /execute, and read back through config().client.
    assert [e["requestId"] for e in jup.sent] == ["req1", "req2"]
    for execution in jup.sent:
        txn = VersionedTransaction.from_bytes(base64.b64decode(execution["signedTransaction"]))
        assert txn.signatures[1].verify(config().public_address, to_bytes_versioned(txn.message))
    assert chain.reads == [_sig(1), _sig(2)]

    buy, sell = payload["buy"], payload["sell"]
    assert (buy["outcome"], sell["outcome"]) == ("confirmed", "confirmed")
    assert (buy["signature"], sell["signature"]) == (_sig(1), _sig(2))
    assert buy["filled_from"] == sell["filled_from"] == "chain"
    assert buy["quoted_out"] == pytest.approx(0.033333333) and buy["filled_out"] == pytest.approx(0.0333)
    assert buy["slippage_bps"] == pytest.approx((33333333 - 33300000) / 33333333 * 1e4)
    assert sell["quoted_out"] == pytest.approx(4.99) and sell["filled_out"] == pytest.approx(4.985)
    # Gasless buy: Jupiter paid. Sell: the wallet paid 15,000 lamports, 10,000 of it priority.
    assert buy["fees"]["charged_lamports"] == 0 and buy["fees"]["complete"]
    assert sell["fees"]["charged_lamports"] == 15_000
    assert sell["fees"]["charged_priority_lamports"] == 10_000
    assert sell["fees"]["estimated"]["prioritization_fee_lamports"] == 10_000
    trip = payload["round_trip"]
    fees_usd = 15_000 / 1e9 * SOL_PRICE
    assert trip["status"] == "realized"
    assert trip["network_fees_usd"] == pytest.approx(fees_usd)
    assert trip["cost_usd"] == pytest.approx(5 - 4.985 + fees_usd)
    assert trip["cost_bps"] == pytest.approx((5 - 4.985 + fees_usd) / 5 * 1e4)
    model = payload["model"]
    assert model["source"] == "default"
    assert model["round_trip_bps"] == pytest.approx(2 * config().costs["default_cost_bps"])
    assert model["actual_minus_model_bps"] == pytest.approx(trip["cost_bps"] - model["round_trip_bps"])
    # Both swaps reach the transaction log, as the bot's own swaps do.
    assert any(f"SolTrade TxID: {_sig(1)}" in r for r in records)
    assert any(f"SolTrade TxID: {_sig(2)}" in r for r in records)
    assert any(r.startswith("swap-smoke: round trip cost") for r in records)


def test_sells_the_chain_receipt_not_the_quote(jup, answer, chain, mints):
    # Jupiter reported 0.0333 SOL, the chain shows 0.0321 arrived: sell what arrived.
    chain.chain[_sig(1)] = _chain_tx(mints, lamports=32_100_000, primary_raw=-5_000_000, fee=10_000,
                                     payer_is_wallet=False)
    code, payload = _run()
    assert code == 0, payload
    assert 32_099_999 <= jup.orders[1]["amount"] <= 32_100_000  # create_order truncates; never more
    assert payload["buy"]["filled_out"] == pytest.approx(0.0321)
    assert payload["buy"]["slippage_bps"] > 300


def test_buy_without_a_chain_receipt_sends_no_sell(jup, answer, chain, mints):
    # Jupiter reports a signed fill, but the RPC never returns the buy: its SOL is not proven.
    del chain.chain[_sig(1)]
    code, payload = _run()
    assert code == 1 and payload["error"] == "buy_unresolved"
    assert len(jup.orders) == 1
    buy = payload["details"][0]["buy"]
    assert buy["attempts"][0]["signature"] == _sig(1) and buy["attempts"][0]["receipt"] is None
    assert buy["fees"]["complete"] is False


def test_sell_falls_back_to_jupiters_reported_fill_and_marks_fees_unknown(jup, answer, chain, mints):
    del chain.chain[_sig(2)]  # the RPC never returns the sell; its proceeds are not sold again
    code, payload = _run()
    assert code == 0, payload
    sell = payload["sell"]
    assert sell["filled_from"] == "jupiter_execute" and sell["filled_out"] == pytest.approx(4.985)
    assert sell["fees"]["complete"] is False and sell["fees"]["charged_lamports"] is None
    assert payload["round_trip"]["status"] == "fees_incomplete"


def _close_wrapped_sol_account(tx: dict, mints, wrapped_lamports: int) -> dict:
    """Add a preexisting wrapped-SOL account of the wallet that the transaction closes into the wallet."""
    wallet_key = str(config().public_address)
    keys = tx["transaction"]["message"]["accountKeys"]
    meta = tx["meta"]
    keys.append({"pubkey": "So1ana11111111111111111111111111111111111111"})
    index = len(keys) - 1
    rent = 2_039_280
    meta["preBalances"].append(rent + wrapped_lamports)
    meta["postBalances"].append(0)
    meta["postBalances"][[k["pubkey"] for k in keys].index(wallet_key)] += rent + wrapped_lamports
    meta["preTokenBalances"].append({"accountIndex": index, "mint": mints["sol"], "owner": wallet_key,
                                     "uiTokenAmount": {"amount": str(wrapped_lamports), "decimals": 9}})
    return tx


def test_receipt_cannot_isolate_sol_when_an_account_closes(mints):
    tx = _close_wrapped_sol_account(
        _chain_tx(mints, lamports=33_300_000, primary_raw=-5_000_000, fee=10_000, payer_is_wallet=False),
        mints, wrapped_lamports=1_000_000,
    )
    receipt = transactions._receipt(_sig(1), tx, [mints["primary"], mints["sol"]])
    assert receipt["change"][mints["sol"]] is None
    assert receipt["change"][mints["primary"]] == pytest.approx(-5.0)  # token deltas stay exact
    assert receipt["rent_accounts"] == ["So1ana11111111111111111111111111111111111111"]


def test_receipt_cannot_isolate_sol_when_balances_do_not_line_up(mints):
    tx = _chain_tx(mints, lamports=33_300_000, primary_raw=-5_000_000, fee=10_000, payer_is_wallet=False)
    tx["meta"]["postBalances"].pop()
    receipt = transactions._receipt(_sig(1), tx, [mints["sol"]])
    assert receipt["change"][mints["sol"]] is None


def test_buy_that_closes_a_wrapped_sol_account_sends_no_sell(jup, answer, chain, mints):
    # The rent refund and the old wrapped SOL would otherwise count as bought SOL and be sold.
    chain.chain[_sig(1)] = _close_wrapped_sol_account(chain.chain[_sig(1)], mints, wrapped_lamports=1_000_000)
    code, payload = _run()
    assert code == 1 and payload["error"] == "buy_unresolved"
    assert len(jup.orders) == 1 and len(jup.sent) == 1
    buy = payload["details"][0]["buy"]
    assert buy["attempts"][0]["receipt"]["rent_accounts"]
    assert buy["filled_out"] is None


def test_success_with_an_amount_but_no_signature_sends_no_sell(jup, answer, mints):
    jup.replies[0] = {"status": "Success", "totalOutputAmount": "33300000"}
    code, payload = _run()
    assert code == 1 and payload["error"] == "buy_unresolved"
    assert len(jup.orders) == 1
    buy = payload["details"][0]["buy"]
    assert buy["signature"] is None and buy["filled_out"] is None
    assert buy["attempts"][0]["status"] == "executed"
    assert buy["attempts"][0]["total_output_amount"] == "33300000"


def test_signed_failure_without_a_receipt_then_success_sends_no_sell(jup, answer, chain, mints):
    # The failed attempt's transaction can't be read, so it may have landed too.
    jup.replies.insert(0, {"status": "Failed", "signature": _sig(9), "error": "timeout"})
    code, payload = _run()
    assert code == 1 and payload["error"] == "buy_unresolved"
    assert len(jup.orders) == 2  # two buy attempts, no sell
    attempts = payload["details"][0]["buy"]["attempts"]
    assert [a["status"] for a in attempts] == ["failed", "executed"]
    assert attempts[0]["receipt"] is None and attempts[1]["receipt"] is not None


def test_signed_failures_without_receipts_are_unresolved_not_failed(jup, answer, mints):
    jup.replies = [{"status": "Failed", "signature": _sig(n), "error": "timeout"} for n in (7, 8, 9)]
    code, payload = _run()
    assert code == 1 and payload["error"] == "buy_unresolved"
    assert "may have landed" in payload["message"]
    attempts = payload["details"][0]["buy"]["attempts"]
    assert [a["signature"] for a in attempts] == [_sig(7), _sig(8), _sig(9)]


def test_signed_failures_proven_by_receipts_are_failed(jup, answer, chain, mints):
    jup.replies = [{"status": "Failed", "signature": _sig(n), "error": "slippage"} for n in (7, 8, 9)]
    for n in (7, 8, 9):
        chain.chain[_sig(n)] = _chain_tx(mints, lamports=0, primary_raw=0, fee=5_000, payer_is_wallet=True,
                                         err={"InstructionError": [0, "SlippageToleranceExceeded"]})
    code, payload = _run()
    assert code == 1 and payload["error"] == "buy_failed"
    assert payload["details"][0]["buy"]["fees"]["charged_lamports"] == 15_000


def test_lost_buy_response_sends_no_sell(jup, answer, mints):
    jup.replies = [httpx.ReadTimeout("lost"), httpx.ReadTimeout("lost"), httpx.ReadTimeout("lost")]
    code, payload = _run()
    assert code == 1 and payload["error"] == "buy_unresolved"
    assert [o["inputMint"] for o in jup.orders] == [mints["primary"]] * 3  # no sell order
    buy = payload["details"][0]["buy"]
    assert buy["outcome"] == "unresolved" and buy["filled_out"] is None
    assert [a["status"] for a in buy["attempts"]] == ["unknown"] * 3
    assert buy["fees"]["unknown_attempts"] == ["req1", "req2", "req3"]


def test_lost_response_with_moved_balance_is_not_sold_from_the_quote(jup, answer, monkeypatch, mints):
    # perform_swap treats this as executed (the balance dropped) and reports the quote; swap-smoke must not.
    jup.replies = [httpx.ReadTimeout("lost")]
    moves = iter([20.0, 15.0])
    monkeypatch.setattr(transactions, "find_balance", lambda mint: next(moves))
    code, payload = _run()
    assert code == 1 and payload["error"] == "buy_unresolved"
    assert len(jup.orders) == 1
    assert payload["details"][0]["buy"]["attempts"][0]["balance_moved"] is True


def test_success_without_an_amount_or_receipt_sends_no_sell(jup, answer, chain, mints):
    jup.replies[0] = {"status": "Success", "signature": _sig(1)}
    del chain.chain[_sig(1)]
    code, payload = _run()
    assert code == 1 and payload["error"] == "buy_unresolved"
    assert len(jup.orders) == 1


def test_retried_buy_counts_the_fee_of_the_failed_attempt(jup, answer, chain, mints):
    # The first buy landed on chain but failed, charging the wallet 5,000 lamports.
    chain.chain[_sig(9)] = _chain_tx(mints, lamports=0, primary_raw=0, fee=5_000, payer_is_wallet=True,
                                     err={"InstructionError": [0, "SlippageToleranceExceeded"]})
    jup.replies.insert(0, {"status": "Failed", "signature": _sig(9), "error": "slippage"})
    code, payload = _run()
    assert code == 0, payload
    buy = payload["buy"]
    assert [a["status"] for a in buy["attempts"]] == ["failed", "executed"]
    assert buy["signature"] == _sig(1)
    assert buy["fees"]["charged_lamports"] == 5_000
    assert payload["round_trip"]["network_fees_usd"] == pytest.approx(20_000 / 1e9 * SOL_PRICE)


def test_failed_attempt_that_landed_anyway_is_unresolved(jup, answer, chain, mints):
    chain.chain[_sig(9)] = _chain_tx(mints, lamports=33_300_000, primary_raw=-5_000_000, fee=5_000,
                                     payer_is_wallet=True)
    jup.replies.insert(0, {"status": "Failed", "signature": _sig(9), "error": "timeout"})
    code, payload = _run()
    assert code == 1 and payload["error"] == "buy_unresolved"


def test_failed_buy_sends_no_sell(jup, answer, mints):
    jup.replies = [{"status": "Failed", "error": "no liquidity"}] * 3
    code, payload = _run()
    assert code == 1 and payload["error"] == "buy_failed"
    assert [o["inputMint"] for o in jup.orders] == [mints["primary"]] * 3


def test_failed_sell_keeps_the_buy_evidence(jup, answer, mints):
    jup.replies[1:] = [{"status": "Failed", "error": "no liquidity"}] * 3
    code, payload = _run()
    assert code == 1 and payload["error"] == "sell_failed"
    legs = payload["details"][0]
    assert legs["buy"]["signature"] == _sig(1) and legs["buy"]["filled_out"] == pytest.approx(0.0333)
    assert legs["sell"]["outcome"] == "failed"
    assert "holds the 0.0333" in payload["message"]


def test_lost_sell_response_is_unresolved_not_held(jup, answer, mints):
    jup.replies[1:] = [httpx.ReadTimeout("lost")] * 3
    code, payload = _run()
    assert code == 1 and payload["error"] == "sell_unresolved"
    assert "holds" not in payload["message"]
    legs = payload["details"][0]
    assert legs["buy"]["outcome"] == "confirmed"
    assert [a["status"] for a in legs["sell"]["attempts"]] == ["unknown"] * 3


def test_sell_exception_keeps_both_legs(jup, answer, monkeypatch):
    real_perform_swap = transactions.perform_swap
    calls = []

    async def boom_on_sell(*args, **kwargs):
        calls.append(args)
        if len(calls) == 2:
            kwargs["attempts"].append({"status": "unknown", "request_id": "reqX", "order": None, "signature": None,
                                       "total_output_amount": None, "error": None, "balance_moved": False})
            raise RuntimeError("decimals lookup failed")
        return await real_perform_swap(*args, **kwargs)

    monkeypatch.setattr(transactions, "perform_swap", boom_on_sell)
    code, payload = _run()
    assert code == 1 and payload["error"] == "sell_unresolved"
    legs = payload["details"][0]
    assert legs["buy"]["signature"] == _sig(1)
    assert legs["sell"]["error"] == "RuntimeError: decimals lookup failed"
    assert legs["sell"]["attempts"][0]["request_id"] == "reqX"


def test_report_failure_says_not_to_run_again(jup, answer, monkeypatch):
    def broken(*args, **kwargs):
        raise KeyError("points")

    monkeypatch.setattr(cli.costs, "model_for", broken)
    code, payload = _run()
    assert code == 1 and payload["error"] == "report_failed"
    assert "Do not run swap-smoke again" in payload["message"]
    legs = payload["details"][0]
    assert (legs["buy"]["signature"], legs["sell"]["signature"]) == (_sig(1), _sig(2))
    assert len(jup.sent) == 2


def test_leaves_books_promotion_and_config_alone(jup, answer):
    config_file = Path(config().path)
    config_text = config_file.read_text()
    before = (ledger.open_positions(), ledger.fills(ledger.LIVE), ledger.fills(ledger.DRY_RUN), promotion.paused())
    code, _ = _run()
    assert code == 0
    assert (ledger.open_positions(), ledger.fills(ledger.LIVE), ledger.fills(ledger.DRY_RUN),
            promotion.paused()) == before
    assert config_file.read_text() == config_text
