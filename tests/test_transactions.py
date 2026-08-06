"""Tests for the Jupiter order/sign path (no network, no real swaps)."""

import asyncio
import base64

import pytest
from solders.hash import Hash
from solders.instruction import AccountMeta, Instruction
from solders.keypair import Keypair
from solders.message import Message, to_bytes_versioned
from solders.system_program import ID as SYS_PROGRAM_ID
from solders.transaction import VersionedTransaction

from sol_trade.config import config
from sol_trade.transactions import OrderError, _sign_order_transaction, execute_order

_SOL = "So11111111111111111111111111111111111111112"


def _two_signer_transaction_b64() -> str:
    """A versioned transaction whose message header requires two signatures."""
    alice, bob = Keypair(), Keypair()
    instruction = Instruction(
        program_id=SYS_PROGRAM_ID,
        data=b"\x00",
        accounts=[
            AccountMeta(pubkey=alice.pubkey(), is_signer=True, is_writable=True),
            AccountMeta(pubkey=bob.pubkey(), is_signer=True, is_writable=True),
        ],
    )
    message = Message.new_with_blockhash(
        [instruction], alice.pubkey(), Hash(bytes(32))
    )
    signed = VersionedTransaction.populate(
        message, [alice.sign_message(b"seed"), bob.sign_message(b"seed")]
    )
    return base64.b64encode(bytes(signed)).decode("utf-8")


def test_execute_order_missing_transaction_raises():
    with pytest.raises(OrderError):
        asyncio.run(execute_order({"requestId": "abc"}))


def test_execute_order_error_code_raises():
    with pytest.raises(OrderError):
        asyncio.run(execute_order({"errorCode": 1, "errorMessage": "boom"}))


def test_sign_preserves_signature_slots():
    signed = _sign_order_transaction(_two_signer_transaction_b64())
    txn = VersionedTransaction.from_bytes(base64.b64decode(signed))
    # The market-maker slot must survive — the header still demands two signers.
    assert len(txn.signatures) == 2
    assert txn.message.header.num_required_signatures == 2


def test_sign_verifies_with_wallet_pubkey():
    signed = _sign_order_transaction(_two_signer_transaction_b64())
    txn = VersionedTransaction.from_bytes(base64.b64decode(signed))
    message = to_bytes_versioned(txn.message)
    assert txn.signatures[0].verify(config().public_address, message)
