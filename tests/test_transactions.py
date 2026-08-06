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

_WALLET_SIGNER = pytest.mark.skipif(
    not config().private_key, reason="SOLTRADE_PRIVATE_KEY not set"
)


def _txn_with_wallet_as_second_signer() -> str:
    """A 2-signer tx where Jupiter's fee payer is slot 0 and the wallet slot 1."""
    fee_payer = Keypair()
    instruction = Instruction(
        program_id=SYS_PROGRAM_ID,
        data=b"\x00",
        accounts=[
            # fee payer's meta dedups with the payer -> required signer 0
            AccountMeta(pubkey=fee_payer.pubkey(), is_signer=True, is_writable=True),
            # the wallet is the second required signer, like a gasless JupiterZ order
            AccountMeta(pubkey=config().public_address, is_signer=True, is_writable=True),
        ],
    )
    message = Message.new_with_blockhash(
        [instruction], fee_payer.pubkey(), Hash(bytes(32))
    )
    signed = VersionedTransaction.populate(
        message, [Keypair().sign_message(b"seed"), Keypair().sign_message(b"seed")]
    )
    return base64.b64encode(bytes(signed)).decode("utf-8")


def test_execute_order_missing_transaction_raises():
    with pytest.raises(OrderError):
        asyncio.run(execute_order({"requestId": "abc"}))


def test_execute_order_error_code_raises():
    with pytest.raises(OrderError):
        asyncio.run(execute_order({"errorCode": 1, "errorMessage": "boom"}))


@_WALLET_SIGNER
def test_sign_preserves_signature_slots():
    signed = _sign_order_transaction(_txn_with_wallet_as_second_signer())
    txn = VersionedTransaction.from_bytes(base64.b64decode(signed))
    # The fee-payer placeholder slot must survive — the header still needs two.
    assert len(txn.signatures) == 2
    assert txn.message.header.num_required_signatures == 2


@_WALLET_SIGNER
def test_sign_lands_in_wallet_slot():
    signed = _sign_order_transaction(_txn_with_wallet_as_second_signer())
    txn = VersionedTransaction.from_bytes(base64.b64decode(signed))
    message = to_bytes_versioned(txn.message)
    wallet_index = next(
        i
        for i in range(txn.message.header.num_required_signatures)
        if txn.message.account_keys[i] == config().public_address
    )
    assert wallet_index == 1  # fee payer is slot 0, wallet is slot 1
    assert txn.signatures[wallet_index].verify(config().public_address, message)
    # The fee-payer slot must NOT carry our signature.
    assert not txn.signatures[0].verify(config().public_address, message)


def test_sign_rejects_wallet_not_a_signer():
    alice, bob = Keypair(), Keypair()
    instruction = Instruction(
        program_id=SYS_PROGRAM_ID,
        data=b"\x00",
        accounts=[
            AccountMeta(pubkey=alice.pubkey(), is_signer=True, is_writable=True),
            AccountMeta(pubkey=bob.pubkey(), is_signer=True, is_writable=True),
        ],
    )
    message = Message.new_with_blockhash([instruction], alice.pubkey(), Hash(bytes(32)))
    signed = VersionedTransaction.populate(
        message, [alice.sign_message(b"seed"), bob.sign_message(b"seed")]
    )
    with pytest.raises(OrderError, match="not a required signer"):
        _sign_order_transaction(base64.b64encode(bytes(signed)).decode("utf-8"))
