import base64
import json
import time
from typing import Any

import httpx
from solana.rpc.commitment import Confirmed
from solders.message import to_bytes_versioned
from solders.signature import Signature
from solders.transaction import VersionedTransaction

from sol_trade import rpc
from sol_trade.config import config
from sol_trade.jupiter import decimals
from sol_trade.log import log_general, log_transaction
from sol_trade.utils import handle_rate_limiting, run_async
from sol_trade.wallet import find_balance

_LAMPORTS_PER_SIGNATURE = 5000


def _input_balance(mint: str) -> float | None:
    """Wallet balance for the input token, or None when the RPC is down."""
    try:
        return find_balance(mint)
    except Exception:  # noqa: BLE001 - reconciliation is best-effort
        return None


def _swap_consumed(mint: str, sent_amount: float, balance_before: float | None) -> bool:
    """True when the wallet balance dropped by roughly the full order size."""
    if balance_before is None:
        return False
    after = _input_balance(mint)
    if after is None:
        return False
    # The swap moves the entire sent amount; require ~90% of it to be gone so
    # dust and fee movements do not create false positives.
    return balance_before - after >= sent_amount * 0.9


# Order fields perform_swap records for each attempt.
_ORDER_AUDIT_KEYS = (
    "requestId",
    "inAmount",
    "outAmount",
    "feeBps",
    "signatureFeeLamports",
    "prioritizationFeeLamports",
    "rentFeeLamports",
    "gasless",
    "router",
)


@handle_rate_limiting()
def _fetch_transaction(signature: str) -> dict[str, Any] | None:
    """A confirmed transaction as parsed JSON, or None when the RPC has none."""
    with rpc.essential():
        response = run_async(
            config().client.get_transaction(
                Signature.from_string(signature),
                encoding="jsonParsed",
                commitment=Confirmed,
                max_supported_transaction_version=0,
            )
        )
    return json.loads(response.to_json()).get("result")


def transaction_receipt(
    signature: str, mints: list[str], tries: int = 3, delay: float = 2.0
) -> dict[str, Any] | None:
    """What a landed transaction did to the wallet, read from the chain.

    ``change`` is the wallet's net change per mint in token units, before
    fees: native SOL counts the main account's lamports plus wrapped SOL,
    with the fee the wallet paid added back. The SOL change is None when
    the transaction opened or closed another account (rent moved with the
    swap) or its balances don't line up with its accounts. Returns None
    when the RPC can't produce the transaction after ``tries`` reads.
    """
    for attempt in range(tries):
        try:
            result = _fetch_transaction(signature)
        except Exception as e:  # noqa: BLE001 - the caller reports it as unknown
            log_general.warning(f"could not read transaction {signature}: {e}")
            return None
        if result:
            return _receipt(signature, result, mints)
        if attempt < tries - 1:
            time.sleep(delay)
    return None


def _receipt(signature: str, result: dict[str, Any], mints: list[str]) -> dict[str, Any]:
    meta = result.get("meta") or {}
    transaction = result.get("transaction") or {}
    keys = [
        key.get("pubkey") if isinstance(key, dict) else key
        for key in (transaction.get("message") or {}).get("accountKeys") or []
    ]
    wallet = str(config().public_address)
    fee = int(meta.get("fee") or 0)
    payer_is_wallet = bool(keys) and keys[0] == wallet
    wallet_fee = fee if payer_is_wallet else 0
    pre, post = meta.get("preBalances") or [], meta.get("postBalances") or []
    aligned = len(keys) == len(pre) == len(post)
    lamports = 0
    if aligned and wallet in keys:
        index = keys.index(wallet)
        lamports = int(post[index]) - int(pre[index])
    # Accounts opened or closed here move rent into or out of the wallet's
    # lamports (closing a wrapped-SOL account refunds its rent), which the
    # lamport delta can't tell apart from swap output.
    rent_accounts = [
        keys[i]
        for i in range(len(keys) if aligned else 0)
        if keys[i] != wallet and (int(pre[i]) == 0) != (int(post[i]) == 0)
    ]

    def tokens(balances: list[dict[str, Any]] | None, mint: str) -> float:
        return sum(
            int(b["uiTokenAmount"]["amount"]) / 10 ** int(b["uiTokenAmount"]["decimals"])
            for b in balances or []
            if b.get("owner") == wallet and b.get("mint") == mint
        )

    change: dict[str, float | None] = {}
    for mint in mints:
        delta = tokens(meta.get("postTokenBalances"), mint) - tokens(meta.get("preTokenBalances"), mint)
        if mint == config().sol_mint:
            if not aligned or rent_accounts:
                change[mint] = None  # the SOL the swap delivered can't be isolated
                continue
            delta += (lamports + wallet_fee) / 1e9
        change[mint] = delta
    base_fee = _LAMPORTS_PER_SIGNATURE * len(transaction.get("signatures") or [])
    return {
        "signature": signature,
        "slot": result.get("slot"),
        "err": meta.get("err"),
        "fee_lamports": fee,
        "fee_payer_is_wallet": payer_is_wallet,
        "wallet_fee_lamports": wallet_fee,
        "wallet_priority_fee_lamports": max(wallet_fee - base_fee, 0),
        "rent_accounts": rent_accounts,
        "change": change,
    }


class OrderError(Exception):
    """Raised when the Jupiter API returns an invalid order response."""


def _sign_order_transaction(transaction_b64: str) -> str:
    """Partially sign a Jupiter order transaction at the wallet's own slot.

    Gasless JupiterZ orders carry two signature slots and make Jupiter's
    signature-fee payer the FIRST required signer — the taker is slot 1, not
    slot 0. Signing the wrong slot yields ``Invalid signature for transaction``
    from /execute, so the wallet's slot is located from the message's account
    keys and the other slots are left as placeholders for /execute.
    """
    raw_txn = VersionedTransaction.from_bytes(base64.b64decode(transaction_b64))
    message = raw_txn.message
    our_pubkey = config().public_address
    signature = config().keypair.sign_message(to_bytes_versioned(message))

    signatures = list(raw_txn.signatures)
    if not signatures:
        raise OrderError("Transaction has no signature slots")

    required = message.header.num_required_signatures
    signer_slots = [
        i for i in range(required) if message.account_keys[i] == our_pubkey
    ]
    if not signer_slots:
        raise OrderError("Wallet is not a required signer of this transaction")
    for slot in signer_slots:
        signatures[slot] = signature

    signed_txn = VersionedTransaction.populate(message, signatures)
    return base64.b64encode(bytes(signed_txn)).decode("utf-8")


def _slippage_attempts(base_bps: int) -> list[dict[str, Any]]:
    """Order attempts: fixed slippage first, dynamic-slippage as fallback.

    JupiterZ RFQ can fail to quote certain sizes even at wide fixed slippage;
    ``dynamicSlippage`` lets Jupiter pick an acceptable level (typically well
    under the user's cap), bounded by ``maxDynamicSlippageBps``.
    """
    dynamic_cap = min(max(base_bps * 4, 100), 400)
    return [
        {"slippageBps": base_bps},
        {"dynamicSlippage": True, "maxDynamicSlippageBps": dynamic_cap},
    ]


async def create_order(
    input_amount: float, input_token_mint: str, output_token_mint: str
) -> dict[str, Any]:
    """
    Creates a swap order using Jupiter Ultra API.

    Args:
        input_amount: The amount of input token to swap (in token units, not lamports)
        input_token_mint: The mint address of the input token
        output_token_mint: The mint address of the output token

    Returns:
        Dictionary containing the order response from Jupiter API
    """
    log_transaction.info(
        f"SolTrade is creating order for {input_amount} {input_token_mint}."
    )

    token_decimals = 10 ** decimals(input_token_mint)
    
    # Convert token amount to smallest unit (lamports for SOL, etc.)
    amount_in_smallest_unit = int(input_amount * token_decimals)
    
    base_params = {
        "inputMint": input_token_mint,
        "outputMint": output_token_mint,
        "amount": amount_in_smallest_unit,
        "taker": str(config().public_address),
    }
    
    headers = {"Content-Type": "application/json"}
    if config().jupiter_api_key:
        headers["x-api-key"] = config().jupiter_api_key
    
    api_link = f"{config().jup_api}/order"
    log_transaction.info(f"SolTrade API Link: {api_link}")
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        last_body = ""
        for attempt in _slippage_attempts(int(config().max_slippage or 50)):
            params = {**base_params, **attempt}
            log_transaction.info(f"Parameters: {params}")
            response = await client.get(api_link, params=params, headers=headers)
            if response.status_code == 200:
                result = response.json()
                log_transaction.info(
                    f"Order created (requestId: {result.get('requestId')}, "
                    f"outAmount: {result.get('outAmount')})"
                )
                return result
            last_body = response.text[:300]
            log_transaction.warning(
                f"order rejected (HTTP {response.status_code}, "
                f"params {attempt}): {last_body}"
            )
        raise OrderError(f"Jupiter order failed: {last_body}")


async def execute_order(order_response: dict) -> dict[str, Any]:
    """
    Signs and executes a swap order using Jupiter Ultra API.
    This replaces the legacy send_transaction function.
    """
    try:
        if "errorCode" in order_response:
            error_msg = order_response.get("errorMessage", "Unknown error")
            log_transaction.error(f"Order failed: {error_msg}")
            raise OrderError(f"Order error: {error_msg}")
        
        transaction_b64 = order_response.get("transaction")
        if not transaction_b64:
            log_transaction.error("no transaction returned in order response")
            raise OrderError("No transaction in order response")
        
        request_id = order_response["requestId"]
        
        # Deserialize and partially sign the transaction (see
        # _sign_order_transaction for why the signature count is preserved).
        signed_txn_b64 = _sign_order_transaction(transaction_b64)
        
        log_transaction.info(
            f"SolTrade is executing order with requestId: {request_id}."
        )
        
        # Prepare headers with Jupiter API key
        headers = {"Content-Type": "application/json"}
        if config().jupiter_api_key:
            headers["x-api-key"] = config().jupiter_api_key
        
        # Execute the transaction via Ultra API
        async with httpx.AsyncClient(timeout=30.0) as client:
            execute_response = await client.post(
                f"{config().jup_api}/execute",
                json={
                    "signedTransaction": signed_txn_b64,
                    "requestId": request_id,
                },
                headers=headers
            )
            try:
                execute_response.raise_for_status()
            except httpx.HTTPStatusError as e:
                # Surface Jupiter's error body (code + message) instead of the
                # generic "400 Bad Request" so failures are diagnosable.
                raise OrderError(
                    f"Jupiter execute failed ({e.response.status_code}): "
                    f"{e.response.text[:300]}"
                ) from e
            result = execute_response.json()
            
            if result.get("status") == "Success":
                log_transaction.info(f"SolTrade TxID: {result.get('signature')}")
            else:
                log_transaction.error(f"transaction failed: {result.get('error')}")
            
            return result
            
    except Exception as e:
        log_transaction.error(f"failed to execute transaction: {e}")
        raise


def _base_units(value: Any) -> int | None:
    """A Jupiter amount (an integer string of base units), or None if it isn't one."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        amount = value
    elif isinstance(value, str):
        try:
            amount = int(value.strip())
        except ValueError:
            return None
    else:
        return None
    return amount if amount >= 0 else None


async def perform_swap(
    sent_amount: float,
    sent_token_mint: str,
    output_token_mint: str,
    sent_token_symbol: str,
    output_token_symbol: str,
    attempts: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Swap tokens via Jupiter; returns fill amounts on success, else None.

    When ``attempts`` is given, one record per order attempt is appended to
    it as the attempt runs, so a caller keeps the evidence even when this
    returns None or raises. ``status`` is ``no_order`` (no order was built,
    nothing was signed), ``unknown`` (an order was built but its execute
    response never arrived, so it may have landed), ``executed`` or
    ``failed`` (Jupiter's /execute status). ``balance_moved`` marks the
    attempt this function treated as executed because the input balance
    dropped.
    """
    log_general.info("SolTrade is taking a market position.")

    # Sample the input balance so a lost/failed execute response can be
    # reconciled against the wallet instead of blindly retrying (which could
    # double-trade a landed swap) or giving up (which would leave the position
    # untracked).
    balance_before = _input_balance(sent_token_mint)

    order = execute_result = None
    is_tx_successful = False

    for i in range(3):
        if not is_tx_successful:
            attempt: dict[str, Any] = {
                "status": "no_order",
                "request_id": None,
                "order": None,
                "signature": None,
                "total_output_amount": None,
                "error": None,
                "balance_moved": False,
            }
            if attempts is not None:
                attempts.append(attempt)
            try:
                order = await create_order(
                    sent_amount, sent_token_mint, output_token_mint
                )
                attempt.update(
                    status="unknown",
                    request_id=order.get("requestId"),
                    order={key: order.get(key) for key in _ORDER_AUDIT_KEYS},
                )
                
                execute_result = await execute_order(order)
                attempt.update(
                    status="executed"
                    if execute_result.get("status") == "Success"
                    else "failed",
                    signature=execute_result.get("signature"),
                    total_output_amount=execute_result.get("totalOutputAmount"),
                    error=execute_result.get("error"),
                )
                
                if execute_result.get("status") == "Success":
                    is_tx_successful = True
                    break
                else:
                    log_general.warning(
                        f"SolTrade failed to complete transaction {i}. Retrying. "
                        f"Error: {execute_result.get('error')}"
                    )
            except Exception as e:  # noqa: BLE001 - retry loop
                attempt["error"] = f"{type(e).__name__}: {e}"
                log_general.warning(
                    f"SolTrade failed to complete transaction {i}. Retrying. Error: {e}"
                )
                if order is not None and _swap_consumed(
                    sent_token_mint, sent_amount, balance_before
                ):
                    attempt["balance_moved"] = True
                    # The order was submitted and the funds moved even though
                    # the response was lost — treat it as executed.
                    log_general.warning(
                        "execute response lost but the input balance moved; "
                        "treating the swap as executed"
                    )
                    is_tx_successful = True
                    break
                continue

    if not is_tx_successful:
        log_general.error(
            "SolTrade failed to complete the transaction after 3 attempts."
        )
        return None

    # Calculate the actual amounts from the execution result
    output_decimals = 10 ** decimals(output_token_mint)

    # The swap has landed, so an odd amount must not raise here. Only a
    # successful execute reports a fill; after a lost response the result
    # may be left over from an earlier failed attempt.
    output_amount = None
    if execute_result and execute_result.get("status") == "Success":
        output_amount = _base_units(execute_result.get("totalOutputAmount"))
    if output_amount is None and order:
        output_amount = _base_units(order.get("outAmount"))
    if output_amount is None:
        log_general.warning(
            "swap landed but Jupiter reported no usable output amount "
            f"(execute {execute_result and execute_result.get('totalOutputAmount')!r}, "
            f"order {order and order.get('outAmount')!r}); recording 0"
        )
        output_amount = 0

    bought_amount = output_amount / output_decimals
    
    log_transaction.info(
        f"SolTrade sold {sent_amount} {sent_token_symbol} for {bought_amount:.2f} {output_token_symbol}."
    )
    return {"out_amount": bought_amount, "sent_amount": sent_amount}
