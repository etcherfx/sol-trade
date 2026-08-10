import base64
from typing import Any

import httpx
from solders.message import to_bytes_versioned
from solders.transaction import VersionedTransaction

from sol_trade.config import config
from sol_trade.log import log_general, log_transaction
from sol_trade.wallet import find_balance


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

    token_decimals = config().decimals(input_token_mint)
    
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


async def perform_swap(
    sent_amount: float,
    sent_token_mint: str,
    output_token_mint: str,
    sent_token_symbol: str,
    output_token_symbol: str,
) -> dict[str, Any] | None:
    """Swap tokens via Jupiter; returns fill amounts on success, else None."""
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
            try:
                order = await create_order(
                    sent_amount, sent_token_mint, output_token_mint
                )
                
                execute_result = await execute_order(order)
                
                if execute_result.get("status") == "Success":
                    is_tx_successful = True
                    break
                else:
                    log_general.warning(
                        f"SolTrade failed to complete transaction {i}. Retrying. "
                        f"Error: {execute_result.get('error')}"
                    )
            except Exception as e:  # noqa: BLE001 - retry loop
                log_general.warning(
                    f"SolTrade failed to complete transaction {i}. Retrying. Error: {e}"
                )
                if order is not None and _swap_consumed(
                    sent_token_mint, sent_amount, balance_before
                ):
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
    decimals = config().decimals(output_token_mint)
    
    output_amount_str = "0"
    if execute_result and execute_result.get("totalOutputAmount"):
        output_amount_str = execute_result.get("totalOutputAmount")
    elif order and order.get("outAmount"):
        output_amount_str = order.get("outAmount")
    
    bought_amount = int(output_amount_str) / decimals
    
    log_transaction.info(
        f"SolTrade sold {sent_amount} {sent_token_symbol} for {bought_amount:.2f} {output_token_symbol}."
    )
    return {"out_amount": bought_amount, "sent_amount": sent_amount}
