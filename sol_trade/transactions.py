import base64
import os

import httpx
from solders.message import to_bytes_versioned
from solders.transaction import VersionedTransaction

from sol_trade.config import config
from sol_trade.log import log_general, log_transaction


class MarketPosition:
    def __init__(self, path):
        self.path = path
        self.ensure_directory_exists()

    def ensure_directory_exists(self):
        directory = os.path.dirname(self.path)
        if not os.path.exists(directory):
            os.makedirs(directory)


_market_instance = None


def market(path=None):
    global _market_instance
    if _market_instance is None and path is not None:
        _market_instance = MarketPosition(path)
    return _market_instance


async def create_order(
    input_amount: float, input_token_mint: str, output_token_mint: str
) -> dict:
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
        f"SolTrade is creating order for {input_amount} {input_token_mint}"
    )

    token_decimals = config().decimals(input_token_mint)
    
    # Convert token amount to smallest unit (lamports for SOL, etc.)
    amount_in_smallest_unit = int(input_amount * token_decimals)
    
    params = {
        "inputMint": input_token_mint,
        "outputMint": output_token_mint,
        "amount": amount_in_smallest_unit,
        "taker": str(config().public_address),
        "slippageBps": int(config().max_slippage or 50),
    }
    
    headers = {"Content-Type": "application/json"}
    if config().jupiter_api_key:
        headers["x-api-key"] = config().jupiter_api_key
    
    api_link = f"{config().jup_api}/order"
    log_transaction.info(f"SolTrade API Link: {api_link}")
    log_transaction.info(f"Parameters: {params}")
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(api_link, params=params, headers=headers)
        response.raise_for_status()
        result = response.json()
        log_transaction.info(f"Order response: {result}")
        return result


async def execute_order(order_response: dict) -> dict:
    """
    Signs and executes a swap order using Jupiter Ultra API.
    This replaces the legacy send_transaction function.
    """
    try:
        if "errorCode" in order_response:
            error_msg = order_response.get("errorMessage", "Unknown error")
            log_transaction.error(f"Order failed: {error_msg}")
            raise Exception(f"Order error: {error_msg}")
        
        transaction_b64 = order_response.get("transaction")
        if not transaction_b64:
            log_transaction.error("No transaction returned in order response")
            raise Exception("No transaction in order response")
        
        request_id = order_response["requestId"]
        
        # Deserialize and sign the transaction
        raw_txn = VersionedTransaction.from_bytes(base64.b64decode(transaction_b64))
        signature = config().keypair.sign_message(to_bytes_versioned(raw_txn.message))
        signed_txn = VersionedTransaction.populate(raw_txn.message, [signature])
        
        # Convert signed transaction back to base64
        signed_txn_b64 = base64.b64encode(bytes(signed_txn)).decode("utf-8")
        
        log_transaction.info(f"SolTrade is executing order with requestId: {request_id}")
        
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
            execute_response.raise_for_status()
            result = execute_response.json()
            
            if result.get("status") == "Success":
                log_transaction.info(f"SolTrade TxID: {result.get('signature')}")
            else:
                log_transaction.error(f"Transaction failed: {result.get('error')}")
            
            return result
            
    except Exception as e:
        log_transaction.error(f"Failed to execute transaction: {e}")
        raise


async def perform_swap(
    sent_amount: float,
    sent_token_mint: str,
    output_token_mint: str,
    sent_token_symbol: str,
    output_token_symbol: str,
):
    log_general.info("SolTrade is taking a market position.")

    order = execute_result = None
    is_tx_successful = False

    for i in range(0, 3):
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
                        f"SolTrade failed to complete transaction {i}. Error: {execute_result.get('error')}. Retrying."
                    )
            except Exception as e:
                log_general.warning(
                    f"SolTrade failed to complete transaction {i}. Retrying. Error: {e}"
                )
                continue

    if not is_tx_successful:
        log_general.error(
            "SolTrade failed to complete the transaction after 3 attempts."
        )
        return False

    # Calculate the actual amounts from the execution result
    decimals = config().decimals(output_token_mint)
    
    output_amount_str = "0"
    if execute_result and execute_result.get("totalOutputAmount"):
        output_amount_str = execute_result.get("totalOutputAmount")
    elif order and order.get("outAmount"):
        output_amount_str = order.get("outAmount")
    
    bought_amount = int(output_amount_str) / decimals
    
    log_transaction.info(
        f"Sold {sent_amount} {sent_token_symbol} for {bought_amount:.2f} {output_token_symbol}"
    )
    return True
