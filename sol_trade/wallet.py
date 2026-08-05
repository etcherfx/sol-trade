import json

from solana.rpc.core import TokenAccountOpts
from solders.pubkey import Pubkey

from sol_trade.config import config
from sol_trade.utils import handle_rate_limiting, run_async


# Returns the current balance of token in the wallet
@handle_rate_limiting()
def find_balance(token_mint: str) -> float:
    if token_mint == config().sol_mint:
        balance_response = run_async(
            config().client.get_balance(config().public_address)
        ).value
        balance_response = balance_response / (10**9)
        if balance_response < 0.02:
            return 0.0
        return balance_response - 0.02

    response = run_async(
        config()
        .client.get_token_accounts_by_owner_json_parsed(
            config().public_address,
            TokenAccountOpts(mint=Pubkey.from_string(token_mint)),
        )
    ).to_json()
    json_response = json.loads(response)
    if len(json_response["result"]["value"]) == 0:
        return 0
    return json_response["result"]["value"][0]["account"]["data"]["parsed"]["info"][
        "tokenAmount"
    ]["uiAmount"]
