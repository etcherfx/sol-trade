import json

from solana.rpc.core import TokenAccountOpts
from solders.pubkey import Pubkey

from sol_trade.config import config
from sol_trade.utils import handle_rate_limiting, run_async


def token_account_ui_amounts(owner: Pubkey, token_mint: str) -> list[float]:
    """UI amounts across all token accounts for a mint (empty when none)."""
    response = run_async(
        config()
        .client.get_token_accounts_by_owner_json_parsed(
            owner, TokenAccountOpts(mint=Pubkey.from_string(token_mint))
        )
    ).to_json()
    json_response = json.loads(response)
    amounts = []
    for account in json_response.get("result", {}).get("value", []):
        ui_amount = account["account"]["data"]["parsed"]["info"]["tokenAmount"][
            "uiAmount"
        ]
        if ui_amount is not None:
            amounts.append(float(ui_amount))
    return amounts


@handle_rate_limiting()
def find_balance(token_mint: str) -> float:
    """Return the spendable token balance of the wallet.

    SOL keeps a 0.02 SOL fee reserve; zero is returned below that floor.
    """
    if token_mint == config().sol_mint:
        balance_response = run_async(
            config().client.get_balance(config().public_address)
        ).value
        balance_response = balance_response / (10**9)
        if balance_response < 0.02:
            return 0.0
        return balance_response - 0.02

    amounts = token_account_ui_amounts(config().public_address, token_mint)
    return amounts[0] if amounts else 0.0
