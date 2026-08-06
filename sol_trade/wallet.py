import json
import time

from solana.rpc.core import TokenAccountOpts
from solders.pubkey import Pubkey

from sol_trade.config import config
from sol_trade.utils import handle_rate_limiting, run_async

_SIGNATURE_FEE_LAMPORTS = 5000
_TOKEN_ACCOUNT_SIZE = 165
_SOL_RESERVE_FALLBACK = 0.02
_SOL_RESERVE_TTL_SECONDS = 60
_sol_reserve_cache: tuple[float, float] | None = None


def minimum_sol_needed() -> float:
    """Smallest SOL reserve that guarantees swaps can settle.

    JupiterZ gasless orders make Jupiter pay the signature and priority fees,
    so the only SOL the wallet can owe is the rent-exempt deposit for a token
    account that does not exist yet. The reserve is therefore: rent for every
    configured token whose ATA is missing (queried from the RPC), plus a small
    signature-fee buffer for non-gasless orders. Falls back to 0.02 SOL when
    the RPC is unreachable (over-reserving is safe).
    """
    global _sol_reserve_cache
    now = time.time()
    if _sol_reserve_cache is not None and now - _sol_reserve_cache[1] < _SOL_RESERVE_TTL_SECONDS:
        return _sol_reserve_cache[0]

    try:
        client = config().client
        owner = config().public_address
        missing = []
        for mint in dict.fromkeys([config().primary_mint, *config().secondary_mints]):
            if mint == config().sol_mint:
                continue  # native SOL needs no token account
            if not token_account_ui_amounts(owner, mint):
                missing.append(mint)
        rent = run_async(
            client.get_minimum_balance_for_rent_exemption(_TOKEN_ACCOUNT_SIZE)
        ).value
        reserve = max((len(missing) * rent + 2 * _SIGNATURE_FEE_LAMPORTS) / 1e9, 0.0)
    except Exception:  # noqa: BLE001 - RPC failure; over-reserving is safe
        reserve = _SOL_RESERVE_FALLBACK

    _sol_reserve_cache = (reserve, now)
    return reserve


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
        reserve = minimum_sol_needed()
        if balance_response < reserve:
            return 0.0
        return balance_response - reserve

    amounts = token_account_ui_amounts(config().public_address, token_mint)
    return amounts[0] if amounts else 0.0
