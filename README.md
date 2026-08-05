<h1 align="center">
  <img src="projectInfo/banner.png" alt="SolTrade Banner" width="850">
</h1>

<div align="center">

[![License](https://img.shields.io/github/license/etcherfx/sol-trade?style=for-the-badge)](https://github.com/etcherfx/sol-trade/blob/main/LICENSE)
[![GitHub issues](https://img.shields.io/github/issues/etcherfx/sol-trade?style=for-the-badge)](https://github.com/etcherfx/sol-trade/issues)
[![GitHub forks](https://img.shields.io/github/forks/etcherfx/sol-trade?style=for-the-badge)](https://github.com/etcherfx/sol-trade/network)
[![GitHub Release](https://img.shields.io/github/release/etcherfx/sol-trade?include_prereleases&style=for-the-badge)](https://github.com/etcherfx/sol-trade/releases/latest)

**Automated trading for Solana.**

A hard fork of [noahtheprogrammer/soltrade](https://github.com/noahtheprogrammer/soltrade).

</div>

> [!WARNING]
> SolTrade trades **real money** on Solana mainnet. Start with small amounts you can afford to lose, test with a new wallet first, and never risk funds you can't spare. Not financial advice — you are responsible for your own trades.

## Links

- [Releases](https://github.com/etcherfx/sol-trade/releases)

## Quick Start

1. Install [uv](https://docs.astral.sh/uv/).

   ```powershell
   # Windows
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

   ```bash
   # Linux / macOS
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Clone the repository and create the configuration file.

   ```bash
   git clone https://github.com/etcherfx/sol-trade.git
   cd sol-trade
   cp config.json.sample config.json
   ```

3. Set the required settings in `config.json`:

   - `private_key` — your Solana wallet private key
   - `api_key` — your [CryptoCompare](https://www.cryptocompare.com/cryptopian/api-keys) API key
   - `secondary_mints` / `secondary_mint_symbols` — the token(s) you want to trade

4. Start the bot.

   ```bash
   uv run main.py
   ```

## Configuration

SolTrade reads its configuration from `config.json` in the project root. Copy `config.json.sample` to `config.json` to create the configuration file.

### Core settings

| Setting | What it does | Default |
| --- | --- | --- |
| `private_key` | Solana wallet private key (base58) | — *(required)* |
| `api_key` | CryptoCompare API key, used for candlestick data | — *(required)* |
| `rpc_https` | Solana RPC endpoint for balances and token data | `https://api.mainnet-beta.solana.com` |
| `jup_api` | Jupiter Swap API endpoint | `https://api.jup.ag/swap/v2` |
| `jupiter_api_key` | Jupiter API key — optional, sent only if set | — |
| `primary_mint` / `primary_mint_symbol` | The token you pay with (usually a stablecoin) | `EPjF..v` / `USDC` |
| `secondary_mints` / `secondary_mint_symbols` | The token(s) you want to trade | `[So11..2]` / `[SOL]` |
| `price_update_seconds` | How often token prices refresh | `60` |
| `trading_interval_minutes` | How often the bot runs its analysis | `1` |
| `max_slippage` | Maximum accepted slippage in BPS (100 BPS = 1%) | `50` |
| `strategy` | The strategy to trade with | `default` |

### Advanced settings

| Setting | What it does | Default |
| --- | --- | --- |
| `whale_tracking_enabled` | Poll configured whale wallets and produce signals | `true` |
| `whale_wallets` | Wallet addresses to watch, per token symbol | `{}` |
| `whale_poll_interval_minutes` | How often whale balances are polled | `5` |
| `confluence_enabled` | Route every trade through the confluence filter | `true` |
| `market_regime_enabled` | Scale position sizes by market regime | `false` |
| `sentiment_enabled` | Pause trading when sentiment crashes | `false` |
| `sentiment_pause_hours` | How long a sentiment block lasts | `4` |
| `sentiment_threshold` | Per-token block threshold (-1 to +1) | `-0.5` |
| `sentiment_crash_threshold` | Market-wide crash threshold (-1 to +1) | `-0.7` |

## How it works

SolTrade runs a continuous loop:

1. **Fetch** — retrieve fresh prices and candlesticks for every configured token.
2. **Analyze** — the active strategy computes indicators (EMA, RSI, and Bollinger Bands by default) and produces `entry` / `exit` signals.
3. **Act** — buy signals open a position; sell signals, stop-losses, take-profits, and trailing stops close it. Every trade is routed through Jupiter's Swap API.
4. **Protect** — open positions are tracked and persisted to disk, so a restart resumes where it left off.

Optional layers — whale tracking, a confluence sizing filter, market regime detection, and a sentiment circuit breaker — operate between the signal and the trade. See [Advanced features](#advanced-features) for details.

## Features

| Feature | What it does |
| --- | --- |
| Technical analysis | EMA, RSI, and Bollinger Bands out of the box — pure Python, no C libraries |
| Multiple tokens | Trade several tokens in the same loop |
| Position management | Stop-loss, take-profit, and trailing stop on every position |
| Custom strategies | Drop in your own strategy file |
| Whale tracking | Watches configured wallets for accumulation or dumping |
| Confluence filter | Sizes every trade from whale activity, market regime, and sentiment |
| Market regime | Scales positions down in bearish markets *(opt-in)* |
| Sentiment breaker | Pauses trading when social sentiment crashes *(opt-in)* |

## Advanced features

<details>
<summary><b>How whale tracking, the confluence filter, market regime, and sentiment work</b></summary>

### Whale wallet tracking

The tracker polls the wallets in `whale_wallets` every `whale_poll_interval_minutes` minutes and compares balances over 1-hour, 4-hour, and 24-hour windows:

| Signal | Meaning |
| --- | --- |
| `ACCUMULATING` | Whales are net buying (>10% balance increase) |
| `DUMPING` | Whales are net selling (>10% balance decrease) |
| `NEUTRAL` | No significant movement |
| `NO_DATA` | No wallets configured, or not enough snapshots yet |

```json
"whale_wallets": {
  "SOL": ["wallet_address_1", "wallet_address_2"]
}
```

Top token holders can be discovered with the built-in CLI:

```bash
uv run -m sol_trade.whale_discovery TOKEN_MINT [LIMIT]
uv run -m sol_trade.whale_discovery So11111111111111111111111111111111111111112 10
```

### Confluence filter

The filter combines the whale signal, market regime, and sentiment to determine position size:

| TA Signal | Whale Activity | Action | Position Size |
| --- | --- | --- | --- |
| BUY | ACCUMULATING | Full entry | 100% |
| BUY | NEUTRAL | Half entry | 50% |
| BUY | DUMPING | Skip | 0% |
| SELL | DUMPING | Full exit | 100% |
| SELL | NEUTRAL | Half exit | 50% |
| SELL | ACCUMULATING | Partial exit | 50% |

> [!NOTE]
> With no whale wallets configured (or while the tracker is still collecting snapshots), trades pass at full size. The matrix only applies once wallets are set up and at least two snapshots exist.

In bearish regimes, all position sizes drop an additional 50%. Protective exits (stop-loss, take-profit, trailing stop) always execute at 100%.

### Market regime detection

The market is classified from the SOL/USDC daily trend (20-day SMA) and DEX volume, and entries are scaled accordingly:

| Regime | Condition | Position Modifier |
| --- | --- | --- |
| BULLISH | Price above 20-day SMA + rising volume | 1.0x |
| NEUTRAL | Mixed signals | 1.0x |
| BEARISH | Price below 20-day SMA + falling volume | 0.5x |

This feature is enabled by setting `"market_regime_enabled": true` in `config.json`.

### Sentiment circuit breaker

The breaker polls social sentiment from Reddit for the tracked tokens and pauses trading when sentiment collapses:

- **Token pause** — a token is blocked when its score drops below `sentiment_threshold`.
- **Market crash** — all new entries pause when every tracked token is below `sentiment_crash_threshold`.
- **Recovery** — blocks expire automatically after `sentiment_pause_hours`.

This feature is enabled by setting `"sentiment_enabled": true` in `config.json`.

</details>

## Custom strategies

> [!NOTE]
> Strategy names must be a single word, lowercase — `momentum`, `trendline`, etc.

1. Create `strategies/{name}_strategy.py`.
2. Define a class `{Name}Strategy(BaseStrategy)` with the following methods:
   - `__init__(self, df)` — store `self.df` and set the risk parameters `stoploss`, `takeprofit`, `trailing_stoploss`, and `trailing_stoploss_target` (percentages).
   - `apply_strategy(self)` — compute indicators, then set `self.df["entry"] = 1` on bars that should buy and `self.df["exit"] = 1` on bars that should sell.
3. Set `"strategy": "{name}"` in `config.json`.

Indicators (`ema`, `sma`, `rsi`) are available from `sol_trade.strategy` — pure-Python, TA-Lib-equivalent implementations.

<details>
<summary><b>Example — a momentum strategy</b></summary>

```python
# strategies/momentum_strategy.py
import pandas as pd

from sol_trade.config import config
from sol_trade.strategy import ema, rsi
from .base_strategy import BaseStrategy


class MomentumStrategy(BaseStrategy):
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.stoploss = 5
        self.takeprofit = 10
        self.trailing_stoploss = 2
        self.trailing_stoploss_target = 5

    def apply_strategy(self):
        if config().strategy == "momentum":
            self.df["ema_fast"] = ema(self.df["close"], 8)
            self.df["ema_slow"] = ema(self.df["close"], 21)
            self.df["rsi"] = rsi(self.df["close"], 14)

            entry = (self.df["ema_fast"] > self.df["ema_slow"]) & (self.df["rsi"] <= 40)
            exit_ = (self.df["ema_fast"] < self.df["ema_slow"]) | (self.df["rsi"] >= 70)

            self.df.loc[entry, "entry"] = 1
            self.df.loc[exit_, "exit"] = 1

        return self.df
```

</details>

New strategies may be contributed via pull request.

## FAQ

**What happens if I stop the bot while I'm holding a position?**
Your open position is saved to `data/{TOKEN}_data.csv`. On restart, the bot picks it up and keeps managing its stop-loss and take-profit.

**Do I need a Jupiter API key?**
No. It is optional and only sent if set — the default `swap/v2` endpoint works without one.

**Can I trade more than one token?**
Yes. Add each token to `secondary_mints` (and its symbol to `secondary_mint_symbols`) and SolTrade trades them all in the same loop.

**Where is my private key stored?**
Only in `config.json` on your machine. The bot loads and signs locally — it is never sent to any server, and `config.json` is git-ignored.

## Glossary

| Term | Meaning |
| --- | --- |
| Primary mint | The token you trade with, usually a stablecoin like USDC |
| Secondary mint | The token you trade for, e.g. SOL |
| Trading interval | Minutes between each technical analysis pass |
| Price update interval | Seconds between price refreshes |
| Slippage | Difference between expected and executed trade price |
| BPS | Basis points — 100 BPS = 1% |
| Whale | A wallet holding a large amount of a token |
