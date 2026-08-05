<h1 align="center">
  <img src="projectInfo/banner.png" alt="SolTrade Banner" width="850">
</h1>

<div align="center">

[![License](https://img.shields.io/github/license/etcherfx/sol-trade?style=for-the-badge)](https://github.com/etcherfx/sol-trade/blob/main/LICENSE)
[![GitHub issues](https://img.shields.io/github/issues/etcherfx/sol-trade?style=for-the-badge)](https://github.com/etcherfx/sol-trade/issues)
[![GitHub Release](https://img.shields.io/github/release/etcherfx/sol-trade?include_prereleases&style=for-the-badge)](https://github.com/etcherfx/sol-trade/releases/latest)

**Automated trading for Solana.** SolTrade watches the tokens you choose, runs technical
analysis on every trading interval, and enters and exits positions for you — with optional
whale tracking, sentiment filters, and market-aware position sizing.

A hard fork of [noahtheprogrammer/soltrade](https://github.com/noahtheprogrammer/soltrade).

</div>

> [!WARNING]
> SolTrade trades **real money** on Solana mainnet. Start with small amounts you can afford
> to lose, and test with a new wallet before trusting it with anything meaningful. This
> software is not financial advice — you are responsible for your own trades.

## Table of contents

- [Quick start](#quick-start)
- [How it works](#how-it-works)
- [Features](#features)
- [Configuration](#configuration)
- [Advanced features in depth](#advanced-features-in-depth)
- [Installation](#installation)
- [Custom strategies](#custom-strategies)
- [FAQ](#faq)
- [Glossary](#glossary)
- [Support](#support)
- [Disclaimer](#disclaimer)

## Quick start

1. Install [uv](https://docs.astral.sh/uv/) (Python runtime manager — the only requirement).
2. Clone the repo and copy the sample config:
   ```bash
   git clone https://github.com/etcherfx/sol-trade.git
   cd sol-trade
   cp config.json.sample config.json
   ```
3. Open `config.json` and fill in the essentials (see [Configuration](#configuration)):
   - `private_key` — your Solana wallet's private key
   - `api_key` — your [CryptoCompare](https://www.cryptocompare.com/cryptopian/api-keys) API key
   - `secondary_mints` / `secondary_mint_symbols` — the token(s) you want to trade
4. Run it:
   ```bash
   uv run main.py
   ```

That's it. `uv` creates the environment from the lockfile, installs everything, and starts
the bot. The first run fetches market data, takes a snapshot of your balances, and begins
analyzing every `trading_interval_minutes`.

## How it works

SolTrade runs a simple loop:

1. **Fetch** fresh prices and candlesticks for every configured token.
2. **Analyze** — your strategy computes indicators (EMA, RSI, Bollinger Bands by default)
   and produces `entry` / `exit` signals.
3. **Act** — buy signals open a position; sell signals, stop-losses, take-profits, and
   trailing stops close it. Every trade is routed through Jupiter's Swap API.
4. **Protect** — open positions are tracked with a stop-loss, take-profit, and trailing
   stop, and persisted to disk so a restart picks up where you left off.

Optional layers sit between the signal and the trade: whale tracking, a confluence sizing
filter, market regime detection, and a sentiment circuit breaker (all detailed in
[Advanced features](#advanced-features-in-depth)).

## Features

**Core**

- **Technical analysis** — EMA, RSI, and Bollinger Bands out of the box, computed in pure
  Python (no C libraries to install).
- **Multiple tokens** — trade several tokens at once instead of waiting for one.
- **Automatic position management** — stop-loss, take-profit, and trailing stop on every
  position.
- **Custom strategies** — drop in your own strategy file with your own indicators and rules
  (see [Custom strategies](#custom-strategies)).

**Advanced — on by default**

- **Whale wallet tracking** — watches wallets you configure, detects accumulation or
  dumping, and feeds that into trade decisions.
- **Confluence filter** — sizes every trade based on whale activity, market regime, and
  sentiment.

**Advanced — opt-in**

- **Market regime detection** — reads the SOL/USDC trend to scale positions in bearish
  markets.
- **Sentiment circuit breaker** — pauses trading when social sentiment on a token crashes.

## Configuration

### 1. Create your config

Copy `config.json.sample` to `config.json` and edit it. The bot reads `config.json` from
the project root — make sure you keep it there.

### 2. Core settings

| Setting | What it does | Default |
| --- | --- | --- |
| `private_key` | Your Solana wallet private key (base58) | — (required) |
| `api_key` | CryptoCompare API key, used for candlestick data | — (required) |
| `rpc_https` | Solana RPC endpoint for balances and token data | `https://api.mainnet-beta.solana.com` |
| `jup_api` | Jupiter Swap API endpoint | `https://api.jup.ag/swap/v2` |
| `jupiter_api_key` | Jupiter API key — optional, sent only if set | — |
| `primary_mint` / `primary_mint_symbol` | The token you pay with (usually a stablecoin) | `EPjF..v` / `USDC` |
| `secondary_mints` / `secondary_mint_symbols` | The token(s) you want to trade | `[So11..2]` / `[SOL]` |
| `price_update_seconds` | How often token prices refresh | `60` |
| `trading_interval_minutes` | How often the bot runs its analysis | `1` |
| `max_slippage` | Maximum accepted slippage in BPS (100 BPS = 1%) | `50` |
| `strategy` | The strategy to trade with | `default` |

### 3. Advanced feature settings

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

## Advanced features in depth

### Whale wallet tracking

The tracker polls the wallets in `whale_wallets` every `whale_poll_interval_minutes` and
compares balances over 1h, 4h, and 24h windows to produce a per-token signal:

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

**Finding wallets to track** — SolTrade ships a discovery CLI that lists the top holders of
any token:

```bash
uv run -m sol_trade.whale_discovery TOKEN_MINT [LIMIT]
```

```bash
uv run -m sol_trade.whale_discovery So11111111111111111111111111111111111111112 10
```

### Confluence filter

Every trade passes through the confluence gate before execution. It combines the whale
signal with market regime and sentiment to decide *how much* to trade:

| TA Signal | Whale Activity | Action | Position Size |
| --- | --- | --- | --- |
| BUY | ACCUMULATING | Full entry | 100% |
| BUY | NEUTRAL | Half entry | 50% |
| BUY | DUMPING | Skip | 0% |
| SELL | DUMPING | Full exit | 100% |
| SELL | NEUTRAL | Half exit | 50% |
| SELL | ACCUMULATING | Partial exit | 50% |

> [!NOTE]
> With no whale wallets configured (or while the tracker is still collecting snapshots),
> trades pass at full size. The matrix above only applies once wallets are set up and at
> least two snapshots exist.

In bearish market regimes, all position sizes are additionally reduced by 50%. Protective
exits (stop-loss, take-profit, trailing stop) always execute at 100% regardless of the
confluence state.

### Market regime detection

Uses the SOL/USDC daily trend (20-day SMA) and DEX volume to classify the market, then
scales entries accordingly:

| Regime | Condition | Position Modifier |
| --- | --- | --- |
| BULLISH | Price above 20-day SMA + rising volume | 1.0x |
| NEUTRAL | Mixed signals | 1.0x |
| BEARISH | Price below 20-day SMA + falling volume | 0.5x |

Enable with `"market_regime_enabled": true`.

### Sentiment circuit breaker

Pulls social sentiment from Reddit for the tokens you track. If sentiment drops below the
threshold, trading for that token pauses automatically:

- **Token pause** — a token is blocked when its score drops below `sentiment_threshold`.
- **Market crash** — all new entries pause when every tracked token is below
  `sentiment_crash_threshold`.
- **Recovery** — blocks expire automatically after `sentiment_pause_hours`.

Enable with `"sentiment_enabled": true`.

## Installation

SolTrade only requires [uv](https://docs.astral.sh/uv/). Everything else — including
Python itself — is managed automatically from the committed lockfile.

**Windows**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Linux / macOS**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then, from the project root:

```bash
uv run main.py
```

> [!TIP]
> `uv sync` is run automatically by `uv run` — you never need to set up a virtual
> environment by hand.

## Custom strategies

> [!NOTE]
> Strategy names must be a single word, lowercase — `momentum`, `trendline`, etc.

1. Create `strategies/{name}_strategy.py`.
2. Define a class `{Name}Strategy(BaseStrategy)` with:
   - `__init__(self, df)` — store `self.df`, and set the risk parameters:
     `stoploss`, `takeprofit`, `trailing_stoploss`, `trailing_stoploss_target` (percentages).
   - `apply_strategy(self)` — compute indicators, then set `self.df["entry"] = 1` on bars
     that should buy and `self.df["exit"] = 1` on bars that should sell.
3. Set `"strategy": "{name}"` in `config.json`.

Indicators are available from `sol_trade.strategy` — pure-python, TA-Lib-equivalent
implementations of `ema`, `sma`, and `rsi`:

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
            # Indicators
            self.df["ema_fast"] = ema(self.df["close"], 8)
            self.df["ema_slow"] = ema(self.df["close"], 21)
            self.df["rsi"] = rsi(self.df["close"], 14)

            # Signals
            entry = (self.df["ema_fast"] > self.df["ema_slow"]) & (self.df["rsi"] <= 40)
            exit_ = (self.df["ema_fast"] < self.df["ema_slow"]) | (self.df["rsi"] >= 70)

            self.df.loc[entry, "entry"] = 1
            self.df.loc[exit_, "exit"] = 1

        return self.df
```

Made something you like? Feel free to open a pull request to add your strategy to the
project.

## FAQ

**What happens if I stop the bot while I'm holding a position?**
Your open position is saved to `data/{TOKEN}_data.csv`. When you restart, the bot picks up
the existing position and keeps managing its stop-loss and take-profit.

**Do I need a Jupiter API key?**
No. The key is optional and only sent if you set it — the default `swap/v2` endpoint works
without one.

**Can I trade more than one token?**
Yes. Add each token to `secondary_mints` (and its symbol to `secondary_mint_symbols`) and
SolTrade analyzes and trades them all in the same loop.

**Where is my private key stored?**
In `config.json` on your machine. The bot loads it locally and signs transactions locally —
it is never sent to any server, and `config.json` is git-ignored.

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

## Support

SolTrade has no platform fee and will stay open-source. If you'd like to support the
project, donations are welcome at:

```
22gwSXc7mvp6UZwgDouhQuJ5AmHN3oxLNGULkARmT3PV
```

## Disclaimer

I am not responsible for any losses you may incur while using this software. Use at your
own risk. Nothing here is financial advice.
