<h1 align="center">
  <img src="projectInfo/banner.png" alt="SolTrade Banner" width="850">
</h1>

<div align="center">

[![CodeFactor](https://www.codefactor.io/repository/github/etcherfx/soltrade/badge/main?style=for-the-badge)](https://www.codefactor.io/repository/github/etcherfx/soltrade/overview/main)
[![License](https://img.shields.io/github/license/etcherfx/soltrade?style=for-the-badge)](https://github.com/etcherfx/soltrade/blob/main/LICENSE)
[![GitHub issues](https://img.shields.io/github/issues/etcherfx/soltrade?style=for-the-badge)](https://github.com/etcherfx/soltrade/issues)
[![GitHub Release](https://img.shields.io/github/release/etcherfx/soltrade?include_prereleases&style=for-the-badge)](https://github.com/etcherfx/soltrade/releases/latest)

**A Solana trading bot with lots of features.**

Hard fork of noahtheprogrammer's [soltrade](https://github.com/noahtheprogrammer/soltrade)

</div>

## 📖 Table of Contents

- [📖 Table of Contents](#-table-of-contents)
- [🔗 Links](#-links)
- [📂 Features](#-features)
- [🔬 Advanced Features](#-advanced-features)
- [📚 Term Definitions](#-term-definitions)
- [🔧 Prerequisites](#-prerequisites)
- [⚙️ Configuration](#️-configuration)
- [🛠️ Installation](#️-installation)
- [📈 Custom Strategies](#-custom-strategies)
- [💸 Donations](#-donations)
- [⚠️ Disclaimer](#️-disclaimer)

## 🔗 Links 

- [Releases](https://github.com/etcherfx/SolTrade/releases)

## 📂 Features 

- **Custom strategies**: Create your own trading strategies and use them with SolTrade. Customize parameters like `stoploss`, `trailing_stoploss`, `takeprofit`, etc to fit your needs
- **Multiple token trading**: Instead of waiting for one token to meet trading conditions, you can analyze multiple tokens to increase trade chances

## 🔬 Advanced Features

### 🐋 Whale Wallet Tracking (Enabled by Default)

Monitors token balance changes for configured whale wallets in real-time. The whale tracker queries Solana RPC to track balance snapshots, then analyzes net balance delta over 1h, 4h, and 24h rolling windows to produce a signal per token:

- **ACCUMULATING**: Whales are net buying (>10% balance increase)
- **DUMPING**: Whales are net selling (>10% balance decrease)
- **NEUTRAL**: No significant movement; **NO_DATA**: no wallets configured or insufficient data

#### Configuration

Add whale wallet addresses to your `config.json`:
```json
"whale_wallets": {
  "SOL": ["wallet_address_1", "wallet_address_2"],
  "TRUMP": ["wallet_address_3"]
}
```

#### Discovering Whale Wallets

Use the built-in whale discovery CLI:
```
uv run -m soltrade.whale_discovery TOKEN_MINT [LIMIT]
```

### ⚖️ Confluence Filter (Enabled by Default)

Every trade routes through the confluence gate before execution. The filter combines whale signals with market regime and sentiment to adjust position sizing:

| TA Signal | Whale Activity | Action | Position Size |
|-----------|---------------|--------|--------------|
| BUY | ACCUMULATING | Full entry | 100% |
| BUY | NEUTRAL | Half entry | 50% |
| BUY | DUMPING | Skip | 0% |
| SELL | DUMPING | Full exit | 100% |
| SELL | NEUTRAL | Half exit | 50% |
| SELL | ACCUMULATING | Partial exit | 50% |

With no whale wallets configured (or while the tracker is still collecting snapshots), trades pass at full size.

In bearish market regimes, all position sizes are further reduced by 50%.

### 📊 Market Regime Detection (Opt-In)

Determines overall Solana market direction using SOL/USDC daily price trend (20-day SMA) and DEX volume trends. Set `market_regime_enabled: true` in config to activate.

| Regime | Condition | Position Modifier |
|--------|-----------|------------------|
| BULLISH | Price above 20-day SMA + rising volume | 1.0x |
| NEUTRAL | Mixed signals | 1.0x |
| BEARISH | Price below 20-day SMA + falling volume | 0.5x |

### 🛑 Sentiment Circuit Breaker (Opt-In)

Pulls social sentiment from Reddit for tracked tokens. If sentiment crashes below a configurable threshold, trading for that token is paused automatically. Set `sentiment_enabled: true` in config to activate.

- **Token pause**: Individual token blocked when sentiment drops below `sentiment_threshold` (default: -0.5)
- **Market crash**: All new entries paused when all tokens drop below `sentiment_crash_threshold` (default: -0.7)
- **Recovery**: Blocks expire after `sentiment_pause_hours` (default: 4 hours)

## 📚 Term Definitions

- **Primary Mint**: The token you want to trade with, usually a stablecoin like USDC
- **Secondary Mint**: The token you want to trade for, like SOL or any other Solana token
- **Trading Intervals**: The time interval between each technical analysis (whether current conditions are fit to trade), in minutes
- **Price Update Interval**: The time interval between each price update, in seconds
- **Max Slippage**: The maximum percentage difference between the expected price and the executed price when making a trade
- **Strategy**: The trading strategy you want to use, like `default` or your own custom strategy

## 🔧 Prerequisites 

- Sign up for a [CryptoCompare API key](https://www.cryptocompare.com/cryptopian/api-keys)
- Sign up for a free [Jupiter API key](https://portal.jup.ag/) (required for Ultra Swap API only)
- Create a new wallet on [Jupiter Wallet](https://jup.ag/) [Phantom](https://phantom.app/) or any other Solana wallet solely for SolTrade
- Deposit however much of the primary token you want to trade with into your wallet and at least `~0.2 $SOL` to cover transaction fees

## ⚙️ Configuration 

- Make a copy of the `config.json.sample` file and rename it to `config.json`
- Fill in / edit the following parameters in the `config.json` file or leave them default:
  | Parameter                  | Description                                                           |                Default                |
  | -------------------------- | --------------------------------------------------------------------- | :-----------------------------------: |
  | `api_key`                  | Your CryptoCompare API key                                            |                `Null`                 |
  | `jupiter_api_key`          | Your Jupiter API key from portal.jup.ag                               |                `Null`                 |
  | `private_key`              | Your Solana wallet private key                                        |                `Null`                 |
  | `rpc_https`                | HTTPS endpoint of your RPC (for balance checks & token info)          | `https://api.mainnet-beta.solana.com` |
  | `jup_api`                  | Jupiter Swap API endpoint                                             |     `https://api.jup.ag/swap/v2`     |
  | `primary_mint`             | Token address of main currency                                        |               `EPjF..v`               |
  | `primary_mint_symbol`      | Token symbol of main token                                            |                `USDC`                 |
  | `secondary_mints`          | Token address of each custom token(s) separated by `,` in a list `[]` |              `[So11..2]`              |
  | `secondary_mint_symbols`   | Token symbol of custom token(s) separated by `,` in a list `[]`       |                `[SOL]`                |
  | `price_update_seconds`     | Second-based time interval between token price updates                |                 `60`                  |
  | `trading_interval_minutes` | Minute-based time interval for technical analysis                     |                  `1`                  |
  | `max_slippage`             | Maximum slippage % in BPS (e.g. `50` = `0.50%`)                       |                 `50`                  |
  | `strategy`                 | The strategy you want to trade with                                   |               `default`               |
  | `whale_tracking_enabled`   | Enable whale wallet tracking                                          |               `true`                 |
  | `whale_wallets`            | Map of token symbols to whale wallet addresses                        |                  `{}`                 |
  | `whale_poll_interval_minutes` | How often to poll whale wallets                                  |                  `5`                  |
  | `confluence_enabled`       | Enable confluence filter for all trades                               |               `true`                 |
  | `market_regime_enabled`    | Enable market regime detection (opt-in)                               |              `false`                 |
  | `sentiment_enabled`        | Enable sentiment circuit breaker (opt-in)                             |              `false`                 |
  | `sentiment_pause_hours`    | Hours to pause trading when sentiment crashes                         |                  `4`                  |
  | `sentiment_threshold`      | Per-token sentiment block threshold (-1 to +1)                        |               `-0.5`                 |
  | `sentiment_crash_threshold`| Market-wide crash threshold (-1 to +1)                              |               `-0.7`                 |

## 🛠️ Installation

- Install Microsoft Visual C++ Build Tools from [here](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
- Install TA-Lib from [here](https://ta-lib.org/install/)
- Install UV:
  - Windows:
    ```
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```
  - Linux / macOS:
    ```
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
- Navigate over to the project root directory and run `main.py`:
  ```
  uv run main.py
  ```

## 📈 Custom Strategies 

> [!NOTE]  
> `{Your Strategy Name}` is just a placeholder for your strategy name. Replace it with your actual strategy name without the `{}`.

- Create a new Python file in the `strategies` directory named `{Your Strategy Name}_strategy.py`
- Create a class named `{Your Strategy Name}Strategy` (all one word with the first letter being a capital letter) that inherits from the `BaseStrategy` class
- Create a `__init__` method that takes in the following parameters:
  ```
  def __init__(self, df: pd.DataFrame):
    self.df = df
    self.stoploss =
    self.takeprofit =
    self.trailing_stoploss =
    self.trailing_stoploss_target =
  ```
- Create a `apply_strategy` method that is called by the bot to apply the strategy:
  ```
  def apply_strategy(self):
    if config().strategy == "{Your Strategy Name}":
      # Your strategy logic here
  ```
- Then, change the config `strategy` parameter to `{Your Strategy Name}`
- Lastly, feel free to make a pull request to add your strategy to the main project

## 💸 Donations

Similar to the original project, SolTrade does not currently include a platform fee and will remain open-source forever. However, if you would like to support the project, you can donate to the following Solana wallet address:

```
22gwSXc7mvp6UZwgDouhQuJ5AmHN3oxLNGULkARmT3PV
```

## ⚠️ Disclaimer

I am not responsible for any losses you may incur while using this software. Use at your own risk.
