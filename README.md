<h1 align="center">
  <img src=".github/assets/banner.png" alt="SolTrade" width="850">
</h1>

<div align="center">

<img alt="python" height="56" src="https://cdn.jsdelivr.net/npm/@intergrav/devins-badges@3/assets/cozy/built-with/python_vector.svg">

[![Release](https://img.shields.io/github/v/release/etcherfx/sol-trade?style=for-the-badge)](https://github.com/etcherfx/sol-trade/releases/latest)

**A Solana trading bot that runs your strategies and swaps through Jupiter.**

[Get started](#getting-started) · [Configuration](#configuration) · [Multi-token portfolios](#multi-token-portfolios) · [Custom strategies](#custom-strategies) · [Research pipeline](#research-pipeline) · [Agent CLI](#agent-cli) · [FAQ](#faq)

</div>

> [!WARNING]
> SolTrade trades **real money** on Solana mainnet. Start with small amounts you can afford to lose, test with a new wallet first, and never risk funds you can't spare. Not financial advice — you are responsible for your own trades.

## Features

| Feature | What it does |
| --- | --- |
| Technical analysis | EMA, RSI, Bollinger Bands, MACD, ATR and more — pure Python, no C libraries |
| Multiple tokens | Trade several tokens in the same loop |
| Position management | Stop-loss, take-profit, and trailing stop on every position |
| JSON strategies | Describe a strategy in a JSON file; no code to run |
| Backtests | Replays stored candles through the same engine the bot trades with, after costs |
| Research pipeline | Gates, a held-out period, paper trading and automatic promotion and demotion |
| Agent CLI | Every research and pipeline command prints one JSON object |
| Kill switch | Flattens everything and pauses at a 15% drawdown |
| Sentiment recorder | Records Reddit posts and comments as strategy inputs |
| Whale tracking | Watches configured wallets for accumulation or dumping *(opt-in)* |
| Confluence filter | Sizes trades from whale activity, market regime, and sentiment *(opt-in)* |
| Market regime | Scales positions down in bearish markets *(opt-in)* |
| Sentiment breaker | Pauses trading when recorded sentiment crashes *(opt-in)* |

## Getting started

1. Download the archive for your system from the [latest release](https://github.com/etcherfx/sol-trade/releases/latest) and extract it: `sol-trade-X.Y.Z-windows.zip` on Windows, `sol-trade-X.Y.Z-macos-linux.tar.gz` on macOS and Linux.

2. Run the launcher in the extracted folder: `run.cmd` on Windows (double-clicking works), `./run.sh` on macOS and Linux. SolTrade runs with [uv](https://docs.astral.sh/uv/), and the launcher offers to install it if it's missing. The first run creates `config.json` and `.env` from the samples and stops.

3. Configure credentials and tokens:

   - In `.env`, set `SOLTRADE_PRIVATE_KEY` (your Solana wallet).
   - In `config.json`, set `primary_mint` / `primary_mint_symbol` (the token you pay with) and `secondary_mints` / `secondary_mint_symbols` (the token(s) you want to trade). An empty token list is fine: the bot then idles and records.

4. Paper trade first by running the launcher with `--dry-run`. Swaps are simulated against a virtual `risk.max_live_capital` and the wallet is never touched, so no private key is needed:

   ```bash
   run.cmd --dry-run     # Windows
   ./run.sh --dry-run    # macOS and Linux
   ```

   Run the launcher without `--dry-run` to trade for real.

To update, extract the new release and copy `config.json`, `.env` and the `data/` folder into it.

### Running from source

With [uv](https://docs.astral.sh/uv/) installed:

```bash
git clone https://github.com/etcherfx/sol-trade.git
cd sol-trade
cp config.json.sample config.json
cp .env.sample .env
uv run main.py --dry-run
```

Fill in `config.json` and `.env` as in step 3, and drop `--dry-run` to trade for real.

## Usage

### How it works

SolTrade runs a continuous loop:

1. **Fetch** — the bot updates each token's candles at its strategy's timeframe and keeps them in a local store.
2. **Analyze** — when a bar closes, the token's strategy computes its indicators and produces `entry` / `exit` signals.
3. **Act** — entries and signal exits fill at the next bar's open. Stop-losses and trailing stops are watched within the bar against levels set at the previous close; take-profits trigger on a close and fill at the next open. Every live trade is routed through the Jupiter Swap API.
4. **Protect** — positions live in `data/research.db`, so a restart resumes where it left off and replays any bars it missed.

The [backtester](#research-pipeline) runs the same engine on stored candles, so a backtest and the live loop make the same decisions from the same bars. Optional layers — whale tracking, a confluence sizing filter, market regime detection, and a sentiment circuit breaker — operate between the signal and the trade for tokens you configure by hand. See [Advanced features](#advanced-features) for details.

### Terminal UI

The bot runs inside a full-screen terminal UI:

| Screen | Key | Shows |
| --- | --- | --- |
| Dashboard | `1` | Live wallet balance, portfolio value, profit, and per-token indicators |
| Logs | `2` | Full log history (scroll with arrow keys, `PgUp`/`PgDn`, `Home`/`End`) |
| Help | `3` | Keybindings and general info |

`Tab` cycles through the screens; `q` or `Ctrl-C` quits. The bot keeps trading in the background while you browse the UI.

### Headless and Docker

`--headless` runs without the terminal UI: logs go to stdout as JSON lines (and to the log files), the bot rewrites `data/status.json` every cycle, and `SIGTERM` or `Ctrl-C` stops it cleanly. `SOLTRADE_LOG_DIR` moves the log files.

```bash
uv run main.py --dry-run --headless
```

The repo and the release archives include a `Dockerfile` and `compose.yaml`. The container runs `main.py --headless --dry-run` unless you replace its command, and its healthcheck is `status --healthcheck`. Put `config.json` (and `.env` if you trade live) in a `config/` folder next to `compose.yaml`, then:

```bash
docker compose up -d --build
```

| Mount | Holds |
| --- | --- |
| `./config` → `/config` | `config.json` and `.env`; a folder, because the pipeline rewrites `config.json` atomically |
| `./strategies` → `/app/strategies` | Your JSON strategies |
| `./data` → `/app/data` | Candles, positions, the pipeline, recordings, the heartbeat |
| `./logs` → `/app/logs` | Log files |

To trade real money, set `command: []` in `compose.yaml`. Run CLI commands inside the container with `docker compose exec soltrade python -m sol_trade.cli status`.

**Unraid:** copy the folder to a share (for example `/mnt/user/appdata/sol-trade`), set `PUID=99` and `PGID=100` in the environment (or an `.env` file next to `compose.yaml`) so files belong to `nobody:users`, and start it with the Compose Manager plugin or `docker compose up -d --build` in that folder.

### RPC budget

Every Solana RPC call goes through one limiter (`rpc.max_cu_per_second`) and counts against `rpc.daily_cu_budget`; `status` shows today's calls and compute units, also kept in `data/rpc_usage.json`. Research commands and the dry run's virtual book make none. With the defaults, live trading of `T` tokens costs about:

| Source | Compute units per day |
| --- | --- |
| Balance reads, reused for 15 minutes | 960 × (T + 1) |
| Dashboard holdings, every 10 minutes (`holdings_refresh_seconds`; 0 turns it off, also in dry runs with a key) | 2,880 |
| SOL fee reserve, hourly | 240 × (T + 2) |
| Whale tracking, if on | 2,880 per wallet at a 5-minute poll |

Three tokens without whale tracking come to about 8,000 CU a day; swaps add a few reads each.

## Configuration

SolTrade reads its configuration from `config.json` and its credentials from the `.env` file, both in the project root. Copy `config.json.sample` to `config.json` and `.env.sample` to `.env` before the first run. Environment variables take precedence over `config.json`, and `SOLTRADE_CONFIG_PATH` and `SOLTRADE_DOTENV_PATH` point the bot and the CLI at files elsewhere.

Config is **hot-reloaded**: edits to `config.json` are picked up automatically on the next trading cycle (≤ 1 minute) — strategy, feature toggles, slippage, whale wallets, polling intervals, and the traded tokens all apply live. Structural settings (the primary token you pay with, RPC, exchange) and `.env` credentials still require a restart. Removing a token that currently has an open position is refused until that position is closed.

### Credentials

Secrets live in the git-ignored `.env` file:

| Variable | Purpose |
| --- | --- |
| `SOLTRADE_PRIVATE_KEY` | Solana wallet private key (base58) |
| `SOLTRADE_JUPITER_API_KEY` | Jupiter API key — optional, sent only if set |

### Core settings

| Setting | What it does | Default |
| --- | --- | --- |
| `rpc_https` | Solana RPC endpoint for balances and token data | `https://api.mainnet-beta.solana.com` |
| `jup_api` | Jupiter Swap API endpoint | `https://api.jup.ag/swap/v2` |
| `primary_mint` / `primary_mint_symbol` | The token you pay with (usually a stablecoin); required | none (the sample uses USDC) |
| `secondary_mints` / `secondary_mint_symbols` | The token(s) you want to trade | `[]` |
| `token_mints` | Mints of tokens you research or promote but don't trade by hand, e.g. `{"JUP": "JUPy..."}` | `{}` |
| `sol_mint` | Mint address of native SOL | `So11111111111111111111111111111111111111112` |
| `price_update_seconds` | How often token prices refresh | `60` |
| `max_slippage` | Maximum accepted slippage in BPS (100 BPS = 1%) | `50` |
| `strategy` | The strategy to trade with (`strategies/{name}.json`, or the built-in `default`) | `default` |
| `strategies_path` | Folder of JSON strategies | `strategies` |
| `secondary_weights` | Portfolio weight per token (parallel to `secondary_mints`); empty = equal split | `[]` |
| `token_strategies` | Per-token strategy overrides, e.g. `{"SOL": "default", "JUP": "momentum"}` | `{}` |
| `token_exchanges` | Per-token candle-data exchange overrides, e.g. `{"POPCAT": "mexc"}`; missing = `data_exchange` | `{}` |
| `data_exchange` | Exchange used for candlestick data (via ccxt) | `okx` |
| `candles_path` | Local SQLite store for candlestick history | `data/candles.db` |
| `research_db_path` | Positions, trades, trials and the pipeline | `data/research.db` |

### Sections

Each section is an object in `config.json`; keys you leave out keep their defaults.

| Section | Key | What it does | Default |
| --- | --- | --- | --- |
| `risk` | `max_live_capital` | Capital the bot may deploy (and the dry run's virtual cash) | `500` |
| | `max_trade_size` | Largest single buy | `100` |
| | `max_drawdown_pct` | Kill switch and every drawdown gate, all measured on `max_live_capital` | `15` |
| | `dust_threshold_usd` | Positions worth less are written off instead of sold | `1` |
| `rpc` | `max_cu_per_second` | Ceiling for all Solana RPC traffic | `30` |
| | `daily_cu_budget` | Past this, only the balance reads swaps need still run | `600000` |
| | `method_cu` | Per-method compute-unit overrides | `{}` |
| | `holdings_refresh_seconds` | Wallet holdings view on the dashboard; `0` turns it off | `600` |
| | `balance_refresh_seconds` | How long a balance read is reused | `900` |
| | `sol_reserve_ttl_seconds` | How long the SOL fee reserve is reused | `3600` |
| `costs` | `default_cost_bps` | Per-side cost for tokens without measured quotes | `50` |
| | `safety_multiplier` | Measured costs are multiplied by this | `1.25` |
| | `probe_sizes_usd` | Trade sizes `measure-costs` quotes | `[25, 100, 250]` |
| | `max_sample_age_days` | Older quotes are stale | `14` |
| `backfill` | `default_timeframes` / `default_days` | What `backfill` downloads without arguments | `["1h"]` / `730` |
| | `page_limit` | Bars per exchange request | `100` |
| `gates` | see [Research pipeline](#research-pipeline) | | |
| `promotion` | see [Research pipeline](#research-pipeline) | | |
| `sentiment` | `subreddits` | Subreddits the recorder reads | `[]` |
| | `record` | Record while the bot runs | `true` |
| | `request_interval_seconds` | Gap between Reddit requests | `65` |
| | `keywords` | Words that count as a mention, per symbol, e.g. `{"SOL": ["SOL", "Solana"]}`; the symbol by default | `{}` |

### Advanced settings

| Setting | What it does | Default |
| --- | --- | --- |
| `whale_tracking_enabled` | Poll configured whale wallets and produce signals | `false` |
| `whale_wallets` | Wallet addresses to watch, per token symbol | `{}` |
| `whale_poll_interval_minutes` | How often whale balances are polled | `5` |
| `confluence_enabled` | Route every trade through the confluence filter | `false` |
| `market_regime_enabled` | Scale position sizes by market regime | `false` |
| `market_regime_pair` | The `BASE/QUOTE` pair whose daily trend sets the regime | `""` |
| `sentiment_enabled` | Pause trading when sentiment crashes | `false` |
| `sentiment_pause_hours` | How long a sentiment block lasts | `4` |
| `sentiment_threshold` | Per-token block threshold (-1 to +1) | `-0.5` |
| `sentiment_crash_threshold` | Market-wide crash threshold (-1 to +1) | `-0.7` |
| `whale_data_path` | Where whale snapshots are stored | `data/whale_data.json` |
| `regime_data_path` | Where market regime data is stored | `data/regime_data.json` |
| `sentiment_data_path` | Where sentiment data is stored | `data/sentiment_data.json` |

## Multi-token portfolios

Trade several tokens with per-token allocation and strategies. Tokens are **hot-reloadable** — add or remove them in `config.json` and the change applies on the next cycle, no restart.

### Portfolio weights

`secondary_weights` is a list parallel to `secondary_mints` — each entry is that token's share of total capital (cash plus all open positions). Weights are normalized to sum to 1 on load; an empty list means an equal split. A weight of `0` makes a token **watch-only** — it is analyzed but never bought.

```json
"secondary_mints": ["So1111...2", "JUPyiwr...gBz"],
"secondary_mint_symbols": ["SOL", "JUP"],
"secondary_weights": [0.6, 0.4]
```

A buy deploys at most `weight × total_capital`, so each token gets its slice instead of the first signal taking everything.

### Per-token strategies

`token_strategies` assigns a different strategy per token; tokens not listed use the global `strategy`.

```json
"token_strategies": { "SOL": "default", "JUP": "momentum" }
```

### Per-token exchanges

`token_exchanges` assigns a different candle-data exchange per token (any [ccxt](https://github.com/ccxt/ccxt) exchange id); tokens not listed use the global `data_exchange`. Use it when one exchange doesn't list all your tokens — e.g. POPCAT is missing on OKX but available on MEXC:

```json
"data_exchange": "okx",
"token_exchanges": { "POPCAT": "mexc" }
```

Both settings hot-reload, but `data_exchange` itself is structural — changing it still requires a restart. Each exchange client is created once and reused; candle history is keyed by symbol, so switching a token's exchange starts fresh data for it.

### Changing tokens while running

- **Adding** a token is always safe — it starts flat, with no position.
- **Removing** a token with an **open position is refused**: the change is rolled back and an error is logged, because a removed token stops being managed (no stop-loss, take-profit, or trailing stop). Close the position first.
- When the token set changes, the P&L baseline is recaptured automatically so profit figures stay correct.

## Advanced features

<details>
<summary><b>How whale tracking, the confluence filter, market regime, and sentiment work</b></summary>

### Whale wallet tracking

The tracker polls the wallets listed in `whale_wallets` at the interval specified by `whale_poll_interval_minutes` and compares balances over 1-hour, 4-hour, and 24-hour windows:

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

Top token holders can be discovered with the built-in CLI. The command below lists the ten largest holders of SOL:

```bash
uv run -m sol_trade.whale_discovery So11111111111111111111111111111111111111112 10
```

Usage: `uv run -m sol_trade.whale_discovery TOKEN_MINT [LIMIT]`

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

The market is classified from the daily trend (20-day SMA) of `market_regime_pair` and DEX volume, and entries are scaled accordingly:

| Regime | Condition | Position Modifier |
| --- | --- | --- |
| BULLISH | Price above 20-day SMA + rising volume | 1.0x |
| NEUTRAL | Mixed signals | 1.0x |
| BEARISH | Price below 20-day SMA + falling volume | 0.5x |

This feature is enabled by setting `"market_regime_enabled": true` and a `market_regime_pair` such as `"SOL/USDC"` in `config.json`.

### Sentiment circuit breaker

The breaker scores the last day of [recorded](#sentiment-recorder) Reddit mentions of each tracked token and pauses trading when sentiment collapses:

- **Token pause** — a token is blocked when its score drops below `sentiment_threshold`.
- **Market crash** — all new entries pause when every tracked token is below `sentiment_crash_threshold`.
- **Recovery** — blocks expire automatically after `sentiment_pause_hours`.

This feature is enabled by setting `"sentiment_enabled": true` in `config.json`. Strategies promoted by the [research pipeline](#research-pipeline) are never filtered by the breaker, the confluence filter or whale signals: they trade exactly as they were tested.

### Sentiment recorder

While the bot runs, it reads the new posts and new comments of each subreddit in `sentiment.subreddits` through Reddit's public RSS feeds, one request a minute (Reddit's limit for anonymous readers), and stores the raw text in `data/sentiment.db`. Strategies read two features from it with the `sentiment` indicator: `mentions` (items naming the token in a trailing window) and `score` (their mean [VADER](https://github.com/cjhutto/vaderSentiment) tone, -1 to +1). Values before recording started are missing, so backtests only see sentiment for the period actually recorded. `status` reports the recorder's items, gaps and errors.

</details>

## Custom strategies

A strategy is a JSON file, `strategies/{name}.json`, where `name` is lowercase with underscores (`momentum`, `mean_reversion`). Set `"strategy": "{name}"` (or a `token_strategies` entry) to trade it. Without a file, `default` is the built-in example; it is not a tested edge.

> [!NOTE]
> Python strategy files (`*_strategy.py`) are no longer loaded. A token whose strategy only exists as Python is skipped with the error `convert to strategies/{name}.json`.

```json
{
  "schema": 1,
  "name": "momentum",
  "description": "Fast EMA over slow EMA while RSI isn't stretched.",
  "timeframe": "1h",
  "lookback": 60,
  "risk": {"stoploss": 5, "takeprofit": 10, "trailing_stoploss": 2, "trailing_stoploss_target": 5},
  "indicators": {
    "fast": {"fn": "ema", "period": 8},
    "slow": {"fn": "ema", "period": 21},
    "rsi": {"fn": "rsi", "period": 14}
  },
  "entry": {"all": [{"crosses_above": ["fast", "slow"]}, ["rsi", "<=", 60]]},
  "exit": {"any": [{"crosses_below": ["fast", "slow"]}, ["rsi", ">=", 75]]}
}
```

| Key | Meaning |
| --- | --- |
| `schema` | Always `1` |
| `name` | Must equal the file name |
| `timeframe` | Bar size the strategy trades on, e.g. `15m`, `1h`, `4h` (any timeframe the exchange offers) |
| `lookback` | Bars of history each decision sees; must cover the indicators' warm-up |
| `risk` | Percentages; `0` or a missing key turns that exit off. The trailing stop sits `trailing_stoploss`% under the highest closed-bar high once a high reaches `trailing_stoploss_target`% above entry. `position_size` (optional, above 0 and up to 1, default `1`) is the share of `risk.max_trade_size` each entry spends, in backtests, paper and live alike |
| `indicators` | Named indicators: `{"fn": ..., parameters, "input": series}`; `input` defaults to `close` |
| `entry` / `exit` | Rules evaluated on each closed bar; `exit` is optional |

| Function | Parameters (default) | Outputs |
| --- | --- | --- |
| `sma`, `ema`, `highest`, `lowest` | `period` | one value |
| `rsi` | `period` (14) | one value |
| `roc` | `period` | percent change |
| `stddev`, `zscore` | `period`, `ddof` (1) | one value |
| `bbands` | `period` (20), `mult` (2), `ddof` (1) | `upper`, `middle`, `lower`, `width` |
| `macd` | `fast` (12), `slow` (26), `signal` (9) | `macd`, `signal`, `hist` |
| `atr` | `period` (14), `method` (`wilder` or `sma`) | one value |
| `stoch` | `k` (14), `d` (3), `smooth` (3) | `k`, `d` |
| `sentiment` | `window_hours` (24) | `score`, `mentions` (see [Sentiment recorder](#sentiment-recorder)) |

- **Series** are `open`, `high`, `low`, `close`, `volume`, an indicator name, or `name.output` for indicators with several outputs (`bb.lower`). Numbers work anywhere a series does.
- **Arithmetic:** `{"add": [a, b]}`, `sub`, `mul`, `div`, and `{"shift": [series, bars]}` for an earlier bar's value.
- **Rules:** `[a, op, b]` with `op` one of `>`, `>=`, `<`, `<=`, `==`, `!=`; `{"all": [rules]}`, `{"any": [rules]}`, `{"not": rule}`; `{"crosses_above": [a, b]}` and `{"crosses_below": [a, b]}` compare the last two bars. A comparison with a missing value is false.
- **Identity:** the strategy's hash is the SHA-256 of its canonical JSON, so any edit makes a new candidate in the [research pipeline](#research-pipeline).

Errors name the exact place, e.g. `indicators.fast.fn: unknown fn 'emaa'` or `lookback: 21 is smaller than the required warm-up of 22 bars`. `backtest` reports them before running anything.

## Research pipeline

The pipeline decides which strategies trade, with evidence for every step. Research uses stored candles and quotes only: backfills, backtests and evaluations make no Solana RPC calls, and every result carries `"rpc": {"calls": 0, "cu": 0}` to prove it.

1. **Backfill** candles with `backfill`; each series is checked for gaps, duplicates and bad bars.
2. **Measure costs** with `measure-costs`: Jupiter quotes at `costs.probe_sizes_usd` in both directions give a per-size cost curve (times `costs.safety_multiplier`). Tokens without quotes use `costs.default_cost_bps` and are flagged `COSTS_DEFAULT`.
3. **Backtest** with `backtest <strategy> <symbol>`. The newest `gates.holdout_days` are held out. The rest runs as one backtest after costs, sliced into `gates.folds` folds, against buy-and-hold and cash. The verdict is `PASS`, `FAIL` or `INSUFFICIENT_EVIDENCE`, with reasons.

   Every simulation (backtest, held-out period, paper book) holds the capital the live bot risks, `risk.max_live_capital`, and each entry spends `risk.max_trade_size` times the strategy's `position_size`. Returns and drawdowns are percentages of that capital, the same base the kill switch uses. Buy-and-hold holds the same position size on the same capital.
4. **Promote** a `PASS` with `promote <strategy> <symbol>`: the held-out period runs from flat. A pass starts a paper book; a failure (a drawdown breach, or a loss over enough closed trades) rejects the hash for good. Too few closed trades to judge is `holdout_insufficient`: the hash stays eligible, its held-out period keeps the start of the first attempt and grows as bars arrive, so `promote` again later.
5. **Paper** books trade the strategy on live candles with modeled costs. The bot moves a paper candidate to live by itself once its evidence qualifies, and retires it after `promotion.max_paper_days` if it never does.
6. **Live** strategies are added to `config.json` (atomically, with a backup in `data/config_backups/`) and trade from the spec that was tested.

| Gate | Passes when | Config |
| --- | --- | --- |
| Evidence | At least 30 trades and under 2% missing bars; otherwise `INSUFFICIENT_EVIDENCE` | `gates.min_trades`, `gates.max_gap_fraction` |
| Drawdown | Max drawdown ≤ 15% of the live capital | `risk.max_drawdown_pct` |
| Sharpe | Above 0 and above buy-and-hold | |
| Folds | At least 3 of 4 folds profitable | `gates.folds`, `gates.min_positive_folds` |
| Deflated Sharpe | ≥ 0.95: the Sharpe ratio still beats the best of the variants tried on that symbol and timeframe | `gates.min_deflated_sharpe` |
| Held out | Drawdown ≤ 15%, then enough closed trades to judge (8, or fewer for a slow strategy: its selection-period trade rate over `gates.holdout_days`, at least 1), then a profit | `gates.holdout_min_trades` |
| Paper → live | 14 days, 90% of bars observed, 2 round trips, return within max(1 point, 25%) of a backtest replay, drawdown ≤ 15%, a free live slot, the token not traded by another strategy | `promotion.*` |

Every strategy variant counts: the deflated Sharpe ratio corrects for the number of distinct hashes tested on a symbol and timeframe, so trying many variants raises the bar.

**Demotion.** A live strategy is flattened and removed from `config.json` when its own drawdown (on `risk.max_live_capital`) reaches `promotion.demote_drawdown_pct` (10%), when its last `demote_window_days` (60) lose money over at least `demote_min_trades` (10) trades, when its fills cost more than `max_cost_divergence_bps` (50) above the model, or on `demote`.

**Kill switch.** When the bot's capital (`risk.max_live_capital` plus realized and open P&L) falls `risk.max_drawdown_pct` below its high, every position is sold and entries pause until a human runs `resume`.

**Approval.** With `promotion.require_approval` set to `true`, paper → live also waits for a human `approve`.

## Agent CLI

`uv run -m sol_trade.cli <command>` prints one JSON object on stdout and nothing else: `{"ok": true, "command": ..., ...}` on success, `{"ok": false, "command": ..., "error": "<code>", "message": ..., "details": [...]}` otherwise. Exit codes are `0` ok, `1` refused or failed, and `2` usage error. Logs go to the log files. The CLI never signs a transaction.

| Command | Output |
| --- | --- |
| `backfill [--symbols A,B] [--timeframes 1h,15m] [--days N]` | Per series: bars added, range, gaps, duplicates, bad bars |
| `backtest <strategy> <symbol>` | `verdict`, `reasons`, metrics, folds, benchmarks, gates, deflated Sharpe, trial number and count, flags, `rpc` |
| `candidates [--symbol S]` | The pipeline, trials, and whether trading is paused |
| `promote <strategy> <symbol>` | The transition, or a refusal code (`not_passed`, `costs_unmeasured`, `paper_full`, `not_ready`, `holdout_insufficient`, `holdout_failed`, ...) |
| `demote <strategy> <symbol> [--reason R]` | The transition; `demote_pending` while the bot still has to sell |
| `measure-costs [--symbols A,B]` | The cost model per token, from fresh Jupiter quotes |
| `status [--healthcheck]` | Bot mode and heartbeat, positions and P&L after costs, paper books, paused, RPC usage today, recorder health, warnings; `--healthcheck` exits 1 without a fresh heartbeat |
| `report` | `{"text": ...}`: a Discord-ready summary of at most 1,900 characters |
| `approve <strategy> <symbol>` | Lets paper → live proceed when `require_approval` is on (interactive terminal only) |
| `reopen <strategy> <symbol>` | Re-opens a hash rejected by the old fixed held-out trade minimum, once, after replaying its stored held-out period under the current rule (interactive terminal only) |
| `resume` | Clears the kill switch (interactive terminal only) |

## FAQ

**What happens if I stop the bot while I'm holding a position?**
Your open position is saved in `data/research.db`. On restart, the bot replays the bars it missed and resumes managing its stop-loss and take-profit. Positions from older versions (`data/{TOKEN}_data.csv`) are imported once and the files renamed to `.migrated`.

**Do I need a Jupiter API key?**
No. It is optional and only sent if set — the default `swap/v2` endpoint works without one.

**Can I trade more than one token?**
Yes. Add each token to `secondary_mints` (and its symbol to `secondary_mint_symbols`) and SolTrade trades them all in the same loop.

**Where is my private key stored?**
Only in the `.env` file on your machine. The bot loads and signs locally — it is never sent to any server, and `.env` is git-ignored.

## Glossary

| Term | Meaning |
| --- | --- |
| Primary mint | The token you trade with, usually a stablecoin like USDC |
| Secondary mint | The token you trade for, e.g. SOL |
| Timeframe | The bar size a strategy trades on, e.g. `1h` |
| Price update interval | Seconds between price refreshes |
| Slippage | Difference between expected and executed trade price |
| BPS | Basis points — 100 BPS = 1% |
| Whale | A wallet holding a large amount of a token |
| Candidate | One strategy hash on one token, on its way through the research pipeline |
| Held-out period | The newest history, kept out of every backtest until promotion |
| Paper book | A virtual book that trades a candidate on live candles without sending swaps |
| Deflated Sharpe | The probability that a Sharpe ratio beats luck, given how many variants were tried |

## Development

Tests run offline against a throwaway config and wallet, so they need no `config.json`, `.env` or funds:

```bash
uv sync
uv run pytest
uv run ruff check .
```

See [Contributing](CONTRIBUTING.md) for the pull request checklist and [AGENTS.md](AGENTS.md) for a map of the code.

## License

SolTrade is licensed under the [GNU General Public License v3.0](LICENSE).

## Credits

SolTrade is a hard fork of [noahtheprogrammer/soltrade](https://github.com/noahtheprogrammer/soltrade).
