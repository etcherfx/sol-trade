# SolTrade

SolTrade is a Python trading bot for Solana. A background thread runs each token's JSON strategy through a closed-bar engine and swaps through the Jupiter Swap API, while a full-screen prompt-toolkit UI shows the dashboard and logs. A research pipeline (backtests, gates, paper books, promotion) decides which strategies trade, driven by an agent through a JSON CLI. It trades real money on mainnet unless started with `--dry-run`.

## Setup and verification

Run from the repo root, cheapest first:

```bash
uv sync                # Python 3.13 per .python-version; requires-python >= 3.11
uv run ruff check .    # lint
uv run pytest          # offline, about 20 seconds
```

- Tests are hermetic: `tests/conftest.py` installs a throwaway config and keypair, points the RPC at a dead local address, clears `SOLTRADE_*` variables and runs from a temporary directory. They never read `config.json` or `.env` and need no network or funds.
- `ruff format` is not enforced; many files are not formatted with it, so don't reformat untouched code.
- CI (`.github/workflows/ci.yml`) runs on every pull request to `main` and every push to `main`. Its `check` job runs the three commands above; its `image` job builds the Dockerfile, runs the container with `--dry-run` against a throwaway config without a key, and waits for Docker to report `healthy`. Both jobs are the merge gate: merging to `main` is the production deploy. Still run both checks yourself before calling a change done. `.github/workflows/release.yml` runs them again on a version tag, before it publishes the release.
- Dependabot PRs auto-merge (`.github/workflows/dependabot-automerge.yml`) after `check` and `image` pass when every update is a semver patch or minor bump. Major bumps, security ones included, and anything it cannot classify get the `needs-review` label instead. Because those merges trigger no push run, CI also re-checks `main` daily.
- Manual smoke test: `uv run main.py --dry-run` (paper trading against a virtual `risk.max_live_capital`; no key needed, but live exchange and Jupiter price access). Point it at a scratch config with `SOLTRADE_CONFIG_PATH`/`SOLTRADE_DOTENV_PATH` rather than the repo's own `config.json`.

## Code map

| Path | What lives there |
|---|---|
| `main.py` | Entry point: parses `--dry-run` and `--headless`, enables file logging, validates config, starts the trading thread and the UI (or waits for SIGTERM/SIGINT headless) |
| `Dockerfile`, `compose.yaml` | Headless container; paper trades unless the command drops `--dry-run` |
| `sol_trade/config.py` | `Config` singleton (`config()`): `config.json` + `.env`, defaults, validation, hot-reload |
| `sol_trade/trading.py` | Trading loop: per token `run_symbol` (missed-bar replay, forming bar), `LiveExecutor`, paper books, governor call, heartbeat, P&L baseline, token-change guard |
| `sol_trade/engine.py` | The one decision core: `Book` (`on_open`/`on_low`/`on_close`), fills, stops, trailing stop, `size_order`, dust |
| `sol_trade/spec.py`, `indicators.py`, `builtin_strategies/` | JSON strategy grammar, validation, canonical hash, evaluation on `(L, N)` windows; the indicators; the built-in `default` |
| `sol_trade/history.py` | Candle backfill, incremental updates, validation report, windows for the live loop |
| `sol_trade/costs.py`, `jupiter.py` | Jupiter quote probes and the cost model; prices, quotes and token metadata over HTTP |
| `sol_trade/backtest.py`, `evaluation.py` | Bar-by-bar backtests through the engine; holdout split, folds, trials, deflated Sharpe, gates |
| `sol_trade/promotion.py`, `ledger.py` | Pipeline state machine, paper readiness, governor, kill switch, atomic `config.json` edits; books, positions, fills, trades in `data/research.db` |
| `sol_trade/rpc.py` | The CU-budgeted Solana RPC client every call goes through, and `data/rpc_usage.json` |
| `sol_trade/cli.py`, `heartbeat.py` | Agent CLI (one JSON envelope per command); the bot's `data/status.json` heartbeat |
| `sol_trade/transactions.py` | Jupiter order, local signing and execution |
| `sol_trade/wallet.py` | Balances, token accounts, the dynamic SOL reserve (`minimum_sol_needed`) |
| `sol_trade/data_source.py` | Candles via ccxt, cached in the SQLite store at `candles_path` |
| `sol_trade/sentiment.py` | Reddit RSS recorder (`data/sentiment.db`), the NDJSON archive import, coverage intervals, point-in-time `features`, and the opt-in breaker |
| `sol_trade/fear_greed.py` | Daily Fear & Greed index in `data/sentiment.db`: backfill, live refresh, the `fear_greed` indicator's values (day D known from D+1 00:00 UTC) |
| `sol_trade/confluence.py`, `whale_tracker.py`, `market_regime.py` | Optional layers that size or block trades of hand-configured tokens |
| `sol_trade/whale_discovery.py` | CLI: `uv run -m sol_trade.whale_discovery TOKEN_MINT [LIMIT]` |
| `sol_trade/ui.py` | Terminal UI and its thread-safe `UIState` |
| `sol_trade/log.py`, `utils.py` | Loggers; shared async loop (`run_async`), rate-limit retry, JSON helpers |
| `strategies/` | Empty in the repo (git-ignored but for `.gitkeep`); users drop `{name}.json` here |
| `tests/` | pytest suite; `conftest.py` makes it hermetic, `fakes.py` has synthetic tick paths and a ccxt-like exchange |
| `packaging/` | `build.sh VERSION` writes the release archives to `dist/` (needs git, tar and zip; the release workflow runs it); `run.sh`, `run.ps1` and `run.cmd` are the launchers shipped in them. A new runtime file or folder must be added to the list in `build.sh` |

## Constraints and conventions

- **Language:** American English in code, comments, docs and user-facing text.
- **Secrets:** they come from environment variables and `.env`; never add a secret field to `config.json`. `config.json`, `.env`, `data/` and `logs/` are git-ignored. Never read, print, log or commit `.env`, the real `config.json` or a private key, in tests or otherwise, and add every new variable to `.env.example` in the same change.
- **Config access:** read settings through `config()` at call time, not at import, so hot-reload applies. Keys in `_STRUCTURAL_KEYS` (primary mint, RPC, exchange, credentials) are pinned on reload; the token set is hot-reloadable.
- **Token changes:** removing a token with an open position is rolled back by `_enforce_token_change_guard`. A token-set change recaptures the P&L baseline. Keep both when touching the loop.
- **One engine:** live, dry run, paper books and backtests all advance `engine.Book` with the same three steps. Change decision logic only in `engine.py`; `tests/test_parity.py` and the paper-replay test in `tests/test_promotion.py` must stay exact.
- **Research makes no RPC calls:** backfill, costs, backtest and evaluation use exchange and Jupiter HTTP only, and results report `rpc: {calls, cu}`. Any Solana RPC goes through `config().client` (the `rpc` limiter).
- **CLI contract:** every command prints exactly one JSON object and exits 0, 1 or 2 (README, Agent CLI). Refusals raise `promotion.Refused`/`CliError` with a stable code; `approve`, `reopen` and `resume` require a TTY.
- **No market literals:** tracked code names no token, mint or pair beyond the native SOL mint and SPL program ids; markets come from `config.json` (the sample may name them).
- **Strategies:** `strategies/{name}.json`, `name` equal to the file name. A promoted strategy trades from the spec stored with its candidate, and confluence, whale and sentiment filters skip it.
- **Errors:** a missing or invalid key raises `ValueError` from `Config.keypair`; `main.py` reports it. Library code raises or logs; only entry points call `sys.exit`.
- **Logging:** importing `sol_trade` writes no files. Entry points call `setup_file_logging()`, which writes to `SOLTRADE_LOG_DIR` or the project-root `logs/`. Console output is muted while the UI owns the terminal; `--headless` logs JSON lines to stdout instead.
- **Paths:** `data/` paths (candle store, `research.db`, `sentiment.db`, costs, heartbeat, whale/regime JSON) are relative to the working directory, so the bot and the CLI run from the runtime folder.
- **Commits:** Conventional Commits (`type(scope): description`) with a DCO sign-off (`git commit -s`).

## Gotchas

- `main.py` without `--dry-run` signs and sends real swaps from the configured wallet. Never run it live to "check" a change.
- Test stubs for `run_async` receive a real coroutine. Close it (`coro.close()`) or the suite warns about a coroutine that was never awaited.
- Tests must not depend on the developer's machine: no reliance on the local token list, local strategy files or an existing `logs/`/`data/`. Pin what a test needs with `monkeypatch` or a fixture. Every test runs in its own `tmp_path` cwd; tests that edit `config.json` copy it and restore the token-set attributes (see the `cfg` fixture in `tests/test_promotion.py`).
- Reddit refuses anonymous `.json` requests (403); the recorder uses the RSS feeds, which allow about one request a minute.
- `minimum_sol_needed()` caches its result in `wallet._sol_reserve_cache`; reset it in tests that call it.
- Jupiter and RPC calls can be rate-limited. Wrap new RPC helpers in `handle_rate_limiting`, which returns `None` after the retries run out, so callers must handle `None`.

## Topic docs

There is no `docs/` folder. The README is the user reference: read its Configuration, Multi-token portfolios, Custom strategies (the JSON grammar), Research pipeline and Agent CLI sections before changing settings, the token-set rules, the strategy grammar, the gates or a CLI command, and update them when behavior changes.
