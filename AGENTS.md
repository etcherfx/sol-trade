# SolTrade

SolTrade is a Python trading bot for Solana. A background thread fetches candles and prices, runs a pluggable strategy, and swaps through the Jupiter Swap API, while a full-screen prompt-toolkit UI shows the dashboard and logs. It trades real money on mainnet unless started with `--dry-run`.

## Setup and verification

Run from the repo root, cheapest first:

```bash
uv sync                # Python 3.13 per .python-version; requires-python >= 3.11
uv run ruff check .    # lint
uv run pytest          # offline, about a second
```

- Tests are hermetic: `tests/conftest.py` installs a throwaway config and keypair, points the RPC at a dead local address, clears `SOLTRADE_*` variables and runs from a temporary directory. They never read `config.json` or `.env` and need no network or funds.
- `ruff format` is not enforced; many files are not formatted with it, so don't reformat untouched code.
- There is no CI for pushes or pull requests. Run both checks yourself before calling a change done. `.github/workflows/release.yml` runs them again on a version tag, before it publishes the release.
- Manual smoke test: `uv run main.py --dry-run` (paper trading). It still needs a valid `SOLTRADE_PRIVATE_KEY` in `.env` and live RPC/exchange access.

## Code map

| Path | What lives there |
|---|---|
| `main.py` | Entry point: parses `--dry-run`, enables file logging, validates config, starts the trading thread and the UI |
| `sol_trade/config.py` | `Config` singleton (`config()`): `config.json` + `.env`, defaults, validation, hot-reload |
| `sol_trade/trading.py` | Trading loop, buy/sell handling, paper-trading balances, P&L baseline, token-change guard |
| `sol_trade/strategy.py` | Strategy loader, `ema`/`sma`/`rsi` indicators, stop-loss/take-profit/trailing-stop columns |
| `sol_trade/transactions.py` | Jupiter order, local signing and execution |
| `sol_trade/wallet.py` | Balances, token accounts, the dynamic SOL reserve (`minimum_sol_needed`) |
| `sol_trade/data_source.py` | Candles via ccxt, cached in the SQLite store at `candles_path` |
| `sol_trade/confluence.py`, `whale_tracker.py`, `market_regime.py`, `sentiment.py` | Optional layers that size or block trades |
| `sol_trade/whale_discovery.py` | CLI: `uv run -m sol_trade.whale_discovery TOKEN_MINT [LIMIT]` |
| `sol_trade/ui.py` | Terminal UI and its thread-safe `UIState` |
| `sol_trade/log.py`, `utils.py` | Loggers; shared async loop (`run_async`), rate-limit retry, JSON helpers |
| `strategies/` | `BaseStrategy` and `DefaultStrategy`; users drop their own `{name}_strategy.py` here |
| `tests/` | pytest suite; `conftest.py` makes it hermetic |
| `packaging/` | `build.sh VERSION` writes the release archives to `dist/` (needs git, tar and zip; the release workflow runs it); `run.sh`, `run.ps1` and `run.cmd` are the launchers shipped in them. A new runtime file or folder must be added to the list in `build.sh` |

## Constraints and conventions

- **Secrets:** `config.json`, `.env`, `data/` and `logs/` are git-ignored. Never commit, print or log a private key, and never read the real `config.json`/`.env` in tests.
- **Config access:** read settings through `config()` at call time, not at import, so hot-reload applies. Keys in `_STRUCTURAL_KEYS` (primary mint, RPC, exchange, credentials) are pinned on reload; the token set is hot-reloadable.
- **Token changes:** removing a token with an open position is rolled back by `_enforce_token_change_guard`. A token-set change recaptures the P&L baseline. Keep both when touching the loop.
- **Strategy names:** `strategies/{name}_strategy.py` with class `{PascalCase}Strategy` (`mean_reversion` → `MeanReversionStrategy`); the legacy `Mean_reversionStrategy` spelling is still accepted. A strategy must not check `config().strategy`, because per-token overrides load it under another global strategy.
- **Errors:** a missing or invalid key raises `ValueError` from `Config.keypair`; `main.py` reports it. Library code raises or logs; only entry points call `sys.exit`.
- **Logging:** importing `sol_trade` writes no files. Entry points call `setup_file_logging()`, which writes to the project-root `logs/`. Console output is muted while the UI owns the terminal.
- **Paths:** `data/` paths (candle store, `data/{SYMBOL}_data.csv` positions, whale/regime/sentiment JSON) are relative to the working directory, so the bot is run from the repo root.
- **Commits:** Conventional Commits (`type(scope): description`) with a DCO sign-off (`git commit -s`).

## Gotchas

- `main.py` without `--dry-run` signs and sends real swaps from the configured wallet. Never run it live to "check" a change.
- Test stubs for `run_async` receive a real coroutine. Close it (`coro.close()`) or the suite warns about a coroutine that was never awaited.
- Tests must not depend on the developer's machine: no reliance on the local token list, local strategy files or an existing `logs/`/`data/`. Pin what a test needs with `monkeypatch` or a fixture.
- `minimum_sol_needed()` caches its result in `wallet._sol_reserve_cache`; reset it in tests that call it.
- Jupiter and RPC calls can be rate-limited. Wrap new RPC helpers in `handle_rate_limiting`, which returns `None` after the retries run out, so callers must handle `None`.

## Topic docs

There is no `docs/` folder. The README is the user reference: read its Configuration, Multi-token portfolios and Custom strategies sections before changing settings, the token-set rules or the strategy interface, and update them when behavior changes.
