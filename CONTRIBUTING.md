# Contributing to SolTrade

Bug reports, fixes, new strategies and documentation improvements are welcome. For a larger change, such as a new trading layer, a different swap route or a change to how positions are stored, open an issue first so the approach can be agreed before you write it.

## Reporting issues

Search [existing issues](https://github.com/etcherfx/sol-trade/issues) first, then open a [bug report or feature request](https://github.com/etcherfx/sol-trade/issues/new/choose). Include reproducible steps, expected and actual behavior, and the release or commit you ran.

Check logs and screenshots before you share them. Never post your private key, `.env`, Jupiter API key or full `config.json`. Remove wallet addresses you don't want public.

Report security problems privately as described in [SECURITY.md](SECURITY.md), not in a public issue.

## Development

Setup and checks are in the README's [Development](README.md#-development) section, and both checks must pass before you open a pull request. [AGENTS.md](AGENTS.md) maps the code and its invariants. To try a change by hand, use the paper-trading mode from [Getting started](README.md#-getting-started).

## Pull requests

The pull request template asks for the why, what you verified and any screenshots. Beyond that:

- Keep each pull request to one change.
- Add or update tests for trading, strategy, config or wallet logic.
- New strategies go in `strategies/{name}_strategy.py` and follow the README's [Custom strategies](README.md#-custom-strategies) rules.

Commits follow [Conventional Commits](https://www.conventionalcommits.org/) (`type(scope): description`) and carry a [DCO](https://developercertificate.org/) sign-off, added with `git commit -s`.

## Licensing

SolTrade is licensed under the [GNU General Public License v3.0](LICENSE). By contributing, you agree that your contributions are licensed under the same terms.
