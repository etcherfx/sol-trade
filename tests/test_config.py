"""Tests for configuration defaults."""

import pytest

from sol_trade.config import config


def test_config_defaults():
    c = config()
    assert isinstance(c.strategy, str) and c.strategy
    assert c.data_exchange == "okx"
    assert c.sol_mint == "So11111111111111111111111111111111111111112"
    assert c.candles_path == "data/candles.db"


def test_keypair_parses_configured_key():
    c = config()
    if not c.private_key:
        import pytest

        pytest.skip("SOLTRADE_PRIVATE_KEY not set")
    keypair = c.keypair
    assert str(keypair.pubkey()) == str(c.public_address)


def test_reload_picks_up_changes_but_pins_structural(tmp_path, monkeypatch):
    import json

    from sol_trade.config import _file_mtime, config

    c = config()
    snapshot = dict(c.__dict__)
    try:
        path = tmp_path / "config.json"
        path.write_text(
            json.dumps(
                {
                    "strategy": "custom",
                    "max_slippage": 123,
                    "secondary_mints": ["AAAA"],
                    "secondary_mint_symbols": ["AAA"],
                    "secondary_weights": [1.0],
                    "token_strategies": {"AAA": "default"},
                }
            )
        )
        monkeypatch.setattr(c, "path", str(path))
        c._config_mtime = 0.0  # force the reload to fire
        old_key = c.private_key

        c.maybe_reload_config()

        assert c.strategy == "custom"  # lightweight -> reloaded
        assert c.max_slippage == 123
        assert c.secondary_mints == ["AAAA"]  # tokens now hot-reloadable
        assert c.secondary_mint_symbols == ["AAA"]
        assert c.secondary_weights == [1.0]
        assert c.token_strategies == {"AAA": "default"}
        assert c.private_key == old_key  # credential -> pinned
        assert c._config_mtime == _file_mtime(str(path))
    finally:
        c.__dict__.clear()
        c.__dict__.update(snapshot)


def test_reload_token_exchanges(tmp_path, monkeypatch):
    import json

    from sol_trade.config import config

    c = config()
    snapshot = dict(c.__dict__)
    try:
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"token_exchanges": {"SOL": "mexc"}}))
        monkeypatch.setattr(c, "path", str(path))
        c._config_mtime = 0.0

        c.maybe_reload_config()

        assert c.token_exchanges == {"SOL": "mexc"}  # lightweight -> reloaded
    finally:
        c.__dict__.clear()
        c.__dict__.update(snapshot)


def test_reload_normalizes_weights(tmp_path, monkeypatch):
    import json

    from sol_trade.config import config

    c = config()
    snapshot = dict(c.__dict__)
    try:
        path = tmp_path / "config.json"
        path.write_text(
            json.dumps(
                {
                    "secondary_mints": ["A", "B"],
                    "secondary_mint_symbols": ["AAA", "BBB"],
                    "secondary_weights": [2.0, 1.0],
                }
            )
        )
        monkeypatch.setattr(c, "path", str(path))
        c._config_mtime = 0.0

        c.maybe_reload_config()

        assert c.secondary_weights == pytest.approx([2 / 3, 1 / 3])
    finally:
        c.__dict__.clear()
        c.__dict__.update(snapshot)


def test_reload_rejects_mismatched_weights(tmp_path, monkeypatch):
    import json

    from sol_trade.config import config

    c = config()
    snapshot = dict(c.__dict__)
    try:
        path = tmp_path / "config.json"
        path.write_text(
            json.dumps(
                {
                    "secondary_mints": ["A", "B"],
                    "secondary_mint_symbols": ["AAA", "BBB"],
                    "secondary_weights": [0.5],  # wrong length
                }
            )
        )
        monkeypatch.setattr(c, "path", str(path))
        c._config_mtime = 0.0
        old_mints = c.secondary_mints

        c.maybe_reload_config()  # must keep the previous config

        assert c.secondary_mints == old_mints
        assert c.secondary_mint_symbols == snapshot["secondary_mint_symbols"]
    finally:
        c.__dict__.clear()
        c.__dict__.update(snapshot)


def test_reload_rejects_mismatched_mint_symbols(tmp_path, monkeypatch):
    import json

    from sol_trade.config import config

    c = config()
    snapshot = dict(c.__dict__)
    try:
        path = tmp_path / "config.json"
        path.write_text(
            json.dumps(
                {
                    "secondary_mints": ["A", "B"],
                    "secondary_mint_symbols": ["AAA"],  # unequal lengths
                }
            )
        )
        monkeypatch.setattr(c, "path", str(path))
        c._config_mtime = 0.0
        old_mints = c.secondary_mints

        c.maybe_reload_config()

        assert c.secondary_mints == old_mints
    finally:
        c.__dict__.clear()
        c.__dict__.update(snapshot)


def test_reload_noop_when_file_unchanged(tmp_path, monkeypatch):
    import json

    from sol_trade.config import _file_mtime, config

    c = config()
    snapshot = dict(c.__dict__)
    try:
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"strategy": "custom"}))
        monkeypatch.setattr(c, "path", str(path))
        c._config_mtime = _file_mtime(str(path))
        c.strategy = "sentinel"

        c.maybe_reload_config()

        assert c.strategy == "sentinel"  # unchanged file -> no reload
    finally:
        c.__dict__.clear()
        c.__dict__.update(snapshot)


def test_reload_survives_malformed_json(tmp_path, monkeypatch):
    from sol_trade.config import config

    c = config()
    snapshot = dict(c.__dict__)
    try:
        path = tmp_path / "config.json"
        path.write_text("{ not valid json")
        monkeypatch.setattr(c, "path", str(path))
        c._config_mtime = 0.0
        c.strategy = "sentinel"

        c.maybe_reload_config()  # must not raise

        assert c.strategy == "sentinel"  # old config kept
    finally:
        c.__dict__.clear()
        c.__dict__.update(snapshot)


def _write(tmp_path, data):
    import json

    path = tmp_path / "config.json"
    path.write_text(json.dumps(data))
    return str(path)


_PRIMARY = {"primary_mint": "PRIMARY_MINT", "primary_mint_symbol": "QUOTE"}


def test_empty_token_set_is_valid(tmp_path):
    import os

    from sol_trade.config import Config

    c = Config(config_path=_write(tmp_path, _PRIMARY), dotenv_path=os.devnull)
    assert c.secondary_mints == []
    assert c.secondary_mint_symbols == []


def test_missing_primary_mint_is_a_clear_error(tmp_path):
    import os

    from sol_trade.config import Config

    with pytest.raises(ValueError, match="primary_mint"):
        Config(config_path=_write(tmp_path, {}), dotenv_path=os.devnull)


def test_no_market_defaults_in_code(tmp_path):
    import os

    from sol_trade.config import Config

    c = Config(config_path=_write(tmp_path, _PRIMARY), dotenv_path=os.devnull)
    assert c.secondary_mints == []
    assert c.market_regime_pair == ""
    assert c.sentiment["subreddits"] == []
    assert c.whale_tracking_enabled is False
    assert c.confluence_enabled is False


def test_sections_deep_merge_with_defaults(tmp_path):
    import os

    from sol_trade.config import SECTION_DEFAULTS, Config

    path = _write(tmp_path, {**_PRIMARY, "risk": {"max_trade_size": 25}})
    c = Config(config_path=path, dotenv_path=os.devnull)
    assert c.risk["max_trade_size"] == 25
    assert c.risk["max_live_capital"] == SECTION_DEFAULTS["risk"]["max_live_capital"]
    assert c.rpc["max_cu_per_second"] == SECTION_DEFAULTS["rpc"]["max_cu_per_second"]


def test_config_path_env_override(tmp_path, monkeypatch):
    import os

    from sol_trade.config import Config

    path = _write(tmp_path, {**_PRIMARY, "strategy": "from_env"})
    monkeypatch.setenv("SOLTRADE_CONFIG_PATH", path)
    c = Config(dotenv_path=os.devnull)
    assert c.path == path
    assert c.strategy == "from_env"


def test_symbol_and_mint_lookups(tmp_path):
    import os

    from sol_trade.config import Config

    path = _write(
        tmp_path,
        {
            **_PRIMARY,
            "secondary_mints": ["MINT_A"],
            "secondary_mint_symbols": ["AAA"],
            "token_mints": {"BBB": "MINT_B"},
        },
    )
    c = Config(config_path=path, dotenv_path=os.devnull)
    assert c.mint_for_symbol("AAA") == "MINT_A"
    assert c.mint_for_symbol("BBB") == "MINT_B"
    assert c.mint_for_symbol("QUOTE") == "PRIMARY_MINT"
    assert c.mint_for_symbol("ZZZ") is None
    assert c.symbol_for_mint("MINT_B") == "BBB"
    assert c.symbol_for_mint("UNKNOWN") is None


def test_token_exchanges_take_an_exchange_and_quote(tmp_path):
    import os

    from sol_trade.config import Config

    sources = {"JUP": {"exchange": "okx", "quote": "USDT"}, "WIF": {"quote": "USDT"}, "SOL": "coinbase"}
    c = Config(config_path=_write(tmp_path, {**_PRIMARY, "token_exchanges": sources}), dotenv_path=os.devnull)
    assert c.token_exchanges == sources
    for bad in ({"exchange": ""}, {"exchange": "okx", "pair": "JUP/USDT"}, ["okx"], {"quote": 1}):
        with pytest.raises(ValueError, match="token_exchanges.JUP"):
            Config(config_path=_write(tmp_path, {**_PRIMARY, "token_exchanges": {"JUP": bad}}), dotenv_path=os.devnull)


_SECRET_FIELDS = {
    "private_key": ("SOLTRADE_PRIVATE_KEY", "file-private-key-value"),
    "jupiter_api_key": ("SOLTRADE_JUPITER_API_KEY", "file-jupiter-key-value"),
}


def _deprecations(caplog):
    return [r.getMessage() for r in caplog.records if "deprecated" in r.getMessage()]


@pytest.mark.parametrize("field", sorted(_SECRET_FIELDS))
def test_secret_in_config_json_warns_without_the_value(tmp_path, caplog, field):
    import os

    from sol_trade.config import Config

    env_var, value = _SECRET_FIELDS[field]
    with caplog.at_level("WARNING", logger="general_logger"):
        c = Config(config_path=_write(tmp_path, {**_PRIMARY, field: value}), dotenv_path=os.devnull)
    assert getattr(c, field) == value  # still accepted
    messages = _deprecations(caplog)
    assert len(messages) == 1
    assert field in messages[0] and env_var in messages[0]
    assert value not in caplog.text


def test_env_wins_over_deprecated_config_json_secrets(tmp_path, monkeypatch, caplog):
    import os

    from sol_trade.config import Config

    data = {**_PRIMARY, **{field: value for field, (_, value) in _SECRET_FIELDS.items()}}
    for env_var, _ in _SECRET_FIELDS.values():
        monkeypatch.setenv(env_var, f"env-{env_var}")
    with caplog.at_level("WARNING", logger="general_logger"):
        c = Config(config_path=_write(tmp_path, data), dotenv_path=os.devnull)
    for field, (env_var, value) in _SECRET_FIELDS.items():
        assert getattr(c, field) == f"env-{env_var}"
        assert value not in caplog.text
        assert f"env-{env_var}" not in caplog.text
    assert len(_deprecations(caplog)) == 2  # one per field


def test_no_deprecation_warning_without_config_json_secrets(tmp_path, monkeypatch, caplog):
    import os

    from sol_trade.config import Config

    monkeypatch.setenv("SOLTRADE_PRIVATE_KEY", "env-key")
    with caplog.at_level("WARNING", logger="general_logger"):
        c = Config(config_path=_write(tmp_path, _PRIMARY), dotenv_path=os.devnull)
    assert c.private_key == "env-key"
    assert _deprecations(caplog) == []
