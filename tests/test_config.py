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


def test_reload_normalizes_weights(tmp_path, monkeypatch):
    import json

    from sol_trade.config import _file_mtime, config

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

    from sol_trade.config import _file_mtime, config

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

    from sol_trade.config import _file_mtime, config

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
