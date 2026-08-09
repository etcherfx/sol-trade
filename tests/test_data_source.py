"""Tests for the market data source."""

import pytest

from sol_trade import data_source
from sol_trade.data_source import _candle_dict


def test_candle_dict_shape():
    candle = _candle_dict(123, 1.0, 2.0, 0.5, 1.5, 100.0)
    assert candle == {
        "open": 1.0,
        "high": 2.0,
        "low": 0.5,
        "close": 1.5,
        "volume": 100.0,
        "totalvolume": 100.0,  # defaults to volume
        "time": 123,
    }


def test_candle_dict_total_volume_override():
    candle = _candle_dict(1, 1.0, 2.0, 0.0, 1.5, 100.0, total_volume=999.0)
    assert candle["totalvolume"] == 999.0


def test_get_exchange_cached_per_id():
    mexc_a = data_source.get_exchange("mexc")
    mexc_b = data_source.get_exchange("mexc")
    okx = data_source.get_exchange("okx")
    assert mexc_a is mexc_b  # same id -> same client instance
    assert mexc_a is not okx  # different ids -> distinct clients


def test_get_exchange_unknown_id_raises():
    with pytest.raises(ValueError):
        data_source.get_exchange("no_such_exchange_xyz")


def test_get_exchange_defaults_to_global_data_exchange(monkeypatch):
    from sol_trade.config import config

    monkeypatch.setattr(config(), "data_exchange", "okx")
    client = data_source.get_exchange()
    assert client.id == "okx"
