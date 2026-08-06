"""Tests for the market data source."""

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
