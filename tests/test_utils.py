"""Tests for shared utilities."""

from sol_trade.utils import load_json_data, save_json_data


def test_json_round_trip(tmp_path):
    path = str(tmp_path / "sub" / "data.json")
    save_json_data(path, {"a": 1, "nested": [1, 2]})
    assert load_json_data(path, {}) == {"a": 1, "nested": [1, 2]}


def test_json_missing_file_returns_default():
    assert load_json_data("/nonexistent/path/data.json", "D") == "D"


def test_json_corrupt_file_returns_default(tmp_path):
    path = tmp_path / "data.json"
    path.write_text("{ not valid json")
    assert load_json_data(str(path), []) == []
