"""The agent CLI contract: one JSON object on stdout, exit codes 0/1/2."""

import io
import json

import pytest

from sol_trade import cli, heartbeat, history, promotion
from sol_trade.config import config
from tests.test_evaluation import SPEC, _market


@pytest.fixture(autouse=True)
def _no_log_files(monkeypatch):
    monkeypatch.setattr(cli, "setup_file_logging", lambda: None)


def _run(*argv):
    lines = []
    code = cli.run(list(argv), out=lines.append)
    assert len(lines) == 1
    return code, json.loads(lines[0])


@pytest.fixture
def stored_strategy(tmp_path):
    folder = tmp_path / "strategies"
    folder.mkdir()
    (folder / "swing.json").write_text(json.dumps(SPEC))
    frame = _market()
    pair = history.pair_for("TKN")
    history._store(pair, "1h", [history.Bar(*row) for row in frame.itertuples(index=False)])
    return "swing"


def test_usage_errors_exit_2_with_json():
    code, payload = _run("bogus")
    assert code == 2
    assert payload["ok"] is False and payload["error"] == "usage"
    code, payload = _run("backtest", "only_one_arg")
    assert code == 2


def test_help_is_json_too():
    code, payload = _run("--help")
    assert code == 2
    assert "backtest" in payload["details"][0]


def test_status_without_a_running_bot():
    code, payload = _run("status")
    assert code == 0
    assert payload["ok"] and payload["mode"] is None
    assert "bot_not_running: no heartbeat" in payload["warnings"]


def test_healthcheck_follows_the_heartbeat():
    assert _run("status", "--healthcheck")[0] == 1
    heartbeat.write("dry_run", 60)
    code, payload = _run("status", "--healthcheck")
    assert code == 0 and payload["heartbeat"]["mode"] == "dry_run"


def test_backtest_reports_a_verdict_with_zero_rpc(stored_strategy, monkeypatch):
    monkeypatch.setattr(config(), "confluence_enabled", False)
    code, payload = _run("backtest", stored_strategy, "TKN")
    assert code == 0, payload
    assert payload["verdict"] in ("PASS", "FAIL", "INSUFFICIENT_EVIDENCE")
    assert payload["rpc"] == {"calls": 0, "cu": 0}
    assert payload["trial"] == 1 and payload["n_trials"] == 1
    assert "COSTS_DEFAULT" in payload["flags"]
    assert payload["candidate"]["state"] in ("candidate", "backtest_passed")
    assert "simulation" not in payload


def test_backtest_names_spec_errors(tmp_path):
    folder = tmp_path / "strategies"
    folder.mkdir()
    (folder / "broken.json").write_text(json.dumps({**SPEC, "name": "broken", "lookback": 1}))
    code, payload = _run("backtest", "broken", "TKN")
    assert code == 1
    assert payload["error"] == "invalid_strategy"
    assert payload["details"][0].startswith("lookback:")


def test_promote_refusal_has_a_code(stored_strategy):
    code, payload = _run("promote", stored_strategy, "TKN")
    assert code == 1
    assert payload["error"] == "not_passed"


def test_human_commands_need_a_terminal(stored_strategy, monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO())
    for argv in (("approve", stored_strategy, "TKN"), ("reopen", stored_strategy, "TKN"), ("resume",)):
        code, payload = _run(*argv)
        assert code == 1 and payload["error"] == "tty_required"


def test_report_fits_discord(stored_strategy, monkeypatch):
    _run("backtest", stored_strategy, "TKN")
    candidate = promotion.candidates()[0]
    for i in range(80):
        promotion._transition(candidate, "candidate", f"long reason {i} " + "x" * 200, {})
    promotion._set_setting("paused", {"at": "now", "drawdown_pct": 16.0})
    code, payload = _run("report")
    assert code == 0
    assert len(payload["text"]) <= cli.REPORT_LIMIT
    assert "KILL SWITCH" in payload["text"]


def test_candidates_lists_trials(stored_strategy):
    _run("backtest", stored_strategy, "TKN")
    code, payload = _run("candidates", "--symbol", "TKN")
    assert code == 0
    assert payload["candidates"][0]["strategy"] == "swing"
    assert len(payload["trials"]) == 1


def test_unexpected_errors_still_print_json(monkeypatch):
    def boom(_args):
        raise RuntimeError("kaput")

    monkeypatch.setattr(cli, "cmd_status", boom)
    code, payload = _run("status")
    assert code == 1 and payload["error"] == "internal" and "kaput" in payload["message"]
