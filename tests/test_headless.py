"""Headless mode: JSON-lines logs, the log folder override, and a clean stop on a signal."""

import json
import logging
import signal
import threading

import main
from sol_trade import heartbeat, log, trading


def test_json_lines_are_valid_json():
    record = logging.LogRecord("transaction_logger", logging.WARNING, __file__, 1, "sold %s", ("SOL",), None)
    entry = json.loads(log.JsonLineFormatter().format(record))
    assert entry["level"] == "warning"
    assert entry["logger"] == "transaction"
    assert entry["msg"] == "sold SOL"


def test_log_dir_override(tmp_path, monkeypatch):
    monkeypatch.setattr(log, "_file_logging_enabled", False)
    monkeypatch.setenv("SOLTRADE_LOG_DIR", str(tmp_path / "custom"))
    handlers = (list(log.log_general.handlers), list(log.log_transaction.handlers))
    try:
        log.setup_file_logging()
        assert (tmp_path / "custom" / "general.log").exists()
    finally:
        for logger, before in zip((log.log_general, log.log_transaction), handlers, strict=True):
            for handler in logger.handlers[len(before):]:
                handler.close()
            logger.handlers[:] = before


def test_headless_stops_cleanly_on_sigterm(monkeypatch):
    started, stopped = [], []
    monkeypatch.setattr(trading, "start_trading", lambda state, dry_run: started.append(dry_run))
    monkeypatch.setattr(trading, "stop_trading", lambda: stopped.append(True))
    monkeypatch.setattr(trading, "is_running", lambda: True)
    handlers = {}
    monkeypatch.setattr(main.signal, "signal", lambda sig, handler: handlers.__setitem__(sig, handler))
    timer = threading.Timer(0.2, lambda: handlers[signal.SIGTERM](signal.SIGTERM, None))
    timer.start()

    assert main.run_headless(dry_run=True) == 0

    assert started == [True] and stopped == [True]
    assert set(handlers) == {signal.SIGTERM, signal.SIGINT}


def test_headless_fails_when_the_trading_thread_dies(monkeypatch):
    monkeypatch.setattr(trading, "start_trading", lambda state, dry_run: None)
    monkeypatch.setattr(trading, "stop_trading", lambda: None)
    monkeypatch.setattr(trading, "is_running", lambda: False)
    monkeypatch.setattr(main.signal, "signal", lambda *_: None)
    assert main.run_headless(dry_run=True) == 1


def test_heartbeat_freshness(monkeypatch):
    heartbeat.write("live", 60, tokens=["SOL"])
    beat = heartbeat.read()
    assert beat is not None and beat["fresh"] and beat["tokens"] == ["SOL"]
    stale = heartbeat.read(now=beat["at"] + 3 * 60 + 61)
    assert stale is not None and not stale["fresh"]
