import logging
import os
import sys
import threading
from collections import deque
from logging import StreamHandler
from logging.handlers import RotatingFileHandler
from typing import ClassVar


# Custom formatter to support colors in console
class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;21m"
    green = "\x1b[32;21m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    fmt = "%(asctime)s       %(message)s"

    FORMATS: ClassVar[dict[int, str]] = {
        logging.DEBUG: grey + fmt + reset,
        logging.INFO: green + fmt + reset,
        logging.WARNING: yellow + fmt + reset,
        logging.ERROR: red + fmt + reset,
        logging.CRITICAL: bold_red + fmt + reset,
    }

    def format(self, record: logging.LogRecord) -> str:
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


# Project-root logs/, independent of the directory the bot is started from.
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")


def setup_logger(name, level=logging.INFO) -> logging.Logger:
    """Set up a logger with console output; file output is added by setup_file_logging()."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    console_handler = StreamHandler(sys.stdout)
    console_handler.setFormatter(CustomFormatter())
    # Console output is off by default: the full-screen UI owns the terminal.
    # CLI entry points opt back in via enable_console_logging().
    console_handler.setLevel(logging.CRITICAL + 1)
    logger.addHandler(console_handler)
    return logger


log_general = setup_logger("general_logger", level=logging.DEBUG)
log_transaction = setup_logger("transaction_logger", level=logging.DEBUG)

_file_logging_enabled = False


def setup_file_logging(log_dir: str | os.PathLike | None = None) -> None:
    """Write logs to rotating files; entry points call this, importing never does.

    general.log receives both loggers, transaction.log only transactions.
    Repeated calls are no-ops.
    """
    global _file_logging_enabled
    if _file_logging_enabled:
        return
    log_dir = log_dir or LOG_DIR
    os.makedirs(log_dir, exist_ok=True)
    file_formatter = logging.Formatter(
        "%(asctime)s     %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    def _rotating(file_name: str) -> RotatingFileHandler:
        handler = RotatingFileHandler(
            os.path.join(log_dir, file_name), maxBytes=1000000, backupCount=5
        )
        handler.setFormatter(file_formatter)
        return handler

    # One shared handler: two handles on general.log break rotation on Windows.
    general_handler = _rotating("general.log")
    log_general.addHandler(general_handler)
    log_transaction.addHandler(general_handler)
    log_transaction.addHandler(_rotating("transaction.log"))
    _file_logging_enabled = True


class MemoryLogHandler(logging.Handler):
    """Captures recent log records in memory for the terminal UI."""

    def __init__(self, capacity: int = 1000) -> None:
        super().__init__()
        self._records: deque = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = self.format(record)
            with self._lock:
                self._records.append((record.levelno, record.created, line))
        except Exception:  # noqa: BLE001 - logging handler error protocol
            self.handleError(record)

    def snapshot(self, limit: int | None = None) -> list[tuple[int, float, str]]:
        """Return records as ``(levelno, created, formatted_line)``."""
        with self._lock:
            items = list(self._records)
        return items if limit is None else items[-limit:]


memory_handler = MemoryLogHandler()
memory_handler.setFormatter(
    logging.Formatter("%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
)
log_general.addHandler(memory_handler)
log_transaction.addHandler(memory_handler)


def get_recent_logs(limit: int | None = None) -> list[tuple[int, float, str]]:
    """Return the most recent log records for the terminal UI."""
    return memory_handler.snapshot(limit)


def enable_console_logging() -> None:
    """Restore console output for CLI entry points."""
    for logger in (log_general, log_transaction):
        for handler in logger.handlers:
            if isinstance(handler, StreamHandler) and handler.stream is sys.stdout:
                handler.setLevel(logging.DEBUG)
