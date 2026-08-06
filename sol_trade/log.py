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
    format = "%(asctime)s       %(message)s"

    FORMATS: ClassVar[dict[int, str]] = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: green + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record) -> str:
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


class AutoFlushStreamHandler(StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()


os.makedirs("logs", exist_ok=True)


def setup_logger(
    name, log_file, level=logging.INFO, add_to_general=False
) -> logging.Logger:
    """Function to set up a logger with rotating file handler and console output."""
    file_formatter = logging.Formatter(
        "%(asctime)s     %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = RotatingFileHandler(
        f"logs/{log_file}", maxBytes=1000000, backupCount=5
    )
    file_handler.setFormatter(file_formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    console_handler = AutoFlushStreamHandler(sys.stdout)
    console_handler.setFormatter(CustomFormatter())
    logger.addHandler(console_handler)

    if add_to_general:
        general_handler = RotatingFileHandler(
            "logs/general.log", maxBytes=1000000, backupCount=5
        )
        general_handler.setFormatter(file_formatter)
        logger.addHandler(general_handler)

    return logger


log_general = setup_logger("general_logger", "general.log", level=logging.DEBUG)
log_transaction = setup_logger(
    "transaction_logger", "transaction.log", add_to_general=True, level=logging.DEBUG
)


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


def silence_console_logging():
    """Raise console handler levels so legacy log lines stay out of the UI."""
    for logger in (log_general, log_transaction):
        for handler in logger.handlers:
            if isinstance(handler, AutoFlushStreamHandler):
                handler.setLevel(logging.CRITICAL + 1)
