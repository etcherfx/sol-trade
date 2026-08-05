import logging
import sys
import os
from logging import StreamHandler
from logging.handlers import RotatingFileHandler


# Custom formatter to support colors in console
class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;21m"
    green = "\x1b[32;21m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s       %(message)s"

    FORMATS = {
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


def silence_console_logging():
    """Raise console handler levels so legacy log lines stay out of the UI."""
    for logger in (log_general, log_transaction):
        for handler in logger.handlers:
            if isinstance(handler, AutoFlushStreamHandler):
                handler.setLevel(logging.CRITICAL + 1)
