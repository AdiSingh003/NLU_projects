"""
Centralised logging configuration for the name-generation project.

Every module imports get_logger() from here so all logs are:
  • printed to the console (coloured, human-readable)
  • written to  logs/<run_name>.log  (plain text, timestamped)
"""

import logging
import os
import sys
from datetime import datetime

# ─────────────────────────────────────────────
#  ANSI colour codes for console output
# ─────────────────────────────────────────────

class _ColourFormatter(logging.Formatter):
    """Adds colour to log level labels in terminal output."""

    GREY    = "\x1b[38;5;240m"
    CYAN    = "\x1b[36m"
    YELLOW  = "\x1b[33m"
    RED     = "\x1b[31m"
    BOLD_RED= "\x1b[1;31m"
    RESET   = "\x1b[0m"

    LEVEL_COLOURS = {
        logging.DEBUG:    GREY,
        logging.INFO:     CYAN,
        logging.WARNING:  YELLOW,
        logging.ERROR:    RED,
        logging.CRITICAL: BOLD_RED,
    }

    FMT = "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s"
    DATEFMT = "%H:%M:%S"

    def format(self, record: logging.LogRecord) -> str:
        colour = self.LEVEL_COLOURS.get(record.levelno, self.RESET)
        formatter = logging.Formatter(
            f"{colour}{self.FMT}{self.RESET}",
            datefmt=self.DATEFMT,
        )
        return formatter.format(record)


# ─────────────────────────────────────────────
#  Plain formatter for log files
# ─────────────────────────────────────────────

_FILE_FMT = logging.Formatter(
    fmt     = "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
)


# ─────────────────────────────────────────────
#  Global state: one FileHandler shared by all loggers
# ─────────────────────────────────────────────

_file_handler: logging.FileHandler | None = None
_log_filepath: str | None = None


def init_logging(
    run_name: str  = "run",
    log_dir:  str  = "logs",
    level:    int  = logging.DEBUG,
) -> str:
    """
    Call once at the start of a script to set the log file destination.

    Parameters
    ----------
    run_name : label embedded in the log filename (e.g. "train_blstm")
    log_dir  : directory to write log files into
    level    : minimum log level captured (DEBUG by default)

    Returns
    -------
    Path to the log file created.
    """
    global _file_handler, _log_filepath

    os.makedirs(log_dir, exist_ok=True)
    timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    _log_filepath = os.path.join(log_dir, f"{run_name}_{timestamp}.log")

    _file_handler = logging.FileHandler(_log_filepath, encoding="utf-8")
    _file_handler.setLevel(level)
    _file_handler.setFormatter(_FILE_FMT)

    # Apply to root so every module picks it up
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(_file_handler)

    return _log_filepath


def get_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    """
    Return a named logger with:
      • coloured StreamHandler  → stdout
      • FileHandler             → shared log file (if init_logging was called)

    Safe to call multiple times with the same name (idempotent).
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    # ── Console handler ────────────────────────
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(_ColourFormatter())
    logger.addHandler(ch)

    # ── File handler (if init_logging was called) ──
    if _file_handler is not None:
        logger.addHandler(_file_handler)

    logger.propagate = False   # avoid double-printing via root logger
    return logger


def get_log_path() -> str | None:
    """Return the current log file path, or None if init_logging was not called."""
    return _log_filepath
