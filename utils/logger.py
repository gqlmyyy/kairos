# Trading Bot V3 - utils/logger.py
"""Centralised logging setup.

Design notes
------------
The previous implementation resolved the log file name at *import* time
(``logs/bot_YYYYMMDD.log``). A bot that ran for several days therefore kept
writing into the file named after its start date and never rolled over.

This module uses a ``TimedRotatingFileHandler`` instead: a single pair of
handlers is attached to the root logger, the file handler rotates at midnight,
and a bounded number of daily backups is kept. Rotated files keep the
historical ``bot_YYYYMMDD.log`` naming so existing archives stay recognisable.

Per-module verbosity is driven by config so hot loops (which previously produced
~26% of all log volume at DEBUG level) can be quietened without touching call
sites.
"""

from __future__ import annotations

import logging
import os
from logging.handlers import TimedRotatingFileHandler

LOG_DIR = "logs"
LOG_BASENAME = "bot"

os.makedirs(LOG_DIR, exist_ok=True)

_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

_configured = False


def _load_settings() -> tuple[str, str, int, dict]:
    """Read logging settings from config, falling back to safe defaults.

    config is imported lazily so this module stays importable from scripts that
    do not have the project root on sys.path yet.
    """
    defaults = ("INFO", "INFO", 14, {})
    try:
        import config
    except Exception:
        return defaults

    return (
        str(getattr(config, "LOG_LEVEL_FILE", defaults[0])).upper(),
        str(getattr(config, "LOG_LEVEL_CONSOLE", defaults[1])).upper(),
        int(getattr(config, "LOG_RETENTION_DAYS", defaults[2])),
        dict(getattr(config, "LOG_LEVEL_PER_MODULE", {}) or {}),
    )


def get_current_log_path() -> str:
    """Path of the *active* log file.

    TimedRotatingFileHandler always writes to this path and renames it on
    rollover, so anything needing the live file should call this helper rather
    than rebuilding a dated name itself.
    """
    return os.path.join(LOG_DIR, f"{LOG_BASENAME}.log")


def _rotated_name(default_name: str) -> str:
    """Render rotated files as logs/bot_YYYYMMDD.log (historical naming)."""
    stamp = default_name.rsplit(".", 1)[-1].replace("-", "")
    return os.path.join(LOG_DIR, f"{LOG_BASENAME}_{stamp}.log")


def _configure_root() -> None:
    global _configured
    if _configured:
        return

    file_level, console_level, retention_days, _ = _load_settings()
    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # handlers do the real filtering

    file_handler = TimedRotatingFileHandler(
        get_current_log_path(),
        when="midnight",
        interval=1,
        backupCount=retention_days,
        encoding="utf-8",
        delay=True,
    )
    file_handler.suffix = "%Y-%m-%d"
    file_handler.namer = _rotated_name
    file_handler.setFormatter(formatter)
    file_handler.setLevel(getattr(logging, file_level, logging.INFO))
    root.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(getattr(logging, console_level, logging.INFO))
    root.addHandler(console_handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a module logger that writes through the shared rotating handlers."""
    _configure_root()

    logger = logging.getLogger(name)
    # Handlers live on the root logger; propagate up rather than duplicating them.
    logger.propagate = True

    _, _, _, per_module = _load_settings()
    if name in per_module:
        logger.setLevel(getattr(logging, str(per_module[name]).upper(), logging.INFO))
    else:
        logger.setLevel(logging.NOTSET)

    return logger
