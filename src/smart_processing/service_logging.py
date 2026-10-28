"""Backend diagnostic logging for #23's service (#53): a persistent,
rotating app.log plus short per-request correlation IDs, configured
once at server startup. Reuses the existing `smart_processing.service`
logger name so #51's existing `_logger.exception(...)` 500 logging
lands in the same file once wired up (service.py wiring is a follow-up
PR, kept separate to stay within the per-PR line budget).
"""
from __future__ import annotations

import logging
import logging.handlers
import os
import sys
import uuid
from pathlib import Path

LOGGER_NAME = "smart_processing.service"
DEFAULT_LOG_DIR = "artifacts/logs"
DEFAULT_LOG_FILE = "app.log"
DEFAULT_MAX_BYTES = 5 * 1024 * 1024
DEFAULT_BACKUP_COUNT = 3

_ENV_LOG_DIR = "SMART_PROCESSING_LOG_DIR"
_ENV_LOG_LEVEL = "SMART_PROCESSING_LOG_LEVEL"


class LoggingConfigError(Exception):
    """Raised when the configured log path can't be created or opened
    for writing -- distinct from a bare OSError so a caller can fail
    server startup with an actionable message instead of silently
    running without persistent logs."""


def resolve_log_path(log_dir: str | os.PathLike | None = None) -> Path:
    """`log_dir` > `SMART_PROCESSING_LOG_DIR` env var > the project's
    default artifact root. Never hard-codes a developer-specific
    absolute path."""
    raw = log_dir if log_dir is not None else os.environ.get(_ENV_LOG_DIR, DEFAULT_LOG_DIR)
    return Path(raw) / DEFAULT_LOG_FILE


def _resolve_level(level: str | int | None) -> int:
    raw = level if level is not None else os.environ.get(_ENV_LOG_LEVEL, "INFO")
    if isinstance(raw, int):
        return raw
    resolved = logging.getLevelName(str(raw).upper())
    if not isinstance(resolved, int):
        raise LoggingConfigError(f"invalid log level: {raw!r}")
    return resolved


def shutdown_logging() -> None:
    """Closes and detaches every handler this module owns on the shared
    logger. Idempotent -- safe to call even if configure_logging() was
    never called, or called multiple times, so tests can always clean
    up between runs without leaking open file descriptors."""
    logger = logging.getLogger(LOGGER_NAME)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def configure_logging(
    log_dir: str | os.PathLike | None = None,
    level: str | int | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
    backup_count: int = DEFAULT_BACKUP_COUNT,
) -> logging.Logger:
    """Configures the shared service logger with a console handler and
    a RotatingFileHandler, idempotently: repeated calls (across tests,
    or multiple run_server() calls in one process) first detach and
    close any handlers this module previously attached, so log lines
    are never duplicated or emitted twice.

    Raises LoggingConfigError if the log directory can't be created or
    the file can't be opened for writing, or if `level` doesn't name a
    real logging level -- the caller must let server startup fail
    loudly rather than claim persistent logging is active when it
    isn't.
    """
    resolved_level = _resolve_level(level)
    shutdown_logging()

    logger = logging.getLogger(LOGGER_NAME)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    log_path = resolve_log_path(log_dir)
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_path, maxBytes=max_bytes, backupCount=backup_count
        )
    except OSError as e:
        logger.removeHandler(console_handler)
        console_handler.close()
        raise LoggingConfigError(f"cannot write logs to {log_path}: {e}") from e
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.setLevel(resolved_level)
    logger.propagate = False
    return logger


def new_request_id() -> str:
    """A short, URL-safe, unique-enough-for-correlation ID -- not a
    security token, just enough to grep one request's log lines and
    echo back in an X-Request-ID response header (service.py wiring,
    follow-up PR)."""
    return uuid.uuid4().hex[:12]
