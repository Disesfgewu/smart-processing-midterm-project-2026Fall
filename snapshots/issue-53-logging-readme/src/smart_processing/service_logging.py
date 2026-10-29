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


# PM-Reviwer finding: shutdown_logging() previously removed every
# handler on the shared logger, including ones attached by unrelated
# code (a caller's own logging.basicConfig(), another library, a test
# harness). Tagging only the handlers this module itself creates lets
# shutdown_logging()/configure_logging() detach exactly those, leaving
# any externally-attached handler on the same logger untouched.
_OWNED_HANDLER_ATTR = "_smart_processing_service_logging_owned"


def shutdown_logging() -> None:
    """Closes and detaches only the handlers this module attached to the
    shared logger (tagged via configure_logging()) -- never a handler
    some other code attached to the same logger name. Idempotent --
    safe to call even if configure_logging() was never called, or
    called multiple times, so tests can always clean up between runs
    without leaking open file descriptors or touching handlers they
    don't own."""
    logger = logging.getLogger(LOGGER_NAME)
    for handler in list(logger.handlers):
        if getattr(handler, _OWNED_HANDLER_ATTR, False):
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
    the file can't be opened for writing, if `level` doesn't name a
    real logging level, or if `max_bytes`/`backup_count` would disable
    bounding (RotatingFileHandler treats maxBytes=0 as "never rotate",
    which would let app.log grow unbounded despite #53's rotation
    requirement) -- the caller must let server startup fail loudly
    rather than claim persistent, bounded logging is active when it
    isn't.
    """
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes <= 0:
        raise LoggingConfigError(f"max_bytes must be a positive int, got {max_bytes!r}")
    if not isinstance(backup_count, int) or isinstance(backup_count, bool) or backup_count < 1:
        raise LoggingConfigError(f"backup_count must be a positive int, got {backup_count!r}")

    resolved_level = _resolve_level(level)
    shutdown_logging()

    logger = logging.getLogger(LOGGER_NAME)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    setattr(console_handler, _OWNED_HANDLER_ATTR, True)
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
    setattr(file_handler, _OWNED_HANDLER_ATTR, True)
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
