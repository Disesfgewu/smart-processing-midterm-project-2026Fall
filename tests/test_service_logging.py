"""Tests for #53's logging module (src/smart_processing/service_logging.py).
Pure stdlib (logging, tempfile) -- no HTTP server, no model, no PIL.
service.py wiring (request IDs in responses, elapsed-time logging) is
a follow-up PR; these tests only cover configure_logging/new_request_id
in isolation."""
import logging
import os
import re

import pytest

from smart_processing.service_logging import (
    LOGGER_NAME,
    LoggingConfigError,
    configure_logging,
    new_request_id,
    resolve_log_path,
    shutdown_logging,
)


@pytest.fixture(autouse=True)
def _clean_logger():
    """Every test starts and ends with the shared logger detached, so
    tests never see each other's handlers or leak file descriptors."""
    shutdown_logging()
    yield
    shutdown_logging()


def test_configure_logging_creates_log_directory_and_file(tmp_path):
    log_dir = tmp_path / "nested" / "logs"
    logger = configure_logging(log_dir=log_dir)
    logger.info("hello from a test")

    log_file = log_dir / "app.log"
    assert log_file.exists()
    assert "hello from a test" in log_file.read_text()


def test_configure_logging_is_idempotent_no_duplicate_lines(tmp_path):
    """Calling configure_logging() twice (e.g. two run_server() calls in
    one process, or re-running a test fixture) must not stack handlers
    -- a single log call must produce exactly one line, not two."""
    configure_logging(log_dir=tmp_path)
    configure_logging(log_dir=tmp_path)
    logger = logging.getLogger(LOGGER_NAME)
    logger.info("only once please")

    log_file = tmp_path / "app.log"
    lines = [line for line in log_file.read_text().splitlines() if "only once please" in line]
    assert len(lines) == 1


def test_shutdown_logging_detaches_and_closes_handlers(tmp_path):
    configure_logging(log_dir=tmp_path)
    logger = logging.getLogger(LOGGER_NAME)
    assert logger.handlers

    shutdown_logging()
    assert logger.handlers == []


def test_shutdown_logging_is_safe_to_call_when_nothing_configured():
    shutdown_logging()  # no prior configure_logging() call in this test
    shutdown_logging()  # calling it again must not raise either


def test_configure_logging_defaults_to_info_level(tmp_path):
    logger = configure_logging(log_dir=tmp_path)
    assert logger.level == logging.INFO
    assert logger.getEffectiveLevel() == logging.INFO


def test_configure_logging_accepts_debug_level_opt_in(tmp_path):
    logger = configure_logging(log_dir=tmp_path, level="DEBUG")
    assert logger.level == logging.DEBUG


def test_configure_logging_rejects_unknown_level_name(tmp_path):
    with pytest.raises(LoggingConfigError, match="invalid log level"):
        configure_logging(log_dir=tmp_path, level="NOT_A_REAL_LEVEL")


def test_configure_logging_reports_actionable_error_on_unwritable_path(tmp_path):
    """On an unwritable log path, startup must fail clearly (per #53's
    acceptance) rather than silently claiming persistent logging is
    active. Simulated by pointing the log dir at a path that collides
    with an existing regular file (can't mkdir a directory over it)."""
    blocking_file = tmp_path / "not_a_directory"
    blocking_file.write_text("occupied")

    with pytest.raises(LoggingConfigError, match="cannot write logs"):
        configure_logging(log_dir=blocking_file / "logs")


def test_resolve_log_path_uses_explicit_arg_over_env(tmp_path, monkeypatch):
    monkeypatch.setenv("SMART_PROCESSING_LOG_DIR", str(tmp_path / "from_env"))
    resolved = resolve_log_path(log_dir=tmp_path / "explicit")
    assert resolved == tmp_path / "explicit" / "app.log"


def test_resolve_log_path_falls_back_to_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv("SMART_PROCESSING_LOG_DIR", str(tmp_path / "from_env"))
    resolved = resolve_log_path()
    assert resolved == tmp_path / "from_env" / "app.log"


def test_resolve_log_path_defaults_to_project_artifact_root(monkeypatch):
    monkeypatch.delenv("SMART_PROCESSING_LOG_DIR", raising=False)
    resolved = resolve_log_path()
    assert resolved == __import__("pathlib").Path("artifacts/logs/app.log")
    assert not os.path.isabs(resolved), "default log path must not hard-code a developer-specific absolute path"


def test_new_request_id_is_a_short_hex_string():
    request_id = new_request_id()
    assert re.fullmatch(r"[0-9a-f]{12}", request_id)


def test_new_request_id_is_unique_across_calls():
    ids = {new_request_id() for _ in range(1000)}
    assert len(ids) == 1000
