"""HTTP-level tests for #53's logging wiring in service.py: request IDs
in the X-Request-ID response header, correlated start/end log lines
with method/route/status/elapsed time, and that private payloads never
land in the log. Uses a fake detector/segmenter (no real model, no
RUN_FEASIBILITY_SMOKE needed) over a real local HTTPServer on a
background thread -- the request-ID/logging wiring itself is
model-independent."""
import json
import threading
import urllib.error
import urllib.request

import pytest

from smart_processing.service import run_server
from smart_processing.types import BoundingBox, DetectionResult, SegmentationMask


class _FakeDetector:
    def detect(self, image, prompts, threshold=0.1):
        return DetectionResult(image_width=10, image_height=10, detections=[])


class _FailingDetector:
    def detect(self, image, prompts, threshold=0.1):
        raise RuntimeError("boom: simulated unexpected failure")


class _LeakyDetector:
    """Simulates a third-party detector that embeds request-derived
    data directly in its exception message -- the realistic privacy
    risk, as opposed to _FailingDetector's hardcoded literal (which
    legitimately appears in a traceback's source-code frame line, not
    because the fix failed to redact it)."""

    def detect(self, image, prompts, threshold=0.1):
        raise RuntimeError("backend rejected prompt=" + prompts[0])


class _FakeSegmenter:
    def set_image(self, image):
        pass

    def segment(self, box):
        import numpy as np

        return SegmentationMask(mask=np.zeros((10, 10), dtype=bool), score=0.5)


def _tiny_png_base64() -> str:
    import base64
    from io import BytesIO

    import numpy as np
    from PIL import Image

    image = Image.fromarray(np.full((10, 10, 3), 128, dtype="uint8"))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


@pytest.fixture
def running_server(tmp_path):
    srv = run_server(_FakeDetector(), device="cpu", port=0, segmenter=_FakeSegmenter(), log_dir=tmp_path)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv, tmp_path
    srv.shutdown()
    thread.join(timeout=5)
    srv.server_close()


def _base_url(server) -> str:
    host, port = server.server_address
    return f"http://127.0.0.1:{port}"


def _log_text(log_dir) -> str:
    return (log_dir / "app.log").read_text()


def test_health_response_carries_an_x_request_id_header(running_server):
    server, log_dir = running_server
    with urllib.request.urlopen(f"{_base_url(server)}/health", timeout=10) as resp:
        request_id = resp.headers.get("X-Request-ID")
    assert request_id and len(request_id) == 12

    log_text = _log_text(log_dir)
    assert f"id={request_id}" in log_text
    assert "request start" in log_text and "request end" in log_text
    assert "status=200" in log_text


def test_successful_infer_logs_correlated_start_and_end_with_matching_id(running_server):
    server, log_dir = running_server
    pytest.importorskip("PIL")
    request_body = json.dumps(
        {"image_base64": _tiny_png_base64(), "prompts": ["a cat"], "threshold": 0.1}
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{_base_url(server)}/infer",
        data=request_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        request_id = resp.headers.get("X-Request-ID")

    log_text = _log_text(log_dir)
    start_lines = [line for line in log_text.splitlines() if f"id={request_id}" in line and "start" in line]
    end_lines = [line for line in log_text.splitlines() if f"id={request_id}" in line and "end" in line]
    assert len(start_lines) == 1 and len(end_lines) == 1
    assert "status=200" in end_lines[0]
    assert "elapsed_ms=" in end_lines[0]
    assert "method=POST" in start_lines[0] and "path=/infer" in start_lines[0]


def test_rejected_input_logs_a_controlled_400_without_the_request_body(running_server):
    server, log_dir = running_server
    request_body = json.dumps({"image_base64": "anything", "prompts": []}).encode("utf-8")
    req = urllib.request.Request(
        f"{_base_url(server)}/infer",
        data=request_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req, timeout=10)
    assert exc_info.value.code == 400

    log_text = _log_text(log_dir)
    assert "status=400" in log_text
    # The raw request body (base64 image bytes, prompts) must never be
    # logged -- only route/status/timing, per #53's privacy requirement.
    assert "anything" not in log_text


def test_forced_server_failure_logs_traceback_but_returns_generic_500(tmp_path):
    pytest.importorskip("PIL")
    srv = run_server(_FailingDetector(), device="cpu", port=0, log_dir=tmp_path)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        request_body = json.dumps(
            {"image_base64": _tiny_png_base64(), "prompts": ["a cat"]}
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{_base_url(srv)}/infer",
            data=request_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req, timeout=10)
        assert exc_info.value.code == 500
        body = json.loads(exc_info.value.read())
        assert body == {"error": "internal error during inference"}  # no internal detail leaked

        log_text = (tmp_path / "app.log").read_text()
        assert "request failed" in log_text
        assert "RuntimeError" in log_text and "boom: simulated unexpected failure" in log_text
        assert "status=500" in log_text
    finally:
        srv.shutdown()
        thread.join(timeout=5)


def test_forced_failure_never_logs_request_derived_exception_text(tmp_path):
    """PM-Reviwer finding: a stub detector raising a RuntimeError whose
    message is built from the actual request's prompt persisted that
    prompt verbatim into app.log via the default `_logger.exception()`
    formatting. The exception TYPE must still appear (useful for
    debugging); the request-derived MESSAGE text must not."""
    pytest.importorskip("PIL")
    srv = run_server(_LeakyDetector(), device="cpu", port=0, log_dir=tmp_path)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        request_body = json.dumps(
            {"image_base64": _tiny_png_base64(), "prompts": ["PROMPT_PRIVATE_SENTINEL"]}
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{_base_url(srv)}/infer",
            data=request_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req, timeout=10)
        assert exc_info.value.code == 500
    finally:
        srv.shutdown()
        thread.join(timeout=5)
        srv.server_close()

    log_text = (tmp_path / "app.log").read_text()
    assert "RuntimeError" in log_text  # exception type preserved
    assert "PROMPT_PRIVATE_SENTINEL" not in log_text  # request-derived message redacted


def test_query_string_is_never_logged(running_server):
    server, log_dir = running_server
    with urllib.request.urlopen(f"{_base_url(server)}/health?token=QUERY_PRIVATE_SENTINEL", timeout=10) as resp:
        resp.read()
    log_text = _log_text(log_dir)
    assert "QUERY_PRIVATE_SENTINEL" not in log_text


def test_unsupported_method_still_gets_logged_and_correlated(running_server):
    """A method with no do_X handler (PUT) never reaches do_GET/do_POST
    -- must still get an X-Request-ID header and a correlated log
    entry via the send_error() override, not silently bypass #53's
    correlation requirement."""
    server, log_dir = running_server
    req = urllib.request.Request(f"{_base_url(server)}/infer", method="PUT")
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req, timeout=10)
    assert exc_info.value.code == 501
    request_id = exc_info.value.headers.get("X-Request-ID")
    assert request_id

    log_text = _log_text(log_dir)
    assert f"id={request_id}" in log_text


def test_server_logs_actual_bound_address_and_shutdown_sequence(tmp_path):
    srv = run_server(_FakeDetector(), device="cpu", port=0, log_dir=tmp_path)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    _, actual_port = srv.server_address
    try:
        log_text = (tmp_path / "app.log").read_text()
        assert f"port={actual_port}" in log_text  # the real bound port, not the requested 0
        assert "server starting" in log_text
    finally:
        srv.shutdown()
        thread.join(timeout=5)
        srv.server_close()

    log_text = (tmp_path / "app.log").read_text()
    assert "server shutdown requested" in log_text
    assert "server closed" in log_text


def test_repeated_run_server_calls_do_not_duplicate_log_lines(tmp_path):
    """Two run_server() calls in one process (e.g. a test module that
    restarts the server) must not stack duplicate handlers -- a single
    /health hit after the second run_server() call produces exactly one
    log line, not two."""
    srv1 = run_server(_FakeDetector(), device="cpu", port=0, log_dir=tmp_path)
    srv2 = run_server(_FakeDetector(), device="cpu", port=0, log_dir=tmp_path)
    thread = threading.Thread(target=srv2.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(f"{_base_url(srv2)}/health", timeout=10) as resp:
            request_id = resp.headers.get("X-Request-ID")
        log_text = (tmp_path / "app.log").read_text()
        matching_lines = [line for line in log_text.splitlines() if f"id={request_id}" in line]
        assert len(matching_lines) == 2  # one start, one end -- never doubled
    finally:
        srv2.shutdown()
        thread.join(timeout=5)
        srv1.server_close()
