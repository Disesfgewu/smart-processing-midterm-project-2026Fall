"""Opt-in real end-to-end test for #23's service: a real HTTP server, real
OwlViTTextDetector inference, a real HTTP client round-trip.

Skipped by default (downloads/loads a real model); run explicitly with
RUN_FEASIBILITY_SMOKE=1, same convention as #2/#4/#5/#6/#11.
"""
import base64
import json
import os
import socket
import threading
import urllib.error
import urllib.request

import pytest

RUN = os.environ.get("RUN_FEASIBILITY_SMOKE") == "1"
pytestmark = pytest.mark.skipif(
    not RUN, reason="set RUN_FEASIBILITY_SMOKE=1 to run the real service end-to-end"
)


@pytest.fixture(scope="module")
def server():
    from smart_processing.adapters.owlvit_text_detector import OwlViTTextDetector
    from smart_processing.service import run_server
    from smart_processing.types import BackendConfig

    detector = OwlViTTextDetector(BackendConfig(device="cpu"))
    srv = run_server(detector, device="cpu", port=0)  # port=0 -> OS picks a free port
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()
    thread.join(timeout=5)


@pytest.fixture(scope="module")
def sample_image_base64():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from sample_image import SAMPLE_IMAGE_PATH, ensure_sample_image

    image_path = ensure_sample_image(SAMPLE_IMAGE_PATH)
    return base64.b64encode(Path(image_path).read_bytes()).decode("ascii")


def _base_url(server) -> str:
    host, port = server.server_address
    return f"http://127.0.0.1:{port}"


def test_health_reports_ok(server):
    with urllib.request.urlopen(f"{_base_url(server)}/health", timeout=10) as resp:
        body = json.loads(resp.read())
    assert body == {"status": "ok", "device": "cpu"}


def test_infer_finds_cats_via_real_http_round_trip(server, sample_image_base64):
    request_body = json.dumps(
        {"image_base64": sample_image_base64, "prompts": ["a photo of a cat"], "threshold": 0.1}
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{_base_url(server)}/infer",
        data=request_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = json.loads(resp.read())

    assert body["image_width"] > 0 and body["image_height"] > 0
    assert len(body["detections"]) >= 1
    assert all(d["label"] == "a photo of a cat" for d in body["detections"])
    assert all(0.0 <= d["score"] <= 1.0 for d in body["detections"])


def test_infer_rejects_malformed_request_with_400(server):
    request_body = json.dumps({"image_base64": "", "prompts": []}).encode("utf-8")
    req = urllib.request.Request(
        f"{_base_url(server)}/infer",
        data=request_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req, timeout=10)
    assert exc_info.value.code == 400


def test_infer_rejects_non_object_json_body_with_400(server):
    """PM-Reviwer's exact repro: a syntactically valid JSON body that
    isn't an object (`[]`) must return 400, not a 500 or a dropped
    connection."""
    request_body = json.dumps([]).encode("utf-8")
    req = urllib.request.Request(
        f"{_base_url(server)}/infer",
        data=request_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req, timeout=10)
    assert exc_info.value.code == 400


def _raw_http_request(server, raw_request_bytes: bytes) -> bytes:
    """Sends raw bytes over a socket (bypassing urllib's well-formed-
    request guarantees) and returns whatever bytes come back, so
    deliberately malformed framing (bad Content-Length, invalid UTF-8
    body) can actually be constructed -- urllib won't let us send these."""
    host, port = server.server_address
    with socket.create_connection((host, port), timeout=10) as sock:
        sock.sendall(raw_request_bytes)
        sock.settimeout(10)
        chunks = []
        try:
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                chunks.append(chunk)
        except socket.timeout:
            pass
        return b"".join(chunks)


def test_infer_rejects_non_numeric_content_length_with_400(server):
    request = (
        b"POST /infer HTTP/1.1\r\n"
        b"Host: localhost\r\n"
        b"Content-Type: application/json\r\n"
        b"Content-Length: not-a-number\r\n"
        b"Connection: close\r\n"
        b"\r\n"
    )
    response = _raw_http_request(server, request)
    assert response.startswith(b"HTTP/1.0 400") or response.startswith(b"HTTP/1.1 400")


def test_infer_rejects_invalid_utf8_body_with_400(server):
    body = b"\xff\xfe\x00invalid-utf8"
    request = (
        b"POST /infer HTTP/1.1\r\n"
        b"Host: localhost\r\n"
        b"Content-Type: application/json\r\n"
        + f"Content-Length: {len(body)}\r\n".encode("ascii")
        + b"Connection: close\r\n"
        b"\r\n" + body
    )
    response = _raw_http_request(server, request)
    assert response.startswith(b"HTTP/1.0 400") or response.startswith(b"HTTP/1.1 400")


def test_service_recovers_after_malformed_request(server):
    """A malformed request must not leave the single HTTPServer unable to
    serve subsequent requests -- confirms recovery with a real /health
    call right after a deliberately bad request."""
    bad_request = (
        b"POST /infer HTTP/1.1\r\n"
        b"Host: localhost\r\n"
        b"Content-Length: garbage\r\n"
        b"Connection: close\r\n"
        b"\r\n"
    )
    _raw_http_request(server, bad_request)

    with urllib.request.urlopen(f"{_base_url(server)}/health", timeout=10) as resp:
        body = json.loads(resp.read())
    assert body == {"status": "ok", "device": "cpu"}
