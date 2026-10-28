"""Local HTTP service (#23): implements `/health` and `/infer` from
docs/frontend_service_api.md. Stdlib-only (`http.server`), no new
dependency -- the frontend is the only planned consumer and this is a
single-process demo/operator tool, not a production API surface.

Image decoding (PIL) stays deferred into `_decode_image` so importing
this module never requires the macos-feasibility optional extra.
"""
from __future__ import annotations

import base64
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO

from smart_processing.types import DetectionResult


class ServiceError(Exception):
    """A client-caused (400) error: distinguished from unexpected (500)
    failures so the handler can report each with the right status code
    and without leaking internal exception details for the latter."""


def _decode_image(image_base64: str):
    # PIL import deliberately deferred past the string/base64 checks below
    # (not just past the top of the function) -- those two checks must
    # stay testable without the macos-feasibility extra installed.
    if not isinstance(image_base64, str) or not image_base64.strip():
        raise ServiceError("image_base64 must be a non-empty base64 string")
    try:
        raw = base64.b64decode(image_base64, validate=True)
    except Exception as e:
        raise ServiceError(f"image_base64 is not valid base64: {e}") from e

    import numpy as np
    from PIL import Image, UnidentifiedImageError

    try:
        image = Image.open(BytesIO(raw)).convert("RGB")
    except UnidentifiedImageError as e:
        raise ServiceError(f"image_base64 does not decode to a valid image: {e}") from e
    return np.array(image)


def _detection_result_to_dict(result: DetectionResult) -> dict:
    return {
        "image_width": result.image_width,
        "image_height": result.image_height,
        "detections": [
            {"label": d.label, "score": d.score, "box_xyxy": list(d.box.as_xyxy())}
            for d in result.detections
        ],
    }


def handle_infer(detector, body: dict) -> dict:
    """Pure request-handling logic, separated from HTTP plumbing so it's
    unit-testable with a fake detector and no real server/socket.

    Validates `prompts`/`threshold` before decoding `image_base64`
    (deliberately ordered this way): those two checks need no PIL import
    at all, so a malformed request fails fast without paying for image
    decoding, and most of this function's error paths stay testable in a
    PIL-free dev venv -- only the "everything valid" path needs a real
    image and the macos-feasibility extra."""
    prompts = body.get("prompts")
    threshold = body.get("threshold", 0.1)
    if not isinstance(prompts, list) or not prompts:
        raise ServiceError("prompts must be a non-empty list of strings")
    if any(not isinstance(p, str) or not p.strip() for p in prompts):
        raise ServiceError("prompts must all be non-blank strings")
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        raise ServiceError("threshold must be a number")
    if not (0.0 <= threshold <= 1.0):
        raise ServiceError(f"threshold must be in [0, 1], got {threshold}")

    image = _decode_image(body.get("image_base64"))

    try:
        result = detector.detect(image, prompts, threshold=threshold)
    except ValueError as e:
        # Detector-level input validation (#4) surfaces as a client error,
        # not a 500 -- the request itself was malformed, not the server.
        raise ServiceError(str(e)) from e

    return _detection_result_to_dict(result)


def make_handler(detector, device: str):
    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send_json(200, {"status": "ok", "device": device})
            else:
                self._send_json(404, {"error": f"no such endpoint: {self.path}"})

        def do_POST(self) -> None:
            if self.path != "/infer":
                self._send_json(404, {"error": f"no such endpoint: {self.path}"})
                return
            length = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError as e:
                self._send_json(400, {"error": f"invalid JSON body: {e}"})
                return
            try:
                result = handle_infer(detector, body)
            except ServiceError as e:
                self._send_json(400, {"error": str(e)})
            except Exception:
                self._send_json(500, {"error": "internal error during inference"})
            else:
                self._send_json(200, result)

        def log_message(self, format, *args) -> None:
            pass  # quiet by default; a real deployment would wire real logging

    return Handler


def run_server(detector, device: str = "cpu", host: str = "127.0.0.1", port: int = 8765) -> HTTPServer:
    server = HTTPServer((host, port), make_handler(detector, device))
    return server
