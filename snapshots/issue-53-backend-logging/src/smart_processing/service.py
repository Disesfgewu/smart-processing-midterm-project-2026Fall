"""Local HTTP service (#23): implements `/health`, `/infer`, and `/segment`
from docs/frontend_service_api.md. Stdlib-only (`http.server`), no new
dependency -- the frontend is the only planned consumer and this is a
single-process demo/operator tool, not a production API surface.

Image/mask decoding and encoding (PIL) stay deferred into their own
functions so importing this module never requires the macos-feasibility
optional extra.
"""
from __future__ import annotations

import base64
import json
import logging
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO

from smart_processing.types import BoundingBox, DetectionResult

_logger = logging.getLogger("smart_processing.service")
if not _logger.handlers:
    _logger.addHandler(logging.StreamHandler(sys.stderr))
    _logger.setLevel(logging.WARNING)


class ServiceError(Exception):
    """A client-caused (400) error: distinguished from unexpected (500)
    failures so the handler can report each with the right status code
    and without leaking internal exception details for the latter."""


def _require_json_object(body, context: str) -> None:
    """Shared by handle_infer/handle_segment: a syntactically valid JSON
    body that isn't an object (e.g. `[]`, `null`, `123`) made the
    per-endpoint body.get() calls raise AttributeError, an unhandled
    exception the HTTP layer reported as a 500 instead of the 400 a
    malformed request deserves. PM-Reviwer/Co-Reviewer finding, found
    first on #47's /infer and confirmed to also affect /segment."""
    if not isinstance(body, dict):
        raise ServiceError(f"{context} request body must be a JSON object, got {type(body).__name__}")


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


def _encode_mask(mask) -> str:
    """Encodes a boolean (H, W) mask as a base64 1-bit PNG. PNG over raw
    base64 bytes or RLE: it's a standard, inspectable format the frontend
    can decode in a browser `<canvas>` or `<img>` with zero custom codec,
    and PIL (already a dependency here) compresses it well for a mostly-
    uniform binary image -- resolved design decision from
    docs/frontend_service_api.md's deferred mask-transport question."""
    from PIL import Image

    buffer = BytesIO()
    Image.fromarray(mask).convert("1").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def handle_infer(detector, body: dict) -> dict:
    """Pure request-handling logic, separated from HTTP plumbing so it's
    unit-testable with a fake detector and no real server/socket.

    Validates the body shape, then `prompts`/`threshold`, before decoding
    `image_base64` (deliberately ordered this way): those checks need no
    PIL import at all, so a malformed request fails fast without paying
    for image decoding, and most of this function's error paths stay
    testable in a PIL-free dev venv -- only the "everything valid" path
    needs a real image and the macos-feasibility extra."""
    _require_json_object(body, "/infer")
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


def handle_segment(segmenter, body: dict) -> dict:
    """Pure request-handling logic for /segment, mirroring handle_infer's
    structure: body shape and box_xyxy are validated before decoding
    image_base64, so most error paths stay testable without PIL/the
    macos-feasibility extra. `box_xyxy` reuses #3's `BoundingBox`
    validation directly (finite, non-negative, correctly-ordered
    coordinates) instead of re-implementing those checks."""
    _require_json_object(body, "/segment")
    box_xyxy = body.get("box_xyxy")
    if not isinstance(box_xyxy, list) or len(box_xyxy) != 4:
        raise ServiceError("box_xyxy must be a list of 4 numbers [x0, y0, x1, y1]")
    try:
        box = BoundingBox(*box_xyxy)
    except (TypeError, ValueError) as e:
        raise ServiceError(f"box_xyxy is invalid: {e}") from e

    image = _decode_image(body.get("image_base64"))

    try:
        segmenter.set_image(image)
        mask_result = segmenter.segment(box)
    except ValueError as e:
        # Segmenter-level input validation (#5) surfaces as a client
        # error, not a 500 -- e.g. a box outside the image bounds.
        raise ServiceError(str(e)) from e

    return {
        "image_width": image.shape[1],
        "image_height": image.shape[0],
        "mask_base64": _encode_mask(mask_result.mask),
        "score": mask_result.score,
    }


def make_handler(detector, device: str, segmenter=None):
    class Handler(BaseHTTPRequestHandler):
        # Defensive upper bound on a claimed request size: a single base64
        # image plus prompts/box is never anywhere near this, and bounding
        # it stops a malformed/hostile Content-Length from making the
        # server attempt an enormous read. PM-Reviwer finding (framing
        # hardening, applies to both /infer and /segment via _handle).
        _MAX_BODY_BYTES = 64 * 1024 * 1024

        def _send_json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _handle(self, handler_fn, *args) -> None:
            # PM-Reviwer/Co-Reviewer finding: Content-Length was parsed
            # with a bare int(), uncaught for a non-numeric or negative
            # header value; and json.loads() on non-UTF-8 bytes raises
            # UnicodeDecodeError, which `except json.JSONDecodeError` did
            # not catch -- both previously fell through to an unhandled
            # exception (and, for UnicodeDecodeError specifically, a
            # connection reset rather than any HTTP response at all).
            # Both now get a clean 400, for every POST endpoint.
            try:
                length = int(self.headers.get("Content-Length", 0))
            except ValueError:
                self._send_json(400, {"error": "Content-Length header is not a valid integer"})
                return
            if length < 0 or length > self._MAX_BODY_BYTES:
                self._send_json(400, {"error": f"Content-Length out of bounds: {length}"})
                return

            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                self._send_json(400, {"error": f"invalid JSON body: {e}"})
                return

            try:
                result = handler_fn(*args, body)
            except ServiceError as e:
                self._send_json(400, {"error": str(e)})
            except Exception:
                # Real exception logged server-side only, never leaked to
                # the client body (docs/frontend_service_api.md's error-
                # response contract) -- previously silently swallowed,
                # a known gap flagged on #47, fixed here.
                _logger.exception("internal error handling %s", self.path)
                self._send_json(500, {"error": "internal error during inference"})
            else:
                self._send_json(200, result)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send_json(200, {"status": "ok", "device": device})
            else:
                self._send_json(404, {"error": f"no such endpoint: {self.path}"})

        def do_POST(self) -> None:
            if self.path == "/infer":
                self._handle(handle_infer, detector)
            elif self.path == "/segment":
                if segmenter is None:
                    self._send_json(503, {"error": "segmentation is not available on this server"})
                else:
                    self._handle(handle_segment, segmenter)
            else:
                self._send_json(404, {"error": f"no such endpoint: {self.path}"})

        def log_message(self, format, *args) -> None:
            pass  # HTTP access log is quiet by default; 500s are logged via _logger above

    return Handler


def run_server(
    detector, device: str = "cpu", host: str = "127.0.0.1", port: int = 8765, segmenter=None
) -> HTTPServer:
    server = HTTPServer((host, port), make_handler(detector, device, segmenter=segmenter))
    return server
