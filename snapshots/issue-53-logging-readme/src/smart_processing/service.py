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
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
from urllib.parse import urlsplit

from smart_processing.service_logging import configure_logging, new_request_id, shutdown_logging
from smart_processing.types import BoundingBox, DetectionResult

# Routes this server actually implements -- used to sanitize what lands
# in logs (#53 forbids arbitrary query strings / unknown paths in route
# logs; a client-supplied query string could carry a secret a client
# embedded in the URL for its own reasons).
_KNOWN_ROUTES = ("/health", "/infer", "/segment")

# Handlers (console + rotating file) are attached by configure_logging()
# in run_server() at startup (#53), not at import time -- importing this
# module (e.g. for unit tests of handle_infer/handle_segment) must not
# itself create artifacts/logs/app.log or print to stderr.
_logger = logging.getLogger("smart_processing.service")


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


def _safe_exception_report(exc: BaseException) -> str:
    """A traceback-only incident report for server-side logs: exception
    type and call-stack frames (file/line/function), but never the
    exception's own message text, walking __cause__/__context__ chains
    the same way. PM-Reviwer finding: a third-party detector/segmenter
    can embed request-derived data (e.g. a prompt) directly in its
    exception's message, which `_logger.exception()`'s default
    formatting would otherwise persist verbatim -- #53 forbids logging
    raw prompts/image data by default even when it arrives this
    indirectly. Frame text (source code lines) is not request data, so
    it is kept; only the exception's args/message are redacted."""
    parts = []
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        frames = "".join(traceback.format_tb(current.__traceback__))
        parts.append(f"{type(current).__module__}.{type(current).__qualname__} (message redacted)\n{frames}")
        current = current.__cause__ or current.__context__
    return "\n-- caused by --\n".join(parts)


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

        def _safe_route(self) -> str:
            # Route for LOGGING only: query/fragment stripped, and an
            # unrecognized path collapses to a generic marker -- #53
            # forbids arbitrary query strings in route logs. Response
            # bodies below still echo self.path verbatim (reflecting a
            # client's own request back to itself is not a cross-
            # request leak the way persisting it to disk would be).
            path = urlsplit(self.path).path
            return path if path in _KNOWN_ROUTES else "<unknown-route>"

        def end_headers(self) -> None:
            # Applies to every response this handler sends, including
            # ones the stdlib dispatches itself (e.g. send_error() for
            # an unsupported HTTP method) -- not just the ones routed
            # through _send_json below.
            request_id = getattr(self, "_request_id", None)
            if request_id is not None:
                self.send_header("X-Request-ID", request_id)
            super().end_headers()

        def _send_json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _begin_request(self) -> None:
            # #53: a short per-request correlation ID + start timestamp,
            # logged here (sanitized route only -- never the request
            # body, which may contain base64 image bytes or prompts)
            # and echoed in every response via end_headers above.
            self._request_id = new_request_id()
            self._request_start = time.monotonic()
            _logger.info(
                "request start id=%s method=%s path=%s", self._request_id, self.command, self._safe_route()
            )

        def _finish_request(self, status: int, payload: dict) -> None:
            elapsed_ms = (time.monotonic() - self._request_start) * 1000
            _logger.info(
                "request end id=%s method=%s path=%s status=%s elapsed_ms=%.1f",
                self._request_id,
                self.command,
                self._safe_route(),
                status,
                elapsed_ms,
            )
            self._send_json(status, payload)

        def send_error(self, code, message=None, explain=None) -> None:
            # PM-Reviwer finding: an unsupported HTTP method (e.g. PUT
            # /infer) never reaches do_GET/do_POST, so it bypassed
            # _begin_request() entirely -- no correlation ID, no log
            # line, for a response the client still received. Covers
            # that and any other stdlib-internal error path that calls
            # send_error() directly.
            if not hasattr(self, "_request_id"):
                self._begin_request()
            elapsed_ms = (time.monotonic() - self._request_start) * 1000
            _logger.info(
                "request end id=%s method=%s path=%s status=%s elapsed_ms=%.1f (stdlib error path)",
                self._request_id,
                self.command,
                self._safe_route(),
                code,
                elapsed_ms,
            )
            super().send_error(code, message, explain)

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
                self._finish_request(400, {"error": "Content-Length header is not a valid integer"})
                return
            if length < 0 or length > self._MAX_BODY_BYTES:
                self._finish_request(400, {"error": f"Content-Length out of bounds: {length}"})
                return

            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                self._finish_request(400, {"error": f"invalid JSON body: {e}"})
                return

            try:
                result = handler_fn(*args, body)
            except ServiceError as e:
                # A validation failure is a controlled client error (#53):
                # logged as part of the request-end line above, not with
                # a traceback -- the request body itself is never logged.
                self._finish_request(400, {"error": str(e)})
            except Exception as e:
                # Real exception logged server-side only, with a type +
                # frame traceback but never the exception's own message
                # text (PM-Reviwer finding: a detector/segmenter could
                # embed request data in its exception message) -- see
                # _safe_exception_report. Never leaked to the client
                # body either way (docs/frontend_service_api.md's
                # error-response contract).
                _logger.error(
                    "request failed id=%s method=%s path=%s\n%s",
                    self._request_id,
                    self.command,
                    self._safe_route(),
                    _safe_exception_report(e),
                )
                self._finish_request(500, {"error": "internal error during inference"})
            else:
                self._finish_request(200, result)

        def do_GET(self) -> None:
            self._begin_request()
            # Routes on the parsed path, ignoring any query string --
            # was previously an exact self.path match, so "/health?x=1"
            # 404'd instead of routing to /health with a query string
            # (noticed while sanitizing route logging; standard HTTP
            # servers route on the path component alone).
            if urlsplit(self.path).path == "/health":
                self._finish_request(200, {"status": "ok", "device": device})
            else:
                self._finish_request(404, {"error": f"no such endpoint: {self.path}"})

        def do_POST(self) -> None:
            self._begin_request()
            route = urlsplit(self.path).path
            if route == "/infer":
                self._handle(handle_infer, detector)
            elif route == "/segment":
                if segmenter is None:
                    self._finish_request(503, {"error": "segmentation is not available on this server"})
                else:
                    self._handle(handle_segment, segmenter)
            else:
                self._finish_request(404, {"error": f"no such endpoint: {self.path}"})

        def log_message(self, format, *args) -> None:
            pass  # HTTP access log is quiet by default; requests are logged via _logger above

    return Handler


class _LoggingHTTPServer(HTTPServer):
    """Logs the shutdown sequence and releases this module's own log
    handlers on close -- PM-Reviwer finding: shutdown()+server_close()
    previously left both logging handlers attached with no shutdown
    event recorded, despite #53 explicitly requiring startup/shutdown
    logging. http.server.HTTPServer has no built-in hook for this, so
    this subclass provides one rather than treating that absence as a
    reason to skip the acceptance criterion."""

    def shutdown(self) -> None:
        host, port = self.server_address
        _logger.info("server shutdown requested host=%s port=%d", host, port)
        super().shutdown()

    def server_close(self) -> None:
        host, port = self.server_address
        super().server_close()
        _logger.info("server closed host=%s port=%d", host, port)
        shutdown_logging()


def run_server(
    detector,
    device: str = "cpu",
    host: str = "127.0.0.1",
    port: int = 8765,
    segmenter=None,
    log_dir=None,
    log_level=None,
) -> HTTPServer:
    # #53: configures the persistent rotating app.log + console logging
    # before the server starts accepting requests. configure_logging()
    # raises LoggingConfigError on an unwritable log path or invalid
    # level, so startup fails loudly here rather than silently running
    # without the persistent log the Owner asked for.
    configure_logging(log_dir=log_dir, level=log_level)
    try:
        server = _LoggingHTTPServer((host, port), make_handler(detector, device, segmenter=segmenter))
    except OSError as e:
        # PM-Reviwer finding: a startup bind failure (e.g. port already
        # in use) must be logged before it propagates, not just raised
        # silently past the logging layer.
        _logger.error("server failed to start host=%s port=%d error=%s", host, port, e)
        raise
    # Logs the ACTUAL bound address, not the requested `port` -- with
    # port=0 (as the opt-in real test suite uses), the OS picks the
    # real port only once the socket is bound.
    bound_host, bound_port = server.server_address
    _logger.info(
        "server starting host=%s port=%d device=%s segmenter=%s",
        bound_host,
        bound_port,
        device,
        segmenter is not None,
    )
    return server
