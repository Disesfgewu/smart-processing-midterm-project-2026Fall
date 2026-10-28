# Frontend/backend service API contract (Issue #23)

Defines the local HTTP API the frontend talks to, before any UI is built,
per #23's "backend service contract" delivery step. This PR implements
only the `/health` and `/infer` (detection-only) endpoints for real;
segmentation, camera streaming, evaluation/benchmark result endpoints are
documented here but left for follow-up PRs, consistent with #23's
"coherent small PRs" guidance.

## Design decisions (resolved before any UI work, per #23's acceptance)

- **Camera ownership: the browser owns the camera**, not the backend.
  The backend has no camera access concept at all -- the frontend
  captures frames (via `getUserMedia` or similar) and POSTs individual
  frames to `/infer`, the same request shape as a static image upload.
  This avoids a second video-transport protocol and keeps the backend
  stateless per request; the tradeoff (one HTTP round-trip per frame) is
  acceptable for a demo/operator tool, not a production streaming
  service.
- **Frame/image transport**: base64-encoded PNG or JPEG bytes in the
  JSON request body (`image_base64`), not multipart form data or a
  separate binary upload -- keeps the contract to one content type
  (`application/json`) for both request and response.
- **Coordinates**: all boxes are pixel `xyxy`, origin top-left, matching
  `#3`'s `BoundingBox.as_xyxy()` contract exactly -- the frontend never
  needs to un-normalize coordinates itself.
- **Cancellation**: not supported in this version. A `/infer` call runs
  to completion or the client's HTTP request times out; there is no
  server-side job queue or cancel endpoint. Documented as a known gap,
  not silently unsupported.
- **Resource cleanup**: the detector/segmenter are loaded once at server
  startup and held for the process lifetime (model loading is the
  expensive part; see `docs/macos_feasibility.md`'s load-time numbers).
  There is no per-request model teardown. Stopping the server process
  releases everything.

## Endpoints

### `GET /health`

Returns `200` with `{"status": "ok", "device": "<cpu|mps>"}` once the
detector has finished loading. Used by the frontend to show a
loading/ready state before allowing inference calls.

### `POST /infer`

Request body:
```json
{
  "image_base64": "<base64-encoded PNG or JPEG bytes>",
  "prompts": ["a cat", "a remote control"],
  "threshold": 0.1
}
```

Response body (`200`):
```json
{
  "image_width": 640,
  "image_height": 480,
  "detections": [
    {"label": "a cat", "score": 0.91, "box_xyxy": [12.0, 18.0, 98.0, 152.0]}
  ]
}
```
This is `DetectionResult` (#3) serialized directly -- `image_width`/
`image_height` echo the decoded image's real dimensions (not the
request's claim, since there isn't one), and each detection is
`{label, score, box_xyxy}`.

Error responses (`4xx`/`5xx`), all `{"error": "<message>"}`:
- `400`: malformed/missing `image_base64`, `prompts` empty or not a list
  of non-blank strings, `threshold` out of `[0, 1]` -- the same
  validation `OwlViTTextDetector`/`DetectionResult` already enforce
  (#3/#4), surfaced as HTTP errors rather than a raised Python exception
  escaping into a stack trace response.
- `500`: unexpected failure during inference (reported with a generic
  message; the real exception is logged server-side only, never leaked
  to the client body, to avoid exposing internal paths/stack details).

### Deferred to follow-up PRs under #23

- `POST /segment`: box-prompted segmentation (#5), returning masks.
  Deferred because mask transport (RLE-encoded PNG vs. a raw boolean
  array) needs its own design decision, not bundled into this PR's
  scope.
- `GET /evaluation/results` and `GET /benchmarks/results`: surfacing
  #10's scorer output and #11's benchmark numbers (both already produce
  JSON via `export_results()`/`export_benchmark_results()`) through the
  API for a results-viewing screen.
- Live camera/prompt-swap session state, if the stateless per-frame
  model above turns out to be too slow in practice once a real UI
  exists to measure it against.

## Not yet built

No actual UI exists yet. This PR is backend-only: the HTTP service and
its real, tested `/infer` wiring. The agent-eye skill and a visible
browser are required for the next PR that adds any frontend code, per
#23's acceptance criteria -- not needed for this backend-only PR.
