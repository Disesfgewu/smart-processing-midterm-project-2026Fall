# Frontend/backend service API contract (Issue #23)

per #23's "backend service contract" delivery step. **This PR (#48) is
documentation-only** (Co-Reviewer finding: an earlier draft of this intro
wrongly said "this PR implements... for real" -- corrected here). The
`/health` and `/infer` (detection-only) endpoints are implemented for
real in stacked PR #47 (hardened against malformed requests in #54);
`/segment` (box-prompted segmentation) in further-stacked PR #51. Camera
streaming and evaluation/benchmark result endpoints remain documented
here but left for later follow-up PRs, consistent with #23's "coherent
small PRs" guidance.

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

### `POST /segment`

Request body:
```json
{"image_base64": "<base64-encoded PNG or JPEG bytes>", "box_xyxy": [12.0, 18.0, 98.0, 152.0]}
```

Response body (`200`):
```json
{"image_width": 640, "image_height": 480, "mask_base64": "<base64 1-bit PNG>", "score": 0.95}
```

**Mask transport decision** (previously deferred): a base64-encoded
1-bit PNG, same envelope style as `image_base64` in the request --
not RLE encoding or a raw boolean array. PNG is a standard format the
frontend can decode with zero custom codec (a browser `<img>`/
`<canvas>` already understands it), and PIL (already a dependency for
`/infer`'s own image decoding) compresses a mostly-uniform binary mask
well. `image_width`/`image_height` echo the real decoded dimensions,
matching `/infer`'s convention.

Error responses: `400` for malformed `box_xyxy` (reuses `#3`'s
`BoundingBox` validation) or a box outside the image bounds (`#5`'s own
check); `500` for unexpected failures (same policy as `/infer`); `503`
if the server was started without a segmenter configured (segmentation
is optional per server instance, since it requires the separate SAM
checkpoint download documented in `docs/macos_feasibility.md`).

### Deferred to follow-up PRs under #23

- `GET /evaluation/results` and `GET /benchmarks/results`: surfacing
  #10's scorer output and #11's benchmark numbers (both already produce
  JSON via `export_results()`/`export_benchmark_results()`) through the
  API for a results-viewing screen.
- Live camera/prompt-swap session state, if the stateless per-frame
  model above turns out to be too slow in practice once a real UI
  exists to measure it against.

## Not yet built

No actual UI exists yet, and (as stated above) this PR itself contains
no implementation code -- it only defines the contract the stacked
implementation PRs (#47, #51) build against. The agent-eye skill and a
visible browser are required for the eventual PR that adds frontend
code, per #23's acceptance criteria -- not needed for this docs-only PR.
