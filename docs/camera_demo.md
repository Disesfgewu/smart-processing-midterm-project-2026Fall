# Live camera demo (Issue #7)

## What is implemented and verified

- `PromptState`: thread-safe current-prompt storage; `.set()` takes effect
  on the *next* frame of `run_loop` without restarting anything.
- `open_camera()`: opens `cv2.VideoCapture(device_index)` and raises a
  clear `RuntimeError` instead of silently returning an unusable capture
  when the camera can't be opened (no device, permission denied, in use).
- `run_loop()`: the camera-agnostic frame-processing loop (pull frame ->
  run the pipeline with the current prompts -> hand off to a display
  callback -> check a stop condition), with cleanup of the frame source
  (closing its generator) guaranteed via `finally`, on both a normal stop
  and an exception from the display callback.

All of the above are covered by fast unit tests (`tests/test_camera_demo.py`,
no camera, no model download) using fake `cv2`/frame-source objects.

## What is NOT verified here, and why

Issue #7 requires "manual macOS demo evidence" of the live interactive
camera display. **This autonomous session cannot provide that**: the
process running this work has no camera permission granted in macOS's TCC
(Transparency, Consent and Control) database, and granting it requires an
interactive user clicking a system permission dialog -- something nobody
can do in an unattended session.

Actual, current verification command and output on this machine
(`pip install -e ".[camera-demo]"` first):

```
$ python -c "
import cv2
cap = cv2.VideoCapture(0)
print('opened:', cap.isOpened())
cap.release()
"
OpenCV: not authorized to capture video (status 0), requesting...
OpenCV: camera failed to properly initialize!
opened: False

$ python -c "
from smart_processing.camera_demo import open_camera
open_camera(0)
"
RuntimeError: could not open camera 0; check that this process has camera
permission (macOS: System Settings > Privacy & Security > Camera) and no
other app is using the camera
```
`open_camera()`'s error-handling path is confirmed real, not a mock: this
is the actual exception raised against the actual unauthorized camera on
this machine.

`system_profiler SPCameraDataType` confirms a camera exists on this machine
("MacBook Air相機"); the TCC database
(`~/Library/Application Support/com.apple.TCC/TCC.db`) has no entry granting
camera access to this terminal/Python process.

`camera_demo_main()` (the thin real-hardware wiring: `cv2.VideoCapture` +
`cv2.imshow` + a stdin-reading thread for prompt changes) is implemented
but **not exercised by any test or by this session**, because doing so
would require the camera access above. To produce the required manual demo
evidence:

1. Run `python -c "from smart_processing.camera_demo import camera_demo_main; ..."`
   (or a thin CLI wrapper, not yet wired into `smart-processing`) locally,
   interactively, as the Owner.
2. Approve the camera permission prompt macOS shows on first access.
3. Type comma-separated prompts into the terminal to change them at runtime
   without restarting; press `q` in the display window to quit.
4. Record the actual command, output, and a screenshot/recording as the
   manual evidence this issue requires.

Issue #7 should **not** be considered closed by this PR alone; the
interactive verification above is Owner-only follow-up work.
