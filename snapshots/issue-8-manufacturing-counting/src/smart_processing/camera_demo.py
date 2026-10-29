"""Live camera demo with runtime prompt changes (#7).

Design note: the frame-processing loop (run_loop) is decoupled from any
specific camera/display backend so it is fully unit-testable without
hardware. Only camera_demo_main() wires real cv2 capture/display/stdin,
which genuinely needs a camera and an authorized TCC permission -- see
README/PR for the real, current verification result on this machine
(camera access is NOT authorized in this autonomous session).
"""
from __future__ import annotations

import threading


def _validate_prompts(prompts) -> None:
    if not isinstance(prompts, (list, tuple)) or not prompts:
        raise ValueError(f"prompts must be a non-empty list of strings, got {prompts!r}")
    if any(not isinstance(p, str) or not p.strip() for p in prompts):
        raise ValueError(f"prompts must be non-empty, non-blank strings, got {prompts!r}")


class PromptState:
    """Thread-safe current prompt list, swappable at runtime without restart."""

    def __init__(self, initial_prompts: list[str]):
        _validate_prompts(initial_prompts)
        self._lock = threading.Lock()
        self._prompts = list(initial_prompts)

    def get(self) -> list[str]:
        with self._lock:
            return list(self._prompts)

    def set(self, prompts: list[str]) -> None:
        _validate_prompts(prompts)
        with self._lock:
            self._prompts = list(prompts)


def open_camera(device_index: int = 0):
    """Opens cv2.VideoCapture(device_index); raises RuntimeError with a
    clear message on failure (permission denied, no device, in use, etc.)
    instead of returning a capture object silently unusable."""
    import cv2

    cap = cv2.VideoCapture(device_index)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            f"could not open camera {device_index}; check that this process has "
            "camera permission (macOS: System Settings > Privacy & Security > "
            "Camera) and no other app is using the camera"
        )
    return cap


def camera_frame_source(device_index: int = 0):
    """Generator yielding RGB frames from the real camera. Always releases
    the capture on exit (normal exhaustion, early .close(), or exception),
    because the cleanup lives in a finally block around the yield loop."""
    import cv2

    cap = open_camera(device_index)
    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                raise RuntimeError("camera read failed mid-stream")
            yield cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def run_loop(pipeline, prompt_state: PromptState, frame_source, on_result, should_stop) -> int:
    """Core loop: pull a frame, run the pipeline with the *current* prompts
    (read fresh every iteration, so prompt_state.set() takes effect on the
    next frame without restarting anything), hand the result to on_result,
    stop when should_stop() is true. Returns the number of frames processed.

    frame_source must be an iterable (typically a generator like
    camera_frame_source()) that performs its own cleanup in a finally block
    around its yields -- breaking out of the for-loop below closes it.
    """
    frames_processed = 0
    try:
        for frame in frame_source:
            result = pipeline.run(frame, prompt_state.get())
            on_result(frame, result)
            frames_processed += 1
            if should_stop():
                break
    finally:
        close = getattr(frame_source, "close", None)
        if close is not None:
            close()  # explicit, not reliant on GC timing: runs frame_source's finally block
    return frames_processed


def camera_demo_main(pipeline, initial_prompts: list[str], device_index: int = 0) -> None:
    """Thin real-hardware wiring: opens the camera, displays boxes/masks
    with cv2.imshow, reads new prompts from stdin in a background thread
    (comma-separated, one line per change), and exits cleanly on 'q' or
    camera failure. Not unit-tested -- genuinely needs a camera and display.
    """
    import cv2

    prompt_state = PromptState(initial_prompts)
    stop_event = threading.Event()

    def read_stdin_prompts():
        import sys

        for line in sys.stdin:
            parsed = [p.strip() for p in line.split(",") if p.strip()]
            if not parsed:
                continue  # blank/comma-only input: ignore, don't crash this thread
            prompt_state.set(parsed)

    threading.Thread(target=read_stdin_prompts, daemon=True).start()

    def on_result(frame_rgb, result):
        from smart_processing.pipeline import render_overlay_image

        overlay_rgb = render_overlay_image(frame_rgb, result)
        cv2.imshow("smart-processing", cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            stop_event.set()

    try:
        run_loop(
            pipeline,
            prompt_state,
            camera_frame_source(device_index),
            on_result,
            stop_event.is_set,
        )
    finally:
        cv2.destroyAllWindows()
