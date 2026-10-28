"""Fast tests (no hardware, no model download): the camera-demo logic that
IS honestly testable without a real camera -- prompt-state thread safety,
camera-open failure handling, and the frame-processing loop's clean
shutdown. camera_demo_main() itself is NOT tested here; it genuinely needs
real camera hardware and display, which this environment doesn't have
authorized (see README/PR for the real, current verification result).
"""
import threading

import numpy as np
import pytest

from smart_processing.camera_demo import PromptState, open_camera, run_loop
from smart_processing.types import DetectionResult


def test_prompt_state_rejects_empty_initial_prompts():
    with pytest.raises(ValueError):
        PromptState([])


def test_prompt_state_get_returns_a_copy():
    state = PromptState(["a cat"])
    prompts = state.get()
    prompts.append("mutated")
    assert state.get() == ["a cat"]  # caller mutating the returned list must not leak back


def test_prompt_state_set_rejects_empty_list():
    state = PromptState(["a cat"])
    with pytest.raises(ValueError):
        state.set([])
    assert state.get() == ["a cat"]  # unchanged


@pytest.mark.parametrize("bad", ["a cat", 123, None, ["a cat", ""], ["a cat", "   "], ["a cat", 5]])
def test_prompt_state_rejects_non_list_or_non_string_items(bad):
    # A bare str is truthy and iterable, so a naive `if not prompts` check
    # would accept PromptState("a cat") and silently explode it into
    # single-character "prompts" -- must be rejected explicitly instead.
    with pytest.raises(ValueError):
        PromptState(bad)
    state = PromptState(["a cat"])
    with pytest.raises(ValueError):
        state.set(bad)
    assert state.get() == ["a cat"]  # unchanged


def test_prompt_state_set_takes_effect_for_next_get():
    state = PromptState(["a cat"])
    state.set(["a dog", "a bird"])
    assert state.get() == ["a dog", "a bird"]


def test_prompt_state_is_thread_safe_under_concurrent_set():
    state = PromptState(["initial"])
    errors = []

    def writer(i):
        try:
            for _ in range(50):
                state.set([f"prompt-{i}"])
        except Exception as e:  # pragma: no cover - failure path
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert state.get()[0].startswith("prompt-")  # ended on some writer's value, not corrupted


class _FakeCv2VideoCapture:
    def __init__(self, opened: bool):
        self._opened = opened
        self.released = False

    def isOpened(self):
        return self._opened

    def release(self):
        self.released = True


def test_open_camera_raises_clear_error_when_not_authorized(monkeypatch):
    import smart_processing.camera_demo as camera_demo_module

    fake_cap = _FakeCv2VideoCapture(opened=False)
    fake_cv2 = type("FakeCv2", (), {"VideoCapture": staticmethod(lambda idx: fake_cap)})
    monkeypatch.setitem(__import__("sys").modules, "cv2", fake_cv2)

    with pytest.raises(RuntimeError, match="could not open camera"):
        open_camera(0)
    assert fake_cap.released  # must not leak a half-open capture on failure


def test_open_camera_returns_capture_when_authorized(monkeypatch):
    fake_cap = _FakeCv2VideoCapture(opened=True)
    fake_cv2 = type("FakeCv2", (), {"VideoCapture": staticmethod(lambda idx: fake_cap)})
    monkeypatch.setitem(__import__("sys").modules, "cv2", fake_cv2)

    assert open_camera(0) is fake_cap


class _StubPipeline:
    def __init__(self):
        self.calls = []

    def run(self, frame, prompts, threshold=0.1):
        self.calls.append((prompts, threshold))
        return DetectionResult(image_width=frame.shape[1], image_height=frame.shape[0], detections=[])


def _fake_frame_source(n_frames, cleanup_log):
    try:
        for _ in range(n_frames):
            yield np.zeros((10, 10, 3), dtype=np.uint8)
    finally:
        cleanup_log.append("cleaned up")


def test_run_loop_processes_each_frame_with_current_prompts():
    pipeline = _StubPipeline()
    prompt_state = PromptState(["a cat"])
    results = []

    def on_result(frame, result):
        results.append(result)
        if len(results) == 2:
            prompt_state.set(["a dog"])  # prompt change mid-stream, no restart

    frames_processed = run_loop(
        pipeline, prompt_state, _fake_frame_source(4, []), on_result, lambda: False
    )

    assert frames_processed == 4
    assert len(results) == 4
    assert [c[0] for c in pipeline.calls] == [["a cat"], ["a cat"], ["a dog"], ["a dog"]]


def test_run_loop_stops_and_cleans_up_frame_source():
    pipeline = _StubPipeline()
    prompt_state = PromptState(["a cat"])
    cleanup_log = []

    count = {"n": 0}

    def should_stop():
        count["n"] += 1
        return count["n"] >= 2  # stop after 2 frames even though source has 10

    run_loop(pipeline, prompt_state, _fake_frame_source(10, cleanup_log), lambda f, r: None, should_stop)

    assert cleanup_log == ["cleaned up"]  # breaking the for-loop closed the generator


def test_run_loop_cleans_up_frame_source_on_exception_in_on_result():
    pipeline = _StubPipeline()
    prompt_state = PromptState(["a cat"])
    cleanup_log = []

    def failing_on_result(frame, result):
        raise RuntimeError("display failed")

    with pytest.raises(RuntimeError, match="display failed"):
        run_loop(pipeline, prompt_state, _fake_frame_source(5, cleanup_log), failing_on_result, lambda: False)

    assert cleanup_log == ["cleaned up"]
