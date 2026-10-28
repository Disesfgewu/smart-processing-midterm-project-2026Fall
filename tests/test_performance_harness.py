"""Tests for #11's performance measurement harness. Synthetic callables
only -- no model coupling here (that's a follow-up PR's real-hardware run)."""
import platform
import time

import pytest

from smart_processing.performance_harness import (
    BenchmarkResult,
    LatencyStats,
    PlatformInfo,
    measure_latency,
    measure_peak_rss_bytes,
    run_benchmark,
)


def test_platform_info_capture_reports_real_values():
    info = PlatformInfo.capture(device="cpu")
    assert info.system == platform.system()
    assert info.machine == platform.machine()
    assert info.python_version == platform.python_version()
    assert info.device == "cpu"


def test_measure_latency_rejects_non_positive_samples():
    with pytest.raises(ValueError):
        measure_latency(lambda: None, samples=0)


def test_measure_latency_rejects_negative_warmup():
    with pytest.raises(ValueError):
        measure_latency(lambda: None, warmup=-1, samples=1)


def test_measure_latency_excludes_warmup_from_samples():
    call_count = {"n": 0}

    def fn():
        call_count["n"] += 1

    stats = measure_latency(fn, warmup=3, samples=5)
    assert call_count["n"] == 8  # 3 warmup + 5 timed
    assert len(stats.samples) == 5
    assert stats.warmup_count == 3


def test_latency_stats_derived_properties():
    stats = LatencyStats(samples=(0.1, 0.2, 0.3), warmup_count=0)
    assert stats.mean_seconds == pytest.approx(0.2)
    assert stats.median_seconds == pytest.approx(0.2)
    assert stats.min_seconds == pytest.approx(0.1)
    assert stats.max_seconds == pytest.approx(0.3)
    assert stats.stdev_seconds > 0
    assert stats.fps == pytest.approx(1.0 / 0.2)


def test_latency_stats_single_sample_has_zero_stdev():
    stats = LatencyStats(samples=(0.1,), warmup_count=0)
    assert stats.stdev_seconds == 0.0


def test_measure_latency_times_a_real_sleep_within_tolerance():
    """Real wall-clock timing of an actual sleep, not a mocked/fabricated
    number -- the measured mean must be close to the real sleep duration."""
    stats = measure_latency(lambda: time.sleep(0.02), warmup=1, samples=5)
    assert 0.015 <= stats.mean_seconds <= 0.1  # generous upper bound for CI jitter


def test_measure_peak_rss_bytes_is_non_negative_for_real_allocation():
    def allocate():
        # Real allocation, not fabricated -- big enough to plausibly move
        # peak RSS, though this function tolerates it not moving (shared
        # process-wide high-water mark, not an isolated measurement).
        _ = bytearray(10 * 1024 * 1024)

    delta = measure_peak_rss_bytes(allocate)
    assert delta >= 0


def test_run_benchmark_returns_full_result_shape():
    result = run_benchmark("synthetic_stage", lambda: time.sleep(0.001), device="cpu", warmup=1, samples=3)
    assert isinstance(result, BenchmarkResult)
    assert result.name == "synthetic_stage"
    assert isinstance(result.platform_info, PlatformInfo)
    assert len(result.latency.samples) == 3
    assert result.peak_rss_delta_bytes >= 0
