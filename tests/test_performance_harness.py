"""Tests for #11's performance measurement harness. Synthetic callables
only -- no model coupling here (that's a follow-up PR's real-hardware run)."""
import os
import platform
import time

import pytest

import smart_processing.performance_harness as performance_harness
from smart_processing.performance_harness import (
    BenchmarkResult,
    LatencyStats,
    PeakRssMeasurement,
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


def test_measure_latency_aggregation_is_correct_with_controlled_clock(monkeypatch):
    """PM-Reviwer finding: a real-sleep test with a tight upper bound is
    flaky under scheduler jitter (observed mean 0.1355s/0.1678s against an
    asserted <=0.1s for a 20ms sleep). Deterministic coverage of the
    aggregation/warmup-boundary logic belongs in a fake-clock test, not a
    real-time assertion. perf_counter is called only around the timed
    samples (start, then after fn()) -- warmup iterations call fn() but
    not the clock -- so 3 timed samples need 6 fake clock values."""
    fake_clock = iter([0.00, 0.01, 1.00, 1.02, 2.00, 2.04])
    monkeypatch.setattr(performance_harness.time, "perf_counter", lambda: next(fake_clock))
    stats = measure_latency(lambda: None, warmup=1, samples=3)
    assert stats.samples == pytest.approx((0.01, 0.02, 0.04))
    assert stats.warmup_count == 1
    assert stats.mean_seconds == pytest.approx(0.07 / 3)


def test_measure_latency_times_a_real_sleep_as_a_loose_smoke_test():
    """Real wall-clock timing of an actual sleep, not a mocked/fabricated
    number -- kept as a loose smoke test (mean roughly in the right
    ballpark, not a tight bound a busy CI runner's scheduler jitter could
    flake on) now that the precise aggregation logic is covered
    deterministically above."""
    stats = measure_latency(lambda: time.sleep(0.02), warmup=1, samples=5)
    assert 0.015 <= stats.mean_seconds <= 1.0


def test_measure_latency_calls_synchronize_after_each_fn_call():
    call_log = []
    stats = measure_latency(
        lambda: call_log.append("fn"),
        warmup=1,
        samples=2,
        synchronize=lambda: call_log.append("sync"),
    )
    assert call_log == ["fn", "sync", "fn", "sync", "fn", "sync"]
    assert len(stats.samples) == 2


def test_run_benchmark_rejects_non_cpu_device_without_synchronize():
    with pytest.raises(ValueError, match="synchronize"):
        run_benchmark("stage", lambda: None, device="mps")


def test_run_benchmark_accepts_non_cpu_device_with_synchronize():
    result = run_benchmark("stage", lambda: None, device="mps", warmup=1, samples=2, synchronize=lambda: None)
    assert result.platform_info.device == "mps"


def test_measure_peak_rss_bytes_reports_plausible_absolute_values():
    def allocate():
        # Real allocation with real (non-zero-page) writes, not fabricated.
        _ = bytearray(os.urandom(10 * 1024 * 1024))

    measurement = measure_peak_rss_bytes(allocate)
    assert isinstance(measurement, PeakRssMeasurement)
    # Any running Python process (with numpy imported) has RSS well above
    # a trivial few KB; this catches an always-zero/broken implementation,
    # which a bare "delta >= 0" assertion would not (PM-Reviwer finding).
    assert measurement.before_bytes > 1_000_000
    assert measurement.after_bytes > 1_000_000
    assert measurement.delta_bytes >= 0


def test_run_benchmark_returns_full_result_shape():
    result = run_benchmark("synthetic_stage", lambda: time.sleep(0.001), device="cpu", warmup=1, samples=3)
    assert isinstance(result, BenchmarkResult)
    assert result.name == "synthetic_stage"
    assert isinstance(result.platform_info, PlatformInfo)
    assert len(result.latency.samples) == 3
    assert isinstance(result.peak_rss, PeakRssMeasurement)
    assert result.peak_rss.delta_bytes >= 0
