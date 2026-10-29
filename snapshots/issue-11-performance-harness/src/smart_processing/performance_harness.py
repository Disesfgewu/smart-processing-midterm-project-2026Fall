"""Reproducible performance measurement harness (#11).

Pure timing/memory utilities: no model coupling here, so they can be
tested cheaply with synthetic callables. A follow-up PR wires these
against the real detection/segmentation/pipeline calls (#4/#5/#6) to
produce macOS demo numbers, kept explicitly separate from any eventual
Jetson results (#11's acceptance: "separate macOS demo data from final
Jetson results").
"""
from __future__ import annotations

import platform
import resource
import statistics
import time
from dataclasses import dataclass, field


@dataclass(frozen=True)
class PlatformInfo:
    """Records the run's platform/configuration so numbers from different
    machines are never silently compared as if equivalent."""

    system: str
    machine: str
    python_version: str
    device: str

    @staticmethod
    def capture(device: str = "cpu") -> "PlatformInfo":
        return PlatformInfo(
            system=platform.system(),
            machine=platform.machine(),
            python_version=platform.python_version(),
            device=device,
        )


@dataclass(frozen=True)
class LatencyStats:
    """Per-call latencies in seconds, plus derived stats. `samples` excludes
    the warmup iterations (a cold first call -- model/JIT/cache warmup --
    would otherwise skew mean/median away from steady-state behavior)."""

    samples: tuple[float, ...]
    warmup_count: int

    @property
    def mean_seconds(self) -> float:
        return statistics.mean(self.samples)

    @property
    def median_seconds(self) -> float:
        return statistics.median(self.samples)

    @property
    def stdev_seconds(self) -> float:
        return statistics.stdev(self.samples) if len(self.samples) > 1 else 0.0

    @property
    def min_seconds(self) -> float:
        return min(self.samples)

    @property
    def max_seconds(self) -> float:
        return max(self.samples)

    @property
    def fps(self) -> float:
        """1 / mean latency. Undefined (0.0) if mean latency is 0, which
        only happens for a measured function that does genuinely nothing."""
        mean = self.mean_seconds
        return 1.0 / mean if mean > 0 else 0.0


def measure_latency(fn, warmup: int = 2, samples: int = 10) -> LatencyStats:
    """Times `fn()` `warmup + samples` times using a monotonic clock
    (`time.perf_counter`, immune to wall-clock adjustments -- the correct
    choice for latency measurement). Discards the warmup timings. Raises on
    non-positive `samples` rather than returning a stats object with no
    samples to compute statistics.mean/median/stdev from."""
    if samples <= 0:
        raise ValueError(f"samples must be positive, got {samples}")
    if warmup < 0:
        raise ValueError(f"warmup must be non-negative, got {warmup}")

    for _ in range(warmup):
        fn()

    timings = []
    for _ in range(samples):
        start = time.perf_counter()
        fn()
        timings.append(time.perf_counter() - start)

    return LatencyStats(samples=tuple(timings), warmup_count=warmup)


def measure_peak_rss_bytes(fn) -> int:
    """Runs `fn()` once and returns the process's peak resident-set size in
    bytes, before vs. after the call, as a lower-bound estimate of `fn`'s
    own allocation (shared with everything else running in-process, so this
    is "peak RSS after minus peak RSS before," not an isolated measurement
    -- documented, not hidden, since `resource` has no per-call RSS delta).
    macOS reports `ru_maxrss` in bytes; Linux reports it in KiB, so this
    normalizes by platform to avoid silently misreporting units.
    """
    unit_multiplier = 1 if platform.system() == "Darwin" else 1024
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * unit_multiplier
    fn()
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * unit_multiplier
    return max(0, after - before)


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    platform_info: PlatformInfo
    latency: LatencyStats
    peak_rss_delta_bytes: int


def run_benchmark(
    name: str,
    fn,
    device: str = "cpu",
    warmup: int = 2,
    samples: int = 10,
) -> BenchmarkResult:
    """Runs the full measurement sequence for one named stage (e.g.
    "detection", "segmentation", "end_to_end") against `fn`: latency
    (with warmup) first, then one additional call wrapped for the RSS
    delta -- so the RSS measurement doesn't also absorb warmup's
    allocations, and the reported latency isn't inflated by the RSS
    call's own `resource.getrusage` overhead."""
    latency = measure_latency(fn, warmup=warmup, samples=samples)
    peak_rss_delta = measure_peak_rss_bytes(fn)
    return BenchmarkResult(
        name=name,
        platform_info=PlatformInfo.capture(device=device),
        latency=latency,
        peak_rss_delta_bytes=peak_rss_delta,
    )
