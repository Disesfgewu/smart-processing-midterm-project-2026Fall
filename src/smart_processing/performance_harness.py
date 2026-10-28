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


def measure_latency(fn, warmup: int = 2, samples: int = 10, synchronize=None) -> LatencyStats:
    """Times `fn()` `warmup + samples` times using a monotonic clock
    (`time.perf_counter`, immune to wall-clock adjustments -- the correct
    choice for latency measurement). Discards the warmup timings. Raises on
    non-positive `samples` rather than returning a stats object with no
    samples to compute statistics.mean/median/stdev from.

    `synchronize`, if given, is called immediately after `fn()` but still
    inside the timed window, every warmup and timed iteration. This
    version has no synchronization of its own: `fn()` is assumed to block
    until its work is truly complete (true for synchronous CPU calls).
    For an accelerator backend that submits async work and returns early
    (e.g. unsynchronized CUDA/MPS kernel launches), pass the real
    completion barrier here (e.g. `torch.mps.synchronize`) -- otherwise
    this measures enqueue time, not completion time, and warmup work can
    leak across the warmup/timed boundary. PM-Reviwer finding on #11;
    `run_benchmark()` enforces this explicitly for `device != "cpu"`."""
    if samples <= 0:
        raise ValueError(f"samples must be positive, got {samples}")
    if warmup < 0:
        raise ValueError(f"warmup must be non-negative, got {warmup}")

    for _ in range(warmup):
        fn()
        if synchronize is not None:
            synchronize()

    timings = []
    for _ in range(samples):
        start = time.perf_counter()
        fn()
        if synchronize is not None:
            synchronize()
        timings.append(time.perf_counter() - start)

    return LatencyStats(samples=tuple(timings), warmup_count=warmup)


@dataclass(frozen=True)
class PeakRssMeasurement:
    """Absolute before/after process peak-RSS readings, not just their
    delta -- PM-Reviwer finding: reporting only a delta silently accepts
    an always-zero implementation (a real allocation that doesn't move
    the process's existing high-water mark reports 0 either way), and
    callers publishing memory numbers need the absolute figures to tell
    "measured zero" apart from "measurement never worked"."""

    before_bytes: int
    after_bytes: int

    @property
    def delta_bytes(self) -> int:
        return max(0, self.after_bytes - self.before_bytes)


def measure_peak_rss_bytes(fn) -> PeakRssMeasurement:
    """Runs `fn()` once and returns the process's peak resident-set size in
    bytes, before vs. after the call. `delta_bytes` is a lower-bound
    estimate of `fn`'s own allocation (shared with everything else
    running in-process, so this is "peak RSS after minus peak RSS
    before," not an isolated measurement -- documented, not hidden, since
    `resource` has no per-call RSS delta). macOS reports `ru_maxrss` in
    bytes; Linux reports it in KiB, so this normalizes by platform to
    avoid silently misreporting units.
    """
    unit_multiplier = 1 if platform.system() == "Darwin" else 1024
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * unit_multiplier
    fn()
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * unit_multiplier
    return PeakRssMeasurement(before_bytes=before, after_bytes=after)


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    platform_info: PlatformInfo
    latency: LatencyStats
    peak_rss: PeakRssMeasurement


def run_benchmark(
    name: str,
    fn,
    device: str = "cpu",
    warmup: int = 2,
    samples: int = 10,
    synchronize=None,
) -> BenchmarkResult:
    """Runs the full measurement sequence for one named stage (e.g.
    "detection", "segmentation", "end_to_end") against `fn`: latency
    (with warmup) first, then one additional call wrapped for the RSS
    measurement -- so the RSS reading doesn't also absorb warmup's
    allocations, and the reported latency isn't inflated by the RSS
    call's own `resource.getrusage` overhead.

    Raises if `device != "cpu"` and no `synchronize` callable is given:
    this version only measures truly synchronous calls correctly (see
    `measure_latency`'s docstring), so an unsynchronized non-CPU device
    claim is rejected rather than silently producing misleading numbers
    (PM-Reviwer finding on #11)."""
    if device != "cpu" and synchronize is None:
        raise ValueError(
            f"device={device!r} requires an explicit synchronize callable "
            "(e.g. torch.mps.synchronize) so async kernel submissions are "
            "fully awaited before the timer stops; pass one or use "
            "device='cpu' for a function known to be fully synchronous"
        )
    latency = measure_latency(fn, warmup=warmup, samples=samples, synchronize=synchronize)
    peak_rss = measure_peak_rss_bytes(fn)
    return BenchmarkResult(
        name=name,
        platform_info=PlatformInfo.capture(device=device),
        latency=latency,
        peak_rss=peak_rss,
    )
