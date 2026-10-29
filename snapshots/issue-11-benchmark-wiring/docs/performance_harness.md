# Performance measurement harness (Issue #11)

Real measured numbers on macOS; no mocked models or fabricated numbers.
These are **macOS demo data**, not Jetson Orin Nano / TensorRT results --
#11's acceptance explicitly requires keeping the two separate, and no
Jetson hardware is available in this session (see #12/#13/#14).

## Methodology

- Clock: `time.perf_counter` (monotonic, immune to wall-clock adjustments).
- Warmup: the first `warmup` calls are run and discarded before any
  timed sample, so one-time costs (first-call JIT/cache effects) don't
  skew steady-state latency.
- Sampling: `samples` timed calls per stage; reports mean/median/stdev/
  min/max and `fps = 1 / mean_seconds`.
- Memory: `resource.getrusage(RUSAGE_SELF).ru_maxrss` before vs. after
  the call -- the process's peak RSS **high-water mark**, not an isolated
  per-call measurement (`resource` has no per-call RSS delta API), so a
  stage that doesn't push the process to a new peak (an earlier stage,
  e.g. model loading, already set a higher one) reports `0`. A real
  limitation, documented rather than worked around with a heavier
  dependency this issue doesn't need.
- Stages measured separately, per #11's acceptance: `detection` (one
  `OwlViTTextDetector.detect()` call), `segmentation` (one
  `SamBoxSegmenter.segment()` call on a fixed box from a real detection,
  isolating SAM's own cost from OWL-ViT's), `end_to_end` (one full
  `DetectSegmentPipeline.run()` call, detection + segmentation together).

## Platform

- macOS (Darwin), arm64, Python 3.11.16, `.venv`, `pip install -e ".[macos-feasibility]"`.
- `BackendConfig(device="cpu")` -- not using Apple's `mps` backend
  (confirmed available per `docs/macos_feasibility.md`); an `mps` run
  would likely be faster, not claimed here.
- SAM checkpoint: `tests/data/sam_vit_b_01ec64.pth` (official Meta
  `vit_b`, ~375MB, not committed; see `docs/macos_feasibility.md`).

## Real benchmark run

Command: `.venv/bin/python scripts/run_pipeline_benchmarks.py --checkpoint tests/data/sam_vit_b_01ec64.pth --prompt "a photo of a cat" --samples 5`

Image: the same COCO sample image used by #2/#4/#5/#6's real-model tests
(two cats, two remote controls; see `docs/macos_feasibility.md` for its
license-verification status).

| Stage | Mean | Median | FPS | Peak RSS delta |
| --- | --- | --- | --- | --- |
| detection | 112.9ms | 112.8ms | 8.86 | 0.00MB (see methodology note above) |
| segmentation | 1302.6ms | 1302.6ms | 0.77 | 0.00MB |
| end_to_end | 1438.4ms | 1438.3ms | 0.70 | 0.00MB |

SAM's box-prompted segmentation (`vit_b`, CPU) dominates end-to-end
latency by roughly 11.5x over OWL-ViT detection alone on this machine --
expected for a ViT-based segmenter at full precision on CPU, and
consistent with `end_to_end ≈ detection + segmentation` (1438.4ms vs.
112.9 + 1302.6 = 1415.5ms; the small gap is pipeline-level overhead, not
measurement noise beyond what's expected at 5 samples).

Reproduce with `RUN_FEASIBILITY_SMOKE=1 pytest tests/test_run_benchmarks_real.py -v`
(opt-in, same convention as #2/#4/#5/#6; skipped by default since it
downloads/loads real models).

## Known limitations

- CPU-only numbers (no `mps` comparison yet); peak-RSS measures a shared
  process-wide high-water mark, not an isolated per-stage allocation (see
  Methodology above); single machine, single run per stage, no repeated-
  run variance analysis.
- Not Jetson/TensorRT numbers. Final assignment benchmarks require Jetson
  Orin Nano / JetPack 6.2 hardware (#12/#13/#14), unavailable here.
