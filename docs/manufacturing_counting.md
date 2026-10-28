# Manufacturing parts-counting scenario (Issue #8)

## Scope and what's real here

- `count_detections(result, label=None)`: pure, reusable counting over a
  `PipelineResult` (#6), optionally filtered to one label. Fully unit-tested
  without any model.
- `run_counting_demo(pipeline, image, prompts, threshold)`: runs the real
  pipeline once and reports a count per prompt.
- A real end-to-end test (`tests/test_manufacturing_counting_real.py`,
  opt-in via `RUN_FEASIBILITY_SMOKE=1`) runs the actual `OwlViTTextDetector`
  + `SamBoxSegmenter` pipeline on a real image, twice, with two different
  target prompts, and asserts the counts differ — demonstrating "changing
  targets" with real models, not a mock.

## Scene: documented substitute, and three real attempts that didn't work

Issue #8 asks for "one repeatable tabletop manufacturing scene." This
session has no camera (#7) and no physical manufacturing parts, so a
photograph had to be sourced. Three real, explicitly-licensed candidates
were tried and their **actual** OWL-ViT (`google/owlvit-base-patch32`)
detection results recorded — not guessed:

| Candidate image | Source / license | Prompt(s) tried | Actual result |
| --- | --- | --- | --- |
| "Bolts,Nuts & Washers.jpg" (flat-lay hardware) | Wikimedia Commons, CC BY-SA 4.0, verified via Commons API `extmetadata` (not just dataset membership) | "a bolt", "a nut", "a washer", "a screw" | 0-2 boxes per prompt, scores 0.11-0.14 (near the 0.1 threshold) at multiple resize/threshold combinations |
| Worker sorting plastic bottles into sacks | Wikimedia Commons, CC BY-SA 4.0, verified | "a bottle" | 0 boxes at thresholds 0.05-0.2 |
| Open carton of a dozen eggs | Wikimedia Commons, CC BY-SA 4.0, verified | "an egg" | 0 boxes at thresholds 0.05-0.2 |

This is real, useful evidence for issue #10's failure-case analysis: OWL-ViT
base-patch32 struggles with small, tightly-packed, top-down-photographed
objects even after resizing -- consistent with hw.md's expected "small
object" and "similar/cluttered object" failure modes, observed here rather
than assumed.

**Documented substitute scene**: rather than ship a counting demo with
near-zero real detections (which would misrepresent the pipeline's actual
capability), this uses the already-validated #2 sample image (COCO val2017
`000000039769.jpg`, fetched on demand, never committed) with "a photo of a
cat" and "a remote control" as the two countable targets. This is **not** a
manufacturing scene; it is an explicitly labeled stand-in chosen because it
has real, previously-verified high-confidence detections (scores 0.7+ for
cats, 0.23-0.26 for remotes), so the counting/changing-targets mechanics
can be demonstrated with real models instead of a mock.

Actual output from `run_counting_demo` on this exact image, two separate
runs with the same pipeline instance (no restart):

```
prompts=['a photo of a cat']    -> counts={'a photo of a cat': 2}    total=2
prompts=['a remote control']    -> counts={'a remote control': 2}    total=2
```

## Limitations

- A genuine manufacturing-parts tabletop photo with reliable OWL-ViT
  detection is still needed for a complete #8 demo; this PR provides the
  counting logic and demonstrates it works end to end with real models on
  a substitute scene, but does not claim the manufacturing scene itself is
  solved.
- No camera involvement (#7's camera hardware limitation applies equally
  here); this is a static-image demo.
- Counting is exact-label-match only; no deduplication of overlapping boxes
  for the same object (not observed as an issue in the tested runs, but not
  specifically handled).
