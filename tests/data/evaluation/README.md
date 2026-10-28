# Evaluation dataset (Issue #9)

Ground-truth annotations for detection accuracy evaluation (#10
implements the scorer against `docs/evaluation_methodology.md`'s spec).

## Status: 1 of 10 images (partial, in progress)

| Image | Category | License | Prompt styles |
| --- | --- | --- | --- |
| `similar_objects_fruits.jpg` | similar_objects | CC0, Wikimedia Commons, verified via Commons API `extmetadata` for `File:Fruits in bowl oranges lime apples (1).jpg`, author Gerardolagunes | basic_noun ("a lime"), fine_grained ("a red and yellow apple"), descriptive_phrase ("a round citrus fruit") |

Ground-truth boxes in `annotations.json` were determined by manual visual
inspection of the downloaded image (not guessed/synthesized), in pixel
coordinates at the image's native 1920x2560 resolution.

## Remaining work

- 9 more images needed to meet #9's 10-image minimum, spanning
  `small_objects` and `occlusion` categories (currently 0 images in
  either) in addition to more `similar_objects` coverage.
- Each additional image needs 3+ prompt styles and manually-verified
  ground-truth boxes, following the same per-file Wikimedia Commons
  license verification pattern used here and in #8 (never infer license
  from a larger dataset's membership alone -- see #25's review history).
