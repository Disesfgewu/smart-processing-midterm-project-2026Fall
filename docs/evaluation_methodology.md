# Detection evaluation methodology (Issue #9)

Defines the accuracy metric and box-matching rules **before** any scoring
happens (#10 implements the scorer against this spec; this issue only
defines it and prepares the dataset it will run on).

## Ground-truth dataset format

`tests/data/evaluation/annotations.json`: a list of entries, one per
(image, prompt-style) pair --

```json
{
  "image": "relative/path/to/image.jpg",
  "image_license": "CC BY-SA 4.0 (Wikimedia Commons, verified per-file via the Commons API)",
  "category": "similar_objects | small_objects | occlusion",
  "prompt_style": "basic_noun | descriptive_phrase | fine_grained",
  "prompt": "a cat",
  "ground_truth_boxes": [
    {"label": "cat", "box_xyxy": [10.0, 20.0, 100.0, 150.0]}
  ]
}
```

- `box_xyxy` is in **pixel coordinates** of the image at its original
  (downloaded) resolution, matching #3's `BoundingBox` contract.
- `ground_truth_boxes` can be empty (a prompt that should correctly find
  nothing in that image -- a valid, useful test case).
- Each image appears once per prompt style it's annotated for (at least 3
  prompt styles per image, per #9's acceptance), so one image contributes
  multiple annotation entries.

## Detection accuracy metric

For one (image, prompt) entry, given the detector's predicted boxes (each
with a label and score) and the ground-truth boxes:

1. **Matching rule**: a predicted box `P` matches a ground-truth box `G` if
   and only if:
   - `IoU(P, G) >= 0.5` (standard PASCAL VOC / COCO-style threshold), and
   - `P.label` equals the prompt's expected label (case-insensitive,
     stripped) -- there is no multi-class confusion matrix here since each
     entry is scored against one target prompt at a time.
2. **Greedy one-to-one matching**: sort predictions by descending score;
   for each prediction in that order, match it to the highest-IoU
   unmatched ground-truth box that clears the threshold. Each ground-truth
   box can be matched at most once (a second prediction overlapping an
   already-matched box counts as a false positive, not a second true
   positive) -- this mirrors COCO/VOC evaluation and prevents trivially
   inflating recall by emitting many overlapping boxes.
3. **Per-entry counts**:
   - `true_positives` = number of matched pairs.
   - `false_positives` = predicted boxes with no match.
   - `false_negatives` = ground-truth boxes with no match.
4. **Per-entry metrics**: `precision = TP / (TP + FP)`, `recall = TP / (TP + FN)`,
   `f1 = 2 * precision * recall / (precision + recall)` (all defined as `0`
   when their denominator is `0`, except precision/recall are both defined
   as `1.0` when an entry has zero ground-truth boxes and zero predictions
   -- a correct "nothing here" result).
5. **Aggregate metrics**: micro-averaged (sum TP/FP/FN across all entries,
   then compute precision/recall/F1 once) rather than macro-averaged
   (averaging per-entry F1 scores), so entries with more ground-truth boxes
   contribute proportionally more weight -- matches how the manufacturing
   counting scenario (#8) cares about absolute counts, not per-image
   consistency between a 1-object and a 20-object test image.
6. **Per-category and per-prompt-style breakdowns**: report the same
   micro-averaged metrics separately for each `category`
   (similar_objects/small_objects/occlusion) and each `prompt_style`, so
   #10's failure analysis can attribute accuracy drops to a specific
   category or prompt phrasing rather than only an overall number.

## Why this instead of full COCO mAP

Full mAP (averaging precision-recall curves across many IoU and confidence
thresholds) needs enough images per category to be statistically
meaningful; #9's minimum 10 images spread across 3 categories and 3+ prompt
styles is too small a sample for a stable mAP curve. A single-threshold
precision/recall/F1 at `IoU >= 0.5` is simpler to compute, simpler to
sanity-check by hand against the ground-truth boxes, and is standard
practice for small, hand-curated evaluation sets.

## Dataset preparation status

See `tests/data/evaluation/README.md` for the current image count, sourcing
method and per-image license verification, and what remains before the
full 10-image/3-category/3-prompt-style minimum is met.
