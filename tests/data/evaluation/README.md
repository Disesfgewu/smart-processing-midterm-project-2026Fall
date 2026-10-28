# Evaluation dataset (Issue #9)

Ground-truth annotations for detection accuracy evaluation (#10
implements the scorer against `docs/evaluation_methodology.md`'s spec).

## Status: 7 of 10 images (partial, in progress); all 3 categories now represented

| Image | Category | License | Prompt styles |
| --- | --- | --- | --- |
| `similar_objects_fruits.jpg` | similar_objects | CC0, Wikimedia Commons, verified via Commons API `extmetadata` for `File:Fruits in bowl oranges lime apples (1).jpg`, author Gerardolagunes | basic_noun ("a lime"), fine_grained ("a red and yellow apple"), descriptive_phrase ("a round citrus fruit") |
| `similar_objects_hardware.jpg` | similar_objects | CC BY-SA 4.0, Wikimedia Commons, verified via Commons API `extmetadata` for `File:Bolts,Nuts & Washers.jpg`, author Sbrennan83 | basic_noun ("a bolt"), fine_grained ("a hexagonal nut..."), descriptive_phrase ("a round metal washer") |
| `occlusion_books.jpg` | occlusion | CC0, Wikimedia Commons, verified via Commons API `extmetadata` for `File:Stack of multicolored books on a table.jpg`, author Alexander Grey | basic_noun ("a book"), fine_grained ("a pink hardcover book..."), descriptive_phrase ("a book spine in a stacked pile...") |
| `small_objects_parking.jpg` | small_objects | CC BY-SA 4.0, Wikimedia Commons, verified via Commons API `extmetadata` for `File:Parking lot, Gross-Gerau (P1090665).jpg`, author Matti Blume | basic_noun ("a car"), fine_grained ("a small red car"), descriptive_phrase ("a vehicle parked in a large aerial parking lot") |
| `occlusion_pens.jpg` | occlusion | CC0, Wikimedia Commons, verified via Commons API `extmetadata` for `File:NIND Pen-pile ISO100.jpg`, author MB-one. Downscaled 3996x2998 -> 1600x1200 with macOS `sips` (7.1MB -> 300KB); boxes are in the downscaled coordinate space | basic_noun ("a pencil"), fine_grained ("a yellow marker pen with a cap"), descriptive_phrase ("a writing instrument lying on a wooden table") |
| `small_objects_paperclips.jpg` | small_objects | CC BY-SA 3.0, Wikimedia Commons, verified via Commons API `extmetadata` for `File:Paperclips.jpg`, author Tdlnx. Downscaled 3264x2448 -> 1600x1200 with macOS `sips` (2.9MB -> 421KB); boxes are in the downscaled coordinate space | basic_noun ("a paperclip"), fine_grained ("a gold-toned paperclip"), descriptive_phrase ("a tangled pile of metal paperclips on a table") |
| `similar_objects_mugs.jpg` | similar_objects | CC BY-SA 3.0, Wikimedia Commons, verified via Commons API `extmetadata` for `File:Align (52873932).jpeg`, author Anoop Mukundan. Already small (1152x864, 282KB); no downscaling needed | basic_noun ("a mug"), fine_grained ("a pastel green mug"), descriptive_phrase ("a row of mugs on a shelf") |

Ground-truth boxes in `annotations.json` were determined by manual visual
inspection of each downloaded image (not guessed/synthesized), in pixel
coordinates at each image's native resolution.

**Known precision limitation**: `small_objects_parking.jpg`'s cars are only
~25-30px across in a 2304x1444 aerial photo, and most are white/gray/black
and visually indistinguishable from their neighbors at this scale. Only the
2 red cars that could be confidently and individually located were
annotated; this difficulty is itself relevant evidence for #10's
small-object failure analysis, not hidden.

**Known precision limitation**: `small_objects_paperclips.jpg`'s pile of
~100 paperclips is too densely overlapping in its central cluster to box
individual clips confidently. Only the 5 clearly isolated clips on the
pile's periphery were annotated for the basic_noun/fine_grained prompts;
the descriptive_phrase prompt instead uses one box covering the whole
pile region. Same rationale as the parking-lot limitation above.

**Known precision limitation**: `similar_objects_mugs.jpg` has ~30
visually near-identical mugs packed edge-to-edge across 4 shelf rows.
Only a representative subset (6 clearly separable mugs across different
rows/colors) is individually boxed for the basic_noun prompt; the rest
are too tightly packed for reliable per-mug boxes. Same rationale as
the paperclips/parking-lot limitations above.

## Remaining work

- 3 more images needed to meet #9's 10-image minimum. All 3 required
  categories (`similar_objects`, `small_objects`, `occlusion`) now have at
  least one image; additional images can go to any category for broader
  coverage.
- Each additional image needs 3+ prompt styles and manually-verified
  ground-truth boxes, following the same per-file Wikimedia Commons
  license verification pattern used here and in #8 (never infer license
  from a larger dataset's membership alone -- see #25's review history).
