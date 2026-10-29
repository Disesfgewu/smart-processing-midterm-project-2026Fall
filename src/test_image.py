"""單張影像測試：python src/test_image.py data/samples/cats.jpg "a cat" "a remote control" """
import argparse
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image

sys.path.insert(0, "src")
from pipeline import OpenVocabSegmenter

ap = argparse.ArgumentParser()
ap.add_argument("image")
ap.add_argument("prompts", nargs="+")
ap.add_argument("--threshold", type=float, default=0.1)
ap.add_argument("--out", default="out.png")
args = ap.parse_args()

seg = OpenVocabSegmenter()
seg.set_prompts(args.prompts)
image = PIL.Image.open(args.image).convert("RGB")
for _ in range(2):  # 第一次含 CUDA 暖機
    dets = seg(image, args.threshold)
    print({k: round(v, 1) for k, v in seg.timing.items()})

plt.figure(figsize=(10, 8)); plt.imshow(image)
for d in dets:
    print(f"{d.label}: {d.score:.2f} box={d.box.round(0)} area={int(d.mask.sum())}px")
    x0, y0, x1, y1 = d.box
    plt.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], "g-")
    plt.text(x0, y0, f"{d.label} {d.score:.2f}", color="w", backgroundcolor="g")
    plt.imshow(np.ma.masked_where(~d.mask, d.mask), alpha=0.5, cmap="autumn")
plt.axis("off"); plt.savefig(args.out, bbox_inches="tight")
print("saved", args.out)
