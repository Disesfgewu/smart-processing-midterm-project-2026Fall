"""只跑 SAM（框或點 → 遮罩），不經過 OWL。
  python src/demo_sam.py data/samples/cats.jpg --box 6 53 320 469
  python src/demo_sam.py data/samples/cats.jpg --point 150 250          # 前景點
"""
import argparse
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import torch
from nanosam.mobile_sam import SamPredictor, sam_model_registry

ap = argparse.ArgumentParser()
ap.add_argument("image")
ap.add_argument("--box", type=float, nargs=4, metavar=("X0", "Y0", "X1", "Y1"))
ap.add_argument("--point", type=float, nargs=2, metavar=("X", "Y"))
ap.add_argument("--checkpoint", default="data/models/mobile_sam.pt")
ap.add_argument("--out", default="out_sam.png")
args = ap.parse_args()
assert args.box or args.point, "請指定 --box 或 --point"

sam = SamPredictor(sam_model_registry["vit_t"](checkpoint=args.checkpoint).cuda().eval())
image = np.asarray(PIL.Image.open(args.image).convert("RGB"))

kw = {}
if args.box:
    kw["box"] = np.array(args.box)
if args.point:
    kw["point_coords"] = np.array([args.point]); kw["point_labels"] = np.array([1])

for i in range(2):  # 第 2 次才是暖機後的時間
    t0 = time.perf_counter()
    sam.set_image(image)                       # image encoder（最重）
    torch.cuda.synchronize(); t1 = time.perf_counter()
    masks, scores, _ = sam.predict(multimask_output=False, **kw)   # prompt encoder + mask decoder
    torch.cuda.synchronize(); t2 = time.perf_counter()
print(f"encoder {(t1 - t0) * 1e3:.0f}ms  decoder {(t2 - t1) * 1e3:.0f}ms  iou_pred={scores[0]:.2f}  area={int(masks[0].sum())}px")

plt.figure(figsize=(8, 6)); plt.imshow(image)
plt.imshow(np.ma.masked_where(~masks[0], masks[0]), alpha=0.5, cmap="autumn")
if args.box:
    x0, y0, x1, y1 = args.box; plt.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], "g-")
if args.point:
    plt.plot(*args.point, "g*", ms=15)
plt.axis("off"); plt.savefig(args.out, bbox_inches="tight"); print("saved", args.out)
