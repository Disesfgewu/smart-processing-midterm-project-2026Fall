"""只跑 OWL（文字 → 物件框）：python src/demo_owl.py data/samples/cats.jpg "a cat" "a remote control"
--engine 指定 TensorRT 引擎時用 TensorRT 後端（Jetson）；不指定則用 PyTorch。"""
import argparse
import time

import PIL.Image
import torch
from torchvision.ops import nms
import PIL.ImageDraw
from nanoowl.owl_predictor import OwlPredictor

ap = argparse.ArgumentParser()
ap.add_argument("image")
ap.add_argument("prompts", nargs="+")
ap.add_argument("--threshold", type=float, default=0.1)
ap.add_argument("--engine", default=None)
ap.add_argument("--out", default="out_owl.jpg")
ap.add_argument("--runs", type=int, default=30, help="量測延遲的重複次數（0 = 不量測）")
args = ap.parse_args()

owl = OwlPredictor(image_encoder_engine=args.engine)
image = PIL.Image.open(args.image).convert("RGB")
enc = owl.encode_text(args.prompts)


def run():
    return owl.predict(image=image, text=args.prompts, text_encodings=enc, threshold=args.threshold, pad_square=False)


out = run()
if len(out.boxes):  # 去除同一物件的重疊框
    keep = nms(out.boxes, out.scores, 0.5)
    out.boxes, out.scores, out.labels = out.boxes[keep], out.scores[keep], out.labels[keep]
for b, s, l in zip(out.boxes.tolist(), out.scores.tolist(), out.labels.tolist()):
    print(f"{args.prompts[l]}: {s:.2f} box={[round(v) for v in b]}")

if args.runs:
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(args.runs):
        run()
    torch.cuda.synchronize(); dt = (time.perf_counter() - t0) / args.runs
    print(f"OWL latency: {dt * 1e3:.1f} ms  ({1 / dt:.1f} FPS)  backend={'TensorRT' if args.engine else 'PyTorch'}")

draw = PIL.ImageDraw.Draw(image)
for b, s_, l in zip(out.boxes.tolist(), out.scores.tolist(), out.labels.tolist()):
    draw.rectangle(b, outline=(0, 200, 0), width=3)
    draw.text((b[0] + 3, b[1] + 3), f"{args.prompts[l]} {s_:.2f}", fill=(255, 255, 0))
image.save(args.out)
print("saved", args.out)
