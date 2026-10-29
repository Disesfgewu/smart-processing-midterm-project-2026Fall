"""即時示範：影片 / 攝影機 → 文字提示 → 物件框 + 遮罩（可在執行中換提示）

  python src/app.py --source data/samples/demo.mp4 --prompt "a cat" "a remote control"
  python src/app.py --source 0 --prompt "a screw"                 # 攝影機（Jetson）
  python src/app.py --source video.mp4 --headless --out out.mp4   # 不開視窗，輸出影片

執行中換提示：在「啟動程式的終端機」輸入新提示後按 Enter（逗號分隔多個，例如  a nut, a bolt），
或在視窗按 q 離開。不需重啟程式。
"""
import argparse
import queue
import sys
import threading
import time

import cv2
import numpy as np
import PIL.Image

sys.path.insert(0, "src")
from pipeline import OpenVocabSegmenter

COLORS = [(0, 200, 0), (0, 128, 255), (255, 80, 80), (200, 0, 200), (0, 200, 200), (200, 200, 0)]

ap = argparse.ArgumentParser()
ap.add_argument("--source", default="0", help="攝影機編號或影片路徑")
ap.add_argument("--prompt", nargs="+", required=True)
ap.add_argument("--threshold", type=float, default=0.1)
ap.add_argument("--sam_checkpoint", default="data/models/mobile_sam.pt")
ap.add_argument("--owl_engine", default=None)
ap.add_argument("--style", choices=["mask", "contour", "both"], default="both", help="遮罩填色 / 只畫輪廓線 / 兩者")
ap.add_argument("--headless", action="store_true")
ap.add_argument("--out", default=None, help="輸出影片路徑")
ap.add_argument("--max_frames", type=int, default=None)
args = ap.parse_args()

prompt_queue: "queue.Queue[list]" = queue.Queue()


def read_stdin():
    for line in sys.stdin:
        prompts = [p.strip() for p in line.split(",") if p.strip()]
        if prompts:
            prompt_queue.put(prompts)


threading.Thread(target=read_stdin, daemon=True).start()


def draw(frame, dets, prompts, timing, fps):
    if args.style in ("mask", "both"):
        overlay = frame.copy()
        for d in dets:
            overlay[d.mask] = COLORS[prompts.index(d.label) % len(COLORS)]
        frame = cv2.addWeighted(overlay, 0.35 if args.style == "both" else 0.45, frame, 0.65 if args.style == "both" else 0.55, 0)
    for d in dets:
        c = COLORS[prompts.index(d.label) % len(COLORS)]
        x0, y0, x1, y1 = [int(v) for v in d.box]
        if args.style in ("contour", "both"):
            cv2.polylines(frame, [pts.reshape(-1, 1, 2).astype(np.int32) for pts in d.contours()], True, c, 2, cv2.LINE_AA)
        else:
            cv2.rectangle(frame, (x0, y0), (x1, y1), c, 2)
        cv2.putText(frame, f"{d.label} {d.score:.2f}", (x0, max(y0 - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
    y = 20
    for i, p in enumerate(prompts):  # 各提示詞的數量與總遮罩面積
        mine = [d for d in dets if d.label == p]
        area = int(sum(d.mask.sum() for d in mine))
        cv2.putText(frame, f"{p}: count={len(mine)} area={area}px", (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    COLORS[i % len(COLORS)], 2, cv2.LINE_AA)
        y += 22
    cv2.putText(frame, f"{fps:.1f} FPS  owl {timing.get('owl_ms', 0):.0f}ms  sam {timing.get('sam_ms', 0):.0f}ms",
                (8, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


seg = OpenVocabSegmenter(sam_checkpoint=args.sam_checkpoint, owl_engine=args.owl_engine)
seg.set_prompts(args.prompt)
print("prompts:", seg.prompts, "（輸入新提示後按 Enter 可即時更換）")

cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
if not cap.isOpened():
    sys.exit(f"無法開啟來源: {args.source}")
writer = None
fps_smooth, n = 0.0, 0
while True:
    ok, bgr = cap.read()
    if not ok:
        break
    try:
        seg.set_prompts(prompt_queue.get_nowait())
        print("prompts ->", seg.prompts)
    except queue.Empty:
        pass
    t0 = time.perf_counter()
    dets = seg(PIL.Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), args.threshold)
    dt = time.perf_counter() - t0
    fps_smooth = 1 / dt if n == 0 else 0.9 * fps_smooth + 0.1 / dt
    vis = draw(bgr, dets, seg.prompts, seg.timing, fps_smooth)
    if args.out:
        if writer is None:
            writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), 15, (vis.shape[1], vis.shape[0]))
        writer.write(vis)
    if not args.headless:
        cv2.imshow("open-vocab seg (q: quit)", vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    n += 1
    if args.max_frames and n >= args.max_frames:
        break
print(f"frames={n} avg_fps={fps_smooth:.1f}")
if writer:
    writer.release()
cap.release()
