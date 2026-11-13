"""失敗模式的受控實驗：小物件（縮小）與遮擋（人工遮蓋）。補 docs/experiments.md 中「沒有受控證據」的兩類。

  python benchmark/bench_failure_modes.py --out data/bench/jetson/failure_modes.json

1) 小物件：把整張影像等比縮小 s 倍後貼在同樣大小的灰色畫布中央，物件在 OWL 輸入（768×768）中的視覺大小就縮小 s 倍（其他都不變）。
   記錄各 s 下：偵測數（門檻 0.1／0.05）、前 N 名（N＝真實數量）的平均分、參考物件「位置也對」（IoU≥0.5）的比例。「視覺大小」＝s=1 時偵測框 sqrt(寬×高) 換算到 768 輸入的像素。
2) 遮擋：以乾淨影像（s=1）中的偵測框當參考；把每個參考框「從左邊起 f 的寬度」整條塗成灰色（f＝0、0.25、0.5、0.75），再偵測。
   「仍偵測到」＝有偵測框與參考框（或其未被遮住的部分）IoU≥0.5；記錄該框的分數與框相對於原框的面積比。
限制：這是**人工**合成（灰色色塊、單一方向），不等於真實遮擋（真實遮擋物有紋理、形狀與前後景關係）；縮小也只改視覺大小，不改細節與模糊。
提示詞用類別式（`a X`）；OWL 為 TensorRT；只要框。
"""
import argparse
import json
import os
import statistics
import sys
import time

import cv2
import numpy as np
import PIL.Image

sys.path.insert(0, "src")
from overlay import draw_overlay  # noqa: E402
from pipeline import OpenVocabSegmenter  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="data/bench/jetson/failure_modes.json")
ap.add_argument("--img_dir", default="data/bench/jetson/failure_modes")
ap.add_argument("--max_side", type=int, default=1280)
ap.add_argument("--owl_engine", default="data/owl_image_encoder_patch32.engine")
ap.add_argument("--power_mode", default="")
args = ap.parse_args()

CASES = [("data/samples/cats.jpg", "cat", "a cat", 2), ("data/samples/cats.jpg", "remote control", "a remote control", 2),
         ("data/samples/eval/buns_f099.jpg", "steamed bun", "a steamed bun", 2), ("data/samples/eval/limes_f346_crop.jpg", "citrus fruit", "a citrus fruit", 3)]
SCALES = [1.0, 0.7, 0.5, 0.35, 0.25, 0.18, 0.125]
OCC = [0.0, 0.25, 0.5, 0.75]
seg = OpenVocabSegmenter(owl_engine=args.owl_engine)
os.makedirs(args.img_dir, exist_ok=True)


def load(path):
    im = PIL.Image.open(path).convert("RGB")
    if max(im.size) > args.max_side:
        k = args.max_side / max(im.size); im = im.resize((round(im.width * k), round(im.height * k)), PIL.Image.LANCZOS)
    return im


def detect(im, concept, query):
    seg.set_prompts([concept], groups={concept: [query]})
    d = seg(im, threshold=0.01, with_masks=False)
    return sorted(((float(x.score), [float(v) for v in x.box]) for x in d), key=lambda t: -t[0])


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]); i = max(0, x1 - x0) * max(0, y1 - y0)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0


def shrink(im, s):
    if s == 1.0:
        return im
    W, H = im.size
    canvas = PIL.Image.new("RGB", (W, H), (127, 127, 127))
    small = im.resize((max(1, round(W * s)), max(1, round(H * s))), PIL.Image.LANCZOS)
    canvas.paste(small, ((W - small.width) // 2, (H - small.height) // 2))
    return canvas


def map_box(b, s, W, H):
    """原影像座標的框 → 縮小 s 倍後貼在畫布中央的座標。"""
    return [(b[0] - W / 2) * s + W / 2, (b[1] - H / 2) * s + H / 2, (b[2] - W / 2) * s + W / 2, (b[3] - H / 2) * s + H / 2]


def draw(im, dets, name, thr=0.1):
    bgr = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
    for sc, b in dets:
        if sc >= thr:
            cv2.rectangle(bgr, tuple(int(v) for v in b[:2]), tuple(int(v) for v in b[2:]), (0, 255, 0), 2)
            cv2.putText(bgr, f"{sc:.2f}", (int(b[0]), max(int(b[1]) - 4, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.imwrite(os.path.join(args.img_dir, name), bgr)


small_rows, occ_rows = [], []
for path, concept, query, gt in CASES:
    base = load(path); W, H = base.size
    tag = os.path.splitext(os.path.basename(path))[0] + "_" + concept.replace(" ", "")
    clean = detect(base, concept, query)
    top = clean[:gt]
    ref_boxes = [b for sc, b in top]
    vis = [float(np.sqrt((b[2] - b[0]) * (b[3] - b[1])) / max(W, H) * 768) for b in ref_boxes]     # s=1 的視覺大小（768 輸入的像素）
    print(f"{tag}: clean top{gt} scores {[round(sc, 3) for sc, _ in top]} apparent size px {[round(v) for v in vis]}", flush=True)
    for s in SCALES:
        dets = detect(shrink(base, s), concept, query)
        sc = [x[0] for x in dets]
        small_rows.append({"image": path, "concept": concept, "scale": s, "gt": gt, "apparent_size_px": round(statistics.fmean(vis) * s, 1),
                           "count_ge_0.1": sum(1 for x in sc if x >= 0.1), "count_ge_0.05": sum(1 for x in sc if x >= 0.05),
                           "mean_top_gt_score": round(statistics.fmean(sc[:gt]), 4) if len(sc) >= gt else round(statistics.fmean(sc + [0] * (gt - len(sc))), 4),
                           "found_gt_ge_0.1": min(gt, sum(1 for x in sc if x >= 0.1)) / gt,
                           # 位置也要對：參考物件（縮小後的位置）有幾個被 IoU>=0.5、分數>=0.1 的偵測框蓋到
                           "located_ge_0.1": sum(1 for rb in ref_boxes if any(sc_ >= 0.1 and iou(d_, map_box(rb, s, W, H)) >= 0.5 for sc_, d_ in dets)) / gt,
                           "located_ge_0.05": sum(1 for rb in ref_boxes if any(sc_ >= 0.05 and iou(d_, map_box(rb, s, W, H)) >= 0.5 for sc_, d_ in dets)) / gt})
        if s in (1.0, 0.35, 0.18):
            draw(shrink(base, s), dets, f"{tag}_scale{s}.jpg")
    for f in OCC:
        arr = np.asarray(base).copy()
        for b in ref_boxes:
            x0, y0, x1, y1 = [int(round(v)) for v in b]
            arr[max(y0, 0):y1, max(x0, 0):int(round(x0 + f * (x1 - x0)))] = 127
        im = PIL.Image.fromarray(arr)
        dets = detect(im, concept, query)
        per = []
        for b in ref_boxes:
            visb = [b[0] + f * (b[2] - b[0]), b[1], b[2], b[3]]
            cands = [(sc, d) for sc, d in dets if iou(d, b) >= 0.5 or iou(d, visb) >= 0.5]
            if cands:
                sc, d = max(cands, key=lambda t: t[0])
                per.append({"score": round(sc, 4), "area_ratio_vs_full": round((d[2] - d[0]) * (d[3] - d[1]) / ((b[2] - b[0]) * (b[3] - b[1])), 3), "found_ge_0.1": sc >= 0.1})
            else:
                per.append({"score": 0.0, "area_ratio_vs_full": None, "found_ge_0.1": False})
        occ_rows.append({"image": path, "concept": concept, "occluded_fraction": f, "gt": gt, "per_object": per,
                         "mean_score": round(statistics.fmean(p["score"] for p in per), 4), "found_rate_ge_0.1": round(sum(p["found_ge_0.1"] for p in per) / len(per), 3),
                         "false_extra_ge_0.1": sum(1 for sc, d in dets if sc >= 0.1 and not any(iou(d, b) >= 0.5 or iou(d, [b[0] + f * (b[2] - b[0]), b[1], b[2], b[3]]) >= 0.5 for b in ref_boxes))})
        if f in (0.0, 0.5, 0.75):
            draw(im, dets, f"{tag}_occ{int(f * 100)}.jpg")
    print("done", tag, flush=True)

json.dump({"date": time.strftime("%Y-%m-%d %H:%M:%S"), "power_mode": args.power_mode, "note": "synthetic: shrink onto gray canvas; gray-patch occlusion", "small_objects": small_rows, "occlusion": occ_rows},
          open(args.out, "w"), ensure_ascii=False, indent=1)
print("saved", args.out)
