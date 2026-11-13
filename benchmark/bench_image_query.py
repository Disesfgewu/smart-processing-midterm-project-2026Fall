"""以圖找圖（image-guided）vs 文字查詢：同一批影像、同一套真值，比較「找出同類的其他實例」的能力。

  python benchmark/bench_image_query.py --out data/bench/jetson/image_query.json

協定（避免洩漏）：
  * 只評估「同一張圖裡有 n >= 2 個實例」的概念（cats 的 cat／remote、電路板的 capacitor／inductor、焊接圖的 steel pipe）。
  * 範例 = 從該圖挑 k 個實例（k=1、2，所有組合），查詢特徵由它們產生；評分時**排除範例本身**
    （與範例框 IoU >= 0.3、或框有一半以上落在範例框內的偵測一律丟掉——後者是範例物件自己的子框，不丟會以近滿分變成假警報；文字基準也套同樣的排除，兩邊比較公平），正例 = 與「其他」同概念真值框 IoU >= 0.5。
  * 每次只查單一概念（避免概念之間互相 NMS 干擾），只要框（with_masks=False），OWL 為 TensorRT。
  * 模式：text（"a {概念}"，現狀）、crop、inplace、text+crop（融合）。
  * 負對照：用範例（第 0 個實例）去搜「沒有該概念」的其他圖，記錄最高分，看影像查詢會不會到處亂報。
單一實例的概念（person、rocket、spring…）在同圖內沒有「其他實例」可找，無法用這個協定評估，會列在 JSON 的 not_evaluable。
真值為目視估計（benchmark/confidence_cases.json 的 gt）。
"""
import argparse
import itertools
import json
import os
import statistics
import sys
import time

import numpy as np
import PIL.Image

sys.path.insert(0, "src")
from pipeline import OpenVocabSegmenter  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cases", default="benchmark/confidence_cases.json")
ap.add_argument("--out", default="data/bench/jetson/image_query.json")
ap.add_argument("--max_side", type=int, default=1280)
ap.add_argument("--owl_engine", default="data/owl_image_encoder_patch32.engine")
ap.add_argument("--power_mode", default="")
args = ap.parse_args()

cases = json.load(open(args.cases))
cases = {k: v for k, v in cases.items() if k.startswith("data/")}
seg = OpenVocabSegmenter(owl_engine=args.owl_engine)


def load(path):
    im = PIL.Image.open(path).convert("RGB")
    if max(im.size) > args.max_side:
        s = args.max_side / max(im.size)
        im = im.resize((round(im.width * s), round(im.height * s)), PIL.Image.LANCZOS)
    return im


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x1 - x0) * max(0, y1 - y0)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def inside(det, ex):
    """偵測框有多少比例落在範例框內（排除範例物件自己的子框，否則它們會以接近滿分的相似度變成假警報）。"""
    x0, y0, x1, y1 = max(det[0], ex[0]), max(det[1], ex[1]), min(det[2], ex[2]), min(det[3], ex[3])
    return max(0, x1 - x0) * max(0, y1 - y0) / max(1e-6, (det[2] - det[0]) * (det[3] - det[1]))


def is_exemplar(det, ex):
    return iou(ex, det) >= 0.3 or inside(det, ex) >= 0.5


def auc(pos, neg):
    if not pos or not neg:
        return None
    p, n = np.asarray(pos)[:, None], np.asarray(neg)[None, :]
    return float(((p > n).sum() + 0.5 * (p == n).sum()) / (p.shape[0] * n.shape[1]))


images = {p: load(p) for p in cases}
MODES = ["text", "crop", "inplace", "text+crop"]
setup_ms = {m: [] for m in MODES}


def configure(mode, concept, image, boxes):
    cur_mode[0] = mode
    t0 = time.perf_counter()
    if mode == "text":
        seg.set_prompts([concept])
    else:
        seg.set_image_queries({concept: [(image, b) for b in boxes]}, mode="inplace" if mode == "inplace" else "crop", with_text=(mode == "text+crop"))
    setup_ms[mode].append((time.perf_counter() - t0) * 1e3)


owl_ms = {m: [] for m in MODES}
cur_mode = [None]


def detect(image):
    d = seg(image, threshold=0.0, with_masks=False)       # 內部以 0.01 取回；0.0 = 不再過濾
    owl_ms[cur_mode[0]].append(seg.timing["official_owl_ms"])
    return d


records, not_eval, controls = [], [], []
for path, case in cases.items():
    image = images[path]
    by_label = {}
    for lab, box in case["gt"]:
        by_label.setdefault(lab, []).append(box)
    for lab, boxes in by_label.items():
        n = len(boxes)
        if n < 2:
            not_eval.append({"image": path, "concept": lab, "instances": n})
            continue
        for k in (1, 2):
            if n - k < 1:
                continue
            for ex in itertools.combinations(range(n), k):
                ex_boxes = [boxes[i] for i in ex]
                rest = [boxes[i] for i in range(n) if i not in ex]
                for mode in MODES:
                    configure(mode, lab, image, ex_boxes)
                    dets = detect(image)
                    keep = [d for d in dets if not any(is_exemplar(d.box.tolist(), eb) for eb in ex_boxes)]
                    keep.sort(key=lambda d: -d.score)
                    is_pos = [any(iou(g, d.box.tolist()) >= 0.5 for g in rest) for d in keep]
                    m = len(rest)
                    ranks = []                                  # 每個「其他實例」第一個匹配偵測的名次（1 起算；找不到 = None）
                    for g in rest:
                        r = next((i + 1 for i, d in enumerate(keep) if iou(g, d.box.tolist()) >= 0.5), None)
                        ranks.append(r)
                    records.append({
                        "image": path, "concept": lab, "n_instances": n, "shots": k, "exemplar_idx": list(ex), "mode": mode,
                        "n_targets": m, "n_dets": len(keep),
                        "precision_at_m": sum(is_pos[:m]) / m,
                        "found_in_top_m": sum(1 for r in ranks if r is not None and r <= m) / m,
                        "ranks": ranks,
                        "auc": auc([d.score for d, p in zip(keep, is_pos) if p], [d.score for d, p in zip(keep, is_pos) if not p]),
                        "pos_scores": [round(d.score, 4) for d, p in zip(keep, is_pos) if p],
                        "max_neg_score": round(max([d.score for d, p in zip(keep, is_pos) if not p], default=0), 4),
                        "top_score": round(keep[0].score, 4) if keep else None,
                    })
        # 負對照：第 0 個實例 → 搜其他圖（那些圖沒有此概念）
        for other, oimg in images.items():
            if other == path or any(l == lab for l, _ in cases[other]["gt"]):
                continue
            for mode in MODES:
                configure(mode, lab, image, [boxes[0]])
                if mode in ("inplace",):
                    continue            # inplace 的範例特徵取自「範例所在圖」，不能拿去搜別張圖
                dets = detect(oimg)
                controls.append({"concept": lab, "from": path, "on": other, "mode": mode, "max_score": round(max((d.score for d in dets), default=0), 4)})
    print("done", path, flush=True)


def agg(rows, key):
    v = [r[key] for r in rows if r[key] is not None]
    return round(statistics.fmean(v), 3) if v else None


summary = {}
for lab_path in sorted({(r["image"], r["concept"]) for r in records}):
    for k in (1, 2):
        for mode in MODES:
            rows = [r for r in records if (r["image"], r["concept"]) == lab_path and r["shots"] == k and r["mode"] == mode]
            if rows:
                allranks = [x for r in rows for x in r["ranks"]]
                summary[f"{os.path.basename(lab_path[0])}|{lab_path[1]}|{k}-shot|{mode}"] = {
                    "runs": len(rows), "precision_at_m": agg(rows, "precision_at_m"), "found_in_top_m": agg(rows, "found_in_top_m"),
                    "auc": agg(rows, "auc"), "median_rank_of_targets": (statistics.median([x for x in allranks if x is not None]) if any(x is not None for x in allranks) else None),
                    "targets_never_found": sum(1 for x in allranks if x is None), "targets_total": len(allranks),
                    "mean_pos_score": round(statistics.fmean([s for r in rows for s in r["pos_scores"]]), 3) if any(r["pos_scores"] for r in rows) else None,
                    "mean_max_neg_score": agg(rows, "max_neg_score")}
out = {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "power_mode": args.power_mode, "max_side": args.max_side, "owl_engine": args.owl_engine,
       "setup_ms_mean": {m: (round(statistics.fmean(v), 1) if v else None) for m, v in setup_ms.items()},
       "owl_ms_median": {m: (round(statistics.median(v), 1) if v else None) for m, v in owl_ms.items()},
       "summary": summary, "controls": controls, "not_evaluable": not_eval, "records": records}
os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=1)
print("saved", args.out, len(records), "records")
