"""以圖找圖對「影像雜訊」的穩定性：同一張圖只改 JPEG 重新壓縮品質，看範例特徵挑選規則（distinct / best_iou / mean）的結果是否變動。

  python benchmark/bench_image_query_robust.py --out data/bench/jetson/image_query_robust.json

對每個（影像、概念、範例）：範例框取自真值；目標＝同圖其他實例。記錄「目標實例的最高分」（越高越好、越穩越好）與 P@m。
品質 orig = 不重新壓縮；99 / 95 / 92 / 85 = 先存成該品質的 JPEG 再讀回。所有規則在同一組影像上比較。只 1-shot、只要框。
"""
import argparse, io, json, statistics, sys, time
import numpy as np
import PIL.Image
sys.path.insert(0, "src")
from pipeline import OpenVocabSegmenter  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cases", default="benchmark/confidence_cases.json")
ap.add_argument("--out", default="data/bench/jetson/image_query_robust.json")
ap.add_argument("--max_side", type=int, default=1280)
ap.add_argument("--owl_engine", default="data/owl_image_encoder_patch32.engine")
args = ap.parse_args()
cases = {k: v for k, v in json.load(open(args.cases)).items() if k.startswith("data/")}
SKIP = {("data/samples/pexels-cmrcn-29224614.jpg", "steel pipe")}
seg = OpenVocabSegmenter(owl_engine=args.owl_engine)


def load(path):
    im = PIL.Image.open(path).convert("RGB")
    if max(im.size) > args.max_side:
        s = args.max_side / max(im.size); im = im.resize((round(im.width * s), round(im.height * s)), PIL.Image.LANCZOS)
    return im


def recompress(im, q):
    if q == "orig":
        return im
    b = io.BytesIO(); im.save(b, "JPEG", quality=q)
    return PIL.Image.open(io.BytesIO(b.getvalue())).convert("RGB")


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]); i = max(0, x1 - x0) * max(0, y1 - y0)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def is_ex(det, ex):
    x0, y0, x1, y1 = max(det[0], ex[0]), max(det[1], ex[1]), min(det[2], ex[2]), min(det[3], ex[3])
    return iou(ex, det) >= 0.3 or max(0, x1 - x0) * max(0, y1 - y0) / max(1e-6, (det[2] - det[0]) * (det[3] - det[1])) >= 0.5


QUAL = ["orig", 99, 95, 92, 85]
RULES = ["distinct", "best_iou", "mean"]
rows = []
for path, case in cases.items():
    base = load(path)
    by = {}
    for l, b in case["gt"]:
        by.setdefault(l, []).append(b)
    for lab, boxes in by.items():
        if len(boxes) < 2 or (path, lab) in SKIP:
            continue
        for q in QUAL:
            im = recompress(base, q)
            for mode in ("crop", "inplace"):
                for rule in RULES:
                    for k, ex in enumerate(boxes):
                        rest = [b for i, b in enumerate(boxes) if i != k]
                        seg.set_image_queries({lab: [(im, ex)]}, mode=mode, select=rule)
                        keep = sorted([d for d in seg(im, threshold=0.0, with_masks=False) if not is_ex(d.box.tolist(), ex)], key=lambda d: -d.score)
                        pos = [any(iou(g, d.box.tolist()) >= 0.5 for g in rest) for d in keep]
                        m = len(rest)
                        tgt = [max([d.score for d in keep if iou(g, d.box.tolist()) >= 0.5], default=0.0) for g in rest]
                        rows.append({"image": path, "concept": lab, "quality": q, "mode": mode, "rule": rule, "exemplar": k,
                                     "p_at_m": sum(pos[:m]) / m, "mean_target_score": float(np.mean(tgt))})
    print("done", path, flush=True)

summary = {}
for mode in ("crop", "inplace"):
    for rule in RULES:
        for q in QUAL:
            r = [x for x in rows if x["mode"] == mode and x["rule"] == rule and x["quality"] == q]
            summary[f"{mode}|{rule}|{q}"] = {"n": len(r), "p_at_m": round(statistics.fmean(x["p_at_m"] for x in r), 3),
                                              "mean_target_score": round(statistics.fmean(x["mean_target_score"] for x in r), 3)}
# 穩定性：同一個（影像、概念、範例、模式、規則）在不同品質下，目標分數的標準差，再取平均（越小越穩）
stab = {}
for mode in ("crop", "inplace"):
    for rule in RULES:
        sds = []
        keys = {(x["image"], x["concept"], x["exemplar"]) for x in rows}
        for kk in keys:
            v = [x["mean_target_score"] for x in rows if (x["image"], x["concept"], x["exemplar"]) == kk and x["mode"] == mode and x["rule"] == rule]
            sds.append(statistics.pstdev(v))
        stab[f"{mode}|{rule}"] = round(statistics.fmean(sds), 3)
json.dump({"date": time.strftime("%Y-%m-%d %H:%M:%S"), "qualities": QUAL, "summary": summary, "score_std_across_quality": stab, "rows": rows},
          open(args.out, "w"), ensure_ascii=False, indent=1)
print("saved", args.out)
for k, v in stab.items():
    print("score std across JPEG quality", k, v)
