"""離線比較「信心訊號」：讀 bench_confidence.py 的輸出 + confidence_cases.json 裡的目視真值框，不跑模型。

  python benchmark/analyze_confidence.py [--backend mobilesam|nanosam] [--prompt plain|template]

正例 = 偵測框與「同標籤」真值框 IoU >= 0.5。指標：
  * AUC：隨機抽一個正例與一個負例，訊號把正例排在前面的機率（0.5 = 沒有資訊，1.0 = 完全分得開），所有圖合併。
  * 每圖每概念的 top-1 是否為正例（= 組內最高分的 primary 是否正確）。
真值是目視估計、圖只有幾張，結果是觀察不是統計。
"""
import argparse
import json
import statistics

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="data/bench/jetson/confidence_baseline.json")
ap.add_argument("--cases", default="benchmark/confidence_cases.json")
ap.add_argument("--backend", default="mobilesam")
ap.add_argument("--prompt", default="plain")
ap.add_argument("--iou", type=float, default=0.5)
args = ap.parse_args()

res = json.load(open(args.results))["images"]
cases = json.load(open(args.cases))


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x1 - x0) * max(0, y1 - y0)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    p, n = np.asarray(pos)[:, None], np.asarray(neg)[None, :]
    return float(((p > n).sum() + 0.5 * (p == n).sum()) / (p.shape[0] * n.shape[1]))


rows = []   # (image, label, label_max_score, features..., is_pos)
for path, entry in res.items():
    gts = cases[path]["gt"]
    w, h = entry["input_size"]
    dets = entry["configs"][f"{args.backend}/{args.prompt}"]["detections"]
    labelled = {l for l, _ in gts}      # 沒標真值的概念（例如 microchip）不納入任何統計
    dets = [d for d in dets if d["label"] in labelled]
    best = {}
    for d in dets:
        best[d["label"]] = max(best.get(d["label"], 0), d["score"])
    for d in dets:
        pos = any(l == d["label"] and iou(g, d["box"]) >= args.iou for l, g in gts)
        af = d["area_px"] / (w * h)
        rows.append(dict(img=path.split("/")[-1], label=d["label"], pos=pos, score=d["score"], sam=d["sam_iou"], area_frac=af,
                         rel=d["score"] / best[d["label"]]))

SIGNALS = {
    "score (現狀)": lambda r: r["score"],
    "sam_iou": lambda r: r["sam"],
    "score × sam_iou": lambda r: r["score"] * r["sam"],
    "score / 組內最高分": lambda r: r["rel"],
    "score × (1-面積比)": lambda r: r["score"] * (1 - r["area_frac"]),
    "score × sam_iou × (1-面積比)": lambda r: r["score"] * r["sam"] * (1 - r["area_frac"]),
    "面積比 (越大越好)": lambda r: r["area_frac"],
}
n_pos = sum(r["pos"] for r in rows)
print(f"後端={args.backend} 提示詞={args.prompt}  偵測 {len(rows)} 個，正例 {n_pos} 個（IoU>={args.iou}、同標籤）")
print(f"{'訊號':32s} {'AUC':>6s}  每圖每概念 top-1 為正例")
for name, f in SIGNALS.items():
    pos = [f(r) for r in rows if r["pos"]]
    neg = [f(r) for r in rows if not r["pos"]]
    ok = tot = 0
    for img in sorted({r["img"] for r in rows}):
        for lab in sorted({r["label"] for r in rows if r["img"] == img}):
            g = [r for r in rows if r["img"] == img and r["label"] == lab]
            tot += 1
            ok += max(g, key=f)["pos"]
    print(f"{name:32s} {auc(pos, neg):6.3f}  {ok}/{tot}")
print("\n逐圖：各概念有沒有任何正例、最高分偵測是否正確")
for img in sorted({r["img"] for r in rows}):
    for lab in sorted({r["label"] for r in rows if r["img"] == img}):
        g = [r for r in rows if r["img"] == img and r["label"] == lab]
        top = max(g, key=lambda r: r["score"])
        found = sorted((r["score"] for r in g if r["pos"]), reverse=True)
        print(f"  {img[:28]:28s} {lab:15s} 偵測{len(g):3d}  最高分={top['score']:.3f}({'對' if top['pos'] else '錯'})  "
              f"真物件分數={[round(x, 3) for x in found[:4]] or '無匹配'}")
