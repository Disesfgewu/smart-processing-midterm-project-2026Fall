"""提示詞寫法比較（題目功能要求 5、繳交內容 3）：≥10 張影像 × ≥3 種提示詞寫法 → 計數正確率與失敗類型。

  python benchmark/bench_prompt_styles.py --out data/bench/jetson/prompt_styles.json

評估集與真值：benchmark/prompt_eval_cases.json（真實數量為人工目視）。每個（影像、概念）用下列寫法各查一次（每次只查一個概念）：
  bare 單詞（`cat`）／category 加冠詞的類別式（`a cat`）／descriptive 外觀描述式／technical 專業名稱式（有的概念沒有）／
  hyde：小型本地 LLM（Ollama qwen2.5:1.5b，CPU）把概念擴成多個描述，同一物件取最高分（server/hyde.py 的 expand_prompts）。
指標：計數正確率（偵測數 == 真實數量）、平均絕對誤差（MAE）、多報／少報比例；門檻 0.05／0.1（系統預設）／0.2，並掃描一組全域門檻看敏感度。
只要框、不跑 SAM（計數與遮罩無關）；OWL 為 TensorRT，影像長邊縮到 --max_side（預設 1280；小圖不放大）。HyDE 的擴展先於載入模型完成，並在載入模型前卸載 LLM（Jetson 共用 8GB）。
"""
import argparse
import json
import os
import statistics
import sys
import time

import requests

sys.path.insert(0, "src")
sys.path.insert(0, ".")

ap = argparse.ArgumentParser()
ap.add_argument("--cases", default="benchmark/prompt_eval_cases.json")
ap.add_argument("--out", default="data/bench/jetson/prompt_styles.json")
ap.add_argument("--max_side", type=int, default=1280)
ap.add_argument("--owl_engine", default="data/owl_image_encoder_patch32.engine")
ap.add_argument("--no_hyde", action="store_true")
ap.add_argument("--reuse_hyde", default=None, help="沿用某個先前輸出 JSON 裡的 HyDE 描述（LLM 每次產生的描述不同，要重現同一組結果就用它）")
ap.add_argument("--gt_boxes", default="benchmark/confidence_cases.json", help="有目視真值框的影像（位置也要對的指標只對這些影像計算）")
ap.add_argument("--power_mode", default="")
args = ap.parse_args()

cfg = json.load(open(args.cases))
images, styles, kind = cfg["images"], cfg["styles"], cfg["kind"]
concepts = sorted({c for v in images.values() for c in v["counts"]})
THR = [0.05, 0.1, 0.2]
GRID = [0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3]

# ---- HyDE：先擴展，再卸載 LLM，才載入模型 ----
hyde_groups, hyde_info = {}, {"used": False}
if args.reuse_hyde:
    prev = json.load(open(args.reuse_hyde))["hyde"]
    hyde_groups, hyde_info = prev["groups"], {**{k: v for k, v in prev.items() if k != "groups"}, "reused_from": args.reuse_hyde}
elif not args.no_hyde:
    try:
        from transformers import OwlViTProcessor
        from server import hyde
        tok = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32").tokenizer
        t0 = time.perf_counter()
        res = hyde.expand_prompts(concepts, token_counter=lambda t: len(tok(t)["input_ids"]))
        hyde_groups = res["groups"]
        hyde_info = {"used": True, "model": res["model"], "elapsed_s": round(time.perf_counter() - t0, 1), "dropped": res.get("dropped"), "n_variants": hyde.N_VARIANTS}
        try:
            requests.post(f"{hyde.OLLAMA_URL}/api/generate", json={"model": res["model"], "keep_alive": 0}, timeout=15)   # 卸載 LLM，讓出記憶體
        except Exception:
            pass
        time.sleep(3)
    except Exception as e:                                                                    # noqa: BLE001
        hyde_info = {"used": False, "error": f"{type(e).__name__}: {e}"}
        print("HyDE 失敗，略過：", hyde_info["error"])
print("HyDE groups:", json.dumps(hyde_groups, ensure_ascii=False))

import PIL.Image  # noqa: E402
from pipeline import OpenVocabSegmenter  # noqa: E402

seg = OpenVocabSegmenter(owl_engine=args.owl_engine)
tok = seg.owl.processor.tokenizer
STYLE_ORDER = ["bare", "category", "descriptive", "technical", "hyde"]


def queries_for(concept, style):
    if style == "hyde":
        return hyde_groups.get(concept)
    return [styles[concept][style]] if styles[concept].get(style) else None


for c in concepts:                                         # 每個查詢都必須 ≤ 16 token（含首尾）
    for s in STYLE_ORDER:
        for q in queries_for(c, s) or []:
            n = len(tok(q)["input_ids"])
            assert n <= 16, (c, s, q, n)


def load(path):
    im = PIL.Image.open(path).convert("RGB")
    if max(im.size) > args.max_side:
        k = args.max_side / max(im.size)
        im = im.resize((round(im.width * k), round(im.height * k)), PIL.Image.LANCZOS)
    return im


gt_boxes = {}
if os.path.exists(args.gt_boxes):
    for p_, c_ in json.load(open(args.gt_boxes)).items():
        if isinstance(c_, dict) and "gt" in c_:
            gt_boxes[p_] = c_["gt"]


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x1 - x0) * max(0, y1 - y0)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def loc_counts(dets, gts, t):
    """位置也要對：依分數由高到低，與尚未配對的同概念真值框 IoU>=0.5 才算 TP；其餘偵測為 FP，沒被配到的真值為 FN。"""
    left, tp, fp = list(gts), 0, 0
    for sc, box in dets:
        if sc < t:
            break
        m = next((g for g in left if iou(g, box) >= 0.5), None)
        if m is not None:
            left.remove(m); tp += 1
        else:
            fp += 1
    return {"tp": tp, "fp": fp, "fn": len(left)}


records = []
for path, spec in images.items():
    im = load(path)
    for concept, gt in spec["counts"].items():
        for style in STYLE_ORDER:
            qs = queries_for(concept, style)
            if not qs:
                continue
            seg.set_prompts([concept], groups={concept: qs})
            dets = seg(im, threshold=0.01, with_masks=False)
            ds = sorted(((float(d.score), [float(v) for v in d.box]) for d in dets), key=lambda x: -x[0])
            scores = [x[0] for x in ds]
            gts = [b for l, b in gt_boxes.get(path, []) if l == concept]
            loc = {str(t): loc_counts(ds, gts, t) for t in THR} if gts else None
            records.append({"image": path, "concept": concept, "kind": kind[concept], "style": style, "queries": qs, "gt": gt,
                            "counts": {str(t): sum(1 for sc in scores if sc >= t) for t in sorted(set(THR + GRID))},
                            "top_scores": [round(sc, 4) for sc in scores[:12]],
                            "gt_boxes": gts or None, "loc": loc, "dets_top": [[round(sc, 4)] + [round(v, 1) for v in b] for sc, b in ds[:40]]})
    print("done", path, flush=True)


def summarize(rows, t):
    errs = [r["counts"][str(t)] - r["gt"] for r in rows]
    n = len(errs)
    return {"pairs": n, "exact_acc": round(sum(e == 0 for e in errs) / n, 3), "mae": round(statistics.fmean(abs(e) for e in errs), 2),
            "over_rate": round(sum(e > 0 for e in errs) / n, 3), "under_rate": round(sum(e < 0 for e in errs) / n, 3)}


summary = {}
for style in STYLE_ORDER:
    for grp in ("all", "everyday", "part"):
        rows = [r for r in records if r["style"] == style and (grp == "all" or r["kind"] == grp)]
        if not rows:
            continue
        locrows = [r for r in rows if r["loc"]]
        locsum = {}
        if locrows:
            for t in THR:
                tp = sum(r["loc"][str(t)]["tp"] for r in locrows); fp = sum(r["loc"][str(t)]["fp"] for r in locrows); fn = sum(r["loc"][str(t)]["fn"] for r in locrows)
                P = tp / (tp + fp) if tp + fp else None; R = tp / (tp + fn) if tp + fn else None
                locsum[str(t)] = {"pairs": len(locrows), "tp": tp, "fp": fp, "fn": fn, "precision": None if P is None else round(P, 3),
                                  "recall": None if R is None else round(R, 3), "f1": None if not (P and R) else round(2 * P * R / (P + R), 3)}
        summary[f"{style}|{grp}"] = {"at_threshold": {str(t): summarize(rows, t) for t in THR}, "location_aware": locsum,
                                     "best_global_threshold": max(({"thr": t, **summarize(rows, t)} for t in GRID), key=lambda x: (x["exact_acc"], -x["mae"]))}
os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump({"date": time.strftime("%Y-%m-%d %H:%M:%S"), "power_mode": args.power_mode, "max_side": args.max_side, "thresholds": THR, "grid": GRID,
           "n_images": len(images), "n_pairs": sum(len(v["counts"]) for v in images.values()), "hyde": {**hyde_info, "groups": hyde_groups},
           "summary": summary, "records": records}, open(args.out, "w"), ensure_ascii=False, indent=1)
print("saved", args.out, len(records), "records")
