"""以圖找圖（網頁流程）的 API 層驗證與量測：對「正在執行的伺服器」打 /api/segment，不載入模型。

  python -m server.main &            # 另一個終端機（或 PORT=8123 HOST=127.0.0.1 python -m server.main）
  python benchmark/bench_image_query_api.py --url http://127.0.0.1:8123 --out data/bench/jetson/image_query_api.json

做四件事：
  1. 回歸測試：文字請求 → 範例請求 → 同一個文字請求，結果必須與第一次完全相同（範例請求不能污染文字查詢的快取）。
  2. 參數驗證：2 個提示詞＋範例、框超出 0~1、模式不合法，都必須回 400。
  3. 實用性掃描：真值框（benchmark/confidence_cases.json）逐一當範例，掃不同分數門檻，記錄每個門檻下的 TP / FP / 召回；
     文字查詢（現狀）用同一組真值與排除規則對照。這回答「網頁上門檻該設多少」。
  4. 端對端延遲（客戶端量到的 round trip）：文字 vs 範例 inplace vs 範例 crop，各 --reps 次。
     注意網頁流程每次請求都會重算範例特徵（一次額外的 OWL 影像編碼），與 bench_image_query.py 裡「換範例才算一次」的成本不同。
影像先縮到長邊 1280（與真值座標一致）。回歸測試與實用性掃描以**無損 PNG** 上傳（排除 JPEG 重新壓縮的雜訊，結果才能和 bench_image_query.py 對照；
crop 模式對雜訊很敏感，見 docs/image-query.md「穩定性」）；延遲以 JPEG q92 上傳（接近實際）。真值為目視估計。
"""
import argparse
import io
import json
import statistics
import time

import PIL.Image
import requests

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8000")
ap.add_argument("--cases", default="benchmark/confidence_cases.json")
ap.add_argument("--out", default="data/bench/jetson/image_query_api.json")
ap.add_argument("--reps", type=int, default=30)
ap.add_argument("--max_side", type=int, default=1280)
ap.add_argument("--power_mode", default="")
args = ap.parse_args()

cases = {k: v for k, v in json.load(open(args.cases)).items() if k.startswith("data/")}
SKIP = {("data/samples/pexels-cmrcn-29224614.jpg", "steel pipe")}     # 真值不可靠（兩個實例外觀差很多），與 bench_image_query.py 一致


def load_images(path):
    """回傳 (尺寸, PNG 位元組, JPEG q92 位元組)。"""
    im = PIL.Image.open(path).convert("RGB")
    if max(im.size) > args.max_side:
        s = args.max_side / max(im.size)
        im = im.resize((round(im.width * s), round(im.height * s)), PIL.Image.LANCZOS)
    a, b = io.BytesIO(), io.BytesIO()
    im.save(a, "PNG"); im.save(b, "JPEG", quality=92)
    return im.size, a.getvalue(), b.getvalue()


def post(jpeg, prompts, threshold=0.01, masks=False, box=None, frac=None, mode="inplace", **extra):
    data = {"prompts": json.dumps(prompts), "threshold": str(threshold), "nms_iou": "0.5", "use_hyde": "false", "quiet": "true",
            "with_masks": "true" if masks else "false", **extra}
    if box is not None:
        w, h = frac
        data["exemplar_box"] = json.dumps([box[0] / w, box[1] / h, box[2] / w, box[3] / h]); data["exemplar_mode"] = mode
    t0 = time.perf_counter()
    r = requests.post(f"{args.url}/api/segment", files={"image": ("a.img", jpeg)}, data=data, timeout=120)
    ms = (time.perf_counter() - t0) * 1e3
    return r, ms


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x1 - x0) * max(0, y1 - y0)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def is_exemplar(det, ex):
    x0, y0, x1, y1 = max(det[0], ex[0]), max(det[1], ex[1]), min(det[2], ex[2]), min(det[3], ex[3])
    inside = max(0, x1 - x0) * max(0, y1 - y0) / max(1e-6, (det[2] - det[0]) * (det[3] - det[1]))
    return iou(ex, det) >= 0.3 or inside >= 0.5


out = {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "url": args.url, "power_mode": args.power_mode, "max_side": args.max_side}
images = {p: load_images(p) for p in cases}
pcb = "data/samples/pexels-muffin-4584612.jpg"
(size, jpeg, jpeg_q92) = images[pcb]
cap = [b for l, b in cases[pcb]["gt"] if l == "capacitor"]

# ---- 1. 回歸：文字 → 範例 → 文字 ----
r1, _ = post(jpeg, ["capacitor"]); r2, _ = post(jpeg, ["capacitor"], box=cap[0], frac=size); r3, _ = post(jpeg, ["capacitor"])
sig = lambda r: [(round(d["score"], 4), [round(v, 1) for v in d["box"]]) for d in r.json()["detections"]]
j2 = r2.json()
out["regression"] = {
    "status": [r1.status_code, r2.status_code, r3.status_code],
    "text_before_equals_text_after": sig(r1) == sig(r3), "n_text": len(sig(r1)), "n_exemplar": len(sig(r2)), "n_text_after": len(sig(r3)),
    "exemplar_flagged": sum(1 for d in j2["detections"] if d["exemplar"]), "exemplar_top_score": max(d["score"] for d in j2["detections"]),
    "exemplar_field": j2["exemplar"], "text_result_has_no_exemplar_field": r1.json()["exemplar"] is None,
    "text_best_score": max(d["score"] for d in r1.json()["detections"])}
print("regression:", {k: v for k, v in out["regression"].items() if k != "exemplar_field"})

# ---- 2. 參數驗證 ----
bad = {}
bad["two_prompts_with_exemplar"] = post(jpeg, ["capacitor", "inductor"], box=cap[0], frac=size)[0].status_code
bad["box_out_of_range"] = requests.post(f"{args.url}/api/segment", files={"image": ("a.img", jpeg)},
                                        data={"prompts": '["capacitor"]', "exemplar_box": "[0.1,0.1,1.5,0.9]"}).status_code
bad["box_inverted"] = requests.post(f"{args.url}/api/segment", files={"image": ("a.img", jpeg)},
                                    data={"prompts": '["capacitor"]', "exemplar_box": "[0.5,0.1,0.2,0.9]"}).status_code
bad["box_not_json"] = requests.post(f"{args.url}/api/segment", files={"image": ("a.img", jpeg)},
                                    data={"prompts": '["capacitor"]', "exemplar_box": "abc"}).status_code
bad["bad_mode"] = requests.post(f"{args.url}/api/segment", files={"image": ("a.img", jpeg)},
                                data={"prompts": '["capacitor"]', "exemplar_box": "[0.1,0.1,0.5,0.5]", "exemplar_mode": "zoom"}).status_code
out["validation_status_codes_expect_400"] = bad
print("validation:", bad)

# ---- 3. 實用性掃描 ----
TH_IMG = [0.5, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995]
TH_TXT = [0.01, 0.02, 0.03, 0.05, 0.1]
sweep = []
for path, case in cases.items():
    size, jpeg, _ = images[path]
    by = {}
    for l, b in case["gt"]:
        by.setdefault(l, []).append(b)
    for lab, boxes in by.items():
        if len(boxes) < 2 or (path, lab) in SKIP:
            continue
        for mode in ("text", "inplace", "crop"):
            per_ex = []
            for k, ex in enumerate(boxes):
                rest = [b for i, b in enumerate(boxes) if i != k]
                r, _ = (post(jpeg, [lab]) if mode == "text" else post(jpeg, [lab], box=ex, frac=size, mode=mode))
                dets = [d for d in r.json()["detections"] if not is_exemplar(d["box"], ex)]
                row = {}
                for t in (TH_TXT if mode == "text" else TH_IMG):
                    kept = [d for d in dets if d["score"] >= t]
                    tp_boxes = {i for i, g in enumerate(rest) if any(iou(g, d["box"]) >= 0.5 for d in kept)}
                    tp = sum(1 for d in kept if any(iou(g, d["box"]) >= 0.5 for g in rest))
                    row[str(t)] = {"kept": len(kept), "tp": tp, "fp": len(kept) - tp, "found_targets": len(tp_boxes), "targets": len(rest)}
                per_ex.append(row)
            agg = {}
            for t in per_ex[0]:
                kept = [x[t]["kept"] for x in per_ex]; tp = [x[t]["tp"] for x in per_ex]
                found = [x[t]["found_targets"] / x[t]["targets"] for x in per_ex]
                agg[t] = {"mean_kept": round(statistics.fmean(kept), 2), "mean_tp": round(statistics.fmean(tp), 2), "mean_fp": round(statistics.fmean(k - p for k, p in zip(kept, tp)), 2),
                          "mean_recall": round(statistics.fmean(found), 2),
                          "mean_precision": round(sum(tp) / sum(kept), 2) if sum(kept) else None}
            sweep.append({"image": path, "concept": lab, "instances": len(boxes), "mode": mode, "by_threshold": agg})
    print("swept", path, flush=True)
out["sweep"] = sweep

# ---- 4. 端對端延遲（JPEG q92 上傳；PCB 圖；概念 capacitor；範例＝第 0 個電容）----
lat = {}
ex = cap[0]
variants = {"text@0.10 (UI 預設門檻)": dict(threshold=0.1),
            "exemplar_inplace@0.95": dict(threshold=0.95, box=ex, frac=size, mode="inplace"),
            "exemplar_inplace@0.01 (最壞：幾百個框)": dict(threshold=0.01, box=ex, frac=size, mode="inplace"),
            "exemplar_crop@0.50": dict(threshold=0.5, box=ex, frac=size, mode="crop")}
for masks in (False, True):
    for name, kw in variants.items():
        for _ in range(3):
            post(jpeg_q92, ["capacitor"], masks=masks, **kw)                                  # 暖機
        client, server, setup, owl, nd = [], [], [], [], []
        for _ in range(args.reps):
            r, ms = post(jpeg_q92, ["capacitor"], masks=masks, **kw)
            j = r.json(); t = j["timing"]
            client.append(ms); server.append(t["server_ms"]); setup.append(t["text_encode_ms"]); owl.append(t["official_owl_ms"]); nd.append(len(j["detections"]))
        pct = lambda v, q: sorted(v)[min(len(v) - 1, int(q * len(v)))]
        lat[f"{name}|masks={masks}"] = {"n": args.reps, "detections": statistics.median(nd), "client_ms_median": round(statistics.median(client), 1), "client_ms_p95": round(pct(client, 0.95), 1),
                                       "server_ms_median": round(statistics.median(server), 1), "query_setup_ms_median": round(statistics.median(setup), 1),
                                       "owl_predict_ms_median": round(statistics.median(owl), 1)}
        print(name, masks, lat[f"{name}|masks={masks}"], flush=True)
out["latency"] = lat
json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=1)
print("saved", args.out)
