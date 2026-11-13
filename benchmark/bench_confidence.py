"""信心分數基準：記錄每個偵測的 OWL score 與 SAM 自評 IoU（sam_iou），之後改信心訊號時用來對照。

  python benchmark/bench_confidence.py --cases benchmark/confidence_cases.json --out data/bench/jetson/confidence_baseline.json

固定做的事：
  * OWL 用 TensorRT 引擎；SAM 後端在「mobilesam（PyTorch fp32，baseline）」與「nanosam（自訓練 ResNet18 + TensorRT decoder）」間切換。
  * 影像與各自的概念由 --cases 的 JSON 指定；提示詞組：plain（"a cat"）與 template（"a photo of a cat"）。
  * 長邊縮到 --max_side（預設 1280）再推論；MobileSAM/plain 另存疊圖到 --overlay_dir 供目視檢查。
  * 以極低門檻（0.01）取回所有 NMS 後的偵測，原始 score 與 sam_iou 全部存下；之後可離線套任何門檻。
  * 每組重複 --runs 次，確認 sam_iou 是否穩定（同輸入應完全相同）。

注意：TensorRT 10.3 的 mask decoder 與其 ONNX 來源結果不同，sam_iou 偏低（約 0.67，PyTorch 約 0.96），見 docs/record.md §13.8。
所以本腳本同時記錄兩個後端的 sam_iou，才能分辨「訊號偏低」是來自後端還是來自遮罩本身。
"""
import argparse
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
ap.add_argument("--cases", default="benchmark/confidence_cases.json", help="{影像路徑: {note, concepts}}")
ap.add_argument("--out", default="data/bench/jetson/confidence_baseline.json")
ap.add_argument("--overlay_dir", default="data/bench/jetson/confidence_overlays")
ap.add_argument("--max_side", type=int, default=1280, help="長邊縮到這個大小再推論（OWL 吃 768、SAM 吃 1024）")
ap.add_argument("--runs", type=int, default=3)
ap.add_argument("--owl_engine", default="data/owl_image_encoder_patch32.engine")
ap.add_argument("--nanosam_encoder", default="data/resnet18_image_encoder.selftrain.engine")
ap.add_argument("--nanosam_decoder", default="data/mobile_sam_mask_decoder.engine")
ap.add_argument("--power_mode", default="", help="記錄用文字")
args = ap.parse_args()

THRESHOLDS = [0.01, 0.03, 0.05, 0.1]   # 離線摘要用；偵測本身以 0.01 取回
cases = json.load(open(args.cases))
seg = OpenVocabSegmenter(owl_engine=args.owl_engine, nanosam_encoder=args.nanosam_encoder,
                         nanosam_decoder=args.nanosam_decoder, nanosam_backend="auto")


def load(path):
    im = PIL.Image.open(path).convert("RGB")
    orig = im.size
    if max(im.size) > args.max_side:
        s = args.max_side / max(im.size)
        im = im.resize((round(im.width * s), round(im.height * s)), PIL.Image.LANCZOS)
    return im, orig


def summarize(dets, concepts):
    out = {}
    for t in THRESHOLDS:
        kept = [d for d in dets if d.score >= t]
        out[str(t)] = {
            "count": {c: sum(d.label == c for d in kept) for c in concepts},
            "mean_score": round(statistics.fmean(d.score for d in kept), 4) if kept else None,
            "mean_sam_iou": round(statistics.fmean(d.sam_iou for d in kept), 4) if kept else None,
        }
    return out


def overlay(image, dets, path, min_score=0.05):
    import cv2
    bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    for d in dets:
        if d.score < min_score:
            continue
        x0, y0, x1, y1 = [int(v) for v in d.box]
        cv2.rectangle(bgr, (x0, y0), (x1, y1), (0, 255, 0) if d.primary else (0, 165, 255), 2)
        cv2.putText(bgr, f"{d.label} {d.score:.2f}", (x0, max(y0 - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(bgr, f"{d.label} {d.score:.2f}", (x0, max(y0 - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, bgr)


results = {"max_side": args.max_side, "power_mode": args.power_mode, "date": time.strftime("%Y-%m-%d %H:%M:%S"),
           "thresholds": THRESHOLDS, "images": {}}

for path, case in cases.items():
    concepts = case["concepts"]
    image, orig = load(path)
    prompt_sets = {"plain": {c: [f"a {c}"] for c in concepts}, "template": {c: [f"a photo of a {c}"] for c in concepts}}
    entry = {"note": case["note"], "orig_size": list(orig), "input_size": list(image.size), "concepts": concepts, "configs": {}}
    results["images"][path] = entry
    print(f"=== {path} {orig} -> {image.size} {concepts}")
    for sam_name in ("mobilesam", "nanosam"):
        seg.set_sam_encoder(sam_name)
        for pset, groups in prompt_sets.items():
            seg.set_prompts(concepts, groups=groups)
            seg(image, threshold=0.01)                                  # 暖機
            runs = [sorted(seg(image, threshold=0.01), key=lambda d: -d.score) for _ in range(args.runs)]
            ref = runs[0]
            stable = all(len(r) == len(ref) and np.allclose([d.score for d in r], [d.score for d in ref], atol=1e-4)
                         and np.allclose([d.sam_iou for d in r], [d.sam_iou for d in ref], atol=1e-3) for r in runs[1:])
            key = f"{sam_name}/{pset}"
            entry["configs"][key] = {
                "sam_encoder": sam_name, "prompt_set": pset, "queries": seg.queries,
                "best_scores": {k: (round(v, 4) if v is not None else None) for k, v in seg.best_scores.items()},
                "stable_across_runs": bool(stable),
                "summary_by_threshold": summarize(ref, concepts),
                "detections": [{"label": d.label, "query": d.query, "score": round(d.score, 4),
                                "sam_iou": round(d.sam_iou, 4), "area_px": d.area,
                                "box": [round(float(x), 1) for x in d.box], "primary": d.primary} for d in ref],
            }
            if key == "mobilesam/plain":
                overlay(image, ref, os.path.join(args.overlay_dir, os.path.splitext(os.path.basename(path))[0] + ".png"))
            print(f"[{key}] {len(ref)} dets, stable={stable}, best={entry['configs'][key]['best_scores']}")
            if sam_name == "mobilesam" and pset == "plain":
                for d in ref[:10]:
                    print(f"    {d.label:15s} score={d.score:.3f} sam_iou={d.sam_iou:.3f} area={d.area:>8d} primary={d.primary}")

os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("saved", args.out)
