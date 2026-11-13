"""用實際的全 TensorRT 流程（OWL + 自訓練 ResNet18 SAM encoder + mask decoder）產生「含遮罩」的結果圖，供報告當偵測與分割截圖。

  python benchmark/render_results.py        # 輸出到 data/bench/jetson/results_with_masks/，並寫 manifest.json

這些是 pipeline 的疊圖輸出（src/overlay.py 的 draw_overlay：遮罩＋輪廓＋框＋分數＋各提示詞數量與面積），**不是網頁介面的截圖**；
Demo 影片與網頁截圖仍需在瀏覽器操作。每張圖的「預期」欄是人工目視的真實數量（見 benchmark/prompt_eval_cases.json）。
"""
import json
import os
import sys
import time

import cv2
import numpy as np
import PIL.Image

sys.path.insert(0, "src")
from overlay import draw_overlay  # noqa: E402
from pipeline import OpenVocabSegmenter  # noqa: E402

OUT = "data/bench/jetson/results_with_masks"
os.makedirs(OUT, exist_ok=True)
hy = json.load(open("data/bench/jetson/prompt_styles.json"))["hyde"]["groups"]
cs = json.load(open("benchmark/prompt_eval_cases.json"))["styles"]

# (檔名, 影像, 概念, 查詢們, 預期數量, 類型, 說明)
JOBS = [
    ("01_cats_success", "data/samples/cats.jpg", "cat", ["a cat"], 2, "成功（日常物件）", "貓與遙控器：框、遮罩都正確"),
    ("02_buns_descriptive_success", "data/samples/eval/buns_f143.jpg", "steamed bun", [cs["steamed bun"]["descriptive"]], 2, "成功（輸送帶，製造情境）", "包子成型線：外觀描述式數出 2 顆"),
    ("03_buns_category_miss", "data/samples/eval/buns_f143.jpg", "steamed bun", [cs["steamed bun"]["category"]], 2, "失敗：漏報", "同一張圖，類別式 `a steamed bun` 只數出 1 顆"),
    ("04_limes_success", "data/samples/eval/limes_f346_crop.jpg", "citrus fruit", [cs["citrus fruit"]["category"]], 3, "成功（輸送帶，製造情境）", "萊姆分選線近處：3 顆"),
    ("05_pcb_capacitor_descriptive", "data/samples/pexels-muffin-4584612.jpg", "capacitor", [cs["capacitor"]["descriptive"]], 7, "部分成功（電路板）", "外觀描述式找到 7 個，多數位置正確（見 prompt_styles.json）"),
    ("06_pcb_capacitor_hyde_overcount", "data/samples/pexels-muffin-4584612.jpg", "capacitor", hy["capacitor"], 7, "失敗：多報（HyDE）", "HyDE 多描述取最高分：偵測數遠多於真實數量，大量誤報"),
    ("07_welding_person_overcount", "data/samples/pexels-cmrcn-29224614.jpg", "person", ["a person"], 1, "失敗：多報（焊接現場）", "真實 1 位，偵測 3 個（局部／重疊框）"),
    ("08_cnc_spindle_wrong_object", "data/samples/pexels-clicksbyrenjith-23948829.jpg", "spindle motor", ["a spindle motor"], 1, "失敗：框到別的物件（相似物件）", "真實 1 個主軸；偵測 3 個，最高分框到彈簧"),
]
seg = OpenVocabSegmenter(owl_engine="data/owl_image_encoder_patch32.engine", nanosam_encoder="data/resnet18_image_encoder.selftrain.engine",
                         nanosam_decoder="data/mobile_sam_mask_decoder.engine", nanosam_backend="auto")
seg.set_sam_encoder("nanosam")
man = []
for name, path, concept, qs, gt, kind, note in JOBS:
    im = PIL.Image.open(path).convert("RGB")
    if max(im.size) > 1280:
        k = 1280 / max(im.size); im = im.resize((round(im.width * k), round(im.height * k)), PIL.Image.LANCZOS)
    seg.set_prompts([concept], groups={concept: qs})
    seg(im, 0.1, 0.5, True)                                   # 暖機
    dets = seg(im, 0.1, 0.5, True)
    t = dict(seg.timing)
    frame = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
    out = draw_overlay(frame, dets, [concept], timing=t, fps=None, style="both", expected={concept: gt})
    cv2.imwrite(f"{OUT}/{name}.jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 92])
    man.append({"file": f"{name}.jpg", "source_image": path, "concept": concept, "queries": qs, "expected_count": gt, "detected": len(dets), "kind": kind, "note": note,
                "scores": [round(float(d.score), 3) for d in dets][:12], "sam_ms": round(t["sam_ms"], 1), "owl_ms": round(t["owl_ms"], 1)})
    print(f"{name}: detected {len(dets)} / expected {gt}  [{kind}]", flush=True)
json.dump({"date": time.strftime("%Y-%m-%d %H:%M:%S"), "pipeline": "OWL TensorRT fp16 + 自訓練 ResNet18 TensorRT fp16 + mask decoder TensorRT fp32", "threshold": 0.1, "images": man},
          open(f"{OUT}/manifest.json", "w"), ensure_ascii=False, indent=1)
print("saved", OUT)
