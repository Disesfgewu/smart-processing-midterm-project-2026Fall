"""API 契約檢查：POST /api/segment 的回應必須包含前端依賴的所有欄位。

語法檢查抓不到「欄位被行尾註解吃掉」「改名後漏改」這類錯誤（2026-10-29 實際發生過：prompts / threshold 欄位消失，
前端結果畫面全部壞掉），所以改動 server/ 之後請實際打一次 API：

    python tests/check_api_contract.py                     # 預設檢查 http://localhost:8000
    BASE=http://localhost:8001 python tests/check_api_contract.py
"""
import json
import os
import sys

import requests

BASE = os.environ.get("BASE", "http://localhost:8000")
IMAGE = os.environ.get("IMAGE", "data/samples/cats.jpg")
NEED = {
    "頂層": {"image", "prompts", "threshold", "nms_iou", "with_masks", "top_per_group", "hyde", "detections", "summary", "timing", "backend", "memory", "request_id"},
    "image": {"width", "height", "original_width", "original_height"},
    "detections[0]": {"id", "label", "score", "box", "box_area", "mask_area", "sam_iou", "query", "primary", "contours"},
    "summary[0]": {"label", "count", "mask_area", "box_area", "mean_score", "best_score", "queries", "primary", "mean_sam_iou"},
    "timing": {"owl_ms", "sam_ms", "total_ms", "text_encode_ms", "serialize_ms", "num_raw_boxes"},
}

with open(IMAGE, "rb") as f:
    r = requests.post(f"{BASE}/api/segment", files={"image": f}, data={"prompts": "a cat,a remote control", "threshold": 0.1}, timeout=60)
r.raise_for_status()
data = r.json()
got = {"頂層": set(data), "image": set(data["image"]), "timing": set(data["timing"])}
if not data["detections"]:
    sys.exit("測試圖沒有偵測到任何物件，無法檢查 detections / summary 欄位")
got["detections[0]"], got["summary[0]"] = set(data["detections"][0]), set(data["summary"][0])
bad = False
for name, need in NEED.items():
    miss = need - got[name]
    print(f"  {name:14s} {'OK' if not miss else '缺少 ' + json.dumps(sorted(miss), ensure_ascii=False)}")
    bad |= bool(miss)
sys.exit(1 if bad else 0)
