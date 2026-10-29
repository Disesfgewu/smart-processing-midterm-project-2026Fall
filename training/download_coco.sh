#!/usr/bin/env bash
# 下載 COCO 影像作為蒸餾訓練資料到 data/coco/
#   ./training/download_coco.sh val     val2017（5k 張，約 1GB）← 建議
#   ./training/download_coco.sh train   train2017（118k 張，約 18GB）
set -euo pipefail
cd "$(dirname "$0")/.."
SPLIT=${1:-val}
mkdir -p data/coco && cd data/coco
case "$SPLIT" in
    val)   wget -nc http://images.cocodataset.org/zips/val2017.zip && unzip -qn val2017.zip ;;
    train) wget -nc http://images.cocodataset.org/zips/train2017.zip && unzip -qn train2017.zip ;;
    *) echo "usage: $0 [val|train]"; exit 1 ;;
esac
ls "${SPLIT}2017" | wc -l
