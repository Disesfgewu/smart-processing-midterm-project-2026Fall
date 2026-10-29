#!/usr/bin/env bash
# 下載 COCO 影像作為蒸餾訓練資料到 data/coco/
#   ./training/download_coco.sh val     val2017（5k 張，約 1GB）← 建議
#   ./training/download_coco.sh train   train2017（118k 張，約 18GB）
#   ./training/download_coco.sh annotations   標註（241MB，只解出 instances_val2017.json；compare_encoders.py 用它算 mIoU）
set -euo pipefail
cd "$(dirname "$0")/.."
SPLIT=${1:-val}
mkdir -p data/coco && cd data/coco
case "$SPLIT" in
    val)   wget -nc http://images.cocodataset.org/zips/val2017.zip && unzip -qn val2017.zip ;;
    train) wget -nc http://images.cocodataset.org/zips/train2017.zip && unzip -qn train2017.zip ;;
    annotations) wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip && unzip -jo annotations_trainval2017.zip annotations/instances_val2017.json && ls -lh instances_val2017.json; exit 0 ;;
    *) echo "usage: $0 [val|train|annotations]"; exit 1 ;;
esac
ls "${SPLIT}2017" | wc -l
