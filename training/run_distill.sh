#!/usr/bin/env bash
# 蒸餾訓練 NanoSAM 的 ResNet18 image encoder（教師：MobileSAM ViT-T）。可中斷、可接續。
#
#   ./training/run_distill.sh                  # 前景執行，預設 10 個 epoch
#   ./training/run_distill.sh --bg             # 背景執行（關掉終端機也不會停），日誌 data/train/train.log
#   EPOCHS=20 ./training/run_distill.sh        # 訓練到第 20 個 epoch（接續時是「總共到第幾個」，不是再多跑幾個）
#   ./training/run_distill.sh --num_images 480 # 其餘參數原樣傳給 train_nanosam.py（例如先用小子集試跑）
#
# 實測（GTX 1060 6GB，COCO val2017）：約 4 張/秒、一個 epoch（4800 張）約 20 分鐘；batch 4 峰值顯存 1.5GB。
# 速度受 GPU 算力限制，batch 加大不會變快（batch 8 同速但要 2.6GB），所以預設 batch 4 + 梯度累積 4（等效 16）。
# 訓練期間會和網頁伺服器 / 其他 GPU 程式搶 GPU；想兼顧請在訓練前先關掉不需要的程式。
#
# 保護 baseline：baseline（data/models/mobile_sam.pt）只被「讀取」；每次實驗請用不同的 RUN 名稱，輸出在 data/train/<RUN>/，
# 不會互相覆蓋，也不會碰 data/models。同一個 RUN 重跑 = 接續訓練；想要「全新的一次」就換 RUN（例如 RUN=resnet18_run2）。
#
# 環境變數：RUN(resnet18_run1) EPOCHS(10) BATCH(4) ACCUM(4) VAL(200) IMAGES(data/coco/val2017) WORKERS(6)  （OUT 可直接指定完整路徑）
set -euo pipefail
cd "$(dirname "$0")/.."

EPOCHS=${EPOCHS:-10}; BATCH=${BATCH:-4}; ACCUM=${ACCUM:-4}; VAL=${VAL:-200}; WORKERS=${WORKERS:-6}
RUN=${RUN:-resnet18_run1}; OUT=${OUT:-data/train/$RUN}; IMAGES=${IMAGES:-data/coco/val2017}
BG=0; if [ "${1:-}" = "--bg" ]; then BG=1; shift; fi

# ---- 前置檢查 ----
python - <<'PY' || { echo "找不到 nanosam / torch：請先 conda activate nanoowl-sam"; exit 1; }
import torch, nanosam.mobile_sam  # noqa
assert torch.cuda.is_available(), "CUDA 不可用"
PY
[ -d "$IMAGES" ] && [ "$(ls "$IMAGES" | wc -l)" -gt 100 ] || { echo "找不到訓練影像 $IMAGES：請先 ./training/download_coco.sh val"; exit 1; }
[ -f data/models/mobile_sam.pt ] || { echo "找不到教師權重 data/models/mobile_sam.pt：請先 ./installation/download_model.sh mobilesam"; exit 1; }
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [ "$FREE" -lt 2000 ]; then echo "警告：GPU 可用顯存只有 ${FREE}MB（訓練 batch=4 約需 1.5GB），請先關掉其他 GPU 程式"; fi
if ss -ltn 2>/dev/null | grep -q ":8000 "; then echo "提示：偵測到 8000 埠的網頁伺服器在執行，它會佔用約 1GB 顯存並與訓練搶 GPU（兩者都會變慢）"; fi

mkdir -p "$OUT"
ARGS=("$IMAGES" "$OUT" --batch_size "$BATCH" --accum_steps "$ACCUM" --num_epochs "$EPOCHS" --val_count "$VAL" --num_workers "$WORKERS" "$@")
if [ -f "$OUT/checkpoint.pth" ]; then echo "發現既有 checkpoint，將從上次中斷處接續（目標：總共 $EPOCHS 個 epoch）"; fi
echo "輸出：$OUT（checkpoint.pth = 最新、best.pth = 驗證 loss 最低、run_info.json = 本次設定）"; echo "baseline：data/models/mobile_sam.pt 只讀取，不會被覆蓋；訓練結束會校驗它的 sha256"

if [ "$BG" = 1 ]; then
    nohup python training/train_nanosam.py "${ARGS[@]}" > "$OUT/../train.log" 2>&1 &
    echo "已在背景執行（pid $!）。看進度：  tail -f $OUT/../train.log    或    cat $OUT/log.txt"
    echo "停止：  kill $!     （之後重跑同一個指令會接續）"
else
    python training/train_nanosam.py "${ARGS[@]}"
    echo; echo "完成。比較 baseline 與自訓練：  python training/compare_encoders.py --student $OUT/best.pth"; echo "匯出 ONNX（檔名帶 .selftrain，不會蓋過官方檔）：  RUN=$RUN ./training/export_encoder.sh"
fi
