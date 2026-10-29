#!/usr/bin/env bash
# 把訓練好的 encoder 匯出成 ONNX（給 Jetson 用 trtexec 轉成 TensorRT 引擎）。
#   ./training/export_encoder.sh                       # 預設用 best.pth（驗證 loss 最低）；沒有就用 checkpoint.pth
#   RUN=resnet18_run2 ./training/export_encoder.sh     # 指定實驗名稱（對應 data/train/<RUN>）
#   CKPT=data/train/resnet18/checkpoint.pth ./training/export_encoder.sh
# 輸出：ONNX_OUT（預設 data/models/resnet18_image_encoder.selftrain.onnx —— 刻意不用官方的檔名 resnet18_image_encoder.onnx，
#   這樣之後手動下載官方檔也不會與自訓練的互相覆蓋，可以並存比較；輸入 image 1x3x1024x1024、輸出 image_embeddings，與 NanoSAM 預期一致）
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=${RUN:-resnet18_run1}; OUT=${OUT:-data/train/$RUN}
CKPT=${CKPT:-$OUT/best.pth}; [ -f "$CKPT" ] || CKPT=$OUT/checkpoint.pth
[ -f "$CKPT" ] || { echo "找不到 checkpoint：請先 ./training/run_distill.sh"; exit 1; }
ONNX_OUT=${ONNX_OUT:-data/models/resnet18_image_encoder.selftrain.onnx}
case "$(basename "$ONNX_OUT")" in resnet18_image_encoder.onnx) [ "${FORCE:-0}" = 1 ] || { echo "拒絕寫入官方檔名 $ONNX_OUT（會蓋過官方檔案）。要這麼做請設 FORCE=1"; exit 1; } ;; esac
mkdir -p "$(dirname "$ONNX_OUT")"
echo "使用 $CKPT"
python -m nanosam.tools.export_image_encoder_onnx --model_name=resnet18 --checkpoint="$CKPT" --output="$ONNX_OUT"
ls -lh "$ONNX_OUT"
