#!/usr/bin/env bash
# 在 Jetson 上建立三個 TensorRT 引擎（引擎綁定 GPU 架構與 TensorRT 版本，不可從 WSL 複製）
#   data/owl_image_encoder_patch32.engine
#   data/<ENC_ONNX 的檔名>.engine（官方: resnet18_image_encoder.engine）
#   data/mobile_sam_mask_decoder.engine
set -euo pipefail
cd "$(dirname "$0")/../.."
source "${VENV:-.venv-jetson}/bin/activate"
TRTEXEC=${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}
M=data/models
# 官方 encoder 為 resnet18_image_encoder.onnx；自行蒸餾的為 resnet18_image_encoder.selftrain.onnx（兩者可並存）。
# 用 ENC_ONNX 選擇；引擎檔名依 ONNX 檔名而定，不會互相覆蓋。
ENC_ONNX=${ENC_ONNX:-$M/resnet18_image_encoder.onnx}
[ -f "$ENC_ONNX" ] || { echo "找不到 $ENC_ONNX。官方檔需手動下載；自訓練的請用 ENC_ONNX=$M/resnet18_image_encoder.selftrain.onnx"; exit 1; }
ENC_ENGINE=data/$(basename "$ENC_ONNX" .onnx).engine
mkdir -p data

# 建引擎很吃記憶體，先關掉不必要的服務；若 OOM 請先跑 04_setup_swap.sh
python -m nanoowl.build_image_encoder_engine data/owl_image_encoder_patch32.engine

"$TRTEXEC" --onnx="$ENC_ONNX" \
    --saveEngine="$ENC_ENGINE" --fp16

"$TRTEXEC" --onnx=$M/mobile_sam_mask_decoder.onnx \
    --saveEngine=data/mobile_sam_mask_decoder.engine \
    --minShapes=point_coords:1x1x2,point_labels:1x1 \
    --optShapes=point_coords:1x1x2,point_labels:1x1 \
    --maxShapes=point_coords:1x10x2,point_labels:1x10

ls -lh data/*.engine
