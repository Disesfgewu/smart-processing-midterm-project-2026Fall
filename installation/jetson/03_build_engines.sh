#!/usr/bin/env bash
# 在 Jetson 上建立三個 TensorRT 引擎（引擎綁定 GPU 架構與 TensorRT 版本，不可從 WSL 複製）
#   data/owl_image_encoder_patch32.engine
#   data/resnet18_image_encoder.engine
#   data/mobile_sam_mask_decoder.engine
set -euo pipefail
cd "$(dirname "$0")/../.."
source "${VENV:-.venv-jetson}/bin/activate"
TRTEXEC=${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}
M=data/models
mkdir -p data

# 建引擎很吃記憶體，先關掉不必要的服務；若 OOM 請先跑 04_setup_swap.sh
python -m nanoowl.build_image_encoder_engine data/owl_image_encoder_patch32.engine

"$TRTEXEC" --onnx=$M/resnet18_image_encoder.onnx \
    --saveEngine=data/resnet18_image_encoder.engine --fp16

"$TRTEXEC" --onnx=$M/mobile_sam_mask_decoder.onnx \
    --saveEngine=data/mobile_sam_mask_decoder.engine \
    --minShapes=point_coords:1x1x2,point_labels:1x1 \
    --optShapes=point_coords:1x1x2,point_labels:1x1 \
    --maxShapes=point_coords:1x10x2,point_labels:1x10

ls -lh data/*.engine
