#!/usr/bin/env bash
# Jetson 上下載模型：OWL-ViT（HuggingFace 快取）+ NanoSAM 兩個 ONNX
# MobileSAM .pt：目前 pipeline 的 SAM 仍走 PyTorch（尚無 SAM TensorRT 後端），所以 Jetson 也需要；已用 pack_for_jetson.sh 傳過來的會略過
# 官方 resnet18_image_encoder.onnx 不在這裡下載（需手動）；用自訓練的 ONNX 時，下面 nanosam 步驟印出的『手動下載』提示可忽略
set -euo pipefail
cd "$(dirname "$0")/../.."
source "${VENV:-.venv-jetson}/bin/activate"
./installation/download_model.sh owlvit
./installation/download_model.sh mobilesam
./installation/download_model.sh nanosam
