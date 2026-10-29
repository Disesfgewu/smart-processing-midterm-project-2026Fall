#!/usr/bin/env bash
# Jetson 上下載模型：OWL-ViT（HuggingFace 快取）+ NanoSAM 兩個 ONNX
# MobileSAM .pt 在 Jetson 上不需要（只給 WSL 的 PyTorch 後端用）
set -euo pipefail
cd "$(dirname "$0")/../.."
source "${VENV:-.venv-jetson}/bin/activate"
./installation/download_model.sh owlvit
./installation/download_model.sh nanosam
