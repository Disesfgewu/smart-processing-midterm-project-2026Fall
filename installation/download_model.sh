#!/usr/bin/env bash
# 下載模型權重到 data/models/（已存在的檔案會略過）
# 注意：resnet18_image_encoder.onnx 需手動下載（Google Drive 需登入）
#   ./installation/download_model.sh            全部下載
#   ./installation/download_model.sh owlvit     只下載 OWL-ViT（HuggingFace 快取）
#   ./installation/download_model.sh mobilesam  只下載 MobileSAM checkpoint（PyTorch 後端用）
#   ./installation/download_model.sh nanosam    只下載 NanoSAM ONNX（Jetson 建 TensorRT 引擎用）
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=data/models
mkdir -p "$OUT"
TARGET=${1:-all}

# 1. OWL-ViT：NanoOWL 的底層模型，由 HuggingFace 下載並放進本機快取（~600MB）
download_owlvit() {
    python - <<'PY'
from transformers import OwlViTForObjectDetection, OwlViTProcessor
name = "google/owlvit-base-patch32"
OwlViTProcessor.from_pretrained(name)
OwlViTForObjectDetection.from_pretrained(name)
print("OWL-ViT ready:", name)
PY
}

# 2. MobileSAM checkpoint（NanoSAM 蒸餾的教師模型；WSL 上 PyTorch 後端用）
download_mobilesam() {
    local f="$OUT/mobile_sam.pt"
    [ -f "$f" ] && { echo "skip $f"; return; }
    wget -O "$f" https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt
}

# 3. NanoSAM ONNX（Jetson 上再用 trtexec 轉成 .engine）
#    mask decoder：HuggingFace 鏡像（Google Drive 已需登入，gdown 無法下載）
#    image encoder：官方只放在 Google Drive，需手動下載（見下方提示）
download_nanosam() {
    local dec="$OUT/mobile_sam_mask_decoder.onnx"
    if [ -f "$dec" ]; then echo "skip $dec"; else
        wget -O "$dec" https://huggingface.co/dragonSwing/nanosam/resolve/main/mobile_sam_mask_decoder.onnx
    fi
    local enc="$OUT/resnet18_image_encoder.onnx"
    if [ ! -f "$enc" ]; then
        cat <<MSG

[手動下載] 請用瀏覽器（登入 Google）開啟以下連結，下載後存成：
    $enc
    https://drive.google.com/file/d/14-SsvoaTl-esC3JOzomHDnI9OGgdO2OR/view
MSG
    fi
}

case "$TARGET" in
    owlvit)    download_owlvit ;;
    mobilesam) download_mobilesam ;;
    nanosam)   download_nanosam ;;
    all)       download_owlvit; download_mobilesam; download_nanosam ;;
    *) echo "unknown target: $TARGET (owlvit|mobilesam|nanosam|all)"; exit 1 ;;
esac
ls -lh "$OUT"
