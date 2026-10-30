#!/usr/bin/env bash
# 【在 Jetson 執行】啟動網頁伺服器（API + 前端）。從筆電瀏覽器開 http://<jetson-ip>:8000
#   ./installation/jetson/07_run_server.sh                 # 引擎都建好 → 全 TensorRT（OWL + NanoSAM），網頁下拉選單仍可切回 PyTorch 的 MobileSAM / 自訓練 encoder 比較
#   OWL_ENGINE= ./installation/jetson/07_run_server.sh     # 強制 OWL 也用 PyTorch（排錯 / 對照用）
#   SAM_STUDENT=data/eval/colab80_best.pth ./installation/jetson/07_run_server.sh   # 預載自訓練 encoder（網頁也會自動掃到 data/eval/*.pth）
# HyDE 預設關閉：Ollama + 模型會吃掉 Orin Nano 大量記憶體，影響量測；要用請先評估記憶體再設 HYDE_AUTOSTART=1。
set -euo pipefail
cd "$(dirname "$0")/../.."
source "${VENV:-.venv-jetson}/bin/activate"
export HOST=${HOST:-0.0.0.0} PORT=${PORT:-8000} HYDE_AUTOSTART=${HYDE_AUTOSTART:-0}
if [ -z "${OWL_ENGINE+x}" ]; then [ -f data/owl_image_encoder_patch32.engine ] && export OWL_ENGINE=data/owl_image_encoder_patch32.engine; fi
export APP_LOG=${APP_LOG:-data/bench/jetson/app.log}; mkdir -p "$(dirname "$APP_LOG")"
ENC=${NANOSAM_ENCODER-data/resnet18_image_encoder.selftrain.engine}; DEC=${NANOSAM_DECODER-data/mobile_sam_mask_decoder.engine}
if [ -n "$ENC" ] && [ -f "$ENC" ] && [ -f "$DEC" ]; then export NANOSAM_ENCODER=$ENC NANOSAM_DECODER=$DEC NANOSAM_BACKEND=${NANOSAM_BACKEND:-trt} SAM_ACTIVE=${SAM_ACTIVE:-nanosam}; fi
[ -f data/models/mobile_sam.pt ] || { echo "找不到 data/models/mobile_sam.pt（目前 SAM 走 PyTorch 需要它）；請依 docs/jetson-data.md 傳過來"; exit 1; }
echo "OWL 後端: ${OWL_ENGINE:-PyTorch}  |  SAM: ${SAM_ACTIVE:-mobilesam}（${NANOSAM_ENCODER:+NanoSAM $NANOSAM_BACKEND 引擎；}網頁可切換 MobileSAM / 自訓練 encoder 比較）  |  http://$(hostname -I | awk '{print $1}'):$PORT  |  日誌 $APP_LOG"
exec python -m server.main
