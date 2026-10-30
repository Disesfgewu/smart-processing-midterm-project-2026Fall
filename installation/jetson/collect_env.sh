#!/usr/bin/env bash
# 【在 Jetson 執行】把環境與版本資訊收集成一份 Markdown，貼進報告的「環境」一節（順便證明版本是實際查到的，不是照文件抄的）。
#   ./installation/jetson/collect_env.sh                 # → data/bench/jetson/env_<時間>.md
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=${OUT:-data/bench/jetson/env_$(date +%Y%m%d_%H%M%S).md}; mkdir -p "$(dirname "$OUT")"
VENV=${VENV:-.venv-jetson}; [ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"
sec() { printf '\n## %s\n\n```\n' "$1"; shift; "$@" 2>&1 | sed 's/\x1b\[[0-9;]*m//g'; printf '```\n'; }
{
echo "# Jetson 環境（$(date '+%F %T')）"
sec "裝置" bash -c 'cat /proc/device-tree/model 2>/dev/null | tr -d "\0"; echo; uname -a'
sec "L4T" bash -c 'cat /etc/nv_tegra_release 2>/dev/null || echo "沒有 /etc/nv_tegra_release（不是 Jetson？）"'
sec "JetPack" bash -c 'dpkg -l 2>/dev/null | grep -E "nvidia-jetpack " || echo "nvidia-jetpack 未安裝"'
sec "CUDA" bash -c 'nvcc --version 2>/dev/null | tail -2 || ls /usr/local | grep cuda'
sec "TensorRT（套件）" bash -c 'dpkg -l | grep -E " (tensorrt|libnvinfer10|python3-libnvinfer) " | awk "{print \$2, \$3}"'
sec "trtexec" bash -c 'ls -l /usr/src/tensorrt/bin/trtexec 2>&1'
sec "電源模式（nvpmodel）" bash -c 'sudo -n nvpmodel -q 2>&1 || nvpmodel -q 2>&1'
sec "時脈（jetson_clocks）" bash -c 'sudo -n jetson_clocks --show 2>&1 | head -20'
sec "記憶體 / swap" free -h
sec "磁碟" df -h .
sec "Python 套件" python3 - <<'PY'
import sys; print("python", sys.version.split()[0])
for m in ["torch", "torchvision", "tensorrt", "torch2trt", "transformers", "numpy", "cv2", "timm", "onnx"]:
    try:
        mod = __import__(m); print(f"{m:14s}", getattr(mod, "__version__", "?"))
    except Exception as e:
        print(f"{m:14s} 匯入失敗: {type(e).__name__}: {e}")
try:
    import torch; print("cuda available:", torch.cuda.is_available(), "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-", "| torch CUDA", torch.version.cuda)
except Exception as e:
    print("torch.cuda 檢查失敗:", e)
PY
sec "專案 commit" bash -c 'git rev-parse --short HEAD; git status --short | head -5'
} > "$OUT"
echo "已寫入 $OUT"; cat "$OUT"
