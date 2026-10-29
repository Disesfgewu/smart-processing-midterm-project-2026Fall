#!/usr/bin/env bash
# Jetson Orin Nano (JetPack 6.2 / L4T 36.4 / CUDA 12.6 / TensorRT 10.3 / Python 3.10) 環境安裝
# 用法: ./installation/jetson/01_install_env.sh
#
# 重點：
#  * venv 必須用 --system-site-packages，才能看到 JetPack 內建的 tensorrt（不要 pip install tensorrt）
#  * torch / torchvision 必須用 Jetson 專用 wheel（PyPI 上的是 x86/CPU 版）
#  * numpy 必須 <2（Jetson wheel 以 numpy 1.x 編譯）
set -euo pipefail
cd "$(dirname "$0")/../.."

JP_INDEX=${JP_INDEX:-https://pypi.jetson-ai-lab.io/jp6/cu126}
VENV=${VENV:-.venv-jetson}

sudo apt-get update
sudo apt-get install -y python3-pip python3-venv python3-dev git wget \
    libopenblas-dev libjpeg-dev zlib1g-dev python3-libnvinfer python3-libnvinfer-dev

# cuSOLVER / cuSPARSELt 缺少時 torch 會出現 libcusolver / "no kernel image" 錯誤
sudo apt-get install -y libcusolver-12-6 libcusolver-dev-12-6 libcusparse-12-6 \
    libcublas-12-6 libcudnn9-cuda-12 || true

python3 -m venv --system-site-packages "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install -U pip setuptools wheel

# 1. numpy 先鎖版本，避免後續套件把它升到 2.x
pip install "numpy<2"

# 2. Jetson 版 PyTorch（已知可用組合：torch 2.8.0 + torchvision 0.23.0；若 index 失效見 docs/jetson-setup.md）
pip install --index-url "$JP_INDEX" --extra-index-url https://pypi.org/simple \
    torch==2.8.0 torchvision==0.23.0

# 3. 其餘純 Python 套件（不含 torch / tensorrt）
pip install transformers==4.46.3 timm==1.0.11 pillow matplotlib requests tqdm gdown onnx
# opencv：JetPack 已有 python3-opencv（含 GStreamer），不要 pip 安裝以免蓋掉
sudo apt-get install -y python3-opencv

# 4. torch2trt（TensorRT 10 需用 master，不要用 pip 上的舊版）
pip install git+https://github.com/NVIDIA-AI-IOT/torch2trt.git

# 5. NanoOWL（editable）；NanoSAM 的 setup.py 無效，改用 .pth
pip install -e ./third_party/nanoowl
SP=$(python -c "import site; print(site.getsitepackages()[0])")
echo "$(pwd)/third_party/nanosam" > "$SP/nanosam_path.pth"

python - <<'PY'
import torch, tensorrt, torch2trt, nanoowl, nanosam.utils.predictor
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))
print("tensorrt", tensorrt.__version__)
PY
echo "OK。之後請先: source $VENV/bin/activate"
