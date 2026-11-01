#!/usr/bin/env bash
# Jetson Orin Nano (JetPack 6.2 / L4T 36.4 / CUDA 12.6 / TensorRT 10.3 / Python 3.10) 環境安裝
# 用法: ./installation/jetson/01_install_env.sh
#
# 重點（每一條都是在實機踩過的，細節見 docs/record.md §13）：
#  * venv 必須用 --system-site-packages，才能看到 JetPack 內建的 tensorrt（不要 pip install tensorrt）
#  * torch / torchvision 必須用 Jetson 專用 wheel，且「只能」指定 Jetson index：
#    加 --extra-index-url pypi.org 會讓 pip 選到 PyPI 的 aarch64 CPU 版（2.8.0+cpu）
#  * numpy 必須固定 1.26.4 並用 constraints 保護：系統是 1.21.5（"numpy<2" 會被它滿足而不安裝），
#    而其他套件（ml_dtypes 0.6）的依賴會把 numpy 升到 2.x，Jetson 的 torch 就會 "Failed to initialize NumPy"
#  * torch2trt 要 --no-build-isolation：它的 setup.py 會 import tensorrt，隔離環境看不到系統的 tensorrt
set -euo pipefail
cd "$(dirname "$0")/../.."

JP_INDEX=${JP_INDEX:-https://pypi.jetson-ai-lab.io/jp6/cu126}
VENV=${VENV:-.venv-jetson}
TORCH_VER=${TORCH_VER:-2.8.0}
TV_VER=${TV_VER:-0.23.0}
ORT_VER=${ORT_VER:-1.23.0}

sudo apt-get update
sudo apt-get install -y python3-pip python3-venv python3-dev git wget \
    libopenblas-dev libjpeg-dev zlib1g-dev python3-libnvinfer python3-libnvinfer-dev

# cuSOLVER / cuSPARSE / cuBLAS 缺少時 torch 會出現 libcusolver / "no kernel image" 錯誤
sudo apt-get install -y libcusolver-12-6 libcusolver-dev-12-6 libcusparse-12-6 \
    libcublas-12-6 libcudnn9-cuda-12 || true

git submodule update --init --recursive

python3 -m venv --system-site-packages "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install -U pip setuptools wheel

# 0. 固定 numpy / ml_dtypes：之後所有 pip（含 torch2trt、nanoowl 的建置環境）都受此限制，想升 numpy 會直接報衝突
CONSTRAINTS="$(pwd)/$VENV/pip-constraints.txt"
printf 'numpy==1.26.4\nml_dtypes<0.6\n' > "$CONSTRAINTS"
export PIP_CONSTRAINT="$CONSTRAINTS"
pip install "numpy==1.26.4"

# 1. Jetson 版 PyTorch：只用 Jetson index、--no-deps（避免從 PyPI 抓到 CPU 版），依賴另外從 PyPI 裝
pip install --no-deps --index-url "$JP_INDEX" "torch==$TORCH_VER" "torchvision==$TV_VER"
pip install filelock "typing-extensions>=4.10" "sympy>=1.13.3" networkx jinja2 fsspec

# 在繼續之前確認拿到的是 CUDA 版（CPU 版版本字串帶 +cpu，且 torch.version.cuda 為 None）
python - <<'PY'
import torch
assert "+cpu" not in torch.__version__ and torch.version.cuda, f"裝到 CPU 版 torch: {torch.__version__}（檢查 index 設定）"
print("torch", torch.__version__, "CUDA", torch.version.cuda)
PY

# 2. 其餘純 Python 套件（不含 torch / tensorrt）+ Web 後端
pip install transformers==4.46.3 timm==1.0.11 pillow matplotlib requests tqdm gdown onnx \
    fastapi "uvicorn[standard]" python-multipart
# opencv：JetPack 已有 python3-opencv（含 GStreamer），不要 pip 安裝以免蓋掉
sudo apt-get install -y python3-opencv

# 3. onnxruntime-gpu：PyPI 沒有 aarch64 的 GPU 版，只能用 Jetson index；--no-deps 後要手動補依賴
pip install --no-deps --index-url "$JP_INDEX" "onnxruntime-gpu==$ORT_VER"
pip install flatbuffers coloredlogs packaging protobuf

# 4. torch2trt（TensorRT 10 需用 master，不要用 pip 上的舊版；需 --no-build-isolation）
pip install --no-build-isolation git+https://github.com/NVIDIA-AI-IOT/torch2trt.git

# 5. NanoOWL（editable）；NanoSAM 的 setup.py 無效，改用 .pth
pip install -e ./third_party/nanoowl
SP=$(python -c "import site; print(site.getsitepackages()[0])")
echo "$(pwd)/third_party/nanosam" > "$SP/nanosam_path.pth"

python - <<'PY'
import numpy, torch, tensorrt, torch2trt, nanoowl, nanosam.utils.predictor, onnxruntime
assert numpy.__version__.startswith("1."), f"numpy 被升到 {numpy.__version__}"
print("numpy", numpy.__version__)
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))
print("tensorrt", tensorrt.__version__, "| torch2trt", getattr(torch2trt, "__version__", "?"), "| onnxruntime", onnxruntime.__version__, onnxruntime.get_available_providers())
PY
echo "OK。之後請先: source $VENV/bin/activate && export PIP_CONSTRAINT=$CONSTRAINTS"
