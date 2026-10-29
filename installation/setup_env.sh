#!/usr/bin/env bash
# 建立 conda 環境 nanoowl-sam，並讓 nanosam 可被 import
set -euo pipefail
cd "$(dirname "$0")/.."
conda env list | grep -q '^nanoowl-sam ' || conda env create -f environment.yml
SP=$(conda run -n nanoowl-sam python -c "import site; print(site.getsitepackages()[0])")
echo "$(pwd)/third_party/nanosam" > "$SP/nanosam_path.pth"
conda run -n nanoowl-sam python -c "import nanosam.mobile_sam, nanoowl; print('ok')"
