# 安裝說明

本專案有兩個環境：**WSL + GTX 1060**（開發用，PyTorch 後端）與 **Jetson Orin Nano**（正式部署，TensorRT 後端）。
所有安裝與下載腳本都在 `installation/`，指令一律從專案根目錄執行。

## 0. 取得程式碼

NanoOWL 與 NanoSAM 以 git submodule 放在 `third_party/`：

```bash
git clone --recurse-submodules <repo-url>
cd smart-processing-midterm-project-2026Fall
# 已經 clone 過但沒帶 submodule：
git submodule update --init --recursive
```

## 1. WSL 開發環境（conda）

### 1.1 環境內容

| 項目 | 版本 | 說明 |
|---|---|---|
| conda 環境名稱 | `nanoowl-sam` | 定義於 `environment.yml` |
| Python | 3.10 | |
| PyTorch / torchvision | 2.5.1 / 0.20.1（cu121） | cu121 仍含 Pascal（GTX 1060, sm_61）可用的 kernel；cu128 之後已移除 |
| transformers | 4.46.3 | OWL-ViT（`OwlViTForObjectDetection`） |
| timm | 1.0.11 | MobileSAM 的 TinyViT |
| numpy | <2 | |
| 其他 | pillow、opencv-python、matplotlib、requests、tqdm、gdown | 完整清單見 `requirements.txt` |

檔案分工：
- `requirements.txt`：pip 套件清單（含 PyTorch 的 cu121 index）。
- `environment.yml`：conda 環境定義，內部引用 `requirements.txt` 並以 editable 安裝 `third_party/nanoowl`。
- `requirements-trt.txt`：選用的 TensorRT 後端（`tensorrt==8.6.1`、`onnx`），WSL 不一定需要。

### 1.2 建立環境

```bash
./installation/setup_env.sh
conda activate nanoowl-sam
```

`setup_env.sh` 做三件事：
1. 若 `nanoowl-sam` 不存在，用 `environment.yml` 建立。
2. 寫入 `nanosam_path.pth`，把 `third_party/nanosam` 加進環境的 import 路徑。
   > NanoSAM 的 `setup.py` 有問題（`nanosam/` 缺少 `__init__.py`，`find_packages` 找不到任何套件），`pip install -e` 實際上什麼都沒裝，所以改用 `.pth`。
3. 測試 `import nanosam.mobile_sam, nanoowl`。

### 1.3 驗證

```bash
conda activate nanoowl-sam
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 預期: True NVIDIA GeForce GTX 1060 6GB
```

PyTorch 回報的 arch list 只有 `sm_60` 沒有 `sm_61`，屬正常現象（同一世代相容），實測 GPU 運算正常。

### 1.4 常用 conda 指令

```bash
conda env list                          # 列出環境
conda activate nanoowl-sam              # 啟用
conda deactivate                        # 離開
conda env remove -n nanoowl-sam         # 刪除（要重建時用）
pip install -r requirements.txt         # 在已啟用的環境中補裝套件
```

### 1.5 （選用）TensorRT 後端

```bash
pip install -r requirements-trt.txt
pip install git+https://github.com/NVIDIA-AI-IOT/torch2trt.git
```
Pascal 只能用 TensorRT 8.x（10.x 已移除支援）。裝不起來可忽略，WSL 以 PyTorch 後端開發即可，`nanosam.utils.predictor` 會因為缺少 `tensorrt` / `torch2trt` 無法 import。

## 2. 下載模型權重

```bash
./installation/download_model.sh              # 全部
./installation/download_model.sh owlvit       # OWL-ViT（HuggingFace 快取，約 600MB）
./installation/download_model.sh mobilesam    # data/models/mobile_sam.pt（約 41MB，WSL 的 PyTorch 後端用）
./installation/download_model.sh nanosam      # data/models/ 下的兩個 ONNX（Jetson 轉引擎用）
```

| 模型 | 用途 | 來源 | 放置位置 |
|---|---|---|---|
| OWL-ViT `google/owlvit-base-patch32` | 文字 → 物件框 | HuggingFace | `~/.cache/huggingface` |
| MobileSAM `mobile_sam.pt` | 框 → 遮罩（WSL 用） | GitHub ChaoningZhang/MobileSAM | `data/models/` |
| `resnet18_image_encoder.onnx` | NanoSAM image encoder | **手動下載**：[Google Drive](https://drive.google.com/file/d/14-SsvoaTl-esC3JOzomHDnI9OGgdO2OR/view)（需登入 Google，gdown 無法下載） | `data/models/` |
| `mobile_sam_mask_decoder.onnx` | NanoSAM mask decoder | HuggingFace `dragonSwing/nanosam`（腳本自動下載，約 16MB） | `data/models/` |

已存在的檔案會略過；`data/` 已在 `.gitignore`。`resnet18_image_encoder.onnx` 請用瀏覽器下載後存成 `data/models/resnet18_image_encoder.onnx`；腳本在檔案缺少時會印出提示。

> 注意：WSL 用的 MobileSAM 與 Jetson 最終用的 NanoSAM（ResNet18 encoder）不是同一個模型，準確度與效能數據一律以 Jetson 為準。

## 3. Jetson Orin Nano（JetPack 6.2）

腳本依編號順序執行：

```bash
./installation/jetson/00_check_jetpack.sh     # 檢查 JetPack / CUDA / TensorRT / 電源模式
./installation/jetson/04_setup_swap.sh        # 建 8GB swap（建議先做，避免 OOM）
./installation/jetson/01_install_env.sh       # 建 .venv-jetson，安裝 Jetson 版 PyTorch、torch2trt 等
./installation/jetson/02_download_models.sh   # OWL-ViT + NanoSAM ONNX
./installation/jetson/03_build_engines.sh     # 建三個 TensorRT 引擎 → data/*.engine
./installation/jetson/05_set_power_mode.sh    # 量測前固定電源模式（可帶模式編號）
source .venv-jetson/bin/activate              # 之後每次使用前啟用
```

Jetson 不使用 conda，用 Python venv（`--system-site-packages`，才能使用 JetPack 內建的 TensorRT）。
版本清單、已知相容性問題與對策見 [jetson-setup.md](jetson-setup.md)。

## 4. 目錄對照

```
installation/
├── setup_env.sh              WSL: 建 conda 環境
├── download_model.sh         下載模型（WSL 與 Jetson 共用）
└── jetson/                   Jetson 部署腳本 00–05
environment.yml               conda 環境定義
requirements.txt              pip 套件（WSL）
requirements-trt.txt          選用 TensorRT 套件
third_party/nanoowl           submodule
third_party/nanosam           submodule
data/                         模型與引擎（不進版控）
```
