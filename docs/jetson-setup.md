# Jetson Orin Nano（JetPack 6.2）部署步驟與版本相容性筆記

> **一步一步的操作請看 [jetson-deploy.md](jetson-deploy.md)；要帶去的資料與量測記錄表請看 [jetson-data.md](jetson-data.md)。** 本文件保留版本預期值與已知衝突清單。

## 目標環境（預期值，請以 `00_check_jetpack.sh` 實際輸出為準並回填報告）

| 項目 | 版本 |
|---|---|
| JetPack / L4T | 6.2 / R36.4.x |
| Ubuntu / Python | 22.04 / 3.10 |
| CUDA / cuDNN | 12.6 / 9.x |
| TensorRT | 10.3（JetPack 內建，**不可 pip 安裝**） |
| PyTorch / torchvision | 2.8.0 / 0.23.0（Jetson 專用 wheel，index: `https://pypi.jetson-ai-lab.io/jp6/cu126`） |
| numpy | <2 |
| transformers | 4.46.3 |

## 執行順序

```bash
git clone --recurse-submodules <repo-url> && cd <repo>
./installation/jetson/00_check_jetpack.sh     # 確認版本
./installation/jetson/04_setup_swap.sh        # 建議先做，避免 OOM
./installation/jetson/01_install_env.sh       # 建 .venv-jetson 並安裝套件
./installation/jetson/02_download_models.sh   # OWL-ViT + NanoSAM ONNX
./installation/jetson/03_build_engines.sh     # 建三個 TensorRT 引擎
./installation/jetson/05_set_power_mode.sh    # 量測前固定電源模式（記錄於報告）
```

## 已知版本衝突與對策（寫報告用）

| 問題 | 原因 | 對策 |
|---|---|---|
| `pip install torch` 得到 CPU 版或 `not a supported wheel` | PyPI 沒有 aarch64+CUDA 的 wheel | 使用 Jetson AI Lab index（見上） |
| `torch.cuda.is_available()` 為 False / `no kernel image` | 裝到 CPU 版，或缺 cuSOLVER 等 CUDA 函式庫 | 安裝 `libcusolver-12-6` 等（01 腳本已包含），重裝 torch |
| numpy 2.x 造成 torch / cv2 import 失敗 | wheel 以 numpy 1.x 編譯 | 鎖定 `numpy<2` |
| `import tensorrt` 失敗（venv 內） | venv 隔離了系統套件 | `python3 -m venv --system-site-packages` |
| `libnvdla_compiler.so` ImportError（只出現在容器內） | 容器未掛載 tegra 函式庫 | 直接在宿主機安裝，不用容器；若用容器需帶 `--runtime nvidia` |
| 安裝 `tensorrt` 的 pip 版覆蓋系統版 | pip 版與 JetPack 的 TRT 版本不一致 | 絕不 `pip install tensorrt` |
| torch2trt 舊版不支援 TensorRT 10 | pip 上的版本過舊 | 從 GitHub master 安裝 |
| `nanosam` 無法 `pip install -e` | `nanosam/__init__.py` 不存在，`find_packages` 找不到 | 以 `.pth` 加入路徑 |
| 建引擎 / 編譯 OOM（8GB 共用記憶體） | GPU 與 CPU 共用記憶體 | 04 腳本建 8GB swap；建引擎時關閉桌面程式 |
| NanoOWL 找不到 `trtexec` | 路徑假設為 `/usr/src/tensorrt/bin/trtexec` | 確認 JetPack 完整安裝（`nvidia-jetpack`） |
| 引擎在 WSL 建好卻無法在 Jetson 載入 | 引擎綁 GPU 架構與 TRT 版本 | 引擎一律在 Jetson 上建 |

## 尚未實機驗證的風險

這些腳本是依 JetPack 6.2 的公開資訊撰寫，**尚未在實機執行**，第一次跑時請注意：
- Jetson AI Lab 的 index 網址與可用版本會變動；若 `torch==2.8.0` 找不到，到該 index 頁面查可用版本，或改設 `JP_INDEX`。
- `nanosam/utils/predictor.py` 的 `TRTModule` 在 TensorRT 10 下是否正常，需實測；若失敗，需改用 TRT 10 的 tensor API（`num_io_tensors`），這也是報告「版本相容性問題」的素材。
- NanoSAM 的 `resnet18_image_encoder.onnx` 只在 Google Drive（需登入，gdown 無法下載），請在電腦下載後放到 Jetson 的 `data/models/`（或用 scp）；`mobile_sam_mask_decoder.onnx` 由腳本從 HuggingFace 下載，建引擎前請確認輸入名稱與官方一致（`point_coords`、`point_labels`）。
