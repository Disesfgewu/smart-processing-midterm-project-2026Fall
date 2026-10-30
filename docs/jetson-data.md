# Jetson 部署：資料清單與量測記錄表

操作步驟在 [jetson-deploy.md](jetson-deploy.md)。這份文件有三個用途：**① 要帶去 Jetson 的資料、② 官方參考數據、③ 要填的量測記錄表**（空白處在 Jetson 上量完後填入，之後直接搬進報告）。

## 1. 要帶去 Jetson 的資料（不在 git 裡）

由 `./installation/jetson/pack_for_jetson.sh` 打包成 `data/jetson_files.tar.gz`（約 160 MB），解壓到 Jetson 的專案根目錄，再 `sha256sum -c SHA256SUMS`。

| 檔案（相對專案根目錄） | 大小 | sha256 開頭 | 用途 |
|---|---|---|---|
| `data/models/resnet18_image_encoder.selftrain.onnx` | 63 MB | `0806681c32d4366d` | 自訓練 NanoSAM encoder → `03_build_engines.sh` 轉 TensorRT。Colab 79 epoch、held-out mIoU 0.704；ONNX 與 PyTorch 輸出 cosine 1.000、最大差 9.5e-7 |
| `data/models/mobile_sam_mask_decoder.onnx` | 16.5 MB | `41e49a2980990481` | mask decoder → 轉 TensorRT（與官方 NanoSAM 共用） |
| `data/models/mobile_sam.pt` | 40.7 MB | `6dbb90523a35330f` | baseline（目前 SAM 走 PyTorch 需要；也是比較基準，**不可被覆寫**） |
| `data/eval/colab80_best.pth` | 63 MB | `6922d08d12830414` | 自訓練 encoder 的 PyTorch 權重，讓 Jetson 網頁也能切換比較 |
| `data/samples/cats.jpg` | 173 KB | — | 量測用影像（與 WSL 同一張） |
| `data/samples/demo_h264.mp4` | 0.7 MB | — | 影片測試 |
| *（選）* `data/models/resnet18_image_encoder.onnx` | — | — | 官方 encoder，手動下載後才有；有的話可與自訓練版並排量 |
| *（選）* `data/jetson_hf_cache.tar.gz` | 約 590 MB | — | OWL-ViT 權重（`WITH_HF_CACHE=1`）；Jetson 連得上 HuggingFace 可省略 |

> 引擎（`*.engine`）**不能**事先建好帶過去，它綁定 GPU 架構與 TensorRT 版本。

## 2. 在 Jetson 上會產生的資料

| 路徑 | 內容 | 帶回來？ |
|---|---|---|
| `data/owl_image_encoder_patch32.engine` | OWL 引擎 | 否 |
| `data/resnet18_image_encoder.selftrain.engine` | 自訓練 encoder 引擎 | 否 |
| `data/mobile_sam_mask_decoder.engine` | decoder 引擎 | 否 |
| `data/bench/jetson/env_*.md` | 環境版本（`collect_env.sh`） | **是** |
| `data/bench/jetson/engines_*/summary.md` + `*.log` | 各引擎 trtexec 延遲 | **是** |
| `data/bench/jetson/pipe_*.json` | 整體流程分段延遲 + 記憶體 | **是** |
| `data/bench/jetson/tegrastats_*.log` | 量測期間功耗 / 溫度 / 記憶體 | **是** |
| `data/bench/jetson/app.log` | 服務日誌 | 視需要 |
| `docs/screenshots/*.png` | 截圖 | **是** |

## 3. 官方參考數據（NanoSAM / NanoOWL README；詳見 [benchmark.md](benchmark.md)）

| 項目 | 官方 | 備註 |
|---|---|---|
| NanoOWL OWL-ViT B/32 | AGX Orin 95 FPS（≈10.5 ms）；**Orin Nano：TBD** | TensorRT |
| MobileSAM | encoder：AGX Orin 35 ms；完整流程 AGX Orin 39 ms、**Orin Nano 146 ms**；mIoU 0.728 | encoder fp32 |
| NanoSAM (ResNet18) | encoder：AGX Orin 4.2 ms；完整流程 AGX Orin 8.1 ms、**Orin Nano 27 ms**；mIoU 0.706 | encoder fp16、decoder fp32 |

官方「完整流程」= image encoder + mask decoder（單一提示），**不含 OWL**。Orin Nano 只有 SAM 完整流程有官方數字；OWL 與 encoder 單項是 TBD，我們實測的就是新數據。

## 4. 量測記錄表（在 Jetson 上填）

### 4.1 環境（貼 `env_*.md` 的重點）

| 項目 | 預期 | **實測** |
|---|---|---|
| 裝置 | Jetson Orin Nano | |
| JetPack / L4T | 6.2 / R36.4.x | |
| CUDA | 12.6 | |
| TensorRT | 10.3 | |
| Python | 3.10 | |
| PyTorch / torchvision | 2.8.0 / 0.23.0（Jetson wheel） | |
| torch2trt | master | |
| 電源模式 | MAXN | |
| 記憶體 / swap | 8 GB / 8 GB | |

### 4.2 引擎建置

| 引擎 | 精度 | 建置時間 | 大小 | 遇到的問題 |
|---|---|---|---|---|
| OWL image encoder (patch32) | fp16 | | | |
| 自訓練 NanoSAM encoder | fp16 | | | |
| mask decoder | fp32 | | | |

### 4.3 各引擎純 GPU 延遲（`06_measure_engines.sh` 的 summary；電源模式：＿＿＿）

| 引擎 | mean (ms) | median | p99 | FPS | 官方（Orin Nano） | 備註 |
|---|---|---|---|---|---|---|
| OWL image encoder | | | | | TBD | |
| 自訓練 NanoSAM encoder | | | | | TBD | |
| mask decoder（2 點） | | | | | — | |
| encoder + decoder（上兩列相加） | | | | | 27 ms | 官方 NanoSAM 完整流程 |
| *（選）* 官方 NanoSAM encoder | | | | | TBD | 有官方 ONNX 時 |

### 4.4 整體流程分段（`bench_pipeline.py`；cats.jpg、`a cat`/`a remote control`、4 個框）

WSL 欄位來自 [benchmark.md](benchmark.md)（GTX 1060、PyTorch fp32，**僅供對照量測方法，不可當成繳交數據**）。

| 階段 | WSL baseline | WSL 自訓練 | Jetson：OWL=TRT + MobileSAM(PyTorch) | Jetson：OWL=TRT + 自訓練(PyTorch) | Jetson：全 TensorRT |
|---|---|---|---|---|---|
| OWL 影像編碼 (ms) | 72.7 | 75.9 | | | |
| OWL 合計 (ms) | 74.5 | 75.3 | | | |
| SAM 影像編碼 (ms) | 88.5 | 32.8 | | | *（SAM 轉接層完成後）* |
| SAM decoder / 框 (ms) | 7.9 | 7.8 | | | |
| **完整流程 (ms)** | 197.2 | 159.1 | | | |
| **FPS** | 5.1 | 6.3 | | | |

### 4.5 記憶體

| 設定 | 系統記憶體增加 (MB) | 本程序 RSS 峰值 (MB) | tegrastats 峰值 RAM | 備註 |
|---|---|---|---|---|
| OWL(TRT) + MobileSAM(PyTorch) | | | | |
| OWL(TRT) + 自訓練(PyTorch) | | | | |
| 全 TensorRT | | | | |
| *（選）* 加上 Ollama（HyDE） | | | | 預計很吃記憶體，量過再決定 Jetson 上是否開 |

Jetson 的 GPU 與 CPU 共用記憶體，所以看系統記憶體，不是 GPU 專屬記憶體。

### 4.6 電源模式比較（至少一種；多做加分）

| 模式 | nvpmodel 編號 | OWL TRT (ms) | encoder TRT (ms) | 完整流程 FPS | 平均功耗 (W)* | 溫度 (°C) |
|---|---|---|---|---|---|---|
| MAXN | | | | | | |
| 25W（若有） | | | | | | |
| 15W | | | | | | |

\* 功耗從 `tegrastats` 的 `VDD_IN` 欄位讀取，取量測期間平均。

### 4.7 準確度（需求 5，另見 [experiments.md](experiments.md)）

| encoder | held-out mIoU（200 張，GT 框） | Small | Medium | Large |
|---|---|---|---|---|
| baseline MobileSAM | 0.744 | 0.685 | 0.756 | 0.822 |
| 自訓練（Colab 79 epoch） | 0.704 | 0.628 | 0.725 | 0.796 |
| 官方 NanoSAM（參考） | 0.706 | 0.624 | 0.738 | 0.796 |

這是在 WSL 以 PyTorch 量的，與 Jetson 的 TensorRT fp16 引擎精度可能略有出入；若要宣稱「引擎精度與 PyTorch 一致」，需在 Jetson 另跑一次（尚未做）。

## 5. 截圖清單（至少 5 張，含失敗案例；存 `docs/screenshots/`）

| # | 內容 | 檔名建議 | 完成 |
|---|---|---|---|
| 1 | 終端機：`00_check_jetpack` / `collect_env` 版本資訊 | `01_env.png` | ☐ |
| 2 | 終端機：三個引擎建置成功與檔案大小 | `02_engines.png` | ☐ |
| 3 | 網頁：即時攝影機，多個物件的框 + 輪廓 + FPS 統計 | `03_live.png` | ☐ |
| 4 | 網頁：製造情境（零件檢查卡 / 報表） | `04_scenario.png` | ☐ |
| 5 | 網頁：串流中更換提示詞前後各一張 | `05_prompt_change_a.png` / `_b.png` | ☐ |
| 6 | **失敗案例**：漏偵測 / 誤偵測 / 輪廓不準（至少 1 張，附原因分析） | `06_fail_*.png` | ☐ |
| 7 | baseline 與自訓練 encoder 同一張圖的差異 | `07_encoder_compare.png` | ☐ |
| 8 | `tegrastats` 量測中的功耗 / 記憶體 | `08_tegrastats.png` | ☐ |

## 6. 版本問題記錄（每遇到一個就填一列；報告要求）

| # | 步驟 | 指令 | 錯誤訊息（關鍵一行） | 原因 | 解法 | 花費時間 |
|---|---|---|---|---|---|---|
| 1 | | | | | | |
| 2 | | | | | | |

> 填寫原則：寫**實際查證到的原因**；沒查證的推測要標明「推測」。解法要能讓別人照做。

## 7. 還沒做、會影響第 4 節的事

- **SAM 的 TensorRT 轉接層**（包裝 `nanosam.utils.predictor.Predictor`）：沒有它，4.4 的「全 TensorRT」欄位量不出來，也無法示範「全 TensorRT 的即時攝影機」。這是 Jetson 實測後最優先補的程式。
- 自訓練 encoder 在 TensorRT fp16 下的準確度驗證（見 4.7 備註）。
- 實體攝影機與真實製造素材：見 [scenario.md](scenario.md)。
