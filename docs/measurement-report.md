# Jetson Orin Nano 量測報告（OWL-ViT + 自訓練 ResNet18 SAM encoder，TensorRT）

> 量測日期：2026-11-01　裝置：NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super（8GB 共用記憶體）
> 資料來源：`data/bench/jetson/` 與 `app.log` 的原始檔（本檔數字由腳本從原始檔計算，非手抄）。限制與尚未完成的項目見最後一節。

> **名詞說明（請先讀）**：本專案與網頁介面裡的「NanoSAM」指的是 **NanoSAM 的推論流程與架構**（官方程式碼：框轉兩個提示點、前後處理、TensorRT 載入；ResNet18 image encoder ＋ 與 MobileSAM 相同的 mask decoder）。**權重不是官方的**：官方預訓練檔 `resnet18_image_encoder.onnx` 放在需要登入的 Google Drive，沒有取得，所以我們自己做知識蒸餾（把 MobileSAM 的 TinyViT encoder 蒸餾進 ResNet18，在 Colab 訓練 79 個 epoch）。部署用的 ONNX／TensorRT 引擎裡就是這份自訓練權重：實測 `resnet18_image_encoder.selftrain.onnx` 的輸出與 `colab80_best.pth`（epoch 78）完全一致（cosine = 1.000000，最大差異 0.00000），與 epoch 2 檢查點不同（0.951）、與隨機初始化約為 0。因此網頁上「自訓練 ResNet18 encoder（PyTorch fp32）」與「（TensorRT）」是**同一份權重**，只差執行方式（fp32 PyTorch ／ fp16 TensorRT）。
> 準確度（COCO 200 張、GT 框當提示，`docs/record.md` §7.4）：baseline MobileSAM mIoU **0.744**、自訓練 epoch 78 **0.704**、官方 NanoSAM 0.706（官方條件不同，僅供參考）。**架構相同 → 延遲可與官方對照；權重不同 → 準確度不可。**

## 0. 摘要

- **端對端延遲**（客戶端送出請求到收到完整結果；真實網頁 API、loopback、每格 50 次、MAXN_SUPER，§6）：全 TensorRT **129 ms**（640×480，約 7.7 請求/秒）、169 ms（1280×960）、628 ms（4096×3072）；PyTorch fp32 baseline 為 461 / 498 / 961 ms，**小圖快 3.6 倍，4K 只快 1.5 倍**（TensorRT 只加速模型，解碼與輪廓序列化等 CPU 端成本沒變）。
- **分階段**（`bench_pipeline.py`，只含推論呼叫，不含圖片解碼、序列化與 HTTP；640×480、4 個框）：整體 99.4 ms（10.1 FPS），其中 OWL 19.5 ms（51.3 FPS）、SAM 影像編碼器 29.7 ms、mask decoder 每個框 9.8 ms。**這組數字比端對端小約 30 ms，不要拿它當使用者感受到的延遲。**
- **功耗**：閒置 5.8 W → 全 TensorRT 負載中約 11.6 W（PyTorch baseline 15.2 W）；每個請求能耗約 1.5 J，PyTorch 約 7.0 J（約 4.7 倍）。接面溫度最高 52.3 °C；`tegrastats` 此版本沒有逐點時脈，**無法保證**整個過程完全沒降頻。
- **大圖時瓶頸不在模型**：手機照片（4284×5712，伺服器縮成 3072×4096）每次請求解碼＋縮放固定約 681 ms，另有 0.4–3.9 秒的上傳／傳輸浮動；SAM 每個框約 36 ms（小圖 10 ms）；偵測數量（閾值）才是 SAM 耗時的主因（§7）。
- **與官方對照**：官方 Orin Nano 的 SAM encoder 與 NanoOWL FPS 都是 TBD；可對照的是 NanoSAM(ResNet18) Full Pipeline 27 ms（架構相同、權重不同）。純 GPU（trtexec）下，MAXN_SUPER 為 18.2 ms（比官方快），切到 15W 為 26.3 ms（與官方只差 2.6%）；官方未公布電源模式與計時範圍，詳見 §8。

## 1. 測試環境

| 項目 | 內容 |
|---|---|
| 裝置 | Jetson Orin Nano Engineering Reference Developer Kit Super，8GB（CPU 與 GPU 共用） |
| L4T / JetPack | R36.4.7（JetPack 6.2.1 對應） |
| 電源模式 | **MAXN_SUPER（nvpmodel 2）**；GPU 時脈 1020 MHz（`/sys` 讀值） |
| CUDA / cuDNN / TensorRT | 12.6.68 / 9.3.0 / 10.3.0 |
| PyTorch / torchvision / torch2trt | 2.8.0 / 0.23.0 / 0.5.0（GitHub master） |
| 模型 | OWL-ViT base patch32（HF `google/owlvit-base-patch32`）；SAM image encoder＝**自行蒸餾的 ResNet18**（`colab80_best.pth`，epoch 78；已用 ONNX 輸出逐項比對確認引擎內是這份權重，見上方名詞說明）；MobileSAM mask decoder |
| 引擎精度 | OWL fp16；自訓練 ResNet18 encoder fp16；decoder fp32 |
| 測試影像 | `data/samples/cats.jpg`（COCO val2017 000000039769，640×480，2 隻貓＋2 個遙控器）；網頁大圖測試為使用者的手機照片（4284×5712，伺服器縮成 3072×4096） |
| 提示詞 | `a cat`、`a remote control`（基準測試，4 個框） |
| 軟體環境注意 | 量測時系統上沒有其他重負載（gazebo 已停止、改用 SSH、Ollama 無模型載入）；`MemAvailable` 起點約 5.0 GB |

## 2. 量測方法

- **引擎層級**：`trtexec`（`--fp16` 或 fp32），純 GPU 推論，不含前後處理。數值取 trtexec 預設的 warmup 與迭代統計。
- **流程層級**：`benchmark/bench_pipeline.py`，每個階段先暖機、再重複 50 次，每次前後 `torch.cuda.synchronize()`。同一組參數連跑兩次以檢查重現性：整體流程 99.1 ms vs 99.4 ms，OWL 合計 19.7 vs 19.5 ms。
- **功耗／溫度／記憶體**：`tegrastats --interval 500`（不需 root）與基準測試同時執行；`VDD_IN` 是板子輸入功耗，`VDD_CPU_GPU_CV` 是 CPU＋GPU＋CV 電源軌。「穩態」＝ GPU 使用率 > 0 的取樣點去掉前後各 20%。
- **網頁 API**：伺服器日誌（`app.log`）逐請求記錄的各階段耗時。

## 3. 引擎層級（trtexec，純 GPU）

| 引擎 | 檔案大小 (MiB) | 建置時間 (s) | GPU 運算平均 (ms) | 延遲平均 (ms) | 延遲中位數 (ms) | 延遲 p99 (ms) | 吞吐 (qps) |
|---|---|---|---|---|---|---|---|
| OWL-ViT image encoder（fp16，768×768） | 175 | 135 | 15.32 | 16.13 | 16.11 | 16.84 | 64.9 |
| SAM image encoder（自訓練 ResNet18，fp16，1024×1024） | 31 | 127 | 9.91 | 11.62 | 11.61 | 11.81 | 100.5 |
| mask decoder（fp32，動態點數 1~10） | 18 | 29 | 8.20 | 8.68 | 8.67 | 8.82 | 121.5 |

延遲包含 host↔device 傳輸，GPU 運算不含。建置時間是整個 `trtexec` 的執行時間（含 ONNX 解析與 TensorRT 的 tactic 搜尋，不含 Python 啟動）；OWL 引擎另需先在 CPU 匯出 ONNX（`installation/jetson/build_owl_engine.py`）。

## 4. 流程層級（`bench_pipeline.py`，50 次，640×480、4 個框）

| 階段 | 平均 (ms) | 中位數 (ms) | p95 (ms) | 標準差 (ms) | FPS |
|---|---|---|---|---|---|
| OWL 前處理 | 1.29 | 1.28 | 1.32 | 0.02 | 776.5 |
| OWL 影像編碼器（TensorRT） | 16.88 | 16.86 | 16.99 | 0.10 | 59.2 |
| OWL 解碼 | 1.92 | 1.91 | 1.98 | 0.04 | 520.5 |
| **OWL 合計** | 19.49 | 19.43 | 19.87 | 0.18 | 51.3 |
| OWL 文字編碼器（只在換提示詞時執行） | 25.72 | 25.59 | 26.15 | 0.26 | 38.9 |
| SAM 影像編碼器（自訓練 ResNet18，含前處理） | 29.69 | 28.86 | 32.04 | 1.30 | 33.7 |
| SAM mask decoder（每個框） | 9.77 | 9.73 | 9.92 | 0.17 | 102.3 |
| **整體流程（4 個框）** | 99.43 | 99.02 | 100.98 | 1.04 | 10.1 |

- 各階段加總（OWL 合計 + SAM 編碼器 + 4 × decoder）= 88.3 ms，與整體 99.4 ms 相差 **11.2 ms**，這是 NMS、遮罩後處理等沒有逐項量的部分。
- 純 GPU（trtexec）的 SAM encoder 只有 11.6 ms，流程中的 29.7 ms 多出的約 18 ms 是影像前處理（縮放、正規化、補邊）與資料搬移。**這個拆分是由兩個數字相減推得，沒有分別計時前處理，屬推測。**
- 整體 FPS 為「4 個框」的結果；框數不同會線性改變（每個框約 9.8 ms）。

## 5. 功耗、溫度、記憶體（`tegrastats`，500 ms 取樣）

| 狀態 | 取樣數 | 板子輸入 `VDD_IN` 平均 (W) | `VDD_IN` 最大 (W) | CPU+GPU+CV 軌平均 (W) | GPU 使用率 平均 % | CPU 平均 % | 接面溫度最高 (°C) | 系統 RAM 已用 (MB) |
|---|---|---|---|---|---|---|---|---|
| 閒置（10 秒） | 19 | 5.80 | 5.84 | 1.63 | 0（最高 0） | 3 | 49.8 | 1557–1558 |
| 基準測試（整段） | 54 | 9.81 | 18.07 | 4.42 | 28（最高 98） | 17 | 52.3 | 1583–3489 |
| 基準測試（GPU 有負載） | 28 | 12.64 | 18.07 | 6.45 | 54（最高 98） | 17 | 52.3 | 1976–3479 |
| **基準測試（穩態）** | 18 | 13.31 | 18.07 | 6.95 | 57（最高 94） | 17 | 52.3 | 3156–3479 |

- 推論時 GPU 使用率平均只有約 57%：流程中有不少 CPU 端工作（前處理、NMS、遮罩後處理）和串行等待，GPU 沒有被吃滿。
- 程序記憶體（`bench_pipeline.py`）：torch 配置峰值 678 MB、保留 696 MB；程序 RSS 峰值 2549 MB（含載入的 CUDA 函式庫對映）；系統已用量 2018 → 3597 MB。Jetson 的 `nvidia-smi` 記憶體欄位為 N/A，所以不報告 GPU 專屬記憶體。
- **記憶體是最大的實務限制**：建 OWL 引擎的 `trtexec` 需要約 3GB 以上共用記憶體，`MemAvailable` 只有 3.3GB 時連續失敗 4 次，約 5.0GB 時一次成功（`docs/record.md` §13.2 第 7 點）。

## 6. 端對端延遲（各模型組合，受控量測）

**定義**：從客戶端送出 `POST /api/segment` 到收到完整 JSON 的時間，涵蓋圖片解碼、OWL、NMS、SAM、遮罩輪廓與序列化、回應組裝與 HTTP。用 `benchmark/bench_e2e.py` 對每種組合各啟動一次真正的 `python -m server.main`，經 loopback 連續送 **50 次**（暖機 5 次，單一客戶端、一次一個請求），提示詞 `a cat, a remote control`、閾值 0.1，電源模式 **MAXN_SUPER（nvpmodel 2）**。loopback 排除了網路浮動，所以**真實使用時還要加上瀏覽器到 Jetson 的上傳與下載時間**。每格都偵測到 4 個物件（2 隻貓、2 個遙控器），組合之間偵測結果一致。

> 影像：`cats.jpg`（640×480）及其放大版（1280×960、4096×3072，JPEG 品質 90）。放大版沒有更多細節，只用來量解碼、傳輸與後處理的成本。**它比真實的手機照片便宜**：使用者的 24 MP 手機照片（4284×5712，3.4 MB）解碼＋縮小要約 655–760 ms（見 §7），而這裡 4096×3072 的「伺服器之外」只有約 142 ms。

### 6.1 端對端延遲

| 組合 | 影像 | 平均 (ms) | 中位數 | p95 | p99 | 最大 | 標準差 | 請求/秒 | 相對 ① |
|---|---|---|---|---|---|---|---|---|---|
| ① PyTorch fp32：OWL + MobileSAM（baseline） | 640×480 | **460.7** | 460.7 | 462.6 | 466.9 | 466.9 | 1.4 | 2.2 | 1.00× |
| ① PyTorch fp32：OWL + MobileSAM（baseline） | 1280×960 | **497.6** | 497.2 | 501.9 | 503.4 | 503.4 | 2.6 | 2.0 | 1.00× |
| ① PyTorch fp32：OWL + MobileSAM（baseline） | 4096×3072 | **961.1** | 963.4 | 965.8 | 967.2 | 967.2 | 5.6 | 1.0 | 1.00× |
| ② OWL TensorRT + MobileSAM（PyTorch） | 640×480 | **341.4** | 340.8 | 345.3 | 358.2 | 358.2 | 2.8 | 2.9 | 1.35× |
| ② OWL TensorRT + MobileSAM（PyTorch） | 1280×960 | **382.3** | 381.9 | 390.9 | 397.3 | 397.3 | 4.5 | 2.6 | 1.30× |
| ② OWL TensorRT + MobileSAM（PyTorch） | 4096×3072 | **853.8** | 854.8 | 858.9 | 871.8 | 871.8 | 5.2 | 1.2 | 1.13× |
| ③ OWL TensorRT + 自訓練 ResNet18 encoder（PyTorch） | 640×480 | **200.4** | 200.4 | 201.6 | 202.1 | 202.1 | 0.6 | 5.0 | 2.30× |
| ③ OWL TensorRT + 自訓練 ResNet18 encoder（PyTorch） | 1280×960 | **237.6** | 237.7 | 239.4 | 241.4 | 241.4 | 1.1 | 4.2 | 2.09× |
| ③ OWL TensorRT + 自訓練 ResNet18 encoder（PyTorch） | 4096×3072 | **656.6** | 655.6 | 664.1 | 677.7 | 677.7 | 4.5 | 1.5 | 1.46× |
| ④ **全 TensorRT：OWL + 自訓練 ResNet18 encoder** | 640×480 | **129.1** | 129.0 | 132.2 | 133.4 | 133.4 | 1.8 | 7.7 | 3.57× |
| ④ **全 TensorRT：OWL + 自訓練 ResNet18 encoder** | 1280×960 | **169.2** | 168.0 | 174.4 | 175.4 | 175.4 | 2.7 | 5.9 | 2.94× |
| ④ **全 TensorRT：OWL + 自訓練 ResNet18 encoder** | 4096×3072 | **628.5** | 626.7 | 649.5 | 656.2 | 656.2 | 10.6 | 1.6 | 1.53× |

### 6.2 時間花在哪裡（平均，ms）

| 組合 | 影像 | OWL | SAM | 序列化（輪廓） | 伺服器內其他 | 伺服器之外* | 合計＝端對端 |
|---|---|---|---|---|---|---|---|
| ① PyTorch fp32：OWL + MobileSAM（baseline） | 640×480 | 144.1 | 286.8 | 3.4 | 0.6 | 25.8 | 460.7 |
| ① PyTorch fp32：OWL + MobileSAM（baseline） | 1280×960 | 148.4 | 304.2 | 10.1 | 0.6 | 34.3 | 497.6 |
| ① PyTorch fp32：OWL + MobileSAM（baseline） | 4096×3072 | 210.6 | 507.4 | 95.0 | 0.6 | 147.5 | 961.1 |
| ② OWL TensorRT + MobileSAM（PyTorch） | 640×480 | 25.8 | 286.3 | 3.3 | 0.5 | 25.4 | 341.4 |
| ② OWL TensorRT + MobileSAM（PyTorch） | 1280×960 | 30.7 | 305.9 | 10.1 | 0.6 | 35.0 | 382.3 |
| ② OWL TensorRT + MobileSAM（PyTorch） | 4096×3072 | 93.3 | 513.3 | 95.7 | 0.6 | 150.8 | 853.8 |
| ③ OWL TensorRT + 自訓練 ResNet18 encoder（PyTorch） | 640×480 | 25.8 | 145.3 | 3.4 | 0.5 | 25.4 | 200.4 |
| ③ OWL TensorRT + 自訓練 ResNet18 encoder（PyTorch） | 1280×960 | 30.3 | 163.3 | 9.8 | 0.6 | 33.6 | 237.6 |
| ③ OWL TensorRT + 自訓練 ResNet18 encoder（PyTorch） | 4096×3072 | 85.4 | 345.6 | 88.8 | 0.6 | 136.2 | 656.6 |
| ④ **全 TensorRT：OWL + 自訓練 ResNet18 encoder** | 640×480 | 26.0 | 73.8 | 3.3 | 0.8 | 25.2 | 129.1 |
| ④ **全 TensorRT：OWL + 自訓練 ResNet18 encoder** | 1280×960 | 30.3 | 93.3 | 9.9 | 0.6 | 35.0 | 169.2 |
| ④ **全 TensorRT：OWL + 自訓練 ResNet18 encoder** | 4096×3072 | 90.1 | 304.0 | 91.3 | 0.6 | 142.5 | 628.5 |

\* 「伺服器之外」＝ 客戶端端對端 − 伺服器回報的 `timing.total_ms`，包含圖片解碼、回應組裝（含 `gpu_stats`）、JSON 編碼、HTTP 與客戶端解析。這個矩陣量測時伺服器還沒有回報 `decode_ms`／`server_ms`，所以當時無法拆開；之後新增並對全 TensorRT 組合重跑，見下表。「伺服器內其他」＝ `total_ms − OWL − SAM − 序列化`（文字編碼、鎖等待等）。

**全 TensorRT 組合的「伺服器之外」拆解**（伺服器新增 `decode_ms`／`server_ms` 後重跑，每格 50 次，單位 ms；與上面矩陣是不同次執行，端對端差在誤差內：132/167/656 vs 129/169/628，4096×3072 這次抖動較大）：

| 影像 | 端對端 | 圖片解碼 | OWL | SAM | 序列化（輪廓） | 伺服器其他（組裝回應等） | **伺服器合計 `server_ms`** | 客戶端 − 伺服器（上傳、JSON 編碼／解析、HTTP） |
|---|---|---|---|---|---|---|---|---|
| 640×480 | 131.5 | 6.5 | 26.3 | 75.2 | 3.6 | 8.0 | 119.5 | 12.0 |
| 1280×960 | 166.9 | 14.3 | 30.3 | 91.4 | 9.8 | 8.1 | 153.9 | 13.0 |
| 4096×3072 | 656.0 | 109.6 | 108.9 | 311.4 | 90.1 | 10.3 | 630.3 | 25.7 |

- 小圖上，原本看不見的約 25 ms 是：圖片解碼約 6.5 ms、伺服器組裝回應約 8 ms、客戶端與伺服器之間（FastAPI 編 JSON、HTTP、客戶端解析；**loopback 上**）約 12 ms。真實網路還要再加上上傳與下載。
- `server_ms` 不含 FastAPI 把回應 dict 編成 JSON 的時間，所以「客戶端 − 伺服器」包含這一段，沒有再細分。

### 6.3 功耗與能耗（`tegrastats` 的 `VDD_IN`＝整板輸入功耗，只取 GPU 有負載的取樣點）

| 組合 | 影像 | 板子功耗（負載中平均 W） | 最大 W | 每個請求能耗 (J) | GPU 使用率 % | 接面溫度最高 °C |
|---|---|---|---|---|---|---|
| ① PyTorch fp32：OWL + MobileSAM（baseline） | 640×480 | 15.2 | 15.6 | 6.98 | 89 | 55 |
| ① PyTorch fp32：OWL + MobileSAM（baseline） | 1280×960 | 14.6 | 15.1 | 7.26 | 90 | 57 |
| ① PyTorch fp32：OWL + MobileSAM（baseline） | 4096×3072 | 11.6 | 12.7 | 11.14 | 78 | 56 |
| ② OWL TensorRT + MobileSAM（PyTorch） | 640×480 | 13.9 | 15.1 | 4.76 | 75 | 57 |
| ② OWL TensorRT + MobileSAM（PyTorch） | 1280×960 | 13.6 | 14.8 | 5.20 | 82 | 57 |
| ② OWL TensorRT + MobileSAM（PyTorch） | 4096×3072 | 11.0 | 11.9 | 9.42 | 80 | 56 |
| ③ OWL TensorRT + 自訓練 ResNet18 encoder（PyTorch） | 640×480 | 13.0 | 13.4 | 2.62 | 72 | 55 |
| ③ OWL TensorRT + 自訓練 ResNet18 encoder（PyTorch） | 1280×960 | 12.3 | 12.7 | 2.93 | 75 | 55 |
| ③ OWL TensorRT + 自訓練 ResNet18 encoder（PyTorch） | 4096×3072 | 9.7 | 10.2 | 6.38 | 58 | 55 |
| ④ **全 TensorRT：OWL + 自訓練 ResNet18 encoder** | 640×480 | 11.6 | 11.9 | 1.49 | 65 | 54 |
| ④ **全 TensorRT：OWL + 自訓練 ResNet18 encoder** | 1280×960 | 11.0 | 11.1 | 1.85 | 52 | 54 |
| ④ **全 TensorRT：OWL + 自訓練 ResNet18 encoder** | 4096×3072 | 8.6 | 8.9 | 5.39 | 56 | 53 |

能耗＝負載中平均功耗 × 端對端延遲，假設單客戶端連續請求時整板都在忙；是估算，不是直接量到的能量。

> **2026-11-10 核對註記**：使用者鎖頻（`sudo jetson_clocks`）後，我把端對端、流程基準、功耗與官方口徑對照全部重跑，結果在 `data/bench/jetson/locked_2026-11-10/`，**本報告的數字全部重現**（端對端 12 格中 9 格差在 ±1.4% 內、最大差 +11.7%（大圖 4096×3072 的③）；全 TensorRT 640×480：129.0 對 129.1 ms、11.61 對 11.6 W；流程基準整體 99.92 對 99.43 ms；官方口徑差在 1.2% 內；穩態功耗 13.37 對 13.31 W）。完整對照表見 [report-data-supplement.md](report-data-supplement.md) §7。**本報告所有延遲／FPS／功耗數字的成立條件是 MAXN_SUPER 且鎖頻（`jetson_clocks`）**：沒鎖頻時同一組設定慢約 2.2 倍（640×480 全 TensorRT 283 ms）、功耗約 7.2 W。另外，§12 的 15W 重現指令切回 `nvpmodel -m 2` 後**沒有再執行 `jetson_clocks`**，照做會得到沒鎖頻的結果；「④ 全 TensorRT」三列當初的原始資料已被後來的重跑覆蓋，但鎖頻重量重現了這些數字。重複量測本身有 ±數% 的變異（大圖更大），引用請寫「約」並附 p95。

### 6.4 記憶體

| 組合 | 伺服器程序 RSS：載入後 → 測完 (MB) | 系統 MemAvailable：載入前 → 載入後 → 測完 (MB) |
|---|---|---|
| ① PyTorch fp32：OWL + MobileSAM（baseline） | 2429 → 2802 | 5284 → 3914 → 3523 |
| ② OWL TensorRT + MobileSAM（PyTorch） | 2702 → 3061 | 5406 → 3744 → 3274 |
| ③ OWL TensorRT + 自訓練 ResNet18 encoder（PyTorch） | 2753 → 3256 | 5318 → 3531 → 3042 |
| ④ **全 TensorRT：OWL + 自訓練 ResNet18 encoder** | 2841 → 3487 | 5322 → 3422 → 2752 |

伺服器永遠把 baseline MobileSAM 也載入（為了網頁可切換編碼器），所以全 TensorRT 組合並不是記憶體最省的，它同時還載了引擎。

### 6.5 觀察（由上表計算）

1. **小圖上，全 TensorRT 比 PyTorch baseline 快 3.6 倍**（461 → 129 ms，約 7.7 請求/秒），**功率也更低**（15.2 → 11.6 W），每個請求的能耗約少 4.7 倍。
2. **加速來自兩處**：OWL 由 144 ms（PyTorch）降到 26 ms（TensorRT fp16）；SAM 由 287 ms（MobileSAM）降到 145 ms（換成 ResNet18 encoder，PyTorch），再降到 74 ms（同一份權重的 TensorRT fp16 引擎）。只換 OWL（組合 ②）只快 1.35 倍，因為 MobileSAM 的 encoder 仍佔 286 ms。
3. **影像越大，加速越小**：4096×3072 時只剩 1.53 倍（961 → 628 ms）。全 TensorRT 組合在這個尺寸上，OWL + SAM 只佔 63%，其餘 234 ms 是模型以外的成本（解碼、輪廓序列化 91 ms、傳輸）；小圖則是 29 ms（23%）。**TensorRT 只加速模型，這些 CPU 端成本沒有變**。
4. **SAM 是全 TensorRT 組合中最大的一塊**：小圖上 74 ms，佔端對端的 57%（4 個框，每個框還要跑 decoder 與遮罩後處理）。框越多越慢（§7）。
5. **穩定度高**：全 TensorRT 小圖 p95 132 ms、最大 133 ms，標準差 1.8 ms；大圖標準差較大（10.6 ms）。

## 7. 網頁 API 端對端（`python -m server.main`，TensorRT 後端）

欄位：`decode` = 圖片解碼＋縮放；`owl`、`sam` = 模型階段；`run` = 伺服器內部整段推論與序列化；`api` = `decode + run`；`http` = 瀏覽器端看到的整次請求；`gap` = `http − api`。

### 7.1 小圖（cats.jpg 640×480）

| 影像 | decode (ms) | owl (ms) | set_image (ms) | sam (ms) | run (ms) | api (ms) | http (ms) |
|---|---|---|---|---|---|---|---|
| 640×480（cats.jpg，4 個偵測，6 次穩態） | 6 | 27.0 | 29 | 80 | 139 | 146 | 153 |

受控的端對端結果見 §6（PyTorch fp32：461 ms；全 TensorRT：129 ms；同樣是 640×480、4 個偵測）。這一節是使用者操作網頁時留下的日誌，量測當下的程式碼比 §6 舊。

> **回應內 `timing.total_ms`、日誌的 `run`、`api` 是三個不同的指標**：`total_ms` 在序列化結束就停止計時；`run` 之後還包含組裝回應；`api` 再加上圖片解碼。**`run` 與 `total_ms` 的 30 ms 左右差距已查證**：回應組裝時會呼叫 `gpu_stats()`，它每次都啟動 `nvidia-smi` 子程序（在這台 Jetson 上約 18 ms，且只回傳 `[N/A]`），整個 `gpu_stats()` 實測平均 26.4 ms。已修正（Jetson 改讀 sysfs，平均 5.9 ms），所以此表之後量到的請求少了約 20 ms；同一個全 TensorRT 組合的端對端由日誌的約 153 ms 降到 §6 的 129 ms。

### 7.2 大圖（使用者的手機照片 4284×5712 → 3072×4096，提示詞 `a dog`、`a bag`、`a person`／`a doll`）

| 時間 | 閾值 | NMS IoU | 偵測數 | decode | owl | set_image | sam | run | api | http | gap |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 19:34:49 | 0.1 | 0.5 | 1 | 759 | 183 | 174 | 210 | 487 | 1250 | 1806 | 556 |
| 19:35:07 | 0.1 | 0.5 | 1 | 732 | 125 | 222 | 270 | 444 | 1178 | 1694 | 515 |
| 19:35:11 | 0.1 | 0.5 | 1 | 657 | 80 | 123 | 210 | 347 | 1006 | 3229 | 2222 |
| 19:35:18 | 0.1 | 0.5 | 1 | 757 | 82 | 174 | 222 | 359 | 1120 | 4600 | 3480 |
| 19:35:24 | 0.1 | 0.5 | 1 | 656 | 79 | 128 | 210 | 338 | 1006 | 3124 | 2118 |
| 19:35:28 | 0.1 | 0.5 | 1 | 756 | 84 | 170 | 221 | 358 | 1118 | 2356 | 1238 |
| 19:35:34 | 0.1 | 0.5 | 1 | 680 | 81 | 192 | 360 | 499 | 1182 | 4838 | 3656 |
| 19:35:39 | 0.1 | 0.5 | 1 | 734 | 80 | 148 | 353 | 525 | 1265 | 5191 | 3926 |
| 19:35:46 | 0.1 | 0.5 | 1 | 694 | 83 | 132 | 176 | 309 | 1006 | 1595 | 589 |
| 19:35:48 | 0.1 | 0.5 | 1 | 660 | 79 | 120 | 164 | 296 | 958 | 1608 | 650 |
| 19:35:52 | 0.1 | 0.5 | 1 | 655 | 79 | 120 | 163 | 295 | 952 | 1544 | 592 |
| 19:36:10 | 0.1 | 0.5 | 1 | 657 | 82 | 121 | 164 | 335 | 995 | 1409 | 415 |
| 19:36:21 | 0.1 | 0.5 | 0 | 663 | 78 | 0 | 0 | 133 | 809 | 1200 | 391 |
| 19:36:26 | 0.01 | 0.5 | 10 | 682 | 101 | 146 | 490 | 852 | 1537 | 1892 | 355 |
| 19:36:34 | 0.08 | 0.5 | 0 | 671 | 77 | 0 | 0 | 109 | 783 | 1137 | 354 |
| 19:36:42 | 0.03 | 0.5 | 1 | 663 | 80 | 121 | 165 | 295 | 961 | 1371 | 410 |
| 19:37:08 | 0.03 | 0.5 | 19 | 658 | 85 | 121 | 807 | 1438 | 2101 | 2846 | 745 |
| 19:37:16 | 0.03 | 0.25 | 14 | 655 | 96 | 121 | 621 | 1051 | 1710 | 2116 | 407 |
| 19:37:25 | 0.03 | 0.15 | 10 | 654 | 82 | 122 | 473 | 803 | 1461 | 1889 | 428 |
| 19:37:38 | 0.06 | 0.15 | 6 | 655 | 81 | 119 | 329 | 569 | 1227 | 1894 | 666 |
| 19:38:08 | 0.2 | 0.15 | 3 | 661 | 190 | 125 | 240 | 542 | 1206 | 1770 | 564 |
| 19:38:24 | 0.2 | 0.15 | 3 | 658 | 81 | 118 | 230 | 459 | 1120 | 1644 | 524 |
| 19:38:30 | 0.13 | 0.15 | 4 | 656 | 81 | 118 | 262 | 460 | 1120 | 1590 | 470 |

（單位 ms，共 23 次請求；來源是使用者操作網頁時伺服器寫下的 `app.log`。）

觀察（皆由上表計算）：

1. **圖片解碼固定約 681 ms**（範圍 654–759），與偵測數無關，每次請求都付一次。
2. **SAM 每個框約 36 ms**（以 `(sam − set_image) / 偵測數` 計，偵測數 ≥ 3 的 8 次請求，範圍 34–38 ms），是小圖基準的 3.7 倍。**推測**多出的部分是把 256×256 的遮罩放大到 3072×4096 並二值化的成本；未單獨計時驗證。`set_image` 也從小圖約 29 ms 變成約 140 ms。
3. **後處理（輪廓與序列化）約每個偵測 31 ms**（`(run − owl − sam) / 偵測數`），偵測數多時明顯。
4. **偵測數量由閾值決定，並線性決定 SAM 耗時**：NMS IoU 0.15 時，閾值 0.03 / 0.06 / 0.13 / 0.2 分別得到 10 / 6 / 4 / 3 個偵測。閾值 0.03 時多出許多分數 0.04–0.21 的額外 `a bag` 偵測；我沒有檢視這張照片，**無法確認它們是局部框還是誤偵測**，但這和 `docs/scenario.md` 提到的「計數偏高」現象同方向。提高閾值才收斂。
5. **瀏覽器端總時間的主要浮動來自伺服器之外**：同一張圖、相同參數（閾值 0.1）且同樣 1 個偵測的 12 次請求，伺服器內部 `api` 為 952–1265 ms，但 `http` 卻是 1409–5191 ms。`gap`（354–3926 ms）在內容相同的請求之間差了約 9 倍，所以不是回應內容大小造成的。**推測**是 3.4 MB 照片的上傳時間受網路狀況影響；沒有直接量測上傳耗時，這是推論。
6. OWL 的耗時在大圖上約 83 ms（排除 2 次 > 120 ms 的離群值），小圖約 27 ms，多出的部分是對大圖做前處理（縮放到 768）。

**建議（未實作）**：瀏覽器端上傳前先把照片縮到長邊 ≤ 1280 px，可同時省掉上傳時間與約 681 ms 的解碼。

## 8. 與官方數據對照

（與 WSL／GTX 1060 的對照見 [wsl-vs-jetson.md](wsl-vs-jetson.md)。）

### 8.1 官方的量測範圍（逐項查證）

- **NanoOWL FPS**：`third_party/nanoowl/examples/owl_predict.py --profile`——前後 `synchronize`，同一張圖連跑 N 次 `predict(..., pad_square=False)`，文字編碼預先算好。範圍 = 前處理 + TensorRT 影像編碼 + decode；**不含** NMS、搬回 CPU、SAM。
- **NanoSAM Image Encoder / Full Pipeline**：README 與程式碼都**沒有**寫計時方法，所以兩種口徑並列：(a) `trtexec` 純 GPU（encoder + 1 框 decoder）；(b) 官方 `nanosam.utils.predictor.Predictor` 的 `set_image` + `predict`（含前處理與 upscale_mask）。MobileSAM 的 encoder 官方用 fp32、NanoSAM 用 fp16、decoder 皆 fp32。

量測腳本 `benchmark/bench_official.py`（640×480，`cats.jpg`，1 個框，單位 ms；結果 `official_compare.json` = MAXN_SUPER、`official_compare_15W.json` = 15W）：

| 項目 | 官方 Orin Nano | MAXN_SUPER：trtexec Full | MAXN_SUPER：Python Full | 15W：trtexec Full | 15W：Python Full |
|---|---|---|---|---|---|
| MobileSAM（TinyViT fp32） Full Pipeline（1 框） | 146 ms | 98.8 | 123.6 | 121.1 | 149.7 |
| MobileSAM（TinyViT fp16） Full Pipeline（1 框） | — | 44.4 | 69.7 | 61.2 | 89.5 |
| ResNet18 自訓練 fp16（NanoSAM 架構） Full Pipeline（1 框） | 27 ms | 18.2 | 42.7 | 26.3 | 54.4 |
| 　└ MobileSAM（TinyViT fp32） encoder | TBD | 90.5（trtexec）/ 113.2（Python） | | 109.5 / 135.7 | |
| 　└ MobileSAM（TinyViT fp16） encoder | TBD | 36.1（trtexec）/ 59.5（Python） | | 49.6 / 75.7 | |
| 　└ ResNet18 自訓練 fp16（NanoSAM 架構） encoder | TBD | 9.9（trtexec）/ 32.7（Python） | | 14.7 / 40.8 | |
| mask decoder（1 框） | — | 8.28 | | 11.58 | |
| NanoOWL ViT-B/32 FPS | TBD | 51.7 FPS（19.3 ms） | | 34.8 FPS（28.7 ms） | |

**電源模式**（`/etc/nvpmodel.conf`）：7W（GPU 408 MHz）、15W（612 MHz）、25W（918 MHz）、MAXN_SUPER（1020 MHz，無功耗上限）。官方 README **沒有寫電源模式、JetPack 或時脈**；Orin Nano 表格的 commit 日期是 2023-09-01（NanoSAM）與 2023-09-19（NanoOWL），早於 Super 模式（2024 年底），所以官方不可能是 Super。MAXN_SUPER 是本機最快的模式；實測的整板功耗約 11.6 W（§5），沒有用到 15W 的上限。

- **電源模式是我們與官方差距的主因**：NanoSAM(ResNet18) Full Pipeline 純 GPU，MAXN_SUPER 18.2 ms → 15W 26.3 ms，與官方 27 ms 只差 2.6%。同一模型 NanoOWL 也從 51.7 降到 34.8 FPS。
- **口徑不能一概而論**：NanoSAM 的官方 27 ms 貼近 15W 的純 GPU（26.3 ms），遠低於含前處理的 Python（54.4 ms）→ 官方很可能不含 CPU 前處理；但 MobileSAM 的官方 146 ms 貼近 15W 的 **Python**（149.7 ms）而不是純 GPU（121.1 ms，快 17%）。兩個模型指向不同口徑，**官方沒寫範圍，我們無法斷定**；可能的原因（官方舊版 TensorRT 8.x 上 fp32 TinyViT 較慢）是推測，沒有驗證。
- 先前的推測「官方 ÷ 我們 = 固定 1.48 倍」只在 MAXN_SUPER 的純 GPU 數字上成立，15W 量測後不再成立，已撤回。
- MobileSAM fp16 官方說 TensorRT 會產生錯誤結果（README 註腳），此列只供速度參考，不能當準確的 baseline。
- 15W 量測時沒有同時記錄 `tegrastats`，所以沒有 15W 的功耗數字。
- 網頁右側新增「官方口徑」卡片，每幀即時顯示同一範圍的數字（`timing.official_*`：OWL predict、SAM encoder、decoder/框、Full Pipeline），與「端對端」分開。pipeline 內量到的 SAM encoder 含前處理，約 29.5 ms，decoder 約 11 ms（含遮罩後處理與搬回 CPU），所以 Full ≈ 40 ms，接近 (b)。

**架構相同、權重不同**：官方 NanoSAM 與本專案都是「ResNet18 image encoder ＋ MobileSAM mask decoder」，所以延遲（由架構與精度決定）可以對照；但官方預訓練權重沒有取得，我們用的是自行蒸餾的權重（mIoU 0.704，官方 0.706 為不同條件的參考值），**準確度不可對照**。

**不能下的結論**：官方數據的測試條件（JetPack 與 TensorRT 版本、電源模式、框數、是否含前後處理）未公布，所以上表只能當量級參考，不能說「比官方快／慢」。

## 9. HyDE 提示詞擴展與 Ollama 模型（Orin Nano，CPU 推論）

HyDE 只在**換提示詞時**呼叫一次 LLM，不影響逐幀 FPS。用專案的 `server/hyde.py`（`expand_prompts`，每組 5 個詞組）測 6 個提示詞（`hex nut`、`screw`、`circuit board`、`remote control`、`螺絲`、`貓`）；Ollama 設 `num_gpu=0`、`num_ctx=1024`。

| 模型 | 大小 (GiB) | 失敗 | 首次載入含推論 (s) | 暖機後平均 (s) | 常駐 (MB) |
|---|---|---|---|---|---|
| `qwen2.5:1.5b` | 0.92 | 0/6 | 7.9 | 1.8 | 1214 |
| `llama3.2:1b` | 1.23 | 0/6 | 8.1 | 2.1 | 1620 |
| `qwen3:1.7b` | 1.27 | 0/6 | 11.1 | 2.5 | 1460 |
| `gemma3:1b` | 0.75 | 0/6 | 8.1 | 5.4 | 1145 |

- **GPU 推論在 Nano 上不可行**：第一輪用預設（GPU）時，伺服器運作中 4 個模型有 3 個每次請求都 `cudaMalloc failed: out of memory`，Ollama 的推論程序崩潰；GPU 配置不能放到 swap。改用 CPU 後 4 個模型 0 次失敗。
- 4 個模型都把中文「螺絲」「貓」翻成正確的英文名稱作為第一個詞組；**4 個模型都會混入少量不相關或過於籠統的詞組**（例如 `qwen2.5:1.5b` 給 `a screwdriver`、`qwen3:1.7b` 給 `a cat toy`、`llama3.2:1b` 給 `a device with a rotating shaft`、`gemma3:1b` 給 `animal`），可能造成 OWL 誤偵測。這是逐筆人工判讀，只有 6 個提示詞，**不是統計結論**。
- 預設選用 `qwen2.5:1.5b`（最快、常駐記憶體小）；已寫入 `server/hyde.py`（Jetson 預設、CPU、閒置 2 分鐘卸載）。

**端對端驗證（伺服器以 TensorRT 引擎運作中，Ollama 在 CPU 上）**：`/api/hyde/expand` 擴展 `螺絲,貓` 第一次含載入模型約 10.0 s，模型已載入時約 3.6 s（單一提示詞約 1.9 s）；`ollama ps` 顯示 `100% CPU`、上下文 1024、常駐約 1.3 GB，擴展後系統 `MemAvailable` 約少 1.2 GB（3709 → 2500 MB）。伺服器與 Ollama 日誌都沒有 OOM；TensorRT 偵測維持 85–94 ms/請求。

| `cats.jpg`、提示詞 `螺絲,貓` | 偵測 | 每幀耗時（穩態） |
|---|---|---|
| 不用 HyDE | 0 個（OWL 只懂英文） | 26 ms（沒有偵測，不跑 SAM） |
| 用 HyDE（`貓` → `a cat, a feline, a domestic cat, …`） | `貓` 2 個（遮罩 110,696 px，與英文 `a cat` 相同）；`螺絲` 0 個（圖中沒有螺絲） | 90–94 ms |

HyDE 的套用條件是「這次請求的**所有**提示詞都已擴展過」（`lookup_cached`），否則整個不套用、改用原提示；網頁流程是先按「擴展提示詞」再偵測，所以正常使用不會遇到，但手動呼叫 API 時要注意。

## 10. 限制與尚未完成

1. 基準測試只有**單一影像（cats.jpg）、4 個框**；電源模式只比較了 MAXN_SUPER 與 15W（§8，且 15W 只量延遲、沒有功耗）；沒有 25W / 7W。
2. MobileSAM baseline 只有端對端（§6）與 SAM 階段的數字，沒有 `bench_pipeline.py` 的逐階段統計；**沒有量官方 NanoSAM 預訓練權重**（`resnet18_image_encoder.onnx` 需從 Google Drive 登入下載，沒有取得），所有「SAM encoder」數字都是自行蒸餾的權重。
3. 功耗取樣只有 18 個穩態樣本（每個 0.5 秒）；`tegrastats` 在此版本不輸出 GPU 時脈。
4. 第 6.2 節大圖資料是使用者操作網頁時留下的日誌，不是受控實驗；`gap` 的成因是推論。
5. **提示詞寫法比較（≥ 10 張影像、≥ 3 種寫法）、攝影機即時 Demo（本機沒有偵測到攝影機）、截圖與影片尚未做。**
6. 與 NanoSAM 官方數據的對照受限於官方數據不完整（Orin Nano 的 encoder 與 NanoOWL 為 TBD）。

## 11. 原始資料

| 內容 | 檔案 |
|---|---|
| 流程基準（最終） | `data/bench/jetson/pipe_all_trt_final.json`（另一次：`pipe_all_trt.json`） |
| 功耗 / 溫度 | `data/bench/jetson/tegrastats_idle.log`、`tegrastats_bench.log` |
| 官方口徑對照 | `data/bench/jetson/official_compare.json`（MAXN_SUPER）、`official_compare_15W.json`（15W）；腳本 `benchmark/bench_official.py` |
| 引擎建置與 trtexec | `build_owl_engine.log`、`build_sam_encoder.log`、`build_sam_decoder.log`（失敗嘗試：`build_owl_engine.attempt*_oom*.log`、`mem_samples*.log`） |
| 網頁逐請求日誌 | `app.log` |
| Ollama 模型評估 | `data/bench/jetson/hyde_model_eval.json` |

（`data/` 與 `app.log` 不進版控；重現方式見 `docs/jetson-deploy.md`。）

## 12. 重現指令（官方口徑量測，§8）

在專案根目錄、`.venv-jetson` 環境下執行。

**事前準備**

```bash
cd ~/midterm_P76154278/smart-processing-midterm-project-2026Fall
source .venv-jetson/bin/activate
pkill -f server.main        # 先關網頁伺服器；Nano 記憶體很緊（建議 free -m 的 available > 3000 MB）
nvpmodel -q                 # 確認電源模式
```

**完整重跑（`benchmark/bench_official.py`）**——預設參數即報告所用：`cats.jpg`、提示詞 `a cat` / `a remote control`、NanoOWL 5 輪 × 200 次、NanoSAM 100 次；`trtexec` 會在載入模型前先跑完（避免 OOM）。

```bash
# MAXN_SUPER（nvpmodel 2）
python benchmark/bench_official.py --power_mode "MAXN_SUPER" --out data/bench/jetson/official_compare.json

# 15W（nvpmodel 0）——量完務必切回
sudo nvpmodel -m 0 && sudo jetson_clocks
python benchmark/bench_official.py --power_mode "15W (nvpmodel 0)" --out data/bench/jetson/official_compare_15W.json
sudo nvpmodel -m 2
```

**單項驗證**

```bash
# NanoOWL：官方 --profile（預設 30 次；旗標已對照 owl_predict.py 原始碼）
cd third_party/nanoowl/examples
python owl_predict.py --prompt="[a cat, a remote control]" --threshold=0.1,0.1 \
    --image=../../../data/samples/cats.jpg --image_encoder_engine=../../../data/owl_image_encoder_patch32.engine \
    --output=/tmp/owl_out.jpg --num_profiling_runs=200 --profile
cd ../../..

# NanoSAM encoder 純 GPU
/usr/src/tensorrt/bin/trtexec --loadEngine=data/resnet18_image_encoder.selftrain.engine \
    --warmUp=1000 --duration=5 --iterations=100 --avgRuns=100 | grep "GPU Compute Time"

# mask decoder 純 GPU（1 個框 = 2 個點）
/usr/src/tensorrt/bin/trtexec --loadEngine=data/mobile_sam_mask_decoder.engine \
    --shapes=point_coords:1x2x2,point_labels:1x2 \
    --warmUp=1000 --duration=5 --iterations=100 --avgRuns=100 | grep "GPU Compute Time"
```

官方 Full Pipeline 的純 GPU 口徑 ＝ encoder 的 `GPU Compute Time` 平均 ＋ decoder 的平均。

**注意**：trtexec 結果有約 ±1% 的波動，不會與表中數字逐位相同；量測時不要同時開瀏覽器、Ollama 推論或網頁伺服器；網頁「官方口徑」卡片的數字是流程內計時（SAM encoder 含前處理），範圍與 trtexec 不同，不要拿來和純 GPU 數字直接比。
