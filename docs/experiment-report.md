# 期中實驗報告：文字提示開放詞彙偵測＋分割（NanoOWL + NanoSAM）在 WSL 與 Jetson Orin Nano 上的實測

> 題目：Topic 4（OWL-ViT 文字提示偵測 → SAM 分割），部署目標 Jetson Orin Nano（JetPack 6.2）。
> 本報告整合兩份資料：**WSL2（GTX 1060）開發機**的量測，與 **Jetson Orin Nano 8GB** 上的量測（Nano 的原始檔與詳細版報告見 [measurement-report.md](measurement-report.md)、[wsl-vs-jetson.md](wsl-vs-jetson.md)）。
> **本檔所有數字都是由 `data/bench/{wsl,jetson}/` 裡的原始 JSON 重新計算的**（WSL 數據在整合 Nano 版本時以同一套腳本重新量過，見 §1.3）。凡是推測、推算、或未驗證的地方，文中都有標明。

## 0. 摘要

1. **端對端（真實 API，640×480，4 個偵測）**：Nano 全 TensorRT **129 ms（7.7 req/s）**，同一台 Nano 的 PyTorch fp32 baseline 為 461 ms（**3.6×**）。
2. **和 WSL 比**：同樣的 PyTorch fp32，Nano 比 GTX 1060 慢約 **2×**（完整流程 427.7 vs 207.0 ms）；但 Nano 用了 TensorRT 之後，反而比 WSL 的 PyTorch 快：端對端 129 vs 251 ms（baseline，快 1.9×）、129 vs 190 ms（WSL 也用自訓練 encoder，快 1.5×）。
3. **大圖不成立**：4096×3072 時 Nano 全 TensorRT 是 628 ms，**比 WSL 的 PyTorch（509–516 ms）還慢**。TensorRT 只加速模型，解碼與輪廓序列化等 CPU 端成本沒有變，而 Nano 的 CPU 較弱。
4. **自訓練 ResNet18 encoder（蒸餾自 MobileSAM）是 Nano 上最大的單一收益**：SAM encoder 在 Nano 上 TensorRT fp32 為 108.8 → 40.5 ms，fp16 再到 29.7 ms；**MobileSAM 在 TensorRT fp16 下輸出全空（IoU 0.000），自訓練的 fp16 沒有損失（IoU 0.973 對 0.973）**。代價是準確度：COCO 200 張 mIoU 0.744 → 0.704（−0.040）。
5. **功耗**：整板負載功耗 15.2 W（PyTorch）→ 11.6 W（全 TensorRT），每個請求能耗 6.98 → 1.49 J（4.7×）。WSL 沒有量功耗。
6. **一個已知的品質問題**：TensorRT 10.3 編出來的 mask decoder 與其來源 ONNX 結果不同（模型自評 IoU 0.67 對 0.96，遮罩 IoU 約 0.97），兩個 SAM encoder 共用這個 decoder，所以**相對比較不受影響，但絕對遮罩品質比 PyTorch 低約 2–3%**（§4.3）。
7. **還沒做**：提示詞寫法比較（≥10 張 × ≥3 種寫法）、攝影機即時 Demo（Nano 上偵測不到攝影機）、15W/25W 電源模式、官方 NanoSAM 預訓練權重（需登入 Google Drive）。

## 1. 實驗設定

### 1.1 平台

| 項目 | WSL（開發機） | Jetson Orin Nano（目標平台） |
|---|---|---|
| GPU | GeForce GTX 1060 6GB（Pascal，獨立顯存） | Orin Nano 8GB（Ampere，與 CPU 共用 8GB 記憶體） |
| 系統 | WSL2 | JetPack 6.2.1（L4T R36.4.7） |
| PyTorch | 2.5.1 + cu121（`requirements.txt` 固定版本；沒有核對 WSL 實際安裝版本） | 2.8.0 + cu126（Jetson 專用 wheel） |
| TensorRT | 無 | 10.3.0（CUDA 12.6、cuDNN 9.3） |
| 電源／時脈 | GPU 1923 MHz、60–62 °C（`nvidia-smi`） | **MAXN_SUPER（nvpmodel 2）**，GPU 1020 MHz、50–58 °C |
| 量測日期 | 2026-11-02（整合時重量） | 2026-11-01 |

### 1.2 模型與方法

- **OWL-ViT** base patch32（HF `google/owlvit-base-patch32`），Nano 上 image encoder 建成 TensorRT fp16 引擎。
- **SAM**：baseline = MobileSAM（TinyViT encoder）；自訓練 = ResNet18 encoder，從 MobileSAM 的 TinyViT encoder 蒸餾（Colab T4、79 epoch、Huber loss）。兩者共用 MobileSAM mask decoder。
- **名詞注意**：專案與網頁裡的「NanoSAM」是指 **NanoSAM 的推論流程與架構**（框→兩個提示點、前後處理、TensorRT 載入）。**權重是我們自己蒸餾的，不是官方的**（官方預訓練檔在需登入的 Google Drive，沒有取得）。Nano 上部署的 ONNX／引擎已用輸出比對確認是 `colab80_best.pth`（epoch 78）：cosine = 1.000000。
- 流程：文字 → OWL 偵測框 → NMS → 每個框丟給 SAM → 遮罩。HyDE（選用）只在偵測**之前**用 LLM 擴展提示詞，逐幀迴圈不呼叫 LLM。
- **輸入**：`data/samples/cats.jpg`（COCO val2017 000000039769，640×480），提示詞 `a cat`、`a remote control`，閾值 0.1，NMS 後 4 個框（2 貓 + 2 遙控器）。放大版（1280×960、4096×3072，LANCZOS）只用來量解碼、傳輸與後處理的成本，沒有更多細節。
- **兩種量法**
  - 分階段：`benchmark/bench_pipeline.py`，暖機 10 次、量測 50 次，每次前後 `torch.cuda.synchronize()`。只含推論呼叫，**不含**圖片解碼、序列化與 HTTP。
  - 端對端：`benchmark/bench_e2e.py`，每種組合啟動一次真正的 `python -m server.main`，loopback 連續打 `POST /api/segment` 50 次（暖機 5 次，單一客戶端）。loopback 排除網路浮動，真實使用還要加上上傳與下載。

### 1.3 這次整合時對 WSL 做的事（以及為什麼）

先前 `wsl-vs-jetson.md` 用的 WSL 數字有兩個問題：(a) 自訓練 encoder 的完整流程 141.0 ms 是**推算**的，不是實測；(b) WSL 沒有端對端數據。所以我在同一台 WSL 上用 Nano 同一套腳本、同樣參數重量（`data/bench/wsl/`）。結果（完整流程）：

| WSL 完整流程（ms） | 先前紀錄 | 本次實測 | 差 |
|---|---|---|---|
| MobileSAM baseline | 197.2（2026-10-29） | **207.0** | +5% |
| 自訓練 ResNet18 | 141.0（推算） | **149.9** | +6%（推算偏低） |

兩點需要誠實說明：

- **baseline 比兩天前的紀錄慢 5%，原因沒有查出來**（GPU 時脈 1923 MHz、62 °C，比當時的 60 °C 略高；日期不同）。本報告一律採用本次實測值。
- **`app.log` 寫在 `/mnt/c`（NTFS）會讓完整流程多 8～9 ms**（baseline 215.4 → 207.0；自訓練 159.1 → 149.9；其他階段不變）。因為量測迴圈每次呼叫都寫約 10 行日誌。Nano 的日誌在 ext4 上沒有這個問題。**WSL 的量測都要把 `APP_LOG` 指到 Linux 檔案系統**（本報告的 WSL 數字都已照此量；先前 WSL 紀錄是否有此偏差未查）。

## 2. 準確度

COCO val 200 張（1349 個物件）、以 GT 框當提示、PyTorch fp32，指標 mIoU（資料：`data/eval/compare_run1_colab80.json`，本檔由該 JSON 重新讀出）：

| SAM image encoder | mIoU 全部 | 小物件 | 中物件 | 大物件 | encoder 延遲（WSL） |
|---|---|---|---|---|---|
| baseline：MobileSAM TinyViT | **0.744** | 0.685 | 0.756 | 0.822 | 87.5 ms |
| 自訓練 ResNet18（epoch 78） | 0.704 | 0.628 | 0.725 | 0.796 | **32.3 ms** |
| 官方 NanoSAM（參考，**非同條件**，見 `docs/record.md` §7.4） | 0.706 | 0.624 | 0.738 | 0.796 | — |

- 自訓練比 baseline 低 0.040 mIoU，**小物件掉得最多（−0.057）**；encoder 快 2.7×。
- 官方 NanoSAM 的 0.706 不是在同一個評估條件下量的，只能當量級參考，不能說「追平官方」。
- 準確度只在 WSL 以 PyTorch fp32 量；**Nano 上沒有重跑 mIoU**，Nano 上只比較「TensorRT 輸出和 PyTorch 輸出的差異」（§4.3）。

## 3. 分階段延遲（`bench_pipeline.py`，640×480、4 個框，平均 ms）

| 階段 | WSL PyTorch：baseline | WSL PyTorch：自訓練 | Nano PyTorch：baseline | Nano PyTorch：自訓練 | Nano TensorRT：baseline（fp32） | Nano TensorRT：自訓練（fp32） | Nano TensorRT：自訓練（fp16） |
|---|---|---|---|---|---|---|---|
| OWL 前處理 | 0.8 | 0.8 | 1.6 | 1.3 | 1.6 | 1.3 | 1.3 |
| OWL 影像編碼器 | 75.7 | 74.9 | 136.0 | 135.5 | 16.8 | 16.8 | 16.9 |
| OWL 解碼 | 0.9 | 1.2 | 1.9 | 1.9 | 1.9 | 1.9 | 1.9 |
| **OWL 合計** | 77.3 | 76.8 | 137.6 | 138.0 | 19.3 | 19.3 | 19.5 |
| OWL 文字編碼器（換提示詞才跑；TensorRT 組合仍是 PyTorch） | 9.4 | 9.5 | 25.0 | 25.5 | 25.6 | 25.6 | 25.7 |
| **SAM 影像編碼器** | 90.7 | 34.4 | 207.4 | 66.1 | 108.8 | 40.5 | **29.7** |
| SAM decoder（每個框） | 7.8 | 8.0 | 18.2 | 18.4 | 9.7 | 10.2 | 9.8 |
| **完整流程（4 個框）** | 207.0 | 149.9 | 427.7 | 289.6 | 178.2 | 117.5 | **99.4** |
| FPS（完整流程） | 4.8 | 6.7 | 2.3 | 3.5 | 5.6 | 8.5 | **10.1** |

- MobileSAM 的 TensorRT **fp16 引擎不可用**（輸出全空），所以 baseline 只有 fp32 欄。
- Nano 的 SAM encoder 數字含前處理（縮放、正規化、補邊）。純 GPU（`trtexec`）只有：自訓練 fp16 **11.6 ms**、自訓練 fp32 22.1 ms、MobileSAM fp32 92.8 ms。流程中的 29.7 ms 比純 GPU 多出約 18 ms，**推測**是前處理與資料搬移（兩數相減推得，沒有分別計時）。
- 各階段加總與完整流程差：Nano TensorRT 為 99.4 − (19.5 + 29.7 + 4 × 9.8) = 11.2 ms（NMS、遮罩後處理，沒有逐項量）。
- 重現性：Nano 同參數連跑兩次，99.1 vs 99.4 ms；標準差 1.0 ms。WSL 完整流程標準差 3.3–4.5 ms。

### 3.1 硬體差距與軟體差距各是多少

| 比較 | 完整流程（ms） | 倍率 | 差在哪裡 |
|---|---|---|---|
| WSL PyTorch baseline → Nano PyTorch baseline | 207.0 → 427.7 | **2.07× 變慢** | 只差硬體（模型與 PyTorch 執行方式相同；軟體版本不同，見 §6） |
| WSL PyTorch 自訓練 → Nano PyTorch 自訓練 | 149.9 → 289.6 | 1.93× 變慢 | 只差硬體 |
| Nano PyTorch baseline → Nano PyTorch 自訓練 | 427.7 → 289.6 | 1.48× 變快 | 換 SAM encoder |
| Nano PyTorch 自訓練 → Nano TensorRT 自訓練 fp16 | 289.6 → 99.4 | **2.91× 變快** | 同一份權重，換 TensorRT＋fp16 |
| WSL PyTorch baseline → Nano TensorRT 自訓練 fp16 | 207.0 → 99.4 | 2.08× 變快 | 硬體慢 2.07×、模型 1.48×、TensorRT 2.91× 三項合成 |
| WSL PyTorch 自訓練 → Nano TensorRT 自訓練 fp16 | 149.9 → 99.4 | 1.51× 變快 | 同一個模型，WSL PyTorch 對 Nano TensorRT |

- 逐階段的「Nano ÷ WSL」（baseline）落在 1.78×（OWL 合計）到 2.65×（OWL 文字編碼器）之間。FP32 理論峰值 GTX 1060 約 4.4 TFLOPS、Orin Nano 約 2.1 TFLOPS（核心數 × 實測時脈估算，**不是量測值**），比值約 2.1×，與整體 2.07× 同量級。小運算（decoder 2.33×、文字編碼器 2.65×）落差較大，**推測**是 kernel 啟動與 Python 開銷對 Nano 的 CPU 影響較大，沒有驗證。
- PyTorch 配置峰值：WSL 901 MB、Nano 902 MB（baseline），自訓練 867 vs 883 MB，間接支持兩邊跑的是同樣的模型與輸入；遮罩面積也一致（WSL 49958 / 58209 / 4121 / 1975 px，Nano 49916 / 58217 / 4121 / 1975 px，差 < 0.1%）。

## 4. 端對端延遲與遮罩正確性

### 4.1 端對端（客戶端送出請求到收到完整 JSON，50 次平均，ms）

| 組合 | 640×480 | 1280×960 | 4096×3072 | 640×480 的 p95 |
|---|---|---|---|---|
| **WSL** PyTorch fp32：OWL + MobileSAM | 250.8 | 268.2 | 515.6 | 265.5 |
| **WSL** PyTorch fp32：OWL + 自訓練 | 189.8 | 210.3 | 509.4 | 202.7 |
| Nano ① PyTorch fp32：OWL + MobileSAM | 460.7 | 497.6 | 961.1 | 462.6 |
| Nano ② OWL TensorRT + MobileSAM（PyTorch） | 341.4 | 382.3 | 853.8 | 345.3 |
| Nano ③ OWL TensorRT + MobileSAM（TensorRT fp32） | 211.0 | 246.5 | 725.0 | 213.2 |
| Nano ④ OWL TensorRT + 自訓練（PyTorch） | 200.4 | 237.6 | 656.6 | 201.6 |
| Nano ⑤ OWL TensorRT + 自訓練（TensorRT fp32） | 141.3 | 177.1 | 614.7 | 147.3 |
| **Nano ⑥ 全 TensorRT：OWL + 自訓練（fp16）** | **129.1** | **169.2** | 628.5 | 132.2 |

所有格子都偵測到 4 個物件（2 貓、2 遙控器），組合之間偵測數一致。

| 觀察 | 數字 |
|---|---|
| Nano 小圖：全 TensorRT 相對 PyTorch baseline | 461 → 129 ms，**3.57×** |
| Nano 小圖：只換 OWL（②） | 1.35×（SAM 的 MobileSAM encoder 仍佔 286 ms） |
| Nano 小圖：只換 SAM encoder 模型（④，PyTorch） | 2.30× |
| Nano ⑥ 對 WSL PyTorch baseline | 640×480：**1.94× 快**；1280×960：1.59× 快；4096×3072：**0.82×（慢 22%）** |
| Nano ⑥ 對 WSL PyTorch 自訓練 | 640×480：1.47× 快；1280×960：1.24× 快；4096×3072：0.81×（慢 23%） |

- **影像越大，TensorRT 的收益越小**：Nano 4096×3072 只剩 1.53×（961 → 628 ms）。此時 OWL + SAM 只佔 63%，其餘約 234 ms 是模型以外的成本：圖片解碼（約 110 ms）、輪廓序列化（約 91 ms）、傳輸。**這些 CPU 端成本沒有被 TensorRT 加速，而且 Nano 的 CPU（Cortex-A78AE）比 WSL 主機弱**，所以大圖時 Nano 反而輸給 WSL。
- ④⑤⑥ 在 4096×3072 看不出差異（657 / 615 / 628 ms）：⑥ 另外重跑過一次是 656 ms，兩次之間的差距比 ⑤⑥ 之間還大。
- 使用者的 24 MP 手機照片（4284×5712 → 伺服器縮成 3072×4096）：Nano 上每次請求解碼＋縮放固定約 681 ms，網頁端另有 0.4–3.9 秒的上傳浮動（來源是使用者操作網頁時留下的日誌，**不是受控實驗**）。**建議（未實作）**：瀏覽器端上傳前先縮到長邊 ≤ 1280 px。

### 4.2 Nano 全 TensorRT 的時間花在哪（640×480，ms）

| 圖片解碼 | OWL | SAM | 輪廓序列化 | 伺服器組裝回應等 | 伺服器合計 | 客戶端 − 伺服器（上傳、JSON、HTTP，loopback） | 端對端 |
|---|---|---|---|---|---|---|---|
| 6.5 | 26.3 | 75.2 | 3.6 | 8.0 | 119.5 | 12.0 | 131.5 |

（取自 `e2e_trt_detail.json`，是伺服器新增 `decode_ms`／`server_ms` 之後的另一次執行，與 §4.1 的 129.1 差在誤差內。）SAM 是最大的一塊（約 57%），框越多越慢：每個框約 +10 ms（小圖）。

### 4.3 遮罩正確性（TensorRT 對同一個模型的 PyTorch fp32，4 個框的遮罩 IoU，`cats.jpg`）

| 模型 | TensorRT fp32 | TensorRT fp16 |
|---|---|---|
| baseline：MobileSAM | 0.981 | **0.000（全空，不可用）** |
| 自訓練 ResNet18 | 0.973 | 0.973（fp16 與 fp32 的遮罩 IoU 0.9730 對 0.9729，沒有損失） |

- **TensorRT 10.3 的 mask decoder 與其來源 ONNX 的結果不同**：模型自評 `iou_pred` 約 0.67（PyTorch／ORT 約 0.96）。決定性實驗：同一個 decoder ONNX 用 ONNX Runtime 的 CPU、CUDA、TensorRT 執行器各跑一次，CPU 與 CUDA 一致，TensorRT 執行器與我們的 torch2trt 引擎幾乎逐位相同地錯。結論：**是 TensorRT 10.3 對這個模型的處理，不是我們的程式**。調整建置選項（最佳化等級、關 tactic、`--noTF32`、固定點數）全都修不好；**根本原因（哪個運算子）沒有查出來**。
- 影響：兩個 encoder 共用同一個 decoder，**相對比較不受影響**；絕對遮罩品質比 PyTorch 低約 2–3%，網頁顯示的 `sam_iou` 偏低。
- 可用的替代：decoder 改用 ONNX Runtime 的 CUDA 後端是精確的（對 PyTorch 遮罩 IoU 0.9997），但每次呼叫慢約 6 ms（15.7 對 9.8 ms，**單次量測**）。目前**沒有採用**，預設仍用較快的 TensorRT decoder。
- TensorRT 預設的「fp32」引擎比加 `--noTF32` 的快 2.2 倍（ResNet18：22.1 對 49.7 ms），符合 Ampere 上預設啟用 TF32 的行為（沒有直接檢查引擎設定）；遮罩一樣。

### 4.4 WSL 上同一套程式的驗證（這次整合做的）

- `tests/check_api_contract.py`：5 個欄位群全部 OK。
- `tests/check_nanosam_backend.py`（ONNX Runtime 後端對 PyTorch）：5 個框的遮罩 IoU 0.995–0.9996，通過。
- 沒有引擎的機器（WSL）上 `python -m server.main` 自動退回 PyTorch：`[server] 後端：OWL = PyTorch fp32 | SAM = mobilesam (PyTorch fp32)`；遮罩面積 49958 / 58209 / 4121 / 1975 px，與先前紀錄完全一致；baseline `mobile_sam.pt` 校驗碼未變。

## 5. 功耗、溫度、記憶體

`tegrastats --interval 500`（Nano 專屬；`VDD_IN` = 整板輸入功耗），只取 GPU 有負載的取樣點。**WSL 沒有量功耗**，所以下表只有 Nano。

| 組合（640×480） | 板子功耗（負載中平均） | 每個請求能耗 | GPU 使用率 | 接面溫度最高 |
|---|---|---|---|---|
| ① PyTorch fp32：OWL + MobileSAM | 15.2 W | 6.98 J | 89% | 55 °C |
| ② OWL TensorRT + MobileSAM（PyTorch） | 13.9 W | 4.76 J | 75% | 57 °C |
| ④ OWL TensorRT + 自訓練（PyTorch） | 13.0 W | 2.62 J | 72% | 55 °C |
| **⑥ 全 TensorRT** | **11.6 W** | **1.49 J** | 65% | 54 °C |
| 閒置 | 5.8 W | — | 0% | 49.8 °C |

- 能耗 = 負載中平均功耗 × 端對端延遲，假設單客戶端連續請求時整板都在忙，**是估算，不是直接量到的能量**。
- ⑥ 的負載取樣點只有約 10 個（5 秒）。我自己從原始 log 重算：⑥ 為 11.76 W（報告 11.58，差在邊界一個取樣點的處理），① 為 15.16 W（報告 15.2）；差異在小樣本雜訊內。
- GPU 使用率只有 55–65%：流程中有不少 CPU 端工作與串行等待，GPU 沒有被吃滿。
- `tegrastats` 在此版本不輸出 GPU 時脈，**無法保證**整個過程完全沒降頻（`/sys` 讀到 1020 MHz、接面溫度最高 52–57 °C）。
- **記憶體是 Nano 最大的實務限制**：CPU／GPU 共用 8GB，GPU 配置不能放到 swap。建 OWL 引擎的 `trtexec` 需要約 3GB 以上，`MemAvailable` 只有 3.3GB 時連續失敗 4 次，約 5.0GB 時一次成功。同時跑 Ollama 也會互搶（§7）。伺服器程序 RSS：PyTorch 組合約 2.8GB、全 TensorRT 約 3.5GB（伺服器永遠也載入 baseline MobileSAM，為了網頁可切換編碼器）。WSL 是獨立 6GB 顯存（量測時 `nvidia-smi` 增加約 1 GB），沒有此限制。

## 6. 與官方數據對照

| 項目 | 官方（Orin Nano） | 本專案 Nano 實測 | 能不能比 |
|---|---|---|---|
| NanoSAM (ResNet18) Full Pipeline | 27 ms | encoder 29.7 + decoder 9.8（1 個框）≈ **39.5 ms** | 架構相同可對照；本次**較慢**。官方未說明是否含前處理；純 GPU 的 encoder + decoder 只有約 18 ms |
| MobileSAM Full Pipeline | 146 ms | SAM 階段（4 個框，PyTorch fp32）287 ms | 框數與定義不同，不可比 |
| NanoSAM Image Encoder | TBD | 29.7 ms（含前處理） | 無法對照 |
| NanoOWL (ViT-B/32) FPS | TBD（AGX Orin 為 95 FPS） | OWL 合計 51.3 FPS（Nano） | 不同硬體 |
| OWL-ViT（PyTorch）在 Orin Nano | 約 3 秒/張 | PyTorch fp32 OWL 約 138 ms | 條件不同（較新 JetPack／TensorRT、MAXN_SUPER） |

**不能下的結論**：官方數據的測試條件（JetPack／TensorRT 版本、電源模式、框數、是否含前後處理）未公布，所以只能當量級參考，不能說「比官方快／慢」。**架構相同 → 延遲可對照；權重不同 → 準確度不可對照。**

## 7. HyDE 提示詞擴展（Ollama，CPU 推論）

HyDE 只在**換提示詞時**呼叫一次 LLM（每個提示詞產生 5 個短英文外觀詞組，原提示加上它們為一組，組內取最高分），不影響逐幀 FPS。

| | WSL | Nano |
|---|---|---|
| 預設模型 | `qwen3:4b`（約 3.4 s／提示詞，n = 5） | `qwen2.5:1.5b`（暖機後約 1.8 s，n = 6；首次載入 7.9 s；常駐 1.2 GB） |
| 其他候選（Nano） | — | `llama3.2:1b` 2.1 s、`qwen3:1.7b` 2.5 s、`gemma3:1b` 5.4 s，全部 0/6 失敗 |

- **Nano 上 Ollama 必須用 CPU（`num_gpu=0`、`num_ctx=1024`）**：伺服器的 TensorRT 引擎運作中，4 個候選模型有 3 個在 GPU 上每次都 `cudaMalloc failed: out of memory`；原因與 §5 相同。Jetson 上 HyDE 預設不自動啟動、閒置 2 分鐘卸載（還記憶體給引擎）。
- 4 個模型都能把中文「螺絲」「貓」翻成正確英文，**也都會混入少量不相關或過於籠統的詞組**（例如 `a screwdriver`、`a cat toy`、`animal`），可能造成誤偵測。這是逐筆人工判讀，只有 6 個提示詞，**不是統計結論**。
- 端對端驗證（Nano，TensorRT 伺服器運作中）：`螺絲,貓` 不用 HyDE 偵測 0 個（OWL 只懂英文）；用 HyDE 後 `貓` 偵測到 2 隻，遮罩與英文 `a cat` 相同；偵測維持 90–94 ms／請求，無 OOM。WSL 先前的驗證結果相同（中文提示詞 0 個 → 2 貓 + 2 遙控器）。
- 兩邊的模型與樣本數不同，**不能直接比較**，只能看量級。

## 8. Nano 部署遇到的問題與解法（重點）

完整排查過程見 `docs/record.md` §13。

| # | 問題 | 原因／解法 |
|---|---|---|
| 1 | `import torch` 段錯誤 | **系統 CUDA 函式庫檔案損毀**（`libcublasLt.so.12` 約 287 bytes 不符）；`dpkg -V` 找出 14 個套件，`apt-get install --reinstall` 後正常。損毀原因未確認。`00_check_jetpack.sh` 現在會自動檢查 |
| 2 | pip 裝到 CPU 版 torch（`2.8.0+cpu`） | 同時指定 PyPI 的 extra index，版本號相同選到 CPU wheel；只用 Jetson index + `--no-deps` |
| 3 | numpy 被升到 2.x | `ml_dtypes 0.6` 要求 numpy ≥ 2；釘 numpy 1.26.4 與 ml_dtypes < 0.6 |
| 4 | 建 OWL 引擎 OOM（`NvMapMemAllocInternalTagged error 12`，被 PyTorch 顯示成 `NVML_SUCCESS == r INTERNAL ASSERT`；後者為推測） | 共用記憶體不足。`build_owl_engine.py` 改在 CPU 匯出 ONNX 再 `trtexec --fp16`；並結束 gazebo／VS Code 遠端，約 5GB 可用時成功 |
| 5 | 載入 OWL 後 SAM 第一個卷積 OOM | 檔案頁面快取佔住「真正空閒」頁面；`src/memutil.py` 載入後 `posix_fadvise(DONTNEED)` 放掉模型檔快取（受控對照下 SAM 階段由失敗轉成功；機制只得到部分支持） |
| 6 | `05_set_power_mode.sh` 預設 `0` 其實是 15W | 此機 0=15W、1=25W、2=MAXN_SUPER；腳本改成依名稱查編號 |
| 7 | torch2trt 安裝失敗（`No module named 'tensorrt'`） | pip 隔離建置看不到系統 tensorrt；`--no-build-isolation` |
| 8 | `gpu_stats()` 每個請求多花 20–26 ms | Jetson 的 `nvidia-smi` 只回傳 `[N/A]` 卻要 18 ms 啟動；改讀 sysfs（5.9 ms） |
| 9 | 網頁主數字不是使用者等待的時間 | 原本只含推論＋序列化，小圖低估 19%、手機照片低估約 10 倍；改為端對端並補 `decode_ms`、`server_ms` |
| 10 | 伺服器攝影機 FPS 高估 26% | 舊算法漏算疊圖與 JPEG 編碼且平均瞬時 FPS；改為平均每幀時間 |
| 11 | MobileSAM 在 TensorRT fp16 輸出全空 | 與官方 README 註記一致；baseline 只能 fp32，自訓練 ResNet18 的 fp16 無損 |
| 12 | TensorRT mask decoder 數值偏差 | 見 §4.3，未解 |

## 9. 驗證範圍（誠實版）

**已驗證**
- Nano 的 6 份分階段 JSON 與端對端矩陣（`e2e_matrix.json`、`e2e_trt_variants.json`）：報告中的 99.4 / 427.7 / 289.6 / 178.2 / 117.5 / 108.8 / 40.5 ms 與端對端 461 / 341 / 200 / 129 / 211 / 141 ms、能耗 6.98 / 4.76 / 2.62 / 1.49 J，**全部從原始 JSON 重算相符**；功耗另從原始 `tegrastats` log 重算（§5）。
- 所有 Python 檔通過編譯、shell 腳本通過語法檢查、`frontend/app.js` 通過 `node --check`。
- WSL 上合併後的程式：API 契約、NanoSAM 後端、自動退回 PyTorch 的行為（§4.4）。

**無法驗證**
- **Nano 的 TensorRT 引擎本身**：收到的 zip 裡 `.engine` 與 `.onnx` 檔被截斷（引擎都剛好 4,194,304 bytes、模型 3,670,016 bytes；zip 內的 `mobile_sam.pt` 也被截斷，校驗碼與本機 baseline 不同），所以**無法在別的機器重現**，也不要把這些檔案放進版控。引擎必須在 Nano 上用 `installation/jetson/03_build_engines.sh` 重建（引擎不可跨 GPU 架構移植）。
- Nano 的數字是我沒有在現場重跑的，只核對了「報告與原始檔一致」，沒有核對「原始檔是否真的來自那台機器」。
- zip 內的 `.git` 損壞（pack 與 index 不符），所以 6 個 commit 的歷史取不到，我是用「檔案差異」移植的；base 就是本機 `fa6e549`，沒有衝突。

## 10. 限制與尚未完成

1. 單一影像（`cats.jpg`）、4 個框、單一電源模式（MAXN_SUPER）；沒有 15W／25W 的比較。框數不同會線性改變 SAM 時間（每個框約 10 ms）。
2. WSL 與 Nano 的軟體版本不同（PyTorch 2.5.1 vs 2.8.0、CUDA 12.1 vs 12.6），硬體差與軟體差混在一起沒有拆開。
3. WSL 為單一天的單次 50 次平均；與兩天前的紀錄差 5% 的原因未查（§1.3）。
4. 準確度只在 WSL 以 PyTorch fp32 量；沒有量「TensorRT 引擎」的 mIoU，只量與 PyTorch 的遮罩 IoU（單張圖、4 個框）。
5. 沒有量官方 NanoSAM 預訓練權重；所有 SAM encoder 數字都是自行蒸餾的權重。
6. **提示詞寫法比較（≥ 10 張影像、≥ 3 種寫法）、攝影機即時 Demo（Nano 上偵測不到 `/dev/video*`）、製造情境真實影像、Demo 影片尚未做。**
7. 重複偵測（高重疊的框同時出現）：閾值 0.1 時 23% 的影格會多算；模擬的「包含抑制」可降到 0%，但**尚未實作進流程**。
8. TensorRT mask decoder 數值偏差的根本原因未查出（§4.3）。

## 11. 重現方式與原始資料

```bash
# WSL（把 APP_LOG 指到 Linux 檔案系統，否則完整流程會多 8~9 ms，見 §1.3）
APP_LOG=/tmp/bench.log python benchmark/bench_pipeline.py data/samples/cats.jpg "a cat" "a remote control" --runs 50
APP_LOG=/tmp/bench.log python benchmark/bench_pipeline.py data/samples/cats.jpg "a cat" "a remote control" --runs 50 --student data/eval/colab80_best.pth
python benchmark/bench_e2e.py --configs pytorch,student --out /tmp/wsl_e2e/e2e_matrix.json

# Nano（先 installation/jetson/05_set_power_mode.sh，再 03_build_engines.sh 建引擎）
python benchmark/bench_pipeline.py data/samples/cats.jpg "a cat" "a remote control" --runs 50 \
    --owl_engine data/owl_image_encoder_patch32.engine \
    --nanosam_encoder data/resnet18_image_encoder.selftrain.engine --nanosam_decoder data/mobile_sam_mask_decoder.engine
python benchmark/bench_e2e.py --configs pytorch,owltrt_mobilesam,owltrt_student,trt
```

| 內容 | 檔案 |
|---|---|
| WSL 分階段（baseline／自訓練） | `data/bench/wsl/pipe_pytorch_fp32.json`、`pipe_pytorch_student.json` |
| WSL 端對端 | `data/bench/wsl/e2e_matrix.json` |
| Nano 分階段 | `data/bench/jetson/pipe_*.json`（`pipe_all_trt_final.json` 為 fp16 全 TensorRT） |
| Nano 端對端、各 SAM encoder 變體 | `data/bench/jetson/e2e_matrix.json`、`e2e_trt_variants.json`、`e2e_trt_detail.json` |
| Nano 功耗／溫度 | `data/bench/jetson/tegrastats_*.log`、`e2e_*_tegra.log` |
| Nano 引擎建置與失敗嘗試 | `data/bench/jetson/build_*.log` |
| HyDE 模型比較（Nano） | `data/bench/jetson/hyde_model_eval.json` |
| 準確度 | `data/eval/compare_run1_colab80.json`（本機，不進版控） |
| 詳細版（逐項、含大圖日誌） | [measurement-report.md](measurement-report.md)、[wsl-vs-jetson.md](wsl-vs-jetson.md)、[record.md](record.md) §13 |
