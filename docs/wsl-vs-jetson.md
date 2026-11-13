# WSL（GTX 1060）與 Jetson Orin Nano 的數據對照

> 目的：把先前在 WSL 開發機上量到的數據，和 Nano 上同樣方法量到的數據放在一起比。**WSL 的數據是開發用，題目要求繳交的是 Nano 的數據**（`docs/benchmark.md`）。
> WSL 的數字取自 `docs/benchmark.md` 與 `docs/record.md`（原始 JSON 在 WSL 那台機器上，這台沒有，所以無法重新計算）；Nano 的數字由 `data/bench/jetson/*.json` 計算。

> **更新（整合）**：下文 WSL 的完整流程 197.2 ms 是 2026-10-29 的紀錄、自訓練的 141.0 ms 是推算。之後在同一台 WSL 上用同一套腳本重量，實測為 **207.0 ms（baseline）／149.9 ms（自訓練）**，並補了 WSL 端對端；整合後的數字與倍率以 [experiment-report.md](experiment-report.md) 為準。本檔保留原樣，供對照當時的紀錄。

## 0. 結論

- **同樣的 PyTorch fp32，Nano 比 GTX 1060 慢約 2 倍**：完整流程 427.7 ms vs 197.2 ms（2.17×）；各階段落在 1.85×（OWL 合計）到 2.68×（OWL 前處理）之間，其中最大的比值多半出在只有幾 ms 的小階段。
- **換成 TensorRT 之後，Nano 反而比 WSL 的 PyTorch 快**：99.4 ms vs 197.2 ms（快 2.0 倍）。但這個比較同時換了 SAM encoder 模型；若 WSL 也用自訓練 encoder（推算 141.0 ms，非實測），則是快 1.4 倍。
- **Nano 上 TensorRT 相對 PyTorch**：OWL 影像編碼器快 8.1 倍；SAM encoder（同一份自訓練權重）快 2.2 倍；整體（同一份 SAM 權重）快 2.9 倍。
- **PyTorch 記憶體配置峰值兩邊幾乎相同**（WSL 901 MB、Nano 902 MB），間接支持兩邊跑的是同樣的模型與輸入。

## 1. 量測條件

| 項目 | WSL | Nano |
|---|---|---|
| GPU | GeForce GTX 1060 6GB（Pascal，獨立顯存） | Orin Nano 8GB（Ampere，與 CPU 共用 8GB） |
| 系統 | WSL2 | JetPack 6.2.1（L4T R36.4.7） |
| PyTorch | 2.5.1 + cu121（依 `requirements.txt` 的固定版本；**沒有核對 WSL 實際安裝的版本**） | 2.8.0 + cu126（Jetson 專用 wheel） |
| 電源／時脈 | 量測期間 GPU 1936 MHz、60 °C | **MAXN_SUPER（nvpmodel 2）**，GPU 1020 MHz、56–58 °C（PyTorch 組合） |
| 腳本與參數 | `benchmark/bench_pipeline.py`，暖機 10 次、量測 50 次 | **同一支腳本、同樣參數** |
| 輸入 | `cats.jpg` 640×480，`a cat`、`a remote control`，NMS 後 4 個框 | 同；Nano 上 MobileSAM 的遮罩面積 49916 / 58217 / 4121 / 1975 px，與 WSL 紀錄的 49958 / 58209 / 4121 / 1975 px 相差 < 0.1% |
| 量測日期 | 2026-10-29 | 2026-11-01 |

## 2. 逐階段對照（平均，ms）

| 階段 | WSL：GTX 1060，PyTorch fp32 | Nano：PyTorch fp32 | Nano：TensorRT | Nano PyTorch ÷ WSL（硬體差） | Nano PyTorch ÷ Nano TensorRT（TensorRT 加速） |
|---|---|---|---|---|---|
| OWL 前處理 | 0.6 | 1.6 | 1.3 | 2.68× | 1.25× |
| OWL 影像編碼器 | 72.7 | 136.0 | 16.9 | 1.87× | 8.06× |
| OWL 解碼 | 0.8 | 1.9 | 1.9 | 2.35× | 0.98× |
| **OWL 合計** | 74.5 | 137.6 | 19.5 | 1.85× | 7.06× |
| OWL 文字編碼器（換提示詞時才跑；TensorRT 組合仍是 PyTorch） | 9.5 | 25.0 | 25.7 | 2.64× | 0.97× |
| SAM 影像編碼器：MobileSAM（baseline） | 88.5 | 207.4 | — | 2.34× | — |
| SAM 影像編碼器：自訓練 ResNet18 | 32.3 | 66.1 | 29.7 | 2.05× | 2.23× |
| SAM decoder（每個框） | 7.9 | 18.2 | 9.8 | 2.31× | 1.87× |
| **完整流程（4 個框）：MobileSAM** | 197.2 | 427.7 | — | 2.17× | — |
| **完整流程（4 個框）：自訓練 ResNet18** | 141.0（推算） | 289.6 | 99.4 | 2.05× | 2.91× |

- 「Nano PyTorch ÷ WSL」只差**硬體**（同樣的 PyTorch fp32 與模型）。
- 「Nano PyTorch ÷ Nano TensorRT」只差**執行方式**；自訓練 ResNet18 與完整流程那兩列，分母用的是同一份 SAM 權重的 PyTorch 數字（66.1 ms、289.6 ms），不是 MobileSAM 的。
- WSL「自訓練 ResNet18」的 encoder 32.3 ms 來自 `compare_encoders.py`（baseline 同法為 87.5 ms，與 `bench_pipeline.py` 的 88.5 ms 很接近），**不是 `bench_pipeline.py` 的逐階段數字**；WSL 的完整流程 141.0 ms 是我用 OWL 74.5 + 32.3 + 4 × 7.9 + 2.6（NMS 與搬移）**推算**的，不是實測。
- TensorRT 組合的 SAM encoder 數字含前處理（縮放、正規化、補邊），純 GPU（trtexec）只有 9.9 ms。

### 2.1 所有 SAM encoder 並排：baseline vs 我們訓練的，各種執行方式

Nano 用 TensorRT，WSL 用它原本的 PyTorch fp32。同一支腳本、同一張圖（`cats.jpg`）、4 個框、50 次。

**SAM image encoder（含前處理，ms）**

| SAM image encoder | WSL（GTX 1060）PyTorch fp32 | Nano PyTorch fp32 | Nano TensorRT fp32 | Nano TensorRT fp16 |
|---|---|---|---|---|
| **baseline：MobileSAM TinyViT** | 88.5 | 207.4 | 108.8 | 輸出錯誤（遮罩全空），不可用 |
| **我們訓練的 ResNet18** | 32.3 | 66.1 | 40.5 | **29.7** |
| 自訓練 ÷ baseline（越小越好） | 0.36 | 0.32 | 0.37 | — |

| 純 GPU 延遲（trtexec） | Nano TensorRT fp32 | Nano TensorRT fp16 |
|---|---|---|
| baseline：MobileSAM TinyViT | 92.8 | 38.6（輸出錯誤） |
| 我們訓練的 ResNet18 | 22.1 | **11.6** |

**完整流程（ms）**

| 完整流程（4 個框，不含解碼與 HTTP） | WSL PyTorch fp32 | Nano PyTorch fp32 | Nano TensorRT fp32 | Nano TensorRT fp16 |
|---|---|---|---|---|
| **baseline：MobileSAM** | 197.2 | 427.7 | 178.2 | 不可用 |
| **我們訓練的 ResNet18** | 141.0（推算） | 289.6 | 117.5 | **99.4** |

**端對端（真實 API，loopback，50 次，ms）**

| 端對端（客戶端送出到收完整 JSON，ms） | 640×480 | 1280×960 | 4096×3072 | 每個請求能耗 (J)，640×480 |
|---|---|---|---|---|
| ① Nano：OWL PyTorch + baseline PyTorch | 461 | 498 | 961 | 6.98 |
| ② Nano：OWL TensorRT + baseline PyTorch | 341 | 382 | 854 | 4.76 |
| ③ Nano：OWL TensorRT + baseline **TensorRT fp32** | 211 | 246 | 725 | 2.86 |
| ④ Nano：OWL TensorRT + 自訓練 PyTorch | 200 | 238 | 657 | 2.62 |
| ⑤ Nano：OWL TensorRT + 自訓練 **TensorRT fp32** | 141 | 177 | 615 | 1.72 |
| ⑥ Nano：OWL TensorRT + 自訓練 **TensorRT fp16** | 129 | 169 | 628 | 1.49 |

**遮罩正確性**（TensorRT 對同一個模型的 PyTorch fp32，4 個框的遮罩 IoU，`cats.jpg`）

| 模型 | TensorRT fp32 | TensorRT fp16 |
|---|---|---|
| baseline：MobileSAM | 0.981 | **0.000（全空）** |
| 我們訓練的 ResNet18 | 0.973 | 0.973 |

分割準確度（WSL，COCO 200 張、GT 框當提示，PyTorch fp32，`docs/record.md` §7.4）：baseline mIoU **0.744**，我們訓練的 **0.704**。

重點：
- **baseline 只能用 fp32；我們訓練的 ResNet18 用 fp16 沒有任何損失**（IoU 0.9730 對 0.9729）。這是它在 Nano 上快的關鍵：fp16 引擎 11.6 ms，baseline 的 fp32 引擎 92.8 ms，純 GPU 差 8 倍。
- **Nano 上 TensorRT 的 SAM encoder**：baseline 108.8 ms，我們訓練的 29.7 ms（fp16）／40.5 ms（fp32）。同一個模型換 TensorRT：baseline 1.9×，ResNet18 2.2×。
- **端對端**：⑥ 比 ③ 快 1.6 倍（211 → 129 ms，640×480），也就是同樣都用 TensorRT，換成我們訓練的模型再用 fp16 的收益；比 ① 快 3.6 倍。
- **4096×3072 這一欄 ④⑤⑥ 看不出差異**（657／615／628 ms）：⑥ 另外重跑過一次是 656 ms，兩次之間的差距比 ⑤⑥ 之間還大。大圖時時間主要花在解碼與輪廓序列化，encoder 的差別被蓋過去。
- TensorRT 預設的「fp32」引擎比加 `--noTF32` 的快 2.2 倍（ResNet18：22.1 對 49.7 ms），這符合 TensorRT 在 Ampere 上預設啟用 TF32 的行為（我沒有直接檢查引擎設定）；兩者的遮罩一樣（IoU 0.9729）。上表的 fp32 都是預設（官方 NanoSAM 也是用預設建的）。
- **TensorRT 10.3 的 mask decoder 引擎與它的來源 ONNX 算出不同的結果**（模型自評的 `iou_pred` 0.67 對 0.96，遮罩 IoU 約 0.97）。ORT 內建的 TensorRT 執行器也一樣，所以是 TensorRT 的問題，不是我們的程式；調整建置選項、精度、形狀都修不好。baseline 與自訓練共用同一個 decoder，所以上面的相對比較不受影響，但**兩者的絕對遮罩品質都比 PyTorch 低約 2～3%**。用 ORT 的 CUDA 後端跑 decoder 是精確的（與 PyTorch 的遮罩 IoU 0.9997），但每次呼叫慢約 6 ms（單次量測：15.7 對 9.8 ms）。

## 3. 從 WSL 到 Nano TensorRT：差距各來自哪裡（完整流程，ms）

| 步驟 | 完整流程 (ms) | 與上一步相差 (ms) | 這一步改變了什麼 |
|---|---|---|---|
| WSL：GTX 1060，PyTorch fp32，MobileSAM | 197.2 | — | 實測（docs/benchmark.md） |
| Nano：PyTorch fp32，MobileSAM | 427.7 | +230.5 | 硬體不同（其餘相同） |
| Nano：PyTorch fp32，自訓練 ResNet18 | 289.6 | -138.2 | 換 SAM encoder 模型（MobileSAM → ResNet18） |
| Nano：TensorRT（OWL fp16 + 自訓練 ResNet18 fp16 + decoder fp32） | 99.4 | -190.1 | 同一份權重，換 TensorRT 與 fp16 |

三步加起來是 -97.8 ms：硬體讓它**變慢** 230.5 ms，換模型**省** 138.2 ms，TensorRT 再**省** 190.1 ms。

## 4. 記憶體、溫度、功耗

| 項目 | WSL（GTX 1060，PyTorch fp32） | Nano：PyTorch fp32 | Nano：TensorRT |
|---|---|---|---|
| PyTorch 配置峰值 (MB) | 901 | 902（MobileSAM）／883（自訓練）| 678（不含 TensorRT 引擎自己的記憶體） |
| 溫度／GPU 時脈 | 60 °C／1936 MHz | 56–58 °C／1020 MHz | 51 °C／1020 MHz |
| 功耗 | **WSL 沒有量** | 整板約 15.2 W（負載中，端對端測試，見 `measurement-report.md` §6.3） | 整板約 11.6 W |

- WSL 的顯存是獨立的 6GB（量測時 `nvidia-smi` 增加約 1006 MB）；Nano 的 CPU 與 GPU 共用 8GB，所以建 TensorRT 引擎與同時跑 Ollama 都會互相搶（`docs/record.md` §13.2 第 7 點、§13.6），這是 WSL 上不存在的限制。
- 功耗沒有 WSL 的數據可比；GTX 1060 的規格 TDP 是 120 W（規格表，非本次量測），而本次 Nano 量到的是整板輸入功耗，兩者定義不同，不能直接相除。

## 5. 其他可比的項目

| 項目 | WSL | Nano | 備註 |
|---|---|---|---|
| HyDE LLM（`qwen2.5:1.5b`，CPU 推論）每個提示詞 | 約 1.0 s（`docs/hyde.md`） | 暖機後約 1.8 s（`measurement-report.md` §9） | WSL 是 n = 3 的測試（`docs/record.md` §6.3），Nano 是 n = 5，條件不完全相同，只能看量級 |
| `qwen3:4b`（CPU）每個提示詞 | 約 3.4 s（n = 5） | **未量**（Nano 預設改用較小的模型） | |
| 只要計數（不跑 SAM）的後端單次 | 201 → 75 ms（有遮罩 → 只留框，`docs/record.md` §4.4） | **未量** | |
| 端對端（HTTP）延遲 | **WSL 沒有受控量測** | 129 ms（全 TensorRT，640×480） | 見 `measurement-report.md` §6 |

## 6. 為什麼 Nano 的 PyTorch 比 GTX 1060 慢約 2 倍

- FP32 理論峰值：GTX 1060 約 4.4 TFLOPS（`docs/benchmark.md` 也用這個數字）；Orin Nano 8GB 為 1024 個 CUDA 核心 × 2 × 1.02 GHz ≈ 2.1 TFLOPS（我用規格表的核心數乘上實測時脈 1020 MHz 算出，**不是量測值**）。比值約 2.1 倍，與各階段的 1.9–2.6 倍同量級。
- 小型運算（SAM decoder 每個框 2.31×、OWL 文字編碼器 2.64×）落差比大型卷積／注意力運算更大。**推測**是這類運算受 CPU 端的 kernel 啟動與 Python 開銷影響較多，而 Nano 的 CPU（Cortex-A78AE）比 WSL 主機的 CPU 弱；沒有驗證。

## 7. 限制

1. WSL 與 Nano 的**軟體版本不同**（PyTorch 2.5.1 vs 2.8.0，CUDA 12.1 vs 12.6），硬體差距與軟體差距混在一起，沒有拆開。
2. WSL 的數字是單一天、單次執行的 50 次平均；沒有在不同日期重複，也沒有重新驗證原始 JSON。
3. WSL 的功耗、端對端延遲沒有量，所以這兩項只有 Nano 的數字。
4. TensorRT 是 fp16（decoder 為 fp32），輸出與 PyTorch fp32 不完全相同（SAM 遮罩 IoU 約 0.96–0.99，見 `docs/record.md` §13.4）。
5. 「完整流程」只含推論呼叫，不含圖片解碼、序列化與 HTTP（端對端見 `measurement-report.md` §6）。

## 8. 原始資料

| 內容 | 來源 |
|---|---|
| WSL 逐階段、記憶體、溫度 | `docs/benchmark.md`「WSL 結果」、`docs/record.md` §4.2 |
| WSL encoder 延遲（baseline 87.5／自訓練 32.3 ms） | `docs/record.md` §7.4（`data/eval/compare_run1_colab80.json` 為其原始檔） |
| Nano PyTorch（MobileSAM） | `data/bench/jetson/pipe_pytorch_fp32.json` |
| Nano PyTorch（自訓練 ResNet18） | `data/bench/jetson/pipe_pytorch_student.json` |
| Nano TensorRT | `data/bench/jetson/pipe_all_trt_final.json` |
