# smart-processing-midterm-project-2026Fall

## 主題四：文字提示之開放詞彙物件偵測與分割系統

授課教授：陳響亮　助教：余冠霖　日期：2026/10/23

### 一、應用場景

傳統偵測模型僅能辨識訓練時定義之類別，產線更換產品或新增檢測對象時，須重新蒐集資料並訓練。開放詞彙模型可依文字描述找出對應物件，不須重新訓練，適用於少量多樣之生產環境。

### 二、專案目標

於 Jetson Orin Nano 建置 NanoOWL 與 NanoSAM 並串接，實現以文字指定物件並取得分割輪廓之系統，並實測其效能。

### 三、專案成果須包括以下五項功能要求（評分用）（50 分）

1. **環境與 TensorRT 引擎**（10 分）：於 Jetson Orin Nano（JetPack 6.2）建置 NanoOWL 與 NanoSAM（GitHub 開源模組）之執行環境，並建立兩模型之 TensorRT 引擎，於報告中說明所遭遇之版本相容性問題與解決方式。
2. **兩模型串接**（10 分）：輸入文字提示 → NanoOWL 輸出物件框 → NanoSAM 以物件框為提示輸出分割遮罩，並於攝影機即時畫面上同時顯示物件框與遮罩。
3. **製造現場應用情境**（10 分）：以 2. 之串接系統設計至少 1 個製造現場應用情境（如依文字指定零件並計算其數量或面積），且可於系統執行中即時更換文字提示，不須重新啟動程式。
4. **效能量測**（10 分）：於 Jetson Orin Nano 分別量測 NanoOWL、NanoSAM 及 2. 整體流程之推論延遲、FPS 與記憶體用量，並與 NanoSAM 官方公布之數據對照。
5. **提示詞比較與失敗分析**（10 分）：自行拍攝或選用至少 10 張影像，以至少 3 種不同提示詞寫法測試 2. 之系統，比較其偵測正確率，並分析失敗案例（相似物件、小物件、遮擋）之原因。

### 四、繳交內容須包括以下五項內容（評分用）（50 分）

1. **程式碼與環境建置說明**（10 分）：串接 NanoOWL 與 NanoSAM 之部署程式碼、TensorRT 引擎建置腳本，並附 README 說明執行環境（JetPack 與套件版本）與執行步驟，以及 JetPack 6.2 下遭遇之版本相容性問題與解決方式。
2. **研究背景與原理**（10 分）：說明開放詞彙偵測於少量多樣生產之需求，繪製系統情境圖與資訊流程圖（文字提示、NanoOWL 物件框、NanoSAM 分割遮罩、畫面輸出），並說明 OWL-ViT 與 Segment Anything 之基本原理。
3. **效能比較表**（10 分）：NanoOWL、NanoSAM 及整體流程之推論延遲、FPS 與記憶體用量量測表（須與 NanoSAM 官方數據對照），以及不同提示詞寫法之偵測正確率比較表，並註明量測時之電源模式。
4. **結果截圖與分析**（10 分）：至少 5 張偵測與分割結果截圖（須含製造現場應用情境與失敗案例），並分析相似物件、小物件與遮擋造成失敗之原因，歸納提示詞設計之原則。
5. **Demo 講解影片**（10 分）：展示於攝影機即時畫面中更換文字提示並取得物件框與遮罩之過程，並講解專案達成的五項重要成果，包括：環境建置、兩模型串接流程、製造現場應用情境、效能量測結果及提示詞比較結果。

### 五、參考資源

- 模型實作｜NanoOWL（NVIDIA-AI-IOT/nanoowl）
- 模型實作｜NanoSAM（NVIDIA-AI-IOT/nanosam）

> 原始題目圖檔見 `image-project-description/`。

---

## 開發計畫（WSL + GTX 1060 先行開發，最後移植 Jetson Orin Nano）

### 開發環境差異

| 項目 | 開發機（WSL） | 目標（Jetson Orin Nano） |
|---|---|---|
| GPU | GTX 1060 6GB（Pascal, sm_61, 驅動 CUDA 12.6） | Ampere sm_87, 8GB 共用記憶體 |
| 系統 | WSL2 Linux | JetPack 6.2 |
| 用途 | 演算法/串接/UI/提示詞測試開發 | TensorRT 引擎、效能量測、Demo（計分關鍵） |

### 需要做的事（洞悉）

**A. WSL 可先完成**
- [x] 建立 conda 環境、下載模型（OWL-ViT、MobileSAM、NanoSAM mask decoder ONNX）
- [x] 跑通 NanoOWL（文字 → 物件框）與 SAM（框 → 遮罩）
- [x] 串接流程與疊圖顯示（框／遮罩／輪廓／各提示詞數量與面積）
- [x] 即時更換文字提示（不需重啟）
- [ ] 自行蒸餾 NanoSAM ResNet18 image encoder（官方檔案需登入 Google Drive）：**訓練進行中**（第一輪因 fp16 教師作廢，已修正重跑，見 [docs/record.md](docs/record.md) §7.2）；完成後用 `training/compare_encoders.py` 與 baseline 比較（baseline mIoU 0.744），再匯出 ONNX。說明見 [docs/training.md](docs/training.md)
- [ ] 計數偏高（局部／巢狀偵測）：IoU-NMS 在預設門檻 0.1 只把多報影格從 26% 降到 23%，剩下的是低分局部框；門檻 ≥ 0.2 或「包含抑制」可消除（單一場景實測，見 [docs/scenario.md](docs/scenario.md)），尚未寫進 pipeline
- [x] HyDE 提示擴展（Ollama + qwen3:4b，每組 5 詞句取組內最高信心，詳見 [docs/hyde.md](docs/hyde.md)）；尚待用真實素材驗證效果
- [x] 攝影機／影片輸入（瀏覽器攝影機、影片檔、伺服器端攝影機或影片路徑）
- [x] 製造情境：零件檢查（工單、預期數量、比例尺校準與 mm² 面積、單件面積範圍、PASS/FAIL、檢查紀錄、可列印報表），詳見 [docs/scenario.md](docs/scenario.md)；尚待用真實製造素材驗證
- [x] **SAM 的 NanoSAM 後端**（`src/sam_backends.py`）：TensorRT 引擎或 ONNX Runtime；pipeline / 網頁 / benchmark 都可選用（`NANOSAM_ENCODER` / `NANOSAM_DECODER`）。前後處理已用 ONNX Runtime 在 WSL 驗證（`tests/check_nanosam_backend.py`，與 PyTorch 遮罩 IoU ≥ 0.995）；**TensorRT 路徑（torch2trt + TensorRT 10）尚未在 Jetson 實測**
- [ ] **批次評估腳本**（影像 × 提示詞寫法 → 偵測正確率表）與提示詞測試集：≥10 張影像、≥3 種提示詞寫法（題目功能要求 5）
- [ ] 蒐集真實製造影像 / 影片（≥10 張 + 一段影片，人工標註實際數量），並用它驗證：製造情境、HyDE 是否有益、面積準確度、計數誤差
- [x] 效能量測腳本 `benchmark/bench_pipeline.py`（分階段延遲 / FPS / 記憶體，**目前只支援 PyTorch baseline**）；[ ] 補上 NanoSAM / TensorRT / 電源模式的選項與官方對照表輸出
- [ ] 準備報告素材：情境圖、OWL-ViT／SAM 原理說明（資訊流程圖見 [docs/workflow.md](docs/workflow.md)；**問題與解決方式的紀錄見 [docs/record.md](docs/record.md)**）

**B. 必須在 Jetson 完成（無法由 WSL 取代）**
- [ ] JetPack 6.2 環境建置，記錄版本相容性問題與解法
- [ ] 於 Jetson 建立 TensorRT 引擎（引擎不可跨 GPU 架構移植）
- [ ] 實測延遲／FPS／記憶體，並記錄電源模式（如 15W / 25W / MAXN），對照 NanoSAM 官方數據
- [ ] 攝影機即時畫面 Demo、錄製講解影片
- [ ] 截圖 ≥5 張（含製造情境與失敗案例）

### 風險與注意

- GTX 1060（sm_61）：新版 PyTorch／TensorRT 可能已不支援或不佳（TensorRT 10 已移除 Pascal 支援），WSL 端可能需用較舊版本或改用 PyTorch / ONNX Runtime 開發；效能數據僅供參考，不可作為繳交數據。
- WSL 看不到 USB 攝影機，開發階段以影片檔／影像資料夾代替輸入。
- 6GB VRAM（目前已被佔用約 1.9GB）需留意 OWL-ViT 與 SAM 同時載入的記憶體。
- 程式碼設計成「後端可抽換」（PyTorch／TensorRT），方便從 WSL 移植到 Jetson。

### 環境與部署

- 完整安裝說明（conda、模型下載、Jetson 步驟）：[docs/installation.md](docs/installation.md)
- WSL 開發環境：`./installation/setup_env.sh`（conda: `nanoowl-sam`），模型：`./installation/download_model.sh`
- HyDE（選用）：`./installation/install_ollama.sh` 安裝 Ollama 並下載 qwen3:4b；server 啟動時會自動拉起，詳見 [docs/hyde.md](docs/hyde.md)
- Jetson 部署：操作手冊 [docs/jetson-deploy.md](docs/jetson-deploy.md)、資料清單與量測記錄表 [docs/jetson-data.md](docs/jetson-data.md)、版本相容性筆記 [docs/jetson-setup.md](docs/jetson-setup.md)；腳本在 `installation/jetson/`（`pack_for_jetson.sh` 於開發機打包，其餘 `00`~`07` 與 `collect_env.sh` 於 Jetson 依序執行）

### 快速開始（WSL）

```bash
git clone --recurse-submodules <repo-url> && cd <repo>
./installation/setup_env.sh && conda activate nanoowl-sam
./installation/download_model.sh owlvit && ./installation/download_model.sh mobilesam

python src/test_image.py <影像> "a cat" "a remote control"      # 單張：框 + 遮罩
python src/demo_owl.py <影像> "a cat"                           # 只跑 OWL
python src/demo_sam.py <影像> --box 6 53 320 469                # 只跑 SAM
python src/app.py --source <影片> --prompt "a cat" --style contour   # 即時；執行中輸入新提示 + Enter 可換
./installation/download_model.sh nanosam                        # 之後給 Jetson 轉引擎用的 mask decoder ONNX（encoder 需手動下載或自行蒸餾，見 docs/training.md）
python -m server.main                                             # Web 介面 http://127.0.0.1:8000（見 docs/web-app.md）
```

### 目錄結構

```
.
├── README.md
├── image-project-description/   # 題目圖檔
├── src/                         # pipeline.py（OWL+SAM 串接）、app.py（即時示範）、demo_owl / demo_sam / test_image
├── server/                      # FastAPI 後端（API、日誌、推論服務）
├── frontend/                    # Web 前端（HTML/JS，由後端提供）
├── benchmark/                   # 分階段延遲／FPS／記憶體量測
├── installation/                # 環境安裝、模型下載（含 jetson/ 部署腳本）
├── training/                    # NanoSAM image encoder 蒸餾訓練、與 baseline 的比較（compare_encoders.py）、COCO 下載
├── tests/                       # API 契約檢查（改 server 後請實際打一次 API）
├── third_party/                 # submodule: nanoowl、nanosam
├── docs/                        # installation / jetson-setup / training / workflow / web-app / hyde / scenario / benchmark / experiments / record（開發紀錄，寫報告用）
├── environment.yml, requirements*.txt
└── data/                        # 模型權重、引擎、測試資料（不進版控）
```


## macOS 階段歷史封存

舊開發的完整 PR／issue／審查紀錄與移植注意事項，見 [DEVELOPMENT_HANDOFF_ZH_TW.md](DEVELOPMENT_HANDOFF_ZH_TW.md)。所有舊分支程式集中保存於 [archive/macos-development-2026-10-29](https://github.com/Disesfgewu/smart-processing-midterm-project-2026Fall/tree/archive/macos-development-2026-10-29) 的 snapshots/；這是多版本參考庫，不是現行 main 的已驗收版本。
