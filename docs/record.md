# 開發紀錄（問題 → 原因 → 解決 → 證據）

> 用途：寫報告（尤其是「遭遇的版本相容性問題與解決方式」「結果分析」「提示詞設計原則」）時的素材庫。
> 範圍：2026-10-29 一整天的開發（WSL + GTX 1060 上先行開發；**Jetson Orin Nano 尚未實機測試**）。
> 記法：每則含 **現象 / 原因 / 解法 / 證據**。數字都是實測值，來源檔案標在括號內；「未驗證」會明講。
> 新增紀錄請接在對應章節，並維持這個格式。

## 目錄
1. [目前狀態與題目對照](#1-目前狀態與題目對照評分項目)
2. [環境與相容性問題](#2-環境與相容性問題)
3. [模型與串接](#3-模型與串接)
4. [效能量測](#4-效能量測)
5. [偵測品質：提示詞、重複偵測、座標](#5-偵測品質提示詞重複偵測座標)
6. [HyDE 提示擴展（Ollama）](#6-hyde-提示擴展ollama)
7. [NanoSAM encoder 蒸餾訓練](#7-nanosam-encoder-蒸餾訓練)
8. [後端、前端與日誌](#8-後端前端與日誌)
9. [製造現場情境](#9-製造現場情境)
10. [開發過程的錯誤與教訓](#10-開發過程的錯誤與教訓)
11. [待辦與未驗證項目](#11-待辦與未驗證項目)

---

## 1. 目前狀態與題目對照（評分項目）

### 功能要求（50 分）

| # | 要求 | 狀態 | 證據 / 缺口 |
|---|---|---|---|
| 1 | Jetson（JetPack 6.2）建置環境與兩模型 TensorRT 引擎，說明相容性問題 | **腳本已寫、未實機執行** | `installation/jetson/`、[jetson-setup.md](jetson-setup.md)。相容性問題表是依公開資料整理，**尚未在 Jetson 上驗證**；TensorRT 引擎一個都還沒建 |
| 2 | 文字提示 → NanoOWL 框 → NanoSAM 遮罩，攝影機即時顯示框與遮罩 | **WSL 上完成**（PyTorch 後端、MobileSAM 暫代 NanoSAM） | `src/pipeline.py`、網頁 / `src/app.py`；攝影機只用「瀏覽器假攝影機」與影片檔測過，**沒有實體 webcam / Jetson 攝影機** |
| 3 | ≥1 個製造現場情境，執行中可換提示 | **功能完成，缺真實素材驗證** | 網頁「製造情境：零件檢查」（[scenario.md](scenario.md)）；全部用貓 / 遙控器的圖測，**沒有真實零件** |
| 4 | Jetson 上量 NanoOWL / NanoSAM / 整體的延遲、FPS、記憶體，與官方對照 | **量測腳本完成，數據只有 WSL** | `benchmark/bench_pipeline.py`、[benchmark.md](benchmark.md)；WSL 數據**不可繳交**（GTX 1060 + PyTorch fp32） |
| 5 | ≥10 張影像 × ≥3 種提示詞寫法比較正確率，分析失敗案例 | **尚未做**（批次評估腳本未寫） | 已有零散的提示詞實測（見 §5.1）與實驗規劃 [experiments.md](experiments.md) |

### 繳交內容（50 分）

| # | 內容 | 狀態 |
|---|---|---|
| 1 | 程式碼與環境建置說明（README、TensorRT 腳本、版本問題） | 程式與 README 有；Jetson 版本問題為推測整理（未實測） |
| 2 | 研究背景與原理（需求、系統情境圖與資訊流程圖、OWL-ViT 與 SAM 原理） | 流程圖有（[workflow.md](workflow.md)）；**情境圖與原理說明還沒寫** |
| 3 | 效能比較表 + 提示詞正確率表 + 電源模式 | WSL 表有（不可繳交）；Jetson 數據、正確率表、電源模式**都還沒有** |
| 4 | ≥5 張截圖（含製造情境與失敗案例）與分析 | 網頁可產生含失敗案例截圖的報表；**真實素材的截圖還沒有** |
| 5 | Demo 影片 | 尚未錄 |

---

## 2. 環境與相容性問題

### 2.1 PyTorch 版本與 GTX 1060（Pascal, sm_61）
- **現象**：擔心新版 PyTorch 不支援舊顯卡。
- **原因**：CUDA 12.8 之後的 PyTorch 已移除 Pascal；TensorRT 10 也已移除 Pascal 支援。
- **解法**：conda 環境固定 `torch 2.5.1 + cu121`、`torchvision 0.20.1`、`numpy<2`、`transformers 4.46.3`、`timm 1.0.11`；TensorRT 後端在 WSL 不裝，只用 PyTorch 後端開發。
- **證據**：`torch.cuda.get_arch_list()` 只列到 `sm_60` 沒有 `sm_61`，但 GPU 矩陣運算正常（同世代相容）。這是正常現象，不用處理。

### 2.2 NanoSAM 的 `setup.py` 裝不起來
- **現象**：`pip install -e third_party/nanosam` 顯示成功，但 `import nanosam` 失敗。
- **原因**：`third_party/nanosam/nanosam/` 缺少 `__init__.py`，`find_packages()` 什麼都找不到，等於沒裝任何東西。
- **解法**：改用 `.pth` 檔把 `third_party/nanosam` 加進 `sys.path`（`installation/setup_env.sh` 會做）。Jetson 腳本同樣處理。

### 2.3 NanoSAM 的 ONNX 在 Google Drive、需登入，gdown 失敗
- **現象**：`gdown` 回報 `Cannot retrieve the public link`；用 `curl` 直接下載拿到的是 Google 登入頁（約 927KB 的 HTML）。
- **原因**：官方 README 的 Drive 連結現在需要登入，不是限流。
- **解法**：
  - **mask decoder**：從 HuggingFace 鏡像 `dragonSwing/nanosam` 下載 `mobile_sam_mask_decoder.onnx`，並用 `onnx` 檢查輸入輸出名稱（`image_embeddings, point_coords, point_labels, mask_input, has_mask_input` → `iou_predictions, low_res_masks`）與 NanoSAM 程式預期一致；也用官方匯出工具從 `mobile_sam.pt` 自行匯出一份，兩者介面相同。
  - **ResNet18 image encoder**：官方只放在 Drive，沒有找到公開鏡像（HF 上找到的是 `.ort` 格式或不同的 encoder）。→ 兩條路：手動下載（`installation/download_model.sh` 會印提示），或**自行蒸餾**（§7）。
- **補充**：把 MobileSAM 原版 ViT-T encoder 匯出當替代品是可行的備案，但它不是題目要的 NanoSAM，速度與官方數據也不能直接對照。

### 2.4 OpenCV（pip 版）沒有 H.264 / VP8 編碼器
- **現象**：`cv2.VideoWriter` 用 `avc1` / `H264` / `VP80` 都開不起來，產不出瀏覽器能播的測試影片。
- **解法**：只為了產生測試影片，在 conda 環境額外裝 `imageio-ffmpeg`（內含 ffmpeg 執行檔），**不列入專案依賴**。影片輸入請用 H.264 mp4 或 webm，否則瀏覽器無法解碼。
- **補充**：`cv2.VideoWriter` 的 `mp4v`（MPEG-4 Part 2）檔案瀏覽器通常也播不出來，所以 `demo.mp4` 只能給伺服器端來源（OpenCV 讀）使用。

### 2.5 Jetson（JetPack 6.2）的預期相容性問題（**依公開資料整理，未實測**）
| 問題 | 對策（寫在 `installation/jetson/01_install_env.sh`） |
|---|---|
| PyPI 的 torch 是 CPU / x86 版 | 使用 Jetson AI Lab 的 `jp6/cu126` index（torch 2.8.0 + torchvision 0.23.0 是公開資料中回報可用的組合；index 網址曾由 `.dev` 改為 `.io`，可能再變） |
| `torch.cuda.is_available()` 為 False / `no kernel image` | 裝 `libcusolver-12-6` 等 CUDA 函式庫 |
| numpy 2.x 讓 wheel 無法 import | 鎖 `numpy<2` |
| venv 內 `import tensorrt` 失敗 | `python3 -m venv --system-site-packages`；**不要 `pip install tensorrt`**（會蓋掉 JetPack 內建的 10.3） |
| torch2trt 舊版不支援 TensorRT 10 | 從 GitHub master 安裝 |
| 容器內 `libnvdla_compiler.so` ImportError | 直接裝在宿主機，不用容器 |
| 建引擎 / 編譯 OOM（8GB 共用記憶體） | `04_setup_swap.sh` 建 8GB swap |
| 引擎不可跨 GPU 架構 | 引擎一律在 Jetson 上建 |
- **未驗證的風險**：`nanosam/utils/predictor.py` 的 `TRTModule` 在 TensorRT 10 下是否正常，要實測；若失敗，就是報告的真實「版本相容性問題」素材。

---

## 3. 模型與串接

### 3.1 官方範例在 WSL 跑不起來
- **現象**：`nanoowl/examples/owl_predict.py` 失敗；`nanosam/examples/segment_from_owl.py` 用的是 HuggingFace 版 OWL-ViT，不是 NanoOWL。
- **原因**：`owl_predict.py` 的 `--image_encoder_engine` 預設值是一個 TensorRT 引擎路徑，沒辦法從命令列關掉；WSL 沒有 TensorRT。它的畫圖函式 `draw_owl_output` 在這版 OpenCV 下又因為「唯讀 NumPy 陣列」丟出 `cv2.error`。
- **解法**：自己寫 `src/demo_owl.py`、`src/demo_sam.py`、`src/pipeline.py`；`OwlPredictor(image_encoder_engine=None)` 就是 PyTorch 後端，PIL 自己畫框。
- **結果**：後端可抽換（`owl_engine` 參數），Jetson 上只要傳入引擎路徑。

### 3.2 SAM 的角色：為什麼不能只用 OWL
- OWL 只給矩形框與分數；SAM 給像素級遮罩，才有輪廓與準確面積。
- **證據（貓圖，同一張）**：左貓框面積 130,774 px，遮罩 49,958 px，框高估約 2.6 倍；遙控器框 6,160 px vs 遮罩 4,121 px，高估約 1.5 倍。
- 題目也明確要求「取得分割輪廓」，所以功能要求 1、2、4 都包含 NanoSAM。
- **釐清**：SAM 自己不知道物件是什麼，只依提示（框 / 點）切形狀；用單點當提示不穩（在右貓身上點一點只切出 4,386 px，信心 0.51），所以串接用 OWL 的框。

### 3.3 WSL 暫用 MobileSAM，最終要用 NanoSAM
- WSL 上用 `mobile_sam.pt`（39MB，完整 SAM：TinyViT encoder + prompt encoder + mask decoder）。
- NanoSAM = MobileSAM 把最重的 image encoder 換成蒸餾出來的 ResNet18，**mask decoder 與 MobileSAM 相同**。所以 WSL 的遮罩品質與 Jetson 最終不完全一樣，**準確度與效能數據一律以 Jetson 為準**。

### 3.4 輪廓
- 流程：框 → SAM 遮罩 → `cv2.findContours`（`Detection.contours()`，可用 `epsilon_ratio` 簡化成多邊形）。

---

## 4. 效能量測

### 4.1 計時方法的錯誤（GPU 非同步）
- **現象**：第一版 `demo_sam.py` 顯示 `encoder 22ms / decoder 88ms`，與直覺不符。
- **原因**：GPU 呼叫是非同步的，encoder 還沒跑完就停表，時間被算進下一階段（decoder）。
- **解法**：每個階段前後都 `torch.cuda.synchronize()`。**修正後 `encoder 101ms / decoder 9ms`**。
- **結論**：SAM 的時間幾乎都在 image encoder（每張圖一次），每多一個框只多約 8ms；這就是 NanoSAM 要換掉 encoder 的原因。

### 4.2 分階段實測（GTX 1060 6GB、PyTorch fp32、640×480、4 個框；`benchmark/bench_pipeline.py`）
| 階段 | 平均 | 官方對照（TensorRT） |
|---|---|---|
| OWL 影像編碼器 | 72.7 ms | AGX Orin 約 10.5 ms（95 FPS）；**Orin Nano 官方為 TBD** |
| OWL 合計 | 74.5 ms（13.4 FPS） | — |
| SAM 影像編碼器 | 88.5 ms | MobileSAM AGX Orin 35 ms；Orin Nano 完整流程 146 ms |
| SAM decoder / 框 | 7.9 ms | — |
| 完整流程 | 197.2 ms（5.1 FPS） | NanoSAM 完整流程 Orin Nano 27 ms、AGX Orin 8.1 ms |
- 一致性：197.2 ≈ 74.5 + 88.5 + 4×7.9（=194.6），差 2.6 ms 是 NMS 與資料搬移；標準差 < 3%，GPU 60°C、時脈 1936MHz，無降頻。
- 記憶體：PyTorch 峰值 901 MB；`nvidia-smi` 增加約 1006 MB，兩種量法相符。
- OWL 的合理性估算：ViT-B/32 於 768×768 約 55~70 GFLOPs；GTX 1060 fp32 理論 4.4 TFLOPS → 理論下限約 16 ms，實測 73 ms ≈ 22% 利用率，符合 Pascal + PyTorch eager 的常見水準。
- **限制**：硬體 / 精度 / 引擎都和官方不同，只能說「數量級合理、方法無誤」，**不能宣稱與官方一致**。

### 4.3 試過但沒用的加速（記錄「為什麼不採用」）
| 做法 | 結果 |
|---|---|
| `torch.backends.cudnn.benchmark=True` | OWL 76.6 / SAM encoder 89.0 ms，**沒有改善** |
| fp16 autocast | OWL 67.8（快 10%），但 **SAM encoder 125.7 ms（變慢）**——GTX 1060 的 fp16 速率只有 fp32 的 1/64。不採用 |
| batch 加大（蒸餾訓練） | 速度不變（GPU 算力是瓶頸），見 §7 |

### 4.4 真正有效的：只要計數就跳過 SAM
- 只跑 OWL：76 ms vs 完整 195 ms（2.6 倍）。
- 網頁加「計算遮罩」開關，取消時只回傳框。**影片分析（24 個取樣點）：有遮罩 239 ms/點（4.2 FPS）→ 只留框 121 ms/點（8.3 FPS）**，後端單次 201 → 75 ms。
- 另外影片迴圈**預取下一幀**（推論第 N 幀時先 seek / 擷取 / 編碼第 N+1 幀），前端額外開銷從約 75 ms 降到約 50 ms。

### 4.5 冷啟動與閒置降頻
- 啟動後第一個請求 OWL 要 263 ms（暖機後約 77 ms）→ 伺服器啟動時先暖機 3 次。
- 閒置一陣子後的第一個請求仍偏慢（OWL 約 305 ms），**推測**為 GPU 閒置降頻（未進一步證實）。量測功能因此捨棄第 1 次。Jetson 上請用 `05_set_power_mode.sh` 鎖定時脈並記錄電源模式。

---

## 5. 偵測品質：提示詞、重複偵測、座標

### 5.1 提示詞寫法實測（`cats.jpg`，2 隻貓、2 個遙控器；單獨查詢，門檻 0.05 看分數）
| 提示詞 | 最高分 | 備註 |
|---|---|---|
| `cat` / `a cat` / `the cat` / `a photo of a cat` | 0.39 / 0.65 / 0.64 / 0.64 | 有無冠詞會影響分數，但**方向不固定** |
| `cats`（複數） | 0.26 | 變差，且 0.1 門檻下多出 2 個誤報（count=4） |
| `kitten` | 0.31 | |
| `remote` / `remote control` / `a remote control` / `a remote` | **0.57 / 0.60** / 0.46 / 0.42 | 這組**不加冠詞反而較高** |
| `貓`、`猫`、`遙控器` | **完全偵測不到**（0 個） | OWL 的文字編碼器是英文 CLIP |
- **結論**：不是「沒加 `a` 就抓不到」，而是**提示詞必須是英文**，其餘寫法只影響分數（有時升有時降）。這是「提示詞設計原則」的第一條證據。
- OWL 文字上限 **16 token**（含首尾特殊 token，約 12 個字）；實測 27 token 的描述會超限。

### 5.2 「統計表是空的」其實是零偵測
- **現象**：使用者回報「各提示詞統計數據都是空的、看不到 SAM 的數據」。
- **原因**：不是配線錯誤——**OWL 沒找到任何框時 SAM 根本不會執行**（沒有框就沒有遮罩），面積與 SAM 品質自然都是空的；最常見的原因是用了中文提示詞或門檻太高。介面卻完全沒說明。
- **解法**：統計表新增「最高分」（含低於門檻者，用 0.01 的低門檻取得所有候選再依使用者門檻過濾）、「SAM 品質」（SAM 對自己遮罩預測的 IoU）；零偵測時直接顯示原因與建議；非英文提示詞（且未啟用 HyDE）顯示警告。
- 例：`a zebra` 在貓圖上最高分 < 0.01，統計表顯示 `<0.01`。

### 5.3 重複偵測 / 計數偏高
- **現象**：貓圖影片中 `a cat` 偶爾數出 3 隻（真實 2 隻）。
- **實測（31 個影格，真實 2 隻貓，`a cat`）**：

| 門檻 | 不用 NMS 多報 | IoU-NMS 0.5 多報 |
|---|---|---|
| 0.05 | 100% | 97% |
| **0.10（預設）** | 26% | **23%（7/31 幀數出 3 隻）** |
| 0.20 | 3% | 0% |
- **原因**：多出的第 3 個框分數只有 0.11~0.13，約佔畫面 8~9%，**其 91~94% 的面積落在某隻貓的框內，但與它的 IoU 只有 0.18**——是貓身上的局部被當成另一隻。IoU 式 NMS 看的是重疊比例，抓不到「小框整個在大框裡」。
- **模擬的解法（尚未寫進 pipeline）**：「包含抑制」——丟棄大部分面積落在更高分框內的低分框。門檻 0.1 時多報 23% → **0%**，漏報 0%（門檻 0.05 時 97% → 29%）。
- **更正**：我原本在文件寫「大框包小框，NMS 去不掉」是**沒驗證的推測**，上面的數據後來證實了這個機制，但**一開始沒有證據**。
- **限制**：單一場景、31 幀幾乎相同的影格、兩隻貓不互相巢狀；工業零件的結論不能由此推出；真正巢狀的同類物件（例如大盤子裡的小盤子）可能被誤殺。

### 5.4 疊圖往左上偏移（只有「圖片」模式）— 2026-10-29 最後一個問題
- **現象**：使用者回報「輪廓形狀是對的，但整組框 / 輪廓往左上偏；只有『圖片』會，影片不會；剛剛不會」。
- **診斷過程（含走過的彎路）**：
  1. 先懷疑 OWL 的框有偏差：把 OWL 框與 SAM 遮罩的緊貼外框比較（12 筆），框的左緣平均多出 2.21%、上緣 0.72%、右緣 −0.15%、下緣 +0.52%——確實有「往左上延伸」的偏差，**但這在影片模式也存在**，而使用者說影片沒問題 → 這不是主因。（SAM 用框當提示，框偏一點輪廓仍會貼著物體，所以「輪廓正確」不代表「框正確」。）
  2. 使用者指出「整個輪廓是對的，只是整體往左上」→ 是疊圖座標與背景圖的對應問題，不是模型預測。
  3. 查 `app.log`：使用者上傳的兩張 Pexels 照片，後端解碼尺寸為 **2304×4096、2725×4096**——長邊剛好 4096。
- **原因**：後端對長邊 > 4096 的圖先縮小再偵測（`MAX_IMAGE_SIDE=4096`），回傳的框 / 輪廓 / 面積都在**縮小後的座標系**；前端卻用**原圖大小**畫圖，沒換算，整組疊圖就縮在原圖左上角。影片 / 攝影機的影格是前端先縮好再送，兩邊一致，所以不會發生；之前測的都是 640×480，沒超過上限。
- **重現**：6144×4608 的圖 → 後端回 `image=4096×3072`，框 `[7,354,2011,3023]`；正確的原圖座標應為 `[58,509,3072,4502]`。
- **解法**：前端 `toDisplayCoords()` 在收到結果時把框、輪廓、面積換算成「畫面上那張圖」的座標系（面積乘 sx·sy），下游（畫圖、面積、比例尺、報表）一次一致；後端回應補 `image.original_width/height` 並在日誌記 `image downscaled` 警告；畫面「影像大小」顯示「後端縮至 …」。若兩軸縮放比例差 > 2%（例如 EXIF 方向不一致）會警告。
- **驗證**：同一張貓圖放大成 6144×4608：輪廓位置與「小圖 × 9.6」差 **0.17%**，面積差 0.7%，設定比例尺後大圖 25,031 mm² vs 小圖 24,922 mm²。
- **未解決的次要現象**：OWL 在不同解析度預測的框本來就有差（右貓下緣差 4.5%，輪廓只差 0.2%）；以及上面 1. 的「左上延伸」偏差（原因未查明，一個**未驗證的假設**是 `pad_square=False` 把圖壓扁成正方形，與 OWL 訓練時的前處理不同，尚未測 `pad_square=True`）。

---

## 6. HyDE 提示擴展（Ollama）

### 6.1 動機與設計
- 動機：英文 CLIP 不懂中文、型號、行話（§5.1）。
- 設計（依使用者要求）：每個提示詞（概念）→ LLM 產生**剛好 5 個**類似詞句（JSON schema 強制格式）+ 原提示（若為英文）= 一組（最多 6 個）→ 整組當 OWL 查詢 → 同一物件取組內信心最高者 → 歸回概念；**組內最高信心者標為「主要輸出」★**（預設保留全部實例以便計數，另有「每組只輸出信心最高的一個」開關）。
- 描述限 8 字、用 OWL 的 tokenizer 驗證 ≤ 16 token；結果依（提示詞, 數量, 模型）快取。
- 多個查詢幾乎不增加 OWL 耗時（10 個查詢 81 ms，與 2 個查詢相同）。

### 6.2 Ollama 安裝與 GPU
- **沒有 Ollama、`sudo` 需要密碼**（非互動環境不能用）→ 改用**使用者層級安裝**到 `~/.local/ollama`（免 root）。
- **官方連結的坑**：`ollama-linux-amd64.tgz` 回 **404**（新版 release 只提供 `.tar.zst`，我一開始只看到重新導向就下載，得到 9 個位元組的錯誤頁）→ 改下載 `.tar.zst`（1.5GB，用 `zstd` 解壓）。
- **GPU**：這版 Ollama 要求 NVIDIA 驅動 ≥ 570，本機是 560，日誌：`NVIDIA driver too old`，自動改用 CPU。GTX 1060 在「Windows 桌面 + OWL + SAM」之後只剩 1.5~2.5 GB 顯存，LLM 放 CPU 反而不會搶顯存。（i5-12400、12 執行緒、可用 RAM 12GB。）

### 6.3 模型選擇（CPU，同一組輸入實測，`n=3` 時的每提示詞耗時）
| 模型 | 大小 | 耗時 | 品質 |
|---|---|---|---|
| `qwen2.5:1.5b` | 0.99 GB | ~1.0 s | 格式正確但有雜訊（`a military-grade hex nut`、`a domestic animal`、`a machine part`） |
| `qwen2.5:3b` | 1.93 GB | ~2.0 s | **反而不穩**：遙控器給出 `wireless`、`plastic` 這種單一形容詞 |
| **`qwen3:4b`（預設）** | 2.50 GB | ~2.7 s（n=3）、**~3.4 s（n=5）**、首次載入 ~11 s | 最好：名詞詞組且彼此有差異 |
| `deepseek-r1:1.5b` | 1.12 GB | — | **6 次全失敗**（輸出為空）：推理模型，用結構化 JSON 輸出時內容為空，`think:false` 也無效 |
- DeepSeek 在 1.5B~4B 只有 R1 蒸餾（推理）版，會先思考再回答，不適合這種短小的改寫任務。

### 6.4 規則：擴展必須在「開始偵測之前」完成（使用者的明確要求）
- **第一版的錯誤**：我把 LLM 呼叫放在每個偵測請求裡（有快取就快），第一幀以及每次改提示詞都會讓偵測卡 7~12 秒，影片 / 攝影機迴圈尤其嚴重。
- **修正**：
  - 唯一會呼叫 LLM 的端點是 `POST /api/hyde/expand`；偵測端點**只查快取**（`hyde.lookup_cached`），沒擴展過就用原始提示並說明原因，不會變慢。
  - 前端按「執行 / 開始分析 / 開啟攝影機 / 開始串流」時先擴展完才送第一幀；偵測用的是**擴展當下凍結的提示詞**。
  - 偵測中改文字（HyDE 開著）不會立刻生效，顯示「待套用」；按「套用」→ 停止送新幀 → **等進行中的那一幀做完** → 擴展 → 繼續（伺服器串流則重啟串流）。HyDE 關閉時改字仍即時生效。
- **驗證**（真實瀏覽器、**全新提示詞**）：開始前擴展約 3.6 s 後才送第一幀；偵測中每幀最慢 237~368 ms；套用時擴展 3.7~3.9 s，期間重疊的偵測請求 0。
- **過程中抓到的兩個問題**：
  1. 第一次驗證套用時有 **1 個偵測請求和擴展重疊**——我只擋了新幀，沒等已送出的那一幀 → 加 `state.inflight` 計數，歸零才擴展。
  2. 修正後重跑，擴展只花 3 ms（伺服器快取已有）→「無重疊」檢查**是空轉的**，沒驗證到修正 → 改用全新的提示詞（真的呼叫 LLM）重跑才算數。

### 6.5 Ollama 自動啟動
- **現象**：使用者重啟伺服器後徽章顯示「Ollama 未連線」。
- **原因**：沒有 Ollama 程序在跑（我收尾時關掉了），而它需要另外手動啟動，很容易忘。
- **解法**：伺服器啟動時偵測到已安裝但沒在跑就自動拉起（實測 3.4 s 就緒），結束時只關掉自己啟動的那個；`HYDE_AUTOSTART=0` 可關閉。

### 6.6 目前的結論（誠實版）
- 中文提示詞：無 HyDE → **0 個偵測**；有 HyDE（含翻譯）→ 2 隻貓、2 個遙控器全找到。這是目前**唯一確定的價值**。
- **常見物件上沒有增益**：貓圖兩組最後贏的都是第一個詞句（`a cat`、`a remote control`）。HyDE 是否能改善型號、行話、少見零件，**沒有證據**，要用真實素材跑 [experiments.md](experiments.md) 的比較。
- 風險：籠統詞句（`a component`）可能搶走正確物件；擴展結果存在伺服器記憶體，重啟後需重新擴展；Jetson 上的 Ollama 未實測。

---

## 7. NanoSAM encoder 蒸餾訓練

- **動機**：官方 ResNet18 encoder 拿不到（§2.3）。
- **官方 `train.py` 不能直接用**：教師模型是 TensorRT 引擎，WSL 沒有 TensorRT → 自寫 `training/train_nanosam.py`（教師為 PyTorch fp16 的 MobileSAM encoder、學生 ResNet18 + ImageNet 預訓練、Huber loss、Adam 3e-4、AMP、梯度累積），checkpoint 格式與官方相容，可用官方 `export_image_encoder_onnx` 匯出。
- **驗證集**：依檔名排序固定留出最後 200 張，每個 epoch 算驗證 loss，並另存 `best.pth`。
- **實測速度（COCO 真實圖片）**：batch 4 與 batch 8 **同速**（96 張 24 s ≈ 4 張/秒），顯存 1.5 vs 2.6 GB → 用 batch 4 + 累積 4（等效 16）。4800 張訓練圖 ≈ **20 分鐘 / epoch**（10 epoch ≈ 3.5 h）。小規模試跑（96 張、2 epoch）驗證 loss 0.0249 → 0.0103，只代表機制正常。
- **訓練資料**：官方是 COCO train2017（11.8 萬張）× 200 epoch；我們只有 val2017（5000 張，778 MB），品質預期會低於官方。**準確度不能直接與官方 mIoU 比**（延遲 / FPS / 記憶體只取決於網路架構，仍可對照）。
- **下載 COCO val2017 很慢**：約 1 小時 45 分，多次逾時重試（`wget` 自動續傳）。
- **匯出驗證**：用 24 張圖、1 epoch 的假 checkpoint 匯出 ONNX，輸入 `image 1×3×1024×1024`、輸出 `image_embeddings`，名稱正確；測試時匯出到暫存位置，避免在 `data/models/` 留下看起來像正式的檔案。
- **狀態**：腳本 `training/run_distill.sh`、`export_encoder.sh` 完成；**完整訓練尚未執行 / 尚無結果**。

### 7.1 保護 baseline 與可比較性（訓練前）
- **要求**：訓練不可蓋過 baseline，且要能比較。baseline = `data/models/mobile_sam.pt`（sha256 開頭 `6dbb90523a35330f`）。
- **檢查出的風險**：① 匯出 ONNX 的預設檔名 `resnet18_image_encoder.onnx` 與**官方檔同名**，之後手動下載官方檔會互相覆蓋；② Jetson 的 `03_build_engines.sh` 也寫死這個名字，引擎檔會撞名；③ 訓練腳本沒有任何防呆，輸出目錄若設成 `data/models` 會和 baseline 混在一起。
- **解法**：自訓練 ONNX 固定為 `resnet18_image_encoder.selftrain.onnx`（寫官方檔名需 `FORCE=1`），Jetson 腳本以 `ENC_ONNX` 選擇、引擎檔名跟著 ONNX 走；訓練腳本拒絕輸出到 `data/models`、每次啟動寫 `run_info.json`、結束時校驗 baseline 的 sha256；每次實驗用 `RUN` 名稱分開存放。四項防護都實測過（拒絕寫入、`未被改動 ✓`、拒絕官方檔名）。
- **比較工具**：`training/compare_encoders.py`（用 COCO GT 框當提示算 mIoU，與官方相同方法）；pipeline / 網頁可載入自訓練 encoder 並即時切換（`SAM_STUDENT`）。實測切換再切回，遮罩面積與第一次完全相同。
- **baseline 基準（先量好、之後訓練結果才有東西可比）**：200 張留出影像、1349 個物件，mIoU All **0.744** / Small 0.685 / Medium 0.756 / Large 0.822，encoder 87.4 ms。官方報的 MobileSAM 為 0.728 / 0.658 / 0.759 / 0.804——差距在 0.02 以內，表示**評估方法複現正確**（但影像子集、後端不同，只是方法驗證，不是數字相等）。
- **已知的訓練 / 推論不一致（沿用官方做法，未改）**：訓練時影像被壓成 1024×1024 正方形（RandomResizedCrop / Resize），推論時 SAM 保持長寬比、縮到長邊 1024 再補零，所以學生在非正方形影像上看到的輸入分佈與訓練不同；是否影響品質未驗證。
- **未驗證**：自訓練 encoder 的準確度（尚未訓練）；速度差距目前只有「假 checkpoint」的結構性數字（SAM 階段約 54 vs 107 ms）。

---

## 8. 後端、前端與日誌

### 8.1 日誌（`app.log`）
- 格式：`時間 | 等級 | request_id | [函式]: input=..., output=... , elapsed=...`；`pipeline.py` 以明確的 `logger.info` 記各階段（OWL 候選、NMS、每個框的 SAM），服務 / API 函式用 `@log_call`；大物件（影像、陣列）只記形狀。
- 連續影格（攝影機 / 影片）會讓 `app.log` 在數分鐘內被洗掉 → **取樣記錄**（前 3 幀與每 10 / 30 幀完整記錄，其餘只記 WARNING 以上）。

### 8.2 前端 bug 與修正
| 現象 | 原因 | 解法 |
|---|---|---|
| 「推論中…」遮罩永遠蓋在結果圖上、滑鼠事件被擋（tooltip 不出現） | CSS 的 `display:flex` 蓋過 HTML 的 `hidden` 屬性 | 加 `[hidden]{display:none!important}` |
| 數量檢查面板第一次渲染就丟例外、被外層 try/catch 吞掉（只看到一行錯誤） | `Object.assign(tr, {dataset:{…}})`——`dataset` 是唯讀屬性，嚴格模式下丟 TypeError | 建好元素後才 `row.dataset.label = p` |
| 伺服器串流模式 HyDE 資訊消失 | 攝影機迴圈每幀重建 `state` 字典，啟動時塞進去的 `hyde` 在第一幀後就被覆蓋 | worker 持有 `hyde_info`，每幀帶入 |
| HyDE 已翻譯成英文、也偵測成功，畫面還顯示「非英文提示詞偵測不到」 | 警告沒考慮 HyDE | HyDE 生效時不顯示這則警告 |
| 零偵測時沒說明原因 | — | 見 §5.2 |

### 8.3 來源與環境限制
- WSL **看不到 USB 攝影機** → 提供兩種來源：瀏覽器攝影機（用 Windows 瀏覽器的 webcam）與**伺服器端來源**（攝影機編號 / 影片路徑 / `gst:` 管線 + MJPEG 串流，Jetson 接攝影機用）。影片與攝影機在程式上是同一套「取一幀 → 推論 → 疊圖」迴圈。
- 瀏覽器只允許 `localhost` / HTTPS 使用攝影機。
- 工單、比例尺存在瀏覽器 localStorage（私密視窗可能被封鎖，已 try/catch）。

---

## 9. 製造現場情境

- **設計**：零件檢查（工單 → 預期數量 / 單件面積範圍 → OK·NG → 紀錄 → 報表）。換工單只改提示詞與預期值，不需重新訓練、不需重啟。細節見 [scenario.md](scenario.md)。
- **面積**：SAM 遮罩像素 × 比例尺²（mm/像素；可手動輸入或在影像上點兩個已知距離的點校準）。未校準但設了面積範圍 → 顯示「**無法判定**」，不會假裝 PASS。**假設鏡頭垂直拍攝平面**。
- **報表**：總覽、各零件統計（預期數量、偵測數量平均 / 最小 / 最大、數量符合率、單件面積平均 / 最小 / 最大）、逐筆明細與失敗案例截圖，可列印。
  - **抓到的統計錯誤**：數量符合率最初用「第一筆紀錄的預期值」算所有紀錄；預期值中途改過時，明明有 3 筆 FAIL 卻顯示 100%。→ 改為每筆紀錄各用當時的預期值。
- **找素材**：沒有找到「公開、有授權、產線影片、零件計數」剛好符合的資料集；最實際是**自己錄模擬產線**（白紙當輸送帶、LEGO 或螺絲螺帽）。可參考的來源（授權請自行確認）：MVTec Screws（384 張，CC BY-NC-SA）、Roboflow Universe 的 nuts/bolts 資料集、Mendeley 的輸送帶異物資料集、LEGO 輸送帶影片、免費影片素材站（Pexels / Pixabay / Coverr / Mixkit；注意畫面中的商標與人臉權利）。
- **狀態**：功能與介面完成；**沒有用真實零件驗證過**，面積準確度未實測（建議用 A4 紙校準、信用卡 85.6×54 mm 驗證）。

---

## 10. 開發過程的錯誤與教訓

| 事件 | 原因 | 影響 | 修正 / 教訓 |
|---|---|---|---|
| 在 `service.py` 把 `result = {...` 那行換成帶**行尾註解**的寫法，原本同一行後面的 `"prompts": …, "threshold": …,` 被註解吃掉 | 字串取代時沒注意同一行後面還有內容；Python 語法照樣合法 | 回傳少了 `prompts`/`threshold`，前端結果畫面全壞；**靠瀏覽器測試的 `indexOf` 例外才發現** | 新增 `tests/check_api_contract.py`（實際打 API 檢查欄位）。**語法檢查不夠，改 server 後必須實際打 API** |
| 舊的瀏覽器測試腳本用「第 N 個輸入框」對應「第 N 列預期數量」，介面改成每列 3 個輸入框後，值填錯格 | 測試腳本沒跟上介面改版；上一輪能過是因為前一個測試留下的比例尺在 localStorage（湊巧過） | 乾淨瀏覽器下「PASS」變「無法判定」 | 改為以列＋欄定位；**測試要在乾淨的瀏覽器工作階段跑**，不可依賴前一輪殘留的狀態 |
| 「套用期間無偵測重疊」檢查因快取讓擴展只花 3 ms | 用了已快取過的提示詞 | 檢查空轉，沒驗證到修正 | 驗證用**全新輸入**，並確認被測的那段時間真的有發生（> 0） |
| `rm -f app.log` 時使用者的伺服器正在寫這個檔案 | 沒意識到那是使用者在用的日誌 | 使用者那次的日誌被清掉 | 測試一律用 `PORT=8001 APP_LOG=/tmp/…`，**不刪使用者的檔案** |
| `pkill -f "server.main"` 把執行這條指令的 shell 自己也殺掉（兩次） | 指令字串本身含有 `server.main` | 指令中斷 | 依「監聽埠」找程序，不用指令字串比對 |
| 下載 Ollama `.tgz` 得到 9 位元組的錯誤頁 | 先前查到重新導向就下載，沒看最終狀態碼（404） | 浪費一輪 | 下載前先 `curl -I` 看最終的 HTTP 狀態與大小 |
| 重建 commit 時根 commit 的雜湊也變了 | 該 commit 是 GitHub 網頁建立、帶 PGP 簽章，重建時簽章被丟掉 | 無謂改寫更多歷史 | 訊息沒變且父節點沒變的 commit **沿用原物件** |
| 文件中寫「大框包小框」「重複偵測」當事實 | 沒有證據的推測 | 之後被數據證實但當時不該寫成事實 | 文件只寫有證據的；推測要標明 |
| 修好 bug 後，新一輪測試的「console errors」裡仍列出同一個例外 | **推測**是 Edge 偵錯介面在連線時重播了上一輪頁面累積的舊紀錄（未直接驗證；依據是：用診斷腳本重現後例外次數為 0，換新的瀏覽器工作階段後也不再出現） | 差點誤判 bug 還在 | 判斷「例外是否還在」要用診斷腳本直接重現；回歸測試用乾淨的瀏覽器工作階段 |

### Git：移除 commit 裡的 Claude co-author 標記
- `f3f124a` 的訊息有 `Co-Authored-By: Claude …`。用 `git commit-tree` 重建該 commit 與其後代（檔案內容 tree 完全不變，只改訊息；作者 / 提交者 / 時間保留），保留備份分支；之後需要 `git push --force-with-lease`（會改寫遠端歷史，由使用者自己執行）。
- force push 後 GitHub API 的貢獻者只剩 repo 擁有者；網頁上的貢獻者圖表是延遲重算的快取，舊 commit 仍可用雜湊存取，**無法由我們控制**，必要時要請 GitHub Support 清除孤兒 commit。

---

## 11. 待辦與未驗證項目

**需要 Jetson / 真實硬體**
- [ ] JetPack 6.2 上建環境、建三個 TensorRT 引擎，**實際記錄遇到的相容性問題**（目前的表是推測）
- [ ] Orin Nano 上量 NanoOWL / NanoSAM / 整體的延遲、FPS、記憶體，記錄電源模式，與官方對照
- [ ] 實體攝影機（USB / CSI）測試；伺服器端來源的 `gst:` 管線
- [ ] Jetson 上的 Ollama（HyDE）

**需要真實製造素材**
- [ ] 錄 / 找 ≥10 張影像 + 一段影片；用 ≥3 種提示詞寫法比較正確率（批次評估腳本尚未寫）
- [ ] 驗證 HyDE 對型號 / 行話 / 少見零件是否有幫助（目前只有「中文翻譯」確定有用）
- [ ] 面積準確度實測（A4 紙校準 + 已知尺寸物件）
- [ ] 評估「包含抑制」是否寫進 pipeline、是否誤殺巢狀的同類物件
- [ ] ≥5 張截圖（含製造情境與失敗案例）、Demo 影片

**訓練與其他**
- [ ] 執行 NanoSAM encoder 蒸餾（`./training/run_distill.sh --bg`），匯出 ONNX、換進 pipeline 的 NanoSAM 後端，並評估品質（需 COCO 標註才能算與官方同方法的 mIoU）
- [ ] 查明 OWL 框「往左上延伸」的偏差；試 `pad_square=True`
- [ ] 報告：研究背景與原理（OWL-ViT、SAM）、系統情境圖
- [ ] 前端在手機寬度與深色模式只寫了 CSS，未實測
