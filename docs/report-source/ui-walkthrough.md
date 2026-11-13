# 網頁介面實操紀錄與截圖（對應 midterm-report 附錄 B.4）

> 2026-11-11，Jetson Orin Nano，MAXN_SUPER 且鎖頻（GPU 1020 MHz，畫面右側「SM 時脈」亦顯示 1020 MHz）。伺服器 `python -m server.main`（OWL＝TensorRT、SAM＝自訓練 ResNet18 TensorRT），瀏覽器是**同一台 Jetson 上的無頭 Chromium**，由 Playwright 腳本操作（`benchmark/ui_walkthrough.py`、`ui_walkthrough2.py`）。截圖在 `data/bench/jetson/ui_screens/`，原始觀察值在同資料夾的 `walkthrough.json`、`walkthrough2.json`。
> **這是真實的網頁介面截圖與畫面上讀到的數字**，但不是攝影機即時 Demo（這台沒有攝影機）。

## 1. 結果總表

| B.4 步驟 | 結果 | 證據 |
|---|---|---|
| 1 上傳圖片、輸入提示詞、框與遮罩疊在畫面 | ✅ | `01_cats_a_cat_remote.png`、`04_workorder_verdict.png`：`a cat ×2`（最高 0.66）、`a remote control ×2`（最高 0.44），遮罩＋輪廓＋分數標籤都畫出 |
| 2 執行中更換提示詞（不重啟） | ✅ | 同一頁面、同一個伺服器連續換：`a cat, a remote control` → `a cat` → `a remote control`（`02`、`03`），伺服器請求數遞增、沒有重啟 |
| 3 套用工單、OK/NG、PASS／FAIL | ✅（面積換算未測） | 工單「貓與遙控器」：`a cat` 預期 2、偵測 2、OK；`a remote control` 預期 2、偵測 2、OK；整體 **PASS**（`04`）。**比例尺校準（兩點校準、mm²）沒有操作**，畫面仍顯示「未校準：面積只能以像素顯示」 |
| 4 記錄、CSV、報表 | ✅ | 「記錄目前結果」✅（`05`）；CSV 匯出 ✅（欄位：`time,source,work_order,verdict,"capacitor count",…`）；**「產生報表」✅（`11_before_report_ng.png`、`12_report.png`）**：先記錄 PASS 一筆，再把貓的預期數量改為 3 → 判 FAIL（`a cat：2/3，NG（缺 1）`）並記錄；報表含基本資訊（工單、期間、偵測設定、比例尺「未校準」）、總覽（2 筆、PASS 1、FAIL 1、合格率 50.0%）、各零件統計（貓預期 2、3，數量符合 1/2＝50%；遙控器 2/2＝100%；單件面積平均／最小／最大 px）、逐筆明細（FAIL 列底色標示並附畫面縮圖）。腳本 `benchmark/ui_report.py`，文字內容 `report_walkthrough.json` |
| 5 點選一個框當範例（以圖找圖） | ✅ | `10_exemplar_image_query.png`：電路板、點選一顆小銀色電容、原位模式、範例門檻 0.95 → 找到 9 個、說明列與「清除範例」按鈕出現、範例本身標 `◎` |
| 6 右側「官方口徑」卡片 | ✅ | 卡片顯示（見 §3） |

## 2. 畫面上讀到的時間（Demo 的「時機」）

640×480 的 `cats.jpg`、兩個提示詞、門檻 0.1，畫面「執行速度」卡與「本次工作階段」表：

| 項目 | 實測 |
|---|---|
| 端對端（含上傳與傳輸） | 第 4 次請求 **171.7 ms（5.8 FPS）**；同圖單一提示詞 145–149 ms |
| 其中：圖片解碼 6.8、文字編碼 29.8、OWL 26.1、SAM 80.8、序列化 3.5、上傳＋傳輸 16.6 ms | 同上請求的分段條 |
| **第一次請求（冷啟動）** | 伺服器處理 **300.9 ms**、工作階段表第一筆總計 **325.0 ms**（OWL 67.2、SAM 152.0）——**錄 Demo 前請先空跑一次暖機** |
| 提示詞切換造成的額外成本 | 文字編碼約 30 ms（換提示詞才算；`a cat` 單一提示詞時 SAM 只處理 2 個框所以更快） |
| 記憶體／功耗（畫面右側即時卡） | 顯存（共用 RAM）4,984 / 7,619 MB；PyTorch 配置 673/692 MB；GPU 閒置功耗 5.96 W；溫度 48 °C |

**注意**：瀏覽器與伺服器在同一台 Jetson，所以這裡的端對端包含瀏覽器自己的繪製成本；與 midterm-report §4 的 loopback 數字（129 ms）不同口徑，不可直接比較。

## 3. 官方口徑卡片（與報告 §4 對照）

小圖 `cats.jpg`（第 1、3 步後）：NanoOWL predict **21.5／20.8／22.2 ms ≈ 46–48 FPS**；SAM image encoder 36.0 ms（含前處理）；mask decoder 11.1–12.2 ms／框；Full Pipeline（1 框）**47.0–47.7 ms**。與報告 §4 的 trtexec／bench_official 數字（19.3 ms、51.7 FPS；Python 口徑 Full 43.2 ms）在同一量級，略高是因為這是伺服器運作中、同時有瀏覽器的狀態。
**第一次請求的冷啟動值**（SAM encoder 102.4 ms、Full 114.6 ms）明顯偏高，不應引用。
**大圖**（電路板 5184×3456，後端縮成 4096×2731）：NanoOWL predict 68.7 ms（14.6 FPS）、Full Pipeline 148.3 ms——**官方口徑數字會隨影像大小增加**（前處理變重），報告引用 19 ms／51 FPS 時要註明是 640×480。

## 4. 以圖找圖的實際操作結果（電路板）

| 步驟 | 畫面上的數字 |
|---|---|
| 文字提示 `capacitor`、門檻 0.1 | **0 個偵測**（最高分 0.03），與報告 §6.2、§5 一致（`09_pcb_text_capacitor.png`） |
| 門檻 0.01 | 35 個偵測，最高分只有 0.03 |
| 點選一顆真電容為範例（原位、範例門檻 0.95） | **9 個**，最高 1.00、平均 0.98；請求端對端 1266 ms（其中 SAM 399.8、圖片解碼 484.8——原圖 3.5 MB 需縮圖；9 個框都要跑遮罩） |
| 命中內容（看 `10_…png`） | 4 顆小銀色電容、2 根深色高電容 ✔；**但也標出 2 個方形電感（CEC R30，0.95、0.97）與左側銅線圈（0.97）**——與報告 §6.4「分數飽和、電感也 ≥0.95」一致；這個畫面就是最直觀的證據 |
| 第一次誤點的例子 | 腳本最初點到分數最高的框（右下 CPU 插槽）當範例，結果只剩 1 個（範例自己）——**範例選錯就找不到同類**，Demo 時要點正確的物件 |

## 5. 發現的問題與限制

1. **只有 `favicon.ico` 的 404**（console 唯一錯誤，無害）；沒有 JS 例外。
2. 比例尺兩點校準、mm² 面積、單件面積範圍**沒有實操**（報表因此以像素 px 顯示；使用者表示比例尺暫不需要）。CSV 以 JS 攔截 Blob 取得，因為 snap 版 Chromium 的下載暫存路徑 Playwright 讀不到——這是測試工具限制，不是網頁問題。
3. 影片模式、瀏覽器攝影機、伺服器攝影機沒有測（無攝影機）。
4. 工單判定只在「貓與遙控器」範本驗證，而這個範本是為範例圖設計的，不代表真實零件。
5. 第一次載入／第一次請求有冷啟動成本（§2）。

## 6. 啟動方式（給 Demo 與重現）

```bash
source .venv-jetson/bin/activate   # 預設 python 沒有 torch
sudo jetson_clocks                 # 需自己輸入密碼；重開機或切換 nvpmodel 後要重做
free -m                            # available 需 ≥ ~4.5GB，否則載入模型會 OOM（VS Code 遠端伺服器常駐會占 2+ GB）
PORT=8123 HOST=127.0.0.1 python -m server.main
# 另一個終端機（需 playwright）：python benchmark/ui_walkthrough.py && python benchmark/ui_walkthrough2.py
```
