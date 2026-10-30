# Web 前後端

後端 FastAPI（`server/`）載入 OWL + SAM，前端為純 HTML/JS（`frontend/`，由後端直接提供，免建置）。

```bash
conda activate nanoowl-sam
python -m server.main                 # http://127.0.0.1:8000
# 環境變數：HOST、PORT、SAM_CHECKPOINT、SAM_STUDENT（自訓練的 ResNet18 encoder，和 baseline 並存、可在網頁切換比較）、OWL_ENGINE（指定 TensorRT 引擎則改用 TensorRT）、APP_LOG（日誌路徑）
```

啟動時會載入模型並暖機 3 次（避免第一個請求被冷啟動拖慢）。

## 輸入來源（四種）

| 來源 | 說明 | 備註 |
|---|---|---|
| 圖片 | 上傳 / 拖曳 / 貼上 | 可「量測速度」重複 N 次 |
| 影片檔 | 瀏覽器逐段取樣（取樣間隔、起始秒數可設），每個取樣點送後端推論 | 需 H.264 mp4 或 webm；產生數量時間軸與 CSV |
| 瀏覽器攝影機 | 使用瀏覽器所在電腦的攝影機，每幀送後端 | 僅 localhost / HTTPS 可用；WSL 開發時可用 Windows 瀏覽器的 webcam |
| 伺服器來源 | 後端自己讀攝影機編號（`0`）、影片路徑或 `gst:` 管線，持續推論並以 MJPEG 串流 | Jetson 接 USB / CSI 攝影機時使用；影片路徑相對於啟動目錄 |

四種來源共用同一套「取一幀 → 推論 → 疊圖」流程；影片與攝影機迴圈**每一幀都重新讀取提示詞**，所以執行中修改提示詞下一幀即生效，不需重啟。伺服器來源則透過 `POST /api/camera/prompts` 即時更新。

## 前端功能

- 上傳（點擊／拖曳／貼上）圖片、輸入提示詞列表（一行一個或逗號分隔）、調整 threshold 與 NMS IoU。
- HyDE 開關：用本機 Ollama 的小型 LLM 把提示改寫成數個短的英文外觀描述（同時會把中文翻成英文），詳見 [hyde.md](hyde.md)。Ollama 不可用時請求不會失敗，而是退回原始提示並在畫面說明原因。
- 「計算遮罩」開關：取消後只輸出物件框、跳過 SAM，影片分析約快一倍（只需要計數時使用）。
- 零偵測診斷：統計表顯示每個提示詞的「最高分」（含低於門檻者）與 SAM 品質；沒有任何偵測時直接說明原因（SAM 不會執行），並警告非英文提示詞。
- 結果：遮罩／輪廓／框三種顯示、點圖例可隱藏某提示詞、滑鼠移到物件上看分數與面積。
- 各提示詞的數量、遮罩面積、框面積、遮罩/框比例、平均分數。
- **製造情境：零件檢查**：工單（範本 / 自訂）、預期數量、比例尺校準與 mm² 面積、單件面積範圍、PASS/FAIL 判定、檢查紀錄、CSV 匯出與可列印報表，詳見 [scenario.md](scenario.md)。
- 執行速度：後端總耗時與 FPS、各階段（文字編碼／OWL／SAM／序列化／其他）堆疊條、含網路往返時間。
- 「量測速度」：重複 N 次（捨棄第 1 次），顯示 mean / median / p95 / min / max / std。
- 即時 GPU 數據（每秒）：顯存、使用率、溫度、時脈、功耗，與近期延遲長條圖。
- 後端日誌：即時顯示 `app.log`，預設只看最近一次請求（以 request id 篩選），可驗證每個請求真的走過哪些函式。

## API

| 方法 | 路徑 | 說明 |
|---|---|---|
| POST | `/api/segment` | multipart：`image`、`prompts`（JSON 陣列或逗號/換行分隔）、`threshold`、`nms_iou`、`use_hyde`。回傳偵測（框、分數、輪廓點、面積）、各提示詞統計、各階段耗時、GPU 記憶體、`request_id` |
| POST | `/api/camera/start` | form：`source`、`prompts`、`threshold`、`nms_iou`、`style`、`max_side`；開始伺服器端串流 |
| POST | `/api/camera/prompts` | 執行中更換提示詞與參數；`expected`（JSON，`{提示詞: 預期數量}`）會在串流畫面顯示 OK/NG |
| POST | `/api/camera/stop` | 停止串流 |
| GET | `/api/camera/state` | 串流狀態、各提示詞統計、耗時（前端每 0.4 秒輪詢，不寫 log） |
| GET | `/api/camera/stream` | MJPEG 串流（`<img src>` 直接顯示） |
| POST | `/api/hyde/expand` | form：`prompts`。HyDE 擴展（會呼叫 LLM，數秒）；**前端必須在開始偵測之前呼叫**，偵測端點只查快取 |
| GET / POST | `/api/sam/encoder` | 查詢 / 切換 SAM image encoder。清單 = baseline `mobilesam` + 後端掃到的自訓練 checkpoint（`data/eval/*.pth`、`data/train/*/best.pth`，略過 `_` 開頭與 `INVALID` 的）；**第一次選用才載入**（約 1 秒、數十 MB 顯存），不必重啟伺服器。只接受掃到的路徑，不能指定任意檔案；也可用 `SAM_STUDENT=<checkpoint>` 啟動時預載（名稱 `student`）。伺服器串流中不可切換 |
| GET | `/api/hyde/status` | Ollama 是否可連線、設定的模型是否已下載（前端 HyDE 徽章用） |
| GET | `/api/stats` | 即時 GPU 數據（前端每秒輪詢，不寫 log） |
| GET | `/api/health` | 後端、裝置、模型載入時間、請求數 |
| GET | `/api/history` | 近 50 次請求摘要 |
| GET | `/api/logs?lines=200&request_id=` | `app.log` 末尾；可依 request id 篩選 |

```bash
curl -X POST localhost:8000/api/segment -F image=@data/samples/cats.jpg -F 'prompts=a cat, a remote control' -F threshold=0.1
```

## 日誌（`app.log`）

格式：`時間 | 等級 | request_id | [函式]: input=(...), output=... [, elapsed=..ms]`；失敗為 `error=型別: 訊息` 並附 traceback。
`app.log` 位於專案根目錄（輪替 5MB × 3，已在 `.gitignore`）。同一請求的所有紀錄共用 request id，可由 API 一路追到每個框的 SAM 結果：

```
[API POST /api/segment]: input=(content_type=multipart/form-data, length=173674)
[service.SegmentService.decode_image]: input=(data=bytes(len=173131)), output=PIL.Image(size=(640, 480), mode=RGB)
[OpenVocabSegmenter.set_prompts]: input=(prompts=['a cat', 'a remote control']), output=text_encodings(2, 512)
[OwlPredictor.predict]: input=(image=(640, 480), threshold=0.1), output=4 raw boxes
[torchvision.nms]: input=(4 boxes, iou=0.5), output=4 boxes kept
[SamPredictor.set_image]: input=(image=(640, 480)), output=image embedding ready, elapsed=162.1ms
[SamPredictor.predict]: input=(label=a cat, score=0.65, box=[6, 53, 320, 469]), output=mask_area=49958px
[API POST /api/segment]: output=(status=200, elapsed=709.9ms)
```

寫法：模型流程（`src/pipeline.py`）以明確的 `logger.info(...)` 記錄各階段；服務與 API 函式用 `@log_call`（`src/applog.py`）統一記錄輸入、輸出／錯誤與耗時，大型物件（影像、陣列、張量）只記形狀。
`/api/stats` 與 `/api/logs` 為前端輪詢用，刻意不逐筆記錄以免洗版。

連續影格（攝影機／影片）的日誌採**取樣**：前 3 幀與每 10 幀（伺服器串流為每 30 幀）完整記錄，其餘只記 WARNING 以上與 API 進出，避免 `app.log` 在數分鐘內被洗掉。因此「只看最近一次請求」在連續模式下可能只看到 API 兩行，屬預期；`POST /api/segment` 的 `quiet` 欄位即為此用途。

## 大圖（長邊 > 4096）的座標系

後端對長邊超過 4096 的圖會先縮小再偵測（OWL 輸入 768、SAM 輸入 1024，更大的解析度對結果沒有幫助，只會拖慢 CPU 端的轉換與遮罩運算）。
**回傳的框、輪廓、面積都在縮小後的座標系**（`image.width × image.height`），`image.original_width/height` 是縮圖前的尺寸，日誌會有 `image downscaled` 警告。
網頁前端收到結果後會自動換算成「畫面上那張圖」的座標系，所以疊圖、面積（原圖像素）、比例尺（mm/原圖像素）都一致；其他 API 用戶端需自行依 `image` 與原圖尺寸換算。
（2026-10-29 曾因為前端沒有換算，大圖的疊圖整組縮在左上角，詳見 [record.md](record.md)。）

## 已知現象與限制

- 閒置一段時間後的第一個請求會明顯較慢（實測 OWL 約 305ms，暖機後約 77ms），推測為 GPU 閒置降頻；「量測速度」因此捨棄第 1 次。Jetson 上量測前請用 `installation/jetson/05_set_power_mode.sh` 鎖定時脈。
- 推論以 lock 序列化，同時多個請求會排隊（`lock_wait_ms`）。
- 後端只有一個模型實例；`threshold` 與 `nms_iou` 為每次請求參數，提示詞變動時才重算文字編碼。
- 前端僅在 Windows Edge（headless，CDP 腳本）測過桌面寬度；瀏覽器攝影機以 Edge 的假裝置（餵入錄好的影格）測過流程，**未用實體 webcam 實測**，伺服器端 USB / CSI 攝影機亦未實測（WSL 看不到 USB 裝置），僅以影片檔路徑測過同一份程式碼。
- 手機寬度與深色模式只做了 CSS，未實測。
