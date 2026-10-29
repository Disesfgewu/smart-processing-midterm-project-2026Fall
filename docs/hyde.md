# HyDE 提示擴展（Ollama + 本機小型 LLM）

## 要解決的問題

OWL-ViT 的文字編碼器是英文 CLIP，只懂「普通的視覺概念」。實測（`cats.jpg`，2 隻貓、2 個遙控器）：

| 提示詞 | 結果 |
|---|---|
| `貓`、`遙控器`（中文） | **0 個偵測**（最高分僅 0.02） |
| `cat` / `a cat` | 最高分 0.39 / 0.65（有無冠詞會影響分數，但方向不固定：`remote` 0.57 反而高於 `a remote control` 0.46） |
| `cats`（複數） | 0.26，且多出誤報 |

另外 OWL 文字上限為 16 token（含首尾特殊 token，約 12 個字），描述必須很短。

## 做法

```
使用者提示（概念）  →  LLM 產生 5 個類似詞句  →  原提示（若為英文）+ 5 個 = 一組（至多 6 個）
        →  整組一起作為 OWL 查詢  →  同一物件取組內信心最高者  →  結果歸回原本的概念
```

- LLM 以 Ollama 的結構化輸出（JSON schema，`phrases` 陣列剛好 5 個）回覆，格式固定；中文會順便翻成英文。
- 每個詞句限 8 字內，並用 OWL 自己的 tokenizer 驗證 ≤ 16 token，不合格的丟棄（記在 `dropped`）。
- OWL 對每個位置本來就取所有查詢中的最高分，所以「取組內最高」是現成行為；多個查詢幾乎不增加 OWL 耗時（10 個查詢 81ms，與 2 個查詢相同）。
- **主要輸出**：每組信心最高的偵測標為 ★（畫面、詳細表、HyDE 面板都會顯示贏的是哪個詞句）。預設保留所有實例（計數需要），勾選「每組只輸出信心最高的一個」則每組只留這一個。
- 結果依（提示詞, 數量, 模型）快取；同樣提示詞第二次不再呼叫 LLM。LLM 只在「換提示」時呼叫，不影響逐幀 FPS。
- Ollama 不可用 / 模型缺失 / 逾時 / 輸出不合格時，請求**不會失敗**，改用原始提示並在畫面與日誌說明原因。

## 規則：擴展一律在「開始偵測之前」完成

擴展要呼叫 LLM（每個提示詞約 3~4 秒，首次載入模型更久），放進偵測迴圈會讓影片 / 攝影機卡住。所以：

| 情況 | 行為 |
|---|---|
| 按「執行 / 開始分析 / 開啟攝影機 / 開始串流」 | **先擴展完**（畫面顯示「HyDE 擴展中」），才送第一幀 |
| 偵測請求（`/api/segment`、`/api/camera/*`） | **只查快取，絕不呼叫 LLM**；沒擴展過就用原始提示並說明原因，不會變慢 |
| 偵測中修改提示詞（HyDE 開著） | **不會立刻生效**，顯示「⚠ 已變更，按『套用』才會生效」；偵測仍用舊提示詞 |
| 按「套用提示詞」 | 停止送新幀 → **等進行中的那一幀做完** → 擴展 → 繼續（伺服器串流則重啟串流） |
| HyDE 關閉 | 沒有 LLM，改文字仍即時生效 |

驗證（真實瀏覽器，全新提示詞、會真的呼叫 LLM）：開始前擴展 3.6 秒後才送第一幀；偵測中每幀最慢 368ms；套用時擴展 3735ms，期間重疊的偵測請求為 0。
唯一會呼叫 LLM 的端點是 `POST /api/hyde/expand`。

## 模型選擇（實測，CPU，i5-12400 / 15GB RAM）

| 模型 | 大小 | 每個提示詞 | 品質 |
|---|---|---|---|
| `qwen2.5:1.5b` | 0.99 GB | 約 1.0 s | 格式正確，但有雜訊（`a military-grade hex nut`、`a domestic animal`、`a machine part`） |
| `qwen2.5:3b` | 1.93 GB | 約 2.0 s | **不穩**：遙控器給出 `wireless`、`plastic` 這種單一形容詞，不是物件詞組 |
| **`qwen3:4b`（預設）** | 2.50 GB | 約 3.4 s（n=5）；首次載入約 11 s | 最好：名詞詞組、彼此有差異（`a handheld remote`、`a surface mount capacitor`） |
| `deepseek-r1:1.5b` | 1.12 GB | — | **全部失敗**：推理模型，結構化輸出時內容為空（`think:false` 也無效） |

DeepSeek 在 1.5B~4B 只有 R1 蒸餾（推理）版，會先思考再回答，不適合這種短小的改寫任務。
要換模型：`HYDE_MODEL=qwen2.5:1.5b python -m server.main`（快約 3 倍，品質較差）。

**GPU 說明**：此版 Ollama 需要 NVIDIA 驅動 ≥ 570 才用 GPU；本機驅動 560，Ollama 自動改用 CPU（日誌：`NVIDIA driver too old`）。
GTX 1060 6GB 在 Windows 桌面 + OWL + SAM 之後只剩約 1.5~2.5GB 顯存，LLM 放 CPU 反而不會與模型搶顯存。

## Ollama 自動啟動

server 啟動時若偵測到 Ollama 已安裝但沒在跑，會自動啟動（日誌 `/tmp/ollama.log`），結束時只關掉自己啟動的那一個；你自己啟動的不動。
手動啟動用 `./installation/start_ollama.sh`；不想自動啟動設 `HYDE_AUTOSTART=0`。
網頁的 HyDE 徽章：`Ollama · qwen3:4b`（可用）、`Ollama 啟動中…`、`未安裝 Ollama`。

## 安裝與啟動

```bash
./installation/install_ollama.sh          # 使用者層級安裝（免 sudo）並 pull qwen3:4b；約 1.5GB + 2.5GB
python -m server.main                     # Ollama 沒在跑會自動啟動；網頁上勾「使用 HyDE」，徽章顯示 Ollama · qwen3:4b
```

| 環境變數 | 預設 | 說明 |
|---|---|---|
| `OLLAMA_URL` | `http://127.0.0.1:11434` | |
| `HYDE_MODEL` | `qwen3:4b` | |
| `HYDE_VARIANTS` | `5` | 每個提示詞產生幾個詞句 |
| `HYDE_TIMEOUT` | `90` | 秒 |
| `HYDE_AUTOSTART` | `1` | Ollama 沒在跑時由 server 自動啟動；`0` 關閉 |
| `OLLAMA_LOG` | `/tmp/ollama.log` | 自動啟動的 Ollama 日誌 |

API：`POST /api/hyde/expand`（**唯一**呼叫 LLM 的端點）；`POST /api/segment`、`/api/camera/start`、`/api/camera/prompts` 的 `use_hyde`（只查快取）與 `top_per_group` 欄位；`GET /api/hyde/status`。

## 初步觀察與限制（待用真實工業素材驗證）

- 中文提示詞：無 HyDE → 0 個偵測；有 HyDE → 2 隻貓、2 個遙控器全部找到。這是 HyDE（含翻譯）最確定的價值。
- **常見物件上沒有額外增益**：貓與遙控器兩組最後贏的都是第一個詞句（`a cat`、`a remote control`），其他描述沒有發揮空間。HyDE 是否能改善型號、行話、少見零件，需要用真實素材跑 [experiments.md](experiments.md) 的比較（目前沒有證據）。
- 擴展需等 LLM（約每個提示詞 3~4 秒，首次載入模型約 11 秒）；結果依（提示詞, 數量, 模型）快取在 server 記憶體，重啟 server 後需重新擴展。
- 組內可能混入雜訊詞句（如 `a component`），最高分機制不保證排除；「取最高」可能讓籠統詞句搶走正確物件，需用實驗確認。
- Jetson 上的 Ollama（aarch64 / JetPack 6）尚未實測；Orin Nano 8GB 的 GPU 記憶體是與 OWL/SAM 共用，需另行評估。
