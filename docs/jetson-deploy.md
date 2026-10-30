# Jetson Orin Nano 部署：指令操作手冊

照順序一步一步做。每一步都有「預期結果」與「失敗時」。**任何錯誤都先把完整輸出存下來**（這就是報告要的「版本相容性問題」），回來貼給開發者整理進 `docs/record.md`。
需要準備的資料與要填的量測表見 [jetson-data.md](jetson-data.md)；已知版本衝突清單見 [jetson-setup.md](jetson-setup.md)。

> **這份文件裡的 Jetson 步驟沒有在實機驗證過**（腳本只在 WSL 做過語法檢查、解析邏輯用假的 `trtexec` 輸出測過）。第一次跑出現的差異，就是要記錄的內容。

## 0. 現況：能做什麼、還做不到什麼

| 項目 | 狀態 |
|---|---|
| OWL 的 TensorRT 引擎 → pipeline / 網頁 | ✅ 設 `OWL_ENGINE` 即可 |
| **NanoSAM 的 TensorRT 引擎 → pipeline / 網頁**（`src/sam_backends.py`） | ⚠️ **程式已寫好，但 TensorRT 路徑未在實機跑過**。前處理 / 框轉點 / 遮罩後處理是用 **ONNX Runtime 在 WSL 驗證過**（與 PyTorch 自訓練 encoder 的遮罩 IoU ≥ 0.995，橫式與直式影像都測），TensorRT 後端沿用官方 `load_image_encoder_engine` / `load_mask_decoder_engine`，只換 encode / decode 兩個呼叫。torch2trt 在 TensorRT 10 的相容性是最大風險，見步驟 5b |
| SAM（baseline / 自訓練 encoder）走 PyTorch | ✅ 可用，網頁可切換比較 |

整體流程有三種組合可以量：① OWL=TRT + MobileSAM(PyTorch)、② OWL=TRT + 自訓練 encoder(PyTorch)、③ **全 TensorRT**（OWL 引擎 + NanoSAM encoder / decoder 引擎）。③ 若 5b 失敗，就退回 ①②，並把失敗訊息記進版本問題表。

## 1. 開發機（WSL）：打包要傳過去的檔案

```bash
cd <專案>
git push                                              # 程式碼走 git
WITH_HF_CACHE=1 ./installation/jetson/pack_for_jetson.sh
# 產生 data/jetson_files.tar.gz（約 160 MB）與 data/jetson_hf_cache.tar.gz（約 590 MB，Jetson 連得上 HuggingFace 可不帶）
```

預期：最後印出打包內容與傳送指令。**失敗時**：`[缺少] …` 表示必要檔案不齊，先回到 `docs/training.md` 匯出 ONNX。

## 2. Jetson：取得程式碼與資料

```bash
git clone --recurse-submodules <repo-url> ~/project && cd ~/project

# 在開發機執行（換成你的使用者與 IP）：
#   scp data/jetson_files.tar.gz data/jetson_hf_cache.tar.gz <user>@<jetson-ip>:~/

tar xzf ~/jetson_files.tar.gz -C ~/project
sha256sum -c SHA256SUMS                                # 每一行都要 OK
# 若有帶 HF 快取：
mkdir -p ~/.cache/huggingface/hub && tar xzf ~/jetson_hf_cache.tar.gz -C ~/.cache/huggingface/hub
```

**失敗時**：`FAILED` = 傳輸損壞，重傳該檔；`--recurse-submodules` 漏了會缺 `third_party/nanoowl`、`nanosam`，補 `git submodule update --init --recursive`。

## 3. Jetson：確認版本、備好記憶體

```bash
./installation/jetson/00_check_jetpack.sh              # 確認 JetPack 6.2 / CUDA 12.6 / TensorRT 10.3 / trtexec 存在
./installation/jetson/collect_env.sh                   # 產生 data/bench/jetson/env_*.md → 報告「環境」一節（此時 torch 還沒裝，先看基底版本）
./installation/jetson/04_setup_swap.sh                 # 8GB swap，建引擎時防 OOM
```

預期：L4T 為 R36.4.x；`trtexec` 在 `/usr/src/tensorrt/bin/trtexec`。**版本與預期不同**（例如 TensorRT 不是 10.3）→ 照實記錄，後面 torch wheel 與引擎相容性都可能受影響。

## 4. Jetson：建環境（約 20~40 分鐘，視網路）

```bash
./installation/jetson/01_install_env.sh                # 建 .venv-jetson（--system-site-packages）、Jetson 版 PyTorch、torch2trt、NanoOWL
source .venv-jetson/bin/activate
```

預期：最後印出 `torch 2.8.0 cuda True Orin…`、`tensorrt 10.3…`。
**失敗時（最常見）**：

| 症狀 | 先試 |
|---|---|
| `torch==2.8.0` 找不到 | 開 https://pypi.jetson-ai-lab.io/ 看 jp6/cu126 實際有哪些版本，`JP_INDEX=… ` 或改版本，並**記錄實際可用的組合** |
| `torch.cuda.is_available()` 為 False | 裝到 CPU 版；`pip uninstall torch` 後重裝；檢查 `libcusolver` |
| `import tensorrt` 失敗 | venv 沒帶 `--system-site-packages`；絕不 `pip install tensorrt` |
| torch2trt 對 TensorRT 10 報錯 | 官方 `nanosam/utils/predictor.py` 用 `TRTModule`；若實測不相容，這就是要寫進報告的版本問題（需改用 TRT 10 的 tensor API） |

```bash
./installation/jetson/collect_env.sh                   # 再跑一次，這次會包含 torch / tensorrt / torch2trt 版本
```

## 5. Jetson：OWL-ViT 權重與建三個引擎

```bash
./installation/jetson/02_download_models.sh            # OWL-ViT（已帶快取就會直接用）；decoder ONNX 已在 data/models 會略過
ENC_ONNX=data/models/resnet18_image_encoder.selftrain.onnx ./installation/jetson/03_build_engines.sh
ls -lh data/*.engine
```

預期產生三個引擎（引擎綁 GPU 架構與 TensorRT 版本，不能從別台複製）：

| 引擎 | 來源 |
|---|---|
| `data/owl_image_encoder_patch32.engine` | NanoOWL，fp16 |
| `data/resnet18_image_encoder.selftrain.engine` | 自訓練 NanoSAM encoder，fp16 |
| `data/mobile_sam_mask_decoder.engine` | mask decoder，fp32，動態點數 1~10 |

**失敗時**：OOM / `Killed` → 確認 swap 已啟用（`free -h`）、關掉桌面與其他程式再重跑；NanoOWL 建引擎找不到 `trtexec` → 確認 `/usr/src/tensorrt/bin/trtexec`；decoder 建引擎報輸入名稱錯誤 → 用 `python -c "import onnx;m=onnx.load('data/models/mobile_sam_mask_decoder.onnx');print([i.name for i in m.graph.input])"` 檢查，應含 `image_embeddings point_coords point_labels mask_input has_mask_input`。
**建引擎所花時間請記錄**（報告素材）。

### 5b. 驗證 NanoSAM 引擎可被 pipeline 載入（全 TensorRT 的關鍵一步）

```bash
python - <<'PY'
import sys, numpy as np, PIL.Image; sys.path.insert(0, "src")
from sam_backends import NanoSamPredictor
p = NanoSamPredictor("data/resnet18_image_encoder.selftrain.engine", "data/mobile_sam_mask_decoder.engine", "trt")
img = np.asarray(PIL.Image.open("data/samples/cats.jpg").convert("RGB"))
p.set_image(img)
m, iou, _ = p.predict(box=[32, 48, 352, 456])
print("mask", m.shape, "px", int(m.sum()), "iou", iou)      # WSL 上同一個框：PyTorch 約 49900 px、sam_iou 約 0.96
PY
```

預期：不報錯，面積接近 49900 px（引擎是 fp16，會有些微差異）。
**失敗時**（把完整 traceback 存下來，這是報告的版本相容性素材）：

| 症狀 | 可能原因 / 先試 |
|---|---|
| `import torch2trt` 或 `TRTModule` 對 TensorRT 10 報錯 | torch2trt 舊版不支援 TRT 10 → 確認裝的是 GitHub master；仍失敗就要改用 TRT 10 的 tensor API 重寫 `_TrtBackend` |
| decoder 報形狀 / 動態維度錯誤 | 引擎建置時 `--maxShapes` 只支援 1~10 個點；我們只傳 2 個點，應在範圍內 → 檢查 `trtexec` 建置 log |
| 遮罩全空或位置歪 | 引擎與 ONNX 輸入名稱 / 順序不一致 → 用 `trtexec --loadEngine=… --verbose` 看 binding 名稱 |
| 面積與 WSL 差很多 | 比對 fp16 與 fp32：用 `trtexec` 重建 encoder 時拿掉 `--fp16` 再比 |

## 6. Jetson：固定電源模式並量測

```bash
sudo nvpmodel -p --verbose | head -20                  # 看有哪些模式（Orin Nano 常見 15W / 25W / MAXN）
./installation/jetson/05_set_power_mode.sh 0           # 0 = 通常是最高效能；實際編號以上一行為準
```

**量測期間電源模式必須固定，且要寫進報告**。要比較多種模式就每種都完整跑一輪（改 `--power_mode` 與輸出檔名）。

### 6a. 各引擎的純 GPU 延遲（trtexec）

```bash
./installation/jetson/06_measure_engines.sh            # → data/bench/jetson/engines_*/summary.md
```

預期：每個引擎一列 mean / median / p99 / FPS。把 `summary.md` 的表貼進 [jetson-data.md](jetson-data.md) 對應位置。

### 6b. 整體流程分段延遲 + 記憶體（①②③ 三種組合）

```bash
python benchmark/bench_pipeline.py data/samples/cats.jpg "a cat" "a remote control" --runs 50 \
    --owl_engine data/owl_image_encoder_patch32.engine \
    --power_mode "MAXN (nvpmodel 0)" --out data/bench/jetson/pipe_owltrt_mobilesam.json
python benchmark/bench_pipeline.py data/samples/cats.jpg "a cat" "a remote control" --runs 50 \
    --owl_engine data/owl_image_encoder_patch32.engine --student data/eval/colab80_best.pth \
    --power_mode "MAXN (nvpmodel 0)" --out data/bench/jetson/pipe_owltrt_student.json
```

```bash
# ③ 全 TensorRT（需 5b 成功）
python benchmark/bench_pipeline.py data/samples/cats.jpg "a cat" "a remote control" --runs 50 \
    --owl_engine data/owl_image_encoder_patch32.engine \
    --nanosam_encoder data/resnet18_image_encoder.selftrain.engine --nanosam_decoder data/mobile_sam_mask_decoder.engine \
    --power_mode "MAXN (nvpmodel 0)" --out data/bench/jetson/pipe_all_trt.json
```

- 同一張圖（`cats.jpg`）、同樣的提示詞，才能與 WSL 的數據對照（[benchmark.md](benchmark.md)）。
- Jetson 的 GPU 與 CPU **共用記憶體**，所以 `nvidia-smi` 的 GPU 記憶體欄位會是 `null`；看 `sys_used_*`（系統記憶體）與 `process_peak_rss_mb`。
- 另開終端機同時記錄功耗與溫度（量測期間）：`tegrastats --interval 500 | tee data/bench/jetson/tegrastats_$(date +%H%M%S).log`

## 7. Jetson：啟動網頁示範

```bash
./installation/jetson/07_run_server.sh                 # 引擎齊全 → 自動用 OWL 引擎 + NanoSAM 引擎；HyDE 預設關閉（Ollama 會吃掉大量記憶體）
```

從筆電瀏覽器開 `http://<jetson-ip>:8000`。

| 要做的事 | 做法 |
|---|---|
| 比較 TensorRT 的 NanoSAM 與 PyTorch 的 baseline / 自訓練 encoder | 網頁「SAM 編碼器」下拉選單切換（`NanoSAM（TensorRT）`、`MobileSAM（baseline）`、掃到的 `data/eval/*.pth`）；同一張圖面積應接近 |
| 即時攝影機 | 分頁「伺服器來源」→ 來源填 `0`（USB）；CSI 填 `gst:nvarguscamerasrc ! video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1 ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1` |
| 影片檔 | 來源填 `data/samples/demo_h264.mp4` |
| 製造情境 | 勾選零件檢查卡（見 [scenario.md](scenario.md)） |
| 即時換提示詞（需求 3） | 串流進行中直接改提示詞（HyDE 關閉時下一幀生效） |

**預期**：畫面有框與輪廓、右側統計顯示 FPS 與各階段毫秒。**失敗時**：攝影機打不開 → `ls /dev/video*`、`v4l2-ctl --list-devices`；CSI 管線無畫面 → 先單獨用 `gst-launch-1.0 nvarguscamerasrc ! nvvidconv ! xvimagesink` 驗證。

## 8. 螢幕截圖與 Demo 影片

至少 5 張（含失敗案例），存到 `docs/screenshots/`，清單見 [jetson-data.md](jetson-data.md) §5。Demo 影片建議用手機拍螢幕＋實物，或用 `ffmpeg` 錄 MJPEG 串流：
`ffmpeg -i http://localhost:8000/api/camera/stream -t 60 -c:v libx264 data/bench/jetson/demo.mp4`（**未驗證**；不行就用手機拍螢幕）

## 9. 做完後帶回來

```bash
tar czf ~/jetson_results.tar.gz data/bench/jetson docs/screenshots   # 在 Jetson
scp <user>@<jetson-ip>:~/jetson_results.tar.gz .                      # 在開發機
```

回來後貼給開發者：`env_*.md`、`summary.md`、兩個 `pipe_*.json`、遇到的每一個錯誤與解法。

## 附：常用指令

| 目的 | 指令 |
|---|---|
| 目前電源模式 | `sudo nvpmodel -q` |
| 鎖定最高時脈 | `sudo jetson_clocks`（還原：`sudo jetson_clocks --restore`） |
| 即時功耗 / 記憶體 / 溫度 | `tegrastats` 或 `jtop`（`sudo pip install jetson-stats`） |
| 溫度過高降頻 | `tegrastats` 看溫度；長時間量測前先等待降溫 |
| 查服務日誌 | `tail -f data/bench/jetson/app.log`（格式 `[function]: input, output/error`） |
