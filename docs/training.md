# 自行訓練 NanoSAM image encoder（蒸餾 MobileSAM）

官方 `resnet18_image_encoder.onnx` 只放在需要求權限的 Google Drive。拿不到，自己蒸餾：
學生 = ResNet18（`nanosam.models.create_model("resnet18")`），教師 = MobileSAM ViT-T image encoder，
目標是讓學生輸出的 `256x64x64` 特徵逼近教師（Huber loss）。

官方 `nanosam.tools.train` 的教師要用 TensorRT 引擎，WSL 無法執行，因此改用
`training/train_nanosam.py`（教師為 PyTorch fp16，其餘設定相同，checkpoint 格式相容）。

## 步驟（用腳本）

```bash
conda activate nanoowl-sam
./training/download_coco.sh val                 # COCO val2017，5000 張、約 780MB（已下載可略過）
./training/run_distill.sh --bg                  # 背景訓練（RUN=resnet18_run1），預設 10 個 epoch；日誌 data/train/train.log
tail -f data/train/train.log                    # 看進度；每個 epoch 結束會印 train / val loss（新實驗請換 RUN 名稱，例如 RUN=resnet18_run2）
RUN=resnet18_run1 ./training/export_encoder.sh     # 匯出 data/models/resnet18_image_encoder.selftrain.onnx（用驗證最佳的 best.pth；不蓋過官方檔名）
```

- 前景執行就不要加 `--bg`；`EPOCHS=20 ./training/run_distill.sh` 改成總共訓練到第 20 個 epoch。
- **可隨時中斷**（`kill <pid>` 或 Ctrl+C），重跑同一個指令會從 `checkpoint.pth` 接續；`best.pth` 每次驗證 loss 創新低就更新，所以隨時都可以匯出。
- 其餘參數會原樣傳給 `train_nanosam.py`，例如先用小子集試跑：`./training/run_distill.sh --num_images 480`。

## 不蓋過 baseline，並且可以比較

**baseline** = `data/models/mobile_sam.pt`（MobileSAM，WSL 目前使用的 SAM）。訓練、匯出、比較都只「讀」它，保護措施：

| 風險 | 防護 |
|---|---|
| 訓練輸出寫進 `data/models` | `train_nanosam.py` 偵測到輸出目錄在 `data/models` 底下就直接拒絕 |
| 多次實驗互相覆蓋 | 每次實驗用不同的 `RUN` 名稱，輸出在 `data/train/<RUN>/`（同名重跑 = 接續）；`run_info.json` 記錄每次啟動的參數、git 版本、教師權重雜湊 |
| baseline 被意外改動 | 訓練開始與結束各算一次 sha256，結束時印出 `baseline … 未被改動 ✓`（也寫進 `log.txt`） |
| 匯出的 ONNX 與官方檔同名 | 自訓練的 ONNX 固定為 `resnet18_image_encoder.selftrain.onnx`；要寫入官方檔名 `resnet18_image_encoder.onnx` 必須 `FORCE=1`。Jetson 的 `03_build_engines.sh` 以 `ENC_ONNX` 選擇，引擎檔名跟著 ONNX 走，兩者可並存 |
| 網頁切換 encoder 影響 baseline | 切換只是換 `image_encoder` 物件，decoder 與 baseline 權重不變；實測切到自訓練再切回，遮罩面積與第一次完全相同 |

baseline 的 sha256 開頭：`6dbb90523a35330f…`（`data/models/mobile_sam.pt`）。

### 比較方式

```bash
# 1) 準確度：用 COCO 的 GT 框當提示，算遮罩與 GT 遮罩的 IoU（與官方相同的評估方法），並比較 encoder 延遲
./training/download_coco.sh annotations                       # 只需要一次（標註 241MB，解出 instances_val2017.json）
python training/compare_encoders.py                            # 只量 baseline（已量好，見下表）
python training/compare_encoders.py --student data/train/resnet18_run1/best.pth
python training/compare_encoders.py --student data/train/run1/best.pth --student data/train/run2/best.pth   # 多個實驗並排

# 2) 肉眼比較：用自訓練的 encoder 開網頁，在「SAM 編碼器」下拉選單即時切換（圖片模式切換後會自動用新的重跑同一張圖）
SAM_STUDENT=data/train/resnet18_run1/best.pth python -m server.main
```

評估的是訓練時固定留出的 200 張影像（沒參與訓練）。結果存 `data/eval/*.json`（不進版控）。

### baseline 基準數字（已量，`data/eval/baseline_mobilesam.json`）

200 張留出影像、1349 個物件（Small 529 / Medium 493 / Large 327），GT 框提示，PyTorch fp32，GTX 1060：

| | mIoU All | Small | Medium | Large | encoder 延遲 |
|---|---|---|---|---|---|
| **baseline（MobileSAM TinyViT）** | **0.744** | 0.685 | 0.756 | 0.822 | 87.4 ms |
| 官方報的 MobileSAM（val2017 全 5000 張、TensorRT） | 0.728 | 0.658 | 0.759 | 0.804 | — |
| 官方報的 NanoSAM（ResNet18） | 0.706 | 0.624 | 0.738 | 0.796 | — |

我們的 baseline 與官方 MobileSAM 差距在 0.02 以內，表示評估方法複現正確。**官方數字的影像、後端與訓練資料都不同，不能與自訓練的直接比較**，只用來確認方法沒錯；真正的比較是「我們的 baseline vs 我們自訓練的」。
速度差異來自架構：即使是只訓練 1 個 epoch 的假 checkpoint，SAM 階段也是約 54 ms（ResNet18）vs 107 ms（TinyViT），這個差距與訓練品質無關。

## 驗證集與訓練設定

- 依檔名排序，**固定留出最後 200 張**當驗證集（不參與訓練、不做隨機裁切），其餘約 4800 張訓練。每個 epoch 結束計算驗證 loss，可看出是否過擬合。
- 學生 ResNet18 使用 ImageNet 預訓練；Adam 3e-4、Huber loss、等效 batch 16（batch 4 × 累積 4）、教師為 PyTorch fp16、學生用 AMP。

## 實測速度（GTX 1060 6GB，COCO 真實圖片）

| 設定 | 速度 | 峰值顯存 |
|---|---|---|
| batch 4 | 約 4 張/秒（96 張 24 秒） | 1.5 GB |
| batch 8 | **同樣速度** | 2.6 GB |

瓶頸是 GPU 算力，加大 batch 不會變快，所以用 batch 4。一個 epoch（4800 張）約 **20 分鐘**，10 個 epoch 約 3.5 小時，30 個約 10 小時。
訓練時會與網頁伺服器 / 其他 GPU 程式搶 GPU（兩邊都會變慢），建議訓練期間先不要開伺服器。

## 已驗證 / 未驗證

- 已驗證：腳本在 GTX 1060 上可執行，接續、背景模式、匯出 ONNX 都測過（輸入輸出名稱正確）；用真實 COCO 圖片實測了速度（上表）。**尚未驗證完整訓練的最終品質**——目前只跑過 96 張、2 個 epoch 的速度試跑。
- **未驗證：** 實際訓練速度與最終品質。官方設定是 COCO train2017、200 epoch、batch 16，在 GTX 1060 上不可行；
  用 5k 張、數十 epoch 訓練出的 encoder 精度會低於官方版本。
- 報告中請如實說明：這是自行蒸餾的 NanoSAM encoder，精度數據不能直接與 NanoSAM 官方 mIoU 對照；延遲／FPS／記憶體只取決於網路架構，仍可與官方對照。
- 品質檢查建議：用自訓練 encoder 與 `mobile_sam.pt` 對同一批影像、同樣的框提示，比較遮罩 IoU。
