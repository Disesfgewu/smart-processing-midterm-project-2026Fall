# 自行訓練 NanoSAM image encoder（蒸餾 MobileSAM）

官方 `resnet18_image_encoder.onnx` 只放在需要求權限的 Google Drive。拿不到，自己蒸餾：
學生 = ResNet18（`nanosam.models.create_model("resnet18")`），教師 = MobileSAM ViT-T image encoder，
目標是讓學生輸出的 `256x64x64` 特徵逼近教師（Huber loss）。

官方 `nanosam.tools.train` 的教師要用 TensorRT 引擎，WSL 無法執行，因此改用
`training/train_nanosam.py`（教師為 PyTorch fp16，其餘設定相同，checkpoint 格式相容）。

## 步驟

```bash
conda activate nanoowl-sam

# 1. 訓練資料（建議 val2017，5k 張；train2017 約 18GB）
./training/download_coco.sh val

# 2. 蒸餾訓練（可中斷，重跑同一指令會從 checkpoint 續訓）
python training/train_nanosam.py data/coco/val2017 data/train/resnet18 \
    --batch_size 4 --accum_steps 4 --num_epochs 30

# 3. 匯出 ONNX（輸入 image 1x3x1024x1024，輸出 image_embeddings，名稱與 NanoSAM 預期一致）
python -m nanosam.tools.export_image_encoder_onnx \
    --model_name=resnet18 \
    --checkpoint=data/train/resnet18/checkpoint.pth \
    --output=data/models/resnet18_image_encoder.onnx
```

之後與官方檔案一樣，在 Jetson 上用 `installation/jetson/03_build_engines.sh` 轉成 TensorRT 引擎。

## 已驗證 / 未驗證

- 已驗證：腳本在 GTX 1060 上可執行（batch 4，峰值顯存約 1.5GB），checkpoint 可匯出 ONNX，輸入輸出名稱正確。這只用 24 張隨機雜訊圖跑了 6 個 batch 做煙霧測試。
- **未驗證：** 實際訓練速度與最終品質。官方設定是 COCO train2017、200 epoch、batch 16，在 GTX 1060 上不可行；
  用 5k 張、數十 epoch 訓練出的 encoder 精度會低於官方版本。
- 報告中請如實說明：這是自行蒸餾的 NanoSAM encoder，精度數據不能直接與 NanoSAM 官方 mIoU 對照；延遲／FPS／記憶體只取決於網路架構，仍可與官方對照。
- 品質檢查建議：用自訓練 encoder 與 `mobile_sam.pt` 對同一批影像、同樣的框提示，比較遮罩 IoU。
