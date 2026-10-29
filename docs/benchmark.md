# 效能量測與官方數據對照

腳本：`benchmark/bench_pipeline.py`（結果 JSON 存於 `data/bench/`，不進版控）。
**WSL 的數據只用來確認量測方法與數量級，不可作為繳交數據；繳交數據須在 Jetson Orin Nano 上以 TensorRT 重量，並註明電源模式。**

```bash
conda activate nanoowl-sam
python benchmark/bench_pipeline.py data/samples/cats.jpg "a cat" "a remote control" --runs 50
```

## 量測方法

- 每階段先暖機 10 次，再重複 50 次；每次計時前後皆 `torch.cuda.synchronize()`（GPU 為非同步，未同步會把時間算到下一階段）。
- 報告 mean / median / p95 / std；FPS = 1000 / mean。
- 輸入：`cats.jpg`（640×480），提示詞 `a cat`、`a remote control`，偵測到 4 個框（NMS 後）。
- 記憶體：PyTorch 配置峰值、`nvidia-smi` 整卡使用量差。

## WSL 結果（GTX 1060 6GB, PyTorch fp32, 2026-10-29）

| 階段 | mean (ms) | median | p95 | std | FPS |
|---|---|---|---|---|---|
| OWL 前處理 | 0.6 | 0.5 | 0.6 | 0.0 | — |
| OWL 影像編碼器 | 72.7 | 72.8 | 73.8 | 0.7 | 13.8 |
| OWL decode | 0.8 | 0.8 | 0.8 | 0.0 | — |
| **OWL 合計** | **74.5** | 74.6 | 75.5 | 0.7 | **13.4** |
| OWL 文字編碼（換提示時才跑） | 9.5 | 8.9 | 12.6 | 2.6 | — |
| **SAM 影像編碼器** | **88.5** | 88.2 | 90.6 | 2.5 | 11.3 |
| SAM decoder（每個框） | 7.9 | 7.4 | 10.2 | 1.3 | 127 |
| **完整流程（4 個框）** | **197.2** | 196.7 | 202.7 | 3.5 | **5.1** |

記憶體：PyTorch 峰值 901MB、reserved 984MB；nvidia-smi 增加約 1006MB（1892 → 2898MB）。兩種量法相符。
GPU 溫度 60°C、SM 時脈 1936MHz，量測期間無降頻。

## 內部一致性檢查

完整流程 197.2ms ≈ OWL 74.5 + SAM encoder 88.5 + 4 × decoder 7.9（= 194.6ms），差 2.6ms 為 NMS 與資料搬移。
完整流程的時間 ≈ 固定成本 + 框數 × 7.9ms：SAM encoder 每張圖只跑一次，框越多單位成本越低。

## 官方數據（source repo README）

| 項目 | 官方數據 | 引擎 / 精度 |
|---|---|---|
| NanoOWL OWL-ViT B/32（768） | AGX Orin **95 FPS**（≈10.5ms）；Orin Nano **TBD** | TensorRT |
| MobileSAM | encoder：AGX Orin 35ms，Orin Nano TBD；完整流程：AGX Orin 39ms，**Orin Nano 146ms**；mIoU 0.728 | TensorRT, encoder fp32 |
| NanoSAM (ResNet18) | encoder：AGX Orin 4.2ms，Orin Nano TBD；完整流程：AGX Orin 8.1ms，**Orin Nano 27ms**；mIoU 0.706 | TensorRT, encoder fp16, decoder fp32 |

官方「完整流程」= image encoder + mask decoder（單一提示）。官方在 Orin Nano 上只有 SAM 完整流程有數字；OWL 與 encoder 單項為 TBD，因此我們在 Orin Nano 實測的 OWL 與 encoder 延遲是新數據，無官方值可對照。

## 合理性判斷

| 項目 | WSL 實測 | 官方 | 判斷 |
|---|---|---|---|
| SAM encoder | 88.5ms | MobileSAM AGX Orin 35ms；Orin Nano 完整流程 146ms | 介於兩者之間，合理 |
| OWL image encoder | 72.7ms | AGX Orin ≈10.5ms | 慢約 7 倍，合理（見下） |

OWL 粗估：ViT-B/32 於 768×768 約 55~70 GFLOPs；GTX 1060 fp32 理論 4.4 TFLOPS → 理論下限約 16ms，實測 73ms ≈ 22% 利用率，符合 Pascal + PyTorch eager 的常見水準。AGX Orin 以 fp16 + TensorRT，算力高約一個數量級，故官方快約 7 倍屬合理。

**限制**：硬體（Pascal vs Ampere）、精度（fp32 vs fp16）、引擎（PyTorch eager vs TensorRT）皆不同，以上僅能佐證「數量級合理、方法無誤」，不能宣稱與官方一致。
