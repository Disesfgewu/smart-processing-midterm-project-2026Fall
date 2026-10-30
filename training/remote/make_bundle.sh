#!/usr/bin/env bash
# 在主機執行：把「獨立訓練包」打成 data/nanosam_train_bundle.tar.gz（只有幾十 KB：訓練腳本 + 一鍵腳本 + baseline 雜湊）。
# 模型權重與資料不在包裡，由包內的 run_remote.sh 在目標機器上自動下載並驗證。
#   ./training/remote/make_bundle.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -f data/models/mobile_sam.pt ] || { echo "找不到 data/models/mobile_sam.pt（baseline），無法產生雜湊"; exit 1; }
OUT=data/nanosam_train_bundle.tar.gz
tmp=$(mktemp -d); d=$tmp/nanosam_train; mkdir -p "$d"
cp training/remote/run_remote.sh training/train_nanosam.py "$d/"
sha256sum data/models/mobile_sam.pt | cut -d' ' -f1 > "$d/BASELINE.sha256"          # 目標機器下載的 mobile_sam.pt 必須與此相同，否則拒絕訓練
{ echo "git: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"; echo "built: $(date '+%F %T')"; echo "baseline sha256: $(cut -c1-16 "$d/BASELINE.sha256")…"; } > "$d/VERSION"
cat > "$d/README.txt" <<'TXT'
NanoSAM image encoder 蒸餾訓練（獨立版）

  tar xzf nanosam_train_bundle.tar.gz && cd nanosam_train
  ./run_remote.sh            # 建環境 → 下載權重與 COCO → 小測（估算每個 epoch 時間）→ 訓練
  ./run_remote.sh status     # 看進度

需要：有 NVIDIA 顯卡的 Linux / WSL2 / Google Colab、Python 3.9~3.12（或 conda）、約 6 GB 磁碟、可連網。
詳細說明與環境變數見 run_remote.sh 開頭的註解。訓練結果在 data/train/<RUN>/（best.pth 為驗證最佳），複製回主機後比較。
TXT
tar -czf "$OUT" -C "$tmp" nanosam_train && rm -rf "$tmp"
ls -lh "$OUT" | awk '{print "已產生:", $9, "("$5")"}'; tar -tzf "$OUT"
