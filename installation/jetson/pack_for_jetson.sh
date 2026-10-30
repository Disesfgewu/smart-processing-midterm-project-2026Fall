#!/usr/bin/env bash
# 【在開發機（WSL）執行】把 Jetson 需要、但不在 git 裡的檔案打成一包（權重 / ONNX / 範例素材），並附 SHA256SUMS 供到了 Jetson 後驗證。
#   ./installation/jetson/pack_for_jetson.sh                    # → data/jetson_files.tar.gz（約 160 MB）
#   WITH_HF_CACHE=1 ./installation/jetson/pack_for_jetson.sh    # 另外產生 data/jetson_hf_cache.tar.gz（OWL-ViT 權重 ~590 MB；Jetson 連不上 HuggingFace 時用）
# 引擎（.engine）不能在這裡建：它綁定 GPU 架構與 TensorRT 版本，一律在 Jetson 上由 03_build_engines.sh 產生。
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=${OUT:-data/jetson_files.tar.gz}

# 必要檔案（路徑相對於專案根目錄；解壓時用 -C <Jetson 上的專案根目錄>）
REQUIRED=(
    data/models/resnet18_image_encoder.selftrain.onnx    # 自訓練 NanoSAM image encoder（Colab 79 epoch，ONNX 已驗證與 PyTorch 一致）
    data/models/mobile_sam_mask_decoder.onnx             # mask decoder（03_build_engines.sh 使用的檔名；與官方 NanoSAM 共用）
    data/models/mobile_sam.pt                            # baseline 權重：目前 pipeline 的 SAM 仍走 PyTorch（尚無 SAM TensorRT 後端）時需要
    data/samples/cats.jpg                                # 量測用影像（與 WSL 的數據同一張，才能對照）
)
# 選用（存在才放）
OPTIONAL=(
    data/models/resnet18_image_encoder.onnx              # 官方 encoder（若你手動下載過）
    data/eval/colab80_best.pth                           # 自訓練 encoder 的 PyTorch 權重（讓 Jetson 的網頁也能切換比較）
    data/samples/demo_h264.mp4                           # 影片測試（H.264，Jetson 的解碼器相容）
)

files=(); missing=0
for f in "${REQUIRED[@]}"; do [ -f "$f" ] && files+=("$f") || { echo "[缺少] $f"; missing=1; }; done
[ $missing -eq 0 ] || { echo "必要檔案不齊，無法打包。"; exit 1; }
for f in "${OPTIONAL[@]}"; do [ -f "$f" ] && files+=("$f") || echo "[略過，不存在] $f"; done

tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
sha256sum "${files[@]}" > "$tmp/SHA256SUMS"
{ echo "git: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)$(git diff --quiet 2>/dev/null || echo ' (有未 commit 的修改)')"
  echo "built: $(date '+%F %T')"; } > "$tmp/PACK_INFO"
tar -czf "$OUT" -C "$tmp" SHA256SUMS PACK_INFO -C "$PWD" "${files[@]}"
echo; echo "已產生 $OUT（$(du -h "$OUT" | cut -f1)），內容："; tar -tzf "$OUT" | sed 's/^/  /'

if [ "${WITH_HF_CACHE:-0}" = 1 ]; then
    hf=${HF_HOME:-$HOME/.cache/huggingface}/hub
    [ -d "$hf/models--google--owlvit-base-patch32" ] || { echo "[錯誤] 找不到 OWL-ViT 快取 $hf/models--google--owlvit-base-patch32（先在本機跑過一次 download_model.sh owlvit）"; exit 1; }
    tar -czf data/jetson_hf_cache.tar.gz -C "$hf" models--google--owlvit-base-patch32
    echo "已產生 data/jetson_hf_cache.tar.gz（$(du -h data/jetson_hf_cache.tar.gz | cut -f1)）"
fi

cat <<MSG

== 傳到 Jetson（把 <user>@<jetson-ip> 換成你的；也可用隨身碟）==
  scp $OUT <user>@<jetson-ip>:~/
$( [ "${WITH_HF_CACHE:-0}" = 1 ] && echo "  scp data/jetson_hf_cache.tar.gz <user>@<jetson-ip>:~/" )
== 在 Jetson 上（<repo> = git clone 下來的專案資料夾）==
  tar xzf ~/$(basename "$OUT") -C <repo>
  cd <repo> && sha256sum -c SHA256SUMS      # 全部要顯示 OK
$( [ "${WITH_HF_CACHE:-0}" = 1 ] && echo "  mkdir -p ~/.cache/huggingface/hub && tar xzf ~/jetson_hf_cache.tar.gz -C ~/.cache/huggingface/hub" )
MSG
