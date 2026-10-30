#!/usr/bin/env bash
# 【在 Jetson 執行】用 trtexec 量每個 TensorRT 引擎的純 GPU 推論延遲（不含前後處理），並記錄記憶體與電源模式。
#   ./installation/jetson/06_measure_engines.sh              # 量 data/ 下全部引擎
#   ITER=200 ./installation/jetson/06_measure_engines.sh
# 結果：data/bench/jetson/engines_<時間>/ 下每個引擎一個完整 log + summary.md（延遲表可直接貼進報告）。
# 先跑 05_set_power_mode.sh 固定電源模式；summary 會記下當時的模式。
set -uo pipefail
cd "$(dirname "$0")/../.."
TRTEXEC=${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}; [ -x "$TRTEXEC" ] || { echo "找不到 $TRTEXEC"; exit 1; }
ITER=${ITER:-100}; WARMUP_MS=${WARMUP_MS:-2000}
D=data/bench/jetson/engines_$(date +%Y%m%d_%H%M%S); mkdir -p "$D"
S=$D/summary.md
{ echo "# TensorRT 引擎延遲（trtexec，$(date '+%F %T')）"; echo
  pm=$(sudo -n nvpmodel -q 2>/dev/null | tr '\n' ' '); echo "- 電源模式：${pm:-（無法查詢，請手動記錄）}"
  echo "- 迭代 $ITER 次，暖機 ${WARMUP_MS} ms；數字為 **GPU Compute Time**（不含主機↔裝置搬移）"; echo
  echo "| 引擎 | 大小 | mean (ms) | median | p99 | min | max | FPS(=1000/mean) | 載入後系統記憶體增加 (MB) |"
  echo "|---|---|---|---|---|---|---|---|---|"; } > "$S"

# 各引擎的輸入形狀（動態形狀的引擎必須給；固定形狀的略過）
shapes_for() {
    case "$(basename "$1")" in
        mobile_sam_mask_decoder*) echo "--shapes=image_embeddings:1x256x64x64,point_coords:1x2x2,point_labels:1x2,mask_input:1x1x256x256,has_mask_input:1" ;;
        *) echo "" ;;
    esac
}

used_mb() { awk '/MemTotal/{t=$2}/MemAvailable/{a=$2}END{printf "%d",(t-a)/1024}' /proc/meminfo; }

shopt -s nullglob
engines=(data/*.engine)
[ ${#engines[@]} -gt 0 ] || { echo "data/ 下沒有 .engine；先跑 03_build_engines.sh"; exit 1; }
for e in "${engines[@]}"; do
    n=$(basename "$e" .engine); log=$D/$n.log
    echo "== $n"; before=$(used_mb)
    "$TRTEXEC" --loadEngine="$e" $(shapes_for "$e") --iterations="$ITER" --warmUp="$WARMUP_MS" --duration=0 --useSpinWait --noDataTransfers > "$log" 2>&1 &
    pid=$!; peak=$before
    while kill -0 $pid 2>/dev/null; do u=$(used_mb); [ "$u" -gt "$peak" ] && peak=$u; sleep 0.5; done
    wait $pid; rc=$?
    line=$(grep -E "GPU Compute Time:" "$log" | tail -1)
    if [ $rc -ne 0 ] || [ -z "$line" ]; then
        echo "  失敗（exit $rc），看 $log 最後幾行："; tail -5 "$log" | sed 's/^/    /'
        echo "| $n | $(du -h "$e" | cut -f1) | **失敗**（見 $log） | | | | | |" >> "$S"; continue
    fi
    g() { echo "$line" | grep -oE "$1 = [0-9.]+" | head -1 | awk '{print $3}'; }
    mean=$(g mean); fps=$(awk -v m="$mean" 'BEGIN{printf "%.1f",1000/m}')
    p99=$(echo "$line" | grep -oE "percentile\(99%\) = [0-9.]+" | awk '{print $3}')
    echo "  mean $mean ms | median $(g median) | p99 $p99 | $fps FPS | 記憶體峰值增加 $((peak - before)) MB"
    echo "| $n | $(du -h "$e" | cut -f1) | $mean | $(g median) | $p99 | $(g min) | $(g max) | $fps | $((peak - before)) |" >> "$S"
done
echo; cat "$S"; echo; echo "完整 log 在 $D/"
