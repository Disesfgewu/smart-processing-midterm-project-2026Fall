#!/usr/bin/env bash
# 量測效能前固定電源模式與時脈，並印出模式（報告需註明）
#   ./05_set_power_mode.sh        # MAXN 最高效能 (mode 0)
#   ./05_set_power_mode.sh 1      # 例如 15W，模式編號用 `sudo nvpmodel -p --verbose` 查
set -euo pipefail
sudo nvpmodel -m "${1:-0}"
sudo jetson_clocks
sudo nvpmodel -q
