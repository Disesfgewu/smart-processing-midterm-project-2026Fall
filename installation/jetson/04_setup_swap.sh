#!/usr/bin/env bash
# Orin Nano 8GB 建 TensorRT 引擎 / 安裝 torch2trt 時可能 OOM，建立 8GB swap 檔
set -euo pipefail
SWAP=/swapfile
if swapon --show | grep -q "$SWAP"; then echo "swap 已存在"; exit 0; fi
sudo fallocate -l 8G $SWAP
sudo chmod 600 $SWAP
sudo mkswap $SWAP
sudo swapon $SWAP
grep -q "$SWAP" /etc/fstab || echo "$SWAP none swap sw 0 0" | sudo tee -a /etc/fstab
free -h
