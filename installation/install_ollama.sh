#!/usr/bin/env bash
# 使用者層級安裝 Ollama（不需要 sudo）並下載 HyDE 用的模型。
#   ./installation/install_ollama.sh              # x86_64 (WSL) 安裝，並 pull 預設模型 qwen3:4b
#   ./installation/install_ollama.sh qwen2.5:1.5b # 改用別的模型（較快、品質較差）
# 安裝位置：~/.local/ollama（執行檔 ~/.local/ollama/bin/ollama）；模型存於 ~/.ollama
#
# 注意：
#  * 此版本需要 NVIDIA 驅動 >= 570 才會使用 GPU；驅動較舊（例如 560）時 Ollama 會自動改用 CPU，
#    對 GTX 1060 6GB 反而是好事——顯存留給 OWL / SAM。
#  * Jetson（aarch64）請改用官方的 JetPack 6 套件：https://github.com/ollama/ollama/releases
#    （ollama-linux-arm64-jetpack6.tar.zst；本專案尚未在 Jetson 上實測）。
set -euo pipefail
MODEL=${1:-qwen3:4b}
DEST="$HOME/.local/ollama"
ARCH=$(uname -m)
[ "$ARCH" = "x86_64" ] || { echo "此腳本只處理 x86_64；$ARCH 請見檔頭說明"; exit 1; }
command -v zstd >/dev/null || { echo "需要 zstd：sudo apt install zstd，或 conda install zstd"; exit 1; }

if [ ! -x "$DEST/bin/ollama" ]; then
    mkdir -p "$DEST" "$HOME/.local/ollama-dl"
    # 注意：新版 release 只提供 .tar.zst（舊的 .tgz 連結會回傳 404）
    curl -L -f -o "$HOME/.local/ollama-dl/ollama-linux-amd64.tar.zst" https://ollama.com/download/ollama-linux-amd64.tar.zst
    zstd -dc "$HOME/.local/ollama-dl/ollama-linux-amd64.tar.zst" | tar -x -C "$DEST"
fi
"$DEST/bin/ollama" --version 2>&1 | tail -1

# 需要服務在跑才能 pull
if ! curl -s -m 2 localhost:11434/api/version >/dev/null; then
    ./installation/start_ollama.sh
fi
"$DEST/bin/ollama" pull "$MODEL"
"$DEST/bin/ollama" list
