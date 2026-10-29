#!/usr/bin/env bash
# 啟動本機 Ollama 服務（背景執行，日誌 /tmp/ollama.log）。server 的 HyDE 功能需要它。
set -euo pipefail
BIN="$HOME/.local/ollama/bin/ollama"
[ -x "$BIN" ] || BIN=$(command -v ollama || true)
[ -n "$BIN" ] || { echo "找不到 ollama，請先執行 ./installation/install_ollama.sh"; exit 1; }
if curl -s -m 2 localhost:11434/api/version >/dev/null; then echo "Ollama 已在執行"; exit 0; fi
OLLAMA_HOST=127.0.0.1:11434 nohup "$BIN" serve > /tmp/ollama.log 2>&1 &
for _ in $(seq 1 20); do curl -s -m 1 localhost:11434/api/version >/dev/null && { echo "Ollama 已啟動：$(curl -s localhost:11434/api/version)"; exit 0; }; sleep 0.5; done
echo "啟動失敗，請看 /tmp/ollama.log"; exit 1
