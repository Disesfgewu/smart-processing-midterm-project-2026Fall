#!/usr/bin/env bash
# 確認 Jetson 的 JetPack / CUDA / TensorRT 版本與電源模式（結果請貼進報告）
echo "== L4T =="; cat /etc/nv_tegra_release 2>/dev/null || echo "not a Jetson?"
echo "== JetPack =="; dpkg -l 2>/dev/null | grep -E "nvidia-jetpack " || echo "nvidia-jetpack 未安裝: sudo apt install nvidia-jetpack"
echo "== CUDA =="; NVCC=$(command -v nvcc || echo /usr/local/cuda/bin/nvcc); "$NVCC" --version 2>/dev/null | tail -2 || ls /usr/local | grep cuda
echo "== TensorRT =="; dpkg -l | grep -E " (tensorrt|python3-libnvinfer) " || true
ls /usr/src/tensorrt/bin/trtexec || echo "找不到 trtexec"
echo "== Python =="; python3 --version
echo "== Power mode =="; sudo nvpmodel -q 2>/dev/null || true
echo "== Memory/Swap =="; free -h
echo "== CUDA / cuDNN / TensorRT 套件檔案完整性（dpkg -V）=="
# 檔案損毀會讓 import torch 段錯誤（實機遇過 libcublasLt 損毀，見 docs/record.md §13.2）；有輸出就先 sudo apt-get install --reinstall 該套件
bad=0
for p in $(dpkg -l 2>/dev/null | awk '/^ii/ && $2 ~ /^(cuda-|libcublas|libcudnn|libcufft|libcurand|libcusolver|libcusparse|libcufile|libnvinfer|libnvrtc|tensorrt|python3-libnvinfer)/ {print $2}'); do
  n=$(dpkg -V "$p" 2>/dev/null | grep -vcE " c /" || true)
  [ "${n:-0}" -gt 0 ] && { echo "損毀: $p（$n 個檔案與套件記錄不符）"; bad=1; }
done
[ "$bad" = 0 ] && echo "OK（未發現與 dpkg 記錄不符的檔案）"
echo "== dpkg 狀態 =="; dpkg -l 2>/dev/null | awk 'NR>5 && $1 !~ /^(ii|hi|rc)/ {print $1,$2}' | head -5 | grep . || echo "OK（沒有未完成設定的套件）"
