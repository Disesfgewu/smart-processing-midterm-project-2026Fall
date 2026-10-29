#!/usr/bin/env bash
# 確認 Jetson 的 JetPack / CUDA / TensorRT 版本與電源模式（結果請貼進報告）
echo "== L4T =="; cat /etc/nv_tegra_release 2>/dev/null || echo "not a Jetson?"
echo "== JetPack =="; dpkg -l 2>/dev/null | grep -E "nvidia-jetpack " || echo "nvidia-jetpack 未安裝: sudo apt install nvidia-jetpack"
echo "== CUDA =="; nvcc --version 2>/dev/null | tail -2 || ls /usr/local | grep cuda
echo "== TensorRT =="; dpkg -l | grep -E " (tensorrt|python3-libnvinfer) " || true
ls /usr/src/tensorrt/bin/trtexec || echo "找不到 trtexec"
echo "== Python =="; python3 --version
echo "== Power mode =="; sudo nvpmodel -q 2>/dev/null || true
echo "== Memory/Swap =="; free -h
