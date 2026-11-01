"""Build the NanoOWL image-encoder TensorRT engine with a low GPU-memory footprint.

`python -m nanoowl.build_image_encoder_engine` loads OWL-ViT onto the GPU (fp32, ~590 MB) just to export the ONNX.
On an Orin Nano 8GB (CPU and GPU share RAM, GPU allocations cannot be swapped) that fails with
`NvMapMemAllocInternalTagged error 12` / `NVML_SUCCESS == r INTERNAL ASSERT` when the system is under memory pressure
(docs/record.md §13.2 item 7).

This script:
  1. exports the ONNX on the CPU (same fp32 ONNX as the official path; CPU memory can use swap),
  2. frees the model, then builds the engine with `trtexec --fp16` (weights stored as fp16 in the engine; engine I/O dtype unchanged),
  3. loads the engine once to check that TRTModule works with this TensorRT version.

Usage: python installation/jetson/build_owl_engine.py data/owl_image_encoder_patch32.engine
"""
import argparse
import gc
import os
import subprocess
import sys
import time

import torch

from nanoowl.owl_predictor import OwlPredictor, _owl_get_image_size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("engine_path")
    ap.add_argument("--model_name", default="google/owlvit-base-patch32")
    ap.add_argument("--onnx_path", default="data/models/owl_image_encoder_patch32.onnx",
                    help="ONNX is kept so that a failed engine build can be retried without re-exporting")
    ap.add_argument("--onnx_opset", type=int, default=16)
    ap.add_argument("--no_fp16", action="store_true", help="build an fp32 engine (default: fp16)")
    ap.add_argument("--workspace_mb", type=int, default=int(os.environ.get("TRT_WORKSPACE_MB", "0")),
                    help="trtexec workspace pool in MB; 0 = TensorRT default (all free memory)")
    ap.add_argument("--trtexec", default=os.environ.get("TRTEXEC", "/usr/src/tensorrt/bin/trtexec"))
    args = ap.parse_args()

    avail = next((int(l.split()[1]) // 1024 for l in open("/proc/meminfo") if l.startswith("MemAvailable")), None)
    if avail is not None and avail < 4000:
        print(f"[警告] 目前可用記憶體只有 {avail} MB。實測 trtexec 建這個引擎需要約 3GB 以上的共用記憶體；"
              f"可用 < 4GB 時曾連續失敗（out of memory，見 docs/record.md §13.2 第 7 點）。"
              f"建議先結束 gazebo / VS Code 遠端連線 / ollama，改用 SSH 終端機。", file=sys.stderr, flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.engine_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.onnx_path)), exist_ok=True)

    t0 = time.time()
    if os.path.exists(args.onnx_path):
        print(f"[1/3] 使用既有 ONNX: {args.onnx_path}", flush=True)
        image_size = _owl_get_image_size(args.model_name)
    else:
        print("[1/3] 在 CPU 匯出 ONNX（不使用 GPU 記憶體）...", flush=True)
        predictor = OwlPredictor(model_name=args.model_name, device="cpu")
        image_size = predictor.image_size
        tmp = args.onnx_path + ".partial"
        with torch.no_grad():
            predictor.export_image_encoder_onnx(tmp, onnx_opset=args.onnx_opset)
        os.replace(tmp, args.onnx_path)  # 只有完整匯出才會留下正式檔名，避免半個檔案被當成可用
        del predictor
        gc.collect()
    t1 = time.time()
    print(f"      ONNX 完成: {os.path.getsize(args.onnx_path) / 2**20:.0f} MB（{t1 - t0:.0f}s）", flush=True)

    print("[2/3] trtexec 建引擎...", flush=True)
    cmd = [args.trtexec, f"--onnx={args.onnx_path}", f"--saveEngine={args.engine_path}",
           f"--shapes=image:1x3x{image_size}x{image_size}"]
    if not args.no_fp16:
        cmd.append("--fp16")
    if args.workspace_mb:
        cmd.append(f"--memPoolSize=workspace:{args.workspace_mb}M")
    print("      " + " ".join(cmd), flush=True)
    rc = subprocess.call(cmd)
    t2 = time.time()
    if rc != 0 or not os.path.exists(args.engine_path):
        print(f"trtexec 失敗 (exit {rc})。常見原因：記憶體不足（Killed / OOM）。可試 TRT_WORKSPACE_MB=512。", file=sys.stderr)
        return rc or 1
    print(f"      引擎完成: {os.path.getsize(args.engine_path) / 2**20:.0f} MB（建引擎 {t2 - t1:.0f}s，總計 {t2 - t0:.0f}s）", flush=True)

    print("[3/3] 載入引擎驗證 TRTModule...", flush=True)
    try:
        enc = OwlPredictor.load_image_encoder_engine(args.engine_path, 1)
        x = torch.rand(1, 3, image_size, image_size, device="cuda")
        out = enc(x)
        print("      OK:", {k: tuple(v.shape) for k, v in out.__dict__.items()} if hasattr(out, "__dict__") else type(out), flush=True)
    except Exception:
        import traceback
        traceback.print_exc()
        print("引擎已建立，但 TRTModule 載入 / 執行失敗（TensorRT 版本相容性，記入 docs/record.md）", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
