"""與 NVIDIA 官方（third_party/nanoowl、third_party/nanosam README）同口徑的量測，數字可以直接對照官方表格。

官方口徑（出處）：
  * NanoOWL 的 FPS：third_party/nanoowl/examples/owl_predict.py --profile
      先 torch.cuda.current_stream().synchronize()，同一張圖連跑 N 次 predictor.predict(image, text, text_encodings, threshold,
      pad_square=False)，再 synchronize；FPS = N / 總時間。文字編碼預先算好；predict = 前處理 + TensorRT 影像編碼 + decode，
      不含 NMS、不搬回 CPU、不畫圖、不含 SAM。
  * NanoSAM 的 Image Encoder / Full Pipeline（ms）：官方 README 與程式碼都沒有寫計時方法，所以兩種口徑都量，並列：
      (a) 純 GPU：trtexec --loadEngine 的 "GPU Compute Time"（encoder；decoder 以 1 個框 = 2 個點）。Full = encoder + decoder。
      (b) Python：官方 nanosam.utils.predictor.Predictor（set_image + predict，1 個框 = 2 個點，含 preprocess_image 與 upscale_mask），
          每次前後 torch.cuda.synchronize()。
    官方表格的 MobileSAM 用 fp32 encoder（README 註腳），NanoSAM(ResNet18) 用 fp16 encoder，decoder 皆 fp32。

用法（Jetson，先把網頁伺服器關掉，避免搶記憶體）：
  python benchmark/bench_official.py --image data/samples/cats.jpg --prompts "a cat" "a remote control" --power_mode "MAXN_SUPER"
"""
import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time

import numpy as np
import PIL.Image
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in ("third_party/nanoowl", "third_party/nanosam"):
    sys.path.insert(0, os.path.join(ROOT, p))

ap = argparse.ArgumentParser()
ap.add_argument("--image", default="data/samples/cats.jpg")
ap.add_argument("--prompts", nargs="+", default=["a cat", "a remote control"])
ap.add_argument("--threshold", type=float, default=0.1)
ap.add_argument("--owl_engine", default="data/owl_image_encoder_patch32.engine")
ap.add_argument("--owl_name", default="google/owlvit-base-patch32")
ap.add_argument("--decoder", default="data/mobile_sam_mask_decoder.engine")
ap.add_argument("--encoders", nargs="+", default=["MobileSAM (TinyViT, fp32)=data/mobile_sam_image_encoder.fp32.engine",
                                                  "MobileSAM (TinyViT, fp16)=data/mobile_sam_image_encoder.fp16.engine",
                                                  "NanoSAM-arch ResNet18 自訓練 (fp16)=data/resnet18_image_encoder.selftrain.engine"],
                help="'顯示名稱=引擎路徑'")
ap.add_argument("--owl_runs", type=int, default=200, help="官方 --num_profiling_runs 的預設值就是這個量級")
ap.add_argument("--owl_repeats", type=int, default=5, help="整個 profile 重複幾次，看穩定度")
ap.add_argument("--sam_runs", type=int, default=100)
ap.add_argument("--power_mode", default="")
ap.add_argument("--trtexec", default="/usr/src/tensorrt/bin/trtexec")
ap.add_argument("--out", default="data/bench/jetson/official_compare.json")
args = ap.parse_args()
os.chdir(ROOT)


def stats(ts):
    s = sorted(ts)
    return {"mean_ms": statistics.mean(ts), "median_ms": statistics.median(ts), "p95_ms": s[int(0.95 * (len(s) - 1))],
            "std_ms": statistics.pstdev(ts), "runs": len(ts)}


def trtexec(engine, shapes=None):
    """回傳 trtexec 的 GPU Compute Time 平均與 Latency 平均（ms）。"""
    cmd = [args.trtexec, f"--loadEngine={engine}", "--warmUp=1000", "--duration=5", "--iterations=100", "--avgRuns=100"]
    if shapes:
        cmd.append(f"--shapes={shapes}")
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    g = re.search(r"GPU Compute Time: .*?mean = ([\d.]+) ms", out)
    lat = re.search(r"\] Latency: .*?mean = ([\d.]+) ms", out)
    if not g:
        raise RuntimeError("trtexec 失敗：\n" + out[-1500:])
    return {"gpu_compute_mean_ms": float(g.group(1)), "latency_mean_ms": float(lat.group(1)) if lat else None, "cmd": " ".join(cmd)}


# trtexec 先全部跑完（此時本程序還沒佔用 GPU 記憶體；Nano 記憶體很緊，之後再跑 trtexec 會 OOM）
dec_trt = trtexec(args.decoder, "point_coords:1x2x2,point_labels:1x2")
enc_trts = {item: trtexec(item.split("=", 1)[1]) for item in args.encoders if os.path.exists(item.split("=", 1)[1])}

res = {"image": args.image, "prompts": args.prompts, "power_mode": args.power_mode, "device": torch.cuda.get_device_name(0),
       "date": time.strftime("%Y-%m-%d %H:%M:%S")}
image = PIL.Image.open(args.image).convert("RGB")
res["image_size"] = list(image.size)

# ---------------- NanoOWL：照 owl_predict.py --profile ----------------
from nanoowl.owl_predictor import OwlPredictor  # noqa: E402

owl = OwlPredictor(args.owl_name, device="cuda", image_encoder_engine=args.owl_engine)
te = owl.encode_text(args.prompts)
kw = dict(image=image, text=args.prompts, text_encodings=te, threshold=args.threshold, pad_square=False)
with torch.no_grad():
    owl.predict(**kw)                                          # 官方也先跑一次再計時
    fps = []
    for _ in range(args.owl_repeats):
        torch.cuda.current_stream().synchronize()
        t0 = time.perf_counter_ns()
        for _ in range(args.owl_runs):
            owl.predict(**kw)
        torch.cuda.current_stream().synchronize()
        fps.append(args.owl_runs / ((time.perf_counter_ns() - t0) / 1e9))
res["nanoowl"] = {"model": "OWL-ViT ViT-B/32, 768, patch 32, TensorRT fp16", "protocol": "owl_predict.py --profile（同一張圖連跑 N 次，前後 synchronize）",
                  "runs_per_repeat": args.owl_runs, "fps_each_repeat": fps, "fps_mean": statistics.mean(fps), "fps_min": min(fps), "fps_max": max(fps),
                  "ms_per_predict": 1e3 / statistics.mean(fps)}
del owl
torch.cuda.empty_cache()

# ---------------- NanoSAM：(a) trtexec  (b) 官方 Predictor ----------------
res["nanosam_decoder_trtexec"] = dec_trt
from nanosam.utils.predictor import Predictor  # noqa: E402

res["nanosam"] = []
box = [100, 100, 400, 400]
points = np.array([[box[0], box[1]], [box[2], box[3]]])
labels = [2, 3]
for item in args.encoders:
    name, eng = item.split("=", 1)
    if not os.path.exists(eng):
        print(f"跳過（找不到）：{eng}")
        continue
    enc_trt = enc_trts[item]
    pred = Predictor(eng, args.decoder)
    for _ in range(10):
        pred.set_image(image)
        pred.predict(points, labels)
    t_enc, t_dec, t_full = [], [], []
    for _ in range(args.sam_runs):
        torch.cuda.synchronize(); a = time.perf_counter()
        pred.set_image(image)
        torch.cuda.synchronize(); b = time.perf_counter()
        pred.predict(points, labels)
        torch.cuda.synchronize(); c = time.perf_counter()
        t_enc.append((b - a) * 1e3); t_dec.append((c - b) * 1e3); t_full.append((c - a) * 1e3)
    res["nanosam"].append({"name": name, "engine": eng,
                           "trtexec_encoder": enc_trt,
                           "trtexec_full_gpu_compute_ms": enc_trt["gpu_compute_mean_ms"] + dec_trt["gpu_compute_mean_ms"],
                           "trtexec_full_latency_ms": (enc_trt["latency_mean_ms"] or 0) + (dec_trt["latency_mean_ms"] or 0),
                           "python_encoder_set_image": stats(t_enc), "python_decoder_1box": stats(t_dec), "python_full_1box": stats(t_full)})
    del pred
    torch.cuda.empty_cache()

os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump(res, open(args.out, "w"), indent=2, ensure_ascii=False)

o = res["nanoowl"]
print(f"\n{res['device']} | power={args.power_mode or '（未標註）'} | image {image.size} | prompts={args.prompts}")
print(f"[NanoOWL ViT-B/32] FPS（官方 --profile 口徑）= {o['fps_mean']:.1f}（{o['fps_min']:.1f}~{o['fps_max']:.1f}，{args.owl_repeats}×{args.owl_runs} 次）≈ {o['ms_per_predict']:.1f} ms/張   官方 Orin Nano：TBD")
print(f"[mask decoder trtexec, 1 框] GPU {dec_trt['gpu_compute_mean_ms']:.2f} ms / latency {dec_trt['latency_mean_ms']:.2f} ms")
print(f"{'NanoSAM encoder':52s} {'trt enc GPU':>11s} {'trt Full GPU':>12s} {'py enc':>8s} {'py dec':>8s} {'py Full':>8s}")
for r in res["nanosam"]:
    print(f"{r['name']:52s} {r['trtexec_encoder']['gpu_compute_mean_ms']:11.2f} {r['trtexec_full_gpu_compute_ms']:12.2f} "
          f"{r['python_encoder_set_image']['mean_ms']:8.2f} {r['python_decoder_1box']['mean_ms']:8.2f} {r['python_full_1box']['mean_ms']:8.2f}")
print("官方 Orin Nano：MobileSAM Full Pipeline 146 ms；NanoSAM(ResNet18) Full Pipeline 27 ms；Image Encoder 皆 TBD")
