"""分階段延遲 / FPS / 記憶體量測（WSL: PyTorch；Jetson 之後換成 TensorRT 後端重跑同一支腳本）

  python benchmark/bench_pipeline.py data/samples/cats.jpg "a cat" "a remote control" --runs 50

  # Jetson：OWL 用 TensorRT 引擎、SAM 用自訓練 encoder（PyTorch），並標註電源模式
  python benchmark/bench_pipeline.py data/samples/cats.jpg "a cat" "a remote control" \
      --owl_engine data/owl_image_encoder_patch32.engine --student data/models/student.pth \
      --power_mode "MAXN (nvpmodel 0)" --out data/bench/jetson/pipeline_owltrt_student.json

方法：
  * 每個階段先暖機（--warmup），再重複 --runs 次；每次計時前後都 torch.cuda.synchronize()
    （GPU 非同步，不同步會把時間算到下一階段）。
  * 報告 mean / median / p95 / std，FPS = 1000 / mean。
  * 階段：OWL 前處理、OWL 影像編碼、OWL decode、SAM 影像編碼（set_image）、SAM decoder（每個框）、完整流程。
  * 記憶體：torch 配置峰值、nvidia-smi 的整卡使用量（含其他程式）。
"""
import argparse
import json
import os
import resource
import statistics
import subprocess
import sys
import time

import numpy as np
import PIL.Image
import torch
from torchvision.ops import nms

sys.path.insert(0, "src")
from pipeline import OpenVocabSegmenter  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("image")
ap.add_argument("prompts", nargs="+")
ap.add_argument("--threshold", type=float, default=0.1)
ap.add_argument("--warmup", type=int, default=10)
ap.add_argument("--runs", type=int, default=50)
ap.add_argument("--out", default="data/bench/bench.json")
ap.add_argument("--owl_engine", default=None, help="OWL image encoder 的 TensorRT 引擎（Jetson 上由 03_build_engines.sh 產生）；不給就用 PyTorch")
ap.add_argument("--student", default=None, help="自訓練 ResNet18 encoder 的 checkpoint；給了就量它而不是 baseline MobileSAM encoder")
ap.add_argument("--power_mode", default="", help="記錄用文字（例如 'MAXN (nvpmodel 0)'），會寫進結果 JSON；報告需註明")
args = ap.parse_args()


def smi_used_mb():
    """[GPU 記憶體使用 MB, 溫度, SM 時脈]。Jetson 沒有可用的 nvidia-smi（且 CPU / GPU 共用記憶體）→ 回傳 NaN，改看 sys_used_mb()。"""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,temperature.gpu,clocks.sm", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout.strip().split(",")
        return [float(x) for x in out]
    except (OSError, ValueError):
        return [float("nan")] * 3


def sys_used_mb():
    """系統記憶體使用量（MemTotal - MemAvailable）。Jetson 的 GPU 與 CPU 共用同一塊記憶體，這才是真正的用量。"""
    m = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/meminfo")}
    return (m["MemTotal"] - m["MemAvailable"]) / 1024


def timeit(fn, warmup, runs):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(runs):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1e3)
    ts_sorted = sorted(ts)
    return {"mean_ms": statistics.mean(ts), "median_ms": statistics.median(ts),
            "p95_ms": ts_sorted[int(0.95 * (len(ts) - 1))], "std_ms": statistics.pstdev(ts),
            "fps": 1e3 / statistics.mean(ts), "runs": runs}


torch.cuda.reset_peak_memory_stats()
smi_before = smi_used_mb()
sys_before = sys_used_mb()
t0 = time.perf_counter()
seg = OpenVocabSegmenter(owl_engine=args.owl_engine, student_checkpoint=args.student)
if args.student:
    seg.set_sam_encoder("student")
seg.set_prompts(args.prompts)
load_s = time.perf_counter() - t0
image = PIL.Image.open(args.image).convert("RGB")
image_np = np.asarray(image)
owl, sam = seg.owl, seg.sam
res = {"image": args.image, "image_size": list(image.size), "prompts": args.prompts, "device": torch.cuda.get_device_name(0),
       "torch": torch.__version__, "model_load_s": load_s, "power_mode": args.power_mode,
       "backend": f"owl={'tensorrt' if args.owl_engine else 'pytorch-fp32'}, sam=pytorch-fp32",
       "owl_engine": args.owl_engine, "sam_encoder": seg.sam_encoder, "student_checkpoint": args.student}

# ---- OWL 分段 ----
with torch.no_grad():
    rois = torch.tensor([[0, 0, image.width, image.height]], dtype=torch.float32, device="cuda")
    x = owl.image_preprocessor.preprocess_pil_image(image)
    enc_img = owl.encode_rois(x, rois, pad_square=False)
    res["owl_preprocess"] = timeit(lambda: owl.image_preprocessor.preprocess_pil_image(image), args.warmup, args.runs)
    res["owl_image_encoder"] = timeit(lambda: owl.encode_rois(x, rois, pad_square=False), args.warmup, args.runs)
    res["owl_decode"] = timeit(lambda: owl.decode(enc_img, seg.text_encodings, args.threshold), args.warmup, args.runs)
    res["owl_total"] = timeit(lambda: owl.predict(image=image, text=args.prompts, text_encodings=seg.text_encodings,
                                                  threshold=args.threshold, pad_square=False), args.warmup, args.runs)
    res["owl_text_encoder"] = timeit(lambda: owl.encode_text(args.prompts), args.warmup, args.runs)

    # ---- SAM 分段 ----
    res["sam_image_encoder"] = timeit(lambda: sam.set_image(image_np), args.warmup, args.runs)
    out = owl.predict(image=image, text=args.prompts, text_encodings=seg.text_encodings, threshold=args.threshold, pad_square=False)
    keep = nms(out.boxes, out.scores, 0.5) if len(out.boxes) else []
    boxes = out.boxes[keep].cpu().numpy() if len(out.boxes) else np.zeros((0, 4))
    res["num_boxes"] = int(len(boxes))
    if len(boxes):
        b = boxes[0]
        res["sam_decoder_per_box"] = timeit(lambda: sam.predict(box=b, multimask_output=False), args.warmup, args.runs)

    # ---- 完整流程 ----
    res["pipeline_total"] = timeit(lambda: seg(image, args.threshold), args.warmup, args.runs)

res["memory"] = {"torch_peak_alloc_mb": torch.cuda.max_memory_allocated() / 2**20,
                 "torch_reserved_mb": torch.cuda.memory_reserved() / 2**20,
                 "nvidia_smi_used_before_mb": smi_before[0], "nvidia_smi_used_after_mb": smi_used_mb()[0],
                 "gpu_temp_c": smi_used_mb()[1], "gpu_sm_clock_mhz": smi_used_mb()[2],
                 "sys_used_before_mb": sys_before, "sys_used_after_mb": sys_used_mb(),                       # 系統記憶體（Jetson 為 CPU+GPU 共用）
                 "process_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}           # 本程序的常駐記憶體峰值（Linux 單位為 KB）

os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump(res, open(args.out, "w"), indent=2)

print(f"\n{res['device']} | {res['backend']} | SAM encoder={res['sam_encoder']} | power={res['power_mode'] or '（未標註）'} | image {image.size} | boxes={res['num_boxes']} | load {load_s:.1f}s")
print(f"{'stage':24s} {'mean':>8s} {'median':>8s} {'p95':>8s} {'std':>7s} {'FPS':>7s}")
for k in ["owl_preprocess", "owl_image_encoder", "owl_decode", "owl_total", "owl_text_encoder",
          "sam_image_encoder", "sam_decoder_per_box", "pipeline_total"]:
    if k in res:
        r = res[k]
        print(f"{k:24s} {r['mean_ms']:8.1f} {r['median_ms']:8.1f} {r['p95_ms']:8.1f} {r['std_ms']:7.1f} {r['fps']:7.1f}")
print("memory(MB):", {k: (round(v) if v == v else None) for k, v in res["memory"].items()})
