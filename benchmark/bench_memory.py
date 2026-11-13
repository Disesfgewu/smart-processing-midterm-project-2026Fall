"""NanoOWL、NanoSAM「各自」與整體流程的記憶體用量（題目功能要求 4：分別量測；先前只有整個程序的數字）。

  python benchmark/bench_memory.py --out data/bench/jetson/memory_per_model.json

每個設定各開一個**獨立子程序**（互不污染），重複 --repeats 次：
  baseline      只 import torch 並初始化 CUDA context（扣掉它，才看得出模型本身多吃多少）
  owl_trt       只載入 NanoOWL（OwlPredictor + TensorRT 影像編碼器引擎）並推論
  sam_trt       只載入 NanoSAM 推論流程（自訓練 ResNet18 encoder 引擎 + mask decoder 引擎）並推論
  pipeline_trt  完整串接（OpenVocabSegmenter，與伺服器相同：另外還會載入 MobileSAM 的 PyTorch 權重，供網頁切換 encoder）
每個設定記錄三個時間點：初始化後、載入後、推論 N 次之後。指標：
  rss_mb              程序常駐記憶體（/proc/self/status VmRSS；含 CUDA 函式庫對映）
  torch_alloc_mb      torch 配置（torch.cuda.memory_allocated）；TensorRT 引擎的記憶體**不在這裡**（由 TensorRT 自己配置）
  sys_used_mb         整機 MemTotal − MemAvailable（Jetson CPU/GPU 共用，這是最接近「GPU 記憶體」的整機數字；會受檔案快取影響，只當參考）
Jetson 的 nvidia-smi 記憶體欄位是 N/A，所以沒有 GPU 專屬記憶體。模型各自的增量＝該設定的 RSS（或 sys_used）減去 baseline。
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def meminfo():
    d = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":", 1)
            d[k] = int(v.split()[0]) / 1024
    return d


def rss_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1024


def child(cfg, n, img_path, owl_engine, enc, dec):
    sys.path.insert(0, os.path.join(ROOT, "src"))
    import numpy as np
    import PIL.Image
    import torch

    def snap():
        torch.cuda.synchronize()
        m = meminfo()
        return {"rss_mb": round(rss_mb(), 1), "torch_alloc_mb": round(torch.cuda.memory_allocated() / 2**20, 1),
                "torch_reserved_mb": round(torch.cuda.memory_reserved() / 2**20, 1), "sys_used_mb": round(m["MemTotal"] - m["MemAvailable"], 1)}

    torch.zeros(1, device="cuda")
    out = {"init": snap()}
    image = PIL.Image.open(img_path).convert("RGB")
    boxes = [[100, 80, 300, 300], [330, 20, 640, 360], [40, 70, 180, 120], [335, 70, 375, 190]]
    if cfg == "baseline":
        out["loaded"] = out["after"] = out["init"]
    elif cfg == "owl_trt":
        from nanoowl.owl_predictor import OwlPredictor
        owl = OwlPredictor("google/owlvit-base-patch32", device="cuda", image_encoder_engine=owl_engine)
        te = owl.encode_text(["a cat", "a remote control"])
        out["loaded"] = snap()
        for _ in range(n):
            owl.predict(image=image, text=["a cat", "a remote control"], text_encodings=te, threshold=0.1, pad_square=False)
        out["after"] = snap()
    elif cfg == "sam_trt":
        from sam_backends import NanoSamPredictor
        sam = NanoSamPredictor(enc, dec, "auto")
        out["loaded"] = snap()
        arr = np.asarray(image)
        for _ in range(n):
            sam.set_image(arr)
            for b in boxes:
                sam.predict(box=np.array(b), multimask_output=False)
        out["after"] = snap()
    elif cfg == "pipeline_trt":
        from pipeline import OpenVocabSegmenter
        seg = OpenVocabSegmenter(owl_engine=owl_engine, nanosam_encoder=enc, nanosam_decoder=dec, nanosam_backend="auto")
        seg.set_sam_encoder("nanosam")
        seg.set_prompts(["a cat", "a remote control"])
        out["loaded"] = snap()
        for _ in range(n):
            seg(image, 0.1)
        out["after"] = snap()
    else:
        raise SystemExit(f"unknown cfg {cfg}")
    import resource
    out["rss_peak_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    out["torch_peak_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    print("RESULT " + json.dumps(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", default=None)
    ap.add_argument("--out", default="data/bench/jetson/memory_per_model.json")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--n", type=int, default=20, help="每個子程序的推論次數")
    ap.add_argument("--image", default="data/samples/cats.jpg")
    ap.add_argument("--owl_engine", default="data/owl_image_encoder_patch32.engine")
    ap.add_argument("--nanosam_encoder", default="data/resnet18_image_encoder.selftrain.engine")
    ap.add_argument("--nanosam_decoder", default="data/mobile_sam_mask_decoder.engine")
    ap.add_argument("--power_mode", default="")
    a = ap.parse_args()
    if a.child:
        return child(a.child, a.n, a.image, a.owl_engine, a.nanosam_encoder, a.nanosam_decoder)
    runs = {}
    for cfg in ("baseline", "owl_trt", "sam_trt", "pipeline_trt"):
        runs[cfg] = []
        for r in range(a.repeats):
            p = subprocess.run([sys.executable, __file__, "--child", cfg, "--n", str(a.n), "--image", a.image, "--owl_engine", a.owl_engine,
                                "--nanosam_encoder", a.nanosam_encoder, "--nanosam_decoder", a.nanosam_decoder], capture_output=True, text=True, cwd=ROOT)
            line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
            if not line:
                print(cfg, "FAILED", p.stderr[-600:])
                continue
            runs[cfg].append(json.loads(line[7:]))
            print(cfg, r, runs[cfg][-1]["loaded"], flush=True)
            time.sleep(3)
    summ = {}
    base = {k: statistics.fmean(x["after"][k] for x in runs["baseline"]) for k in ("rss_mb", "sys_used_mb")}
    for cfg, rs in runs.items():
        if not rs:
            continue
        g = lambda stage, k: [x[stage][k] for x in rs]
        f = lambda v: {"mean": round(statistics.fmean(v), 1), "min": round(min(v), 1), "max": round(max(v), 1)}
        summ[cfg] = {"n": len(rs), **{f"{st}_{k}": f(g(st, k)) for st in ("loaded", "after") for k in ("rss_mb", "torch_alloc_mb", "sys_used_mb")},
                     "rss_peak_mb": f([x["rss_peak_mb"] for x in rs]), "torch_peak_mb": f([x["torch_peak_mb"] for x in rs]),
                     "delta_vs_baseline_after_rss_mb": round(statistics.fmean(g("after", "rss_mb")) - base["rss_mb"], 1),
                     "delta_vs_baseline_after_sys_used_mb": round(statistics.fmean(g("after", "sys_used_mb")) - base["sys_used_mb"], 1)}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"date": time.strftime("%Y-%m-%d %H:%M:%S"), "power_mode": a.power_mode, "inference_runs_per_child": a.n, "summary": summ, "runs": runs},
              open(a.out, "w"), ensure_ascii=False, indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()
