"""端對端延遲：對每種「模型組合」啟動一次真正的 `python -m server.main`，用 loopback 打 /api/segment。

量的是客戶端從送出請求到收到完整 JSON 的時間（圖片解碼 + OWL + NMS + SAM + 遮罩輪廓 + 序列化 + HTTP），
同時記錄伺服器回報的各階段、tegrastats 的功耗 / 溫度 / 記憶體。loopback 排除了網路浮動；真實網路還要加上上傳時間。

    python benchmark/bench_e2e.py                       # 全部組合 × 全部影像尺寸（約 10 分鐘）
    python benchmark/bench_e2e.py --configs trt         # 只量全 TensorRT
    python benchmark/bench_e2e.py --runs 100 --images data/samples/cats.jpg

組合用環境變數切換（server/main.py 的 resolve_backends：設成空字串 = 明確關閉）。
"""
import argparse
import json
import os
import re
import statistics as st
import subprocess
import sys
import time

import requests
import PIL.Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OFF = {"NANOSAM_ENCODER": "", "NANOSAM_DECODER": "", "SAM_ACTIVE": ""}
CONFIGS = {   # key → (顯示名稱, 環境變數)
    "pytorch": ("PyTorch fp32：OWL + MobileSAM", {"OWL_ENGINE": "", **OFF}),
    "student": ("PyTorch fp32：OWL + 自訓練 ResNet18 encoder（WSL / 無引擎的機器）", {"OWL_ENGINE": "", **{k: v for k, v in OFF.items() if k != "SAM_ACTIVE"}, "SAM_STUDENT": "data/eval/colab80_best.pth", "SAM_ACTIVE": "student"}),
    "owltrt_mobilesam": ("OWL TensorRT + MobileSAM（PyTorch fp32）", dict(OFF)),
    "owltrt_student": ("OWL TensorRT + 自訓練 ResNet18 encoder（PyTorch fp32）", {**OFF, "SAM_STUDENT": "data/eval/colab80_best.pth", "SAM_ACTIVE": "student"}),
    "trt": ("全 TensorRT：OWL + 自訓練 ResNet18 encoder（fp16）", {}),
    # 以下三個需要先建對應引擎（見 docs/measurement-report.md）：SAM encoder 的「模型」與「精度」分開比
    "trt_mobilesam_fp32": ("全 TensorRT：OWL + MobileSAM encoder（fp32，baseline）", {"NANOSAM_ENCODER": "data/mobile_sam_image_encoder.fp32.engine", "NANOSAM_DECODER": "data/mobile_sam_mask_decoder.engine", "SAM_ACTIVE": "nanosam"}),
    "trt_mobilesam_fp16": ("全 TensorRT：OWL + MobileSAM encoder（fp16）", {"NANOSAM_ENCODER": "data/mobile_sam_image_encoder.fp16.engine", "NANOSAM_DECODER": "data/mobile_sam_mask_decoder.engine", "SAM_ACTIVE": "nanosam"}),
    "trt_student_fp32": ("全 TensorRT：OWL + 自訓練 ResNet18 encoder（fp32）", {"NANOSAM_ENCODER": "data/resnet18_image_encoder.selftrain.fp32.engine", "NANOSAM_DECODER": "data/mobile_sam_mask_decoder.engine", "SAM_ACTIVE": "nanosam"}),
}


def make_images(base):
    """由 base 影像放大出不同尺寸（只用來量解碼 / 傳輸 / 後處理成本；放大不會增加細節，偵測內容與原圖相同）。"""
    out = [base]
    im = PIL.Image.open(base).convert("RGB")
    for w, h in ((1280, 960), (4096, 3072)):
        p = os.path.join(os.path.dirname(base), f"{os.path.splitext(os.path.basename(base))[0]}_{w}x{h}.jpg")
        if not os.path.exists(p):
            im.resize((w, h), PIL.Image.LANCZOS).save(p, quality=90)
        out.append(p)
    return out


def tegrastats(duration_log):
    """非 Jetson（WSL / 桌機）沒有 tegrastats → 回傳 None，該格就沒有功耗欄位，其餘照量。"""
    try:
        return subprocess.Popen(["tegrastats", "--interval", "500"], stdout=open(duration_log, "w"), stderr=subprocess.DEVNULL)
    except OSError:
        return None


def parse_tegra(path):
    rows = []
    if not os.path.exists(path):
        return {}
    for line in open(path):
        if "VDD_IN" not in line:
            continue
        g = lambda p: re.search(p, line)
        rows.append((int(g(r"RAM (\d+)/").group(1)), int(g(r"GR3D_FREQ (\d+)%").group(1)), float(g(r"tj@([\d.]+)C").group(1)), int(g(r"VDD_IN (\d+)mW").group(1))))
    if not rows:
        return {}
    # 取樣窗口前後各有 1 秒的 sleep（閒置），會把平均功耗往閒置值拉低；只用 GPU 有負載（GR3D_FREQ > 0）的取樣點算功耗與使用率
    act = [r for r in rows if r[1] > 0] or rows
    return {"samples": len(rows), "active_samples": len(act), "power_w_mean": st.mean(r[3] for r in act) / 1000, "power_w_max": max(r[3] for r in act) / 1000,
            "gpu_util_mean": st.mean(r[1] for r in act), "tj_max_c": max(r[2] for r in rows), "ram_used_mb_max": max(r[0] for r in rows)}


def stats(xs):
    xs = sorted(xs)
    pct = lambda p: xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]
    return {"n": len(xs), "mean": st.mean(xs), "median": st.median(xs), "p95": pct(95), "p99": pct(99), "min": xs[0], "max": xs[-1],
            "std": st.pstdev(xs), "fps": 1000 / st.mean(xs)}


def mem_available_mb():
    return next(int(l.split()[1]) // 1024 for l in open("/proc/meminfo") if l.startswith("MemAvailable"))


def rss_mb(pid):
    return int(re.search(r"VmRSS:\s+(\d+)", open(f"/proc/{pid}/status").read()).group(1)) // 1024


def run_config(key, images, prompts, threshold, runs, warmup, port, logdir):
    name, env_add = CONFIGS[key]
    env = {**os.environ, "HOST": "127.0.0.1", "PORT": str(port), "HYDE_AUTOSTART": "0", "APP_LOG": os.path.join(logdir, f"e2e_{key}_app.log"), **env_add}
    out_path = os.path.join(logdir, f"e2e_{key}_server.out")
    avail_before = mem_available_mb()
    p = subprocess.Popen([sys.executable, "-m", "server.main"], cwd=ROOT, env=env, stdout=open(out_path, "w"), stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(120):
            if "就緒" in open(out_path, errors="ignore").read():
                break
            if p.poll() is not None:
                raise RuntimeError(f"server exited: {open(out_path, errors='ignore').read()[-400:]}")
            time.sleep(1)
        else:
            raise RuntimeError("server not ready in 120 s")
        backend_line = next((l.strip() for l in open(out_path, errors="ignore") if l.startswith("[server] 後端")), "")
        health = requests.get(f"{base}/api/health", timeout=10).json()
        res = {"name": name, "backend_line": backend_line, "health_backend": health.get("backend"), "env": env_add,
               "mem_available_before_mb": avail_before, "mem_available_after_load_mb": mem_available_mb(), "server_rss_mb": rss_mb(p.pid), "images": {}}
        for img in images:
            size = "×".join(map(str, PIL.Image.open(img).size))
            data = open(img, "rb").read()

            def one():
                t = time.perf_counter()
                r = requests.post(f"{base}/api/segment", files={"image": (os.path.basename(img), data, "image/jpeg")},
                                  data={"prompts": prompts, "threshold": threshold}, timeout=300)
                r.raise_for_status()
                j = r.json()
                return (time.perf_counter() - t) * 1e3, j
            for _ in range(warmup):
                one()
            tlog = os.path.join(logdir, f"e2e_{key}_{size}_tegra.log")
            tp = tegrastats(tlog)
            time.sleep(1)
            rows = [one() for _ in range(runs)]
            time.sleep(1)
            if tp:
                tp.terminate(); tp.wait()
            tm = lambda k: [j["timing"][k] for _, j in rows if k in j["timing"]]
            res["images"][size] = {"file": os.path.basename(img), "bytes": len(data), "client_wall_ms": stats([w for w, _ in rows]),
                                   "server_total_ms": stats(tm("total_ms")), "owl_ms": stats(tm("owl_ms")), "sam_ms": stats(tm("sam_ms")),
                                   "serialize_ms": stats(tm("serialize_ms")), "decode_ms": stats(tm("decode_ms")) if tm("decode_ms") else None,
                                   "server_ms": stats(tm("server_ms")) if tm("server_ms") else None, "detections": sorted({len(j["detections"]) for _, j in rows}),
                                   "summary": {s["label"]: s["count"] for s in rows[-1][1]["summary"]}, "tegrastats": parse_tegra(tlog)}
            w = res["images"][size]["client_wall_ms"]
            print(f"   {size:>10}: end-to-end mean {w['mean']:.0f} ms (median {w['median']:.0f}, p95 {w['p95']:.0f}, max {w['max']:.0f}) = {w['fps']:.1f} req/s | "
                  f"owl {res['images'][size]['owl_ms']['mean']:.0f} + sam {res['images'][size]['sam_ms']['mean']:.0f} ms | detections {res['images'][size]['detections']}", flush=True)
        res["server_rss_mb_end"] = rss_mb(p.pid)
        res["mem_available_end_mb"] = mem_available_mb()
        return res
    finally:
        p.terminate()
        try:
            p.wait(timeout=20)
        except subprocess.TimeoutExpired:
            p.kill()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="pytorch,owltrt_mobilesam,owltrt_student,trt")
    ap.add_argument("--images", nargs="*", help="影像清單；不給就用 data/samples/cats.jpg 與其放大版（1280×960、4096×3072）")
    ap.add_argument("--prompts", default="a cat,a remote control")
    ap.add_argument("--threshold", type=float, default=0.1)
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--port", type=int, default=8101)
    ap.add_argument("--out", default="data/bench/jetson/e2e_matrix.json")
    a = ap.parse_args()
    os.chdir(ROOT)
    images = a.images or make_images("data/samples/cats.jpg")
    logdir = os.path.dirname(a.out); os.makedirs(logdir, exist_ok=True)
    try:
        power = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True).stdout.split("\n")[0:2]
    except OSError:
        power = ["（非 Jetson：沒有 nvpmodel）"]
    print(f"電源模式：{' '.join(power)} | 影像：{images} | 每組 {a.runs} 次（暖機 {a.warmup}）", flush=True)
    results = {"power_mode": " ".join(power), "prompts": a.prompts, "threshold": a.threshold, "runs": a.runs, "warmup": a.warmup, "configs": {}}
    for i, key in enumerate(a.configs.split(",")):
        print(f"== {key}: {CONFIGS[key][0]}", flush=True)
        results["configs"][key] = run_config(key, images, a.prompts, a.threshold, a.runs, a.warmup, a.port + i, logdir)
        json.dump(results, open(a.out, "w"), indent=1, ensure_ascii=False)   # 每組做完就存，中途失敗也不會全丟
    print("saved", a.out)


if __name__ == "__main__":
    main()
