"""Jetson 記憶體輔助：載入模型後放掉「模型檔案」的頁面快取。

背景（docs/record.md §13.2 第 7 點）：Jetson 的 CPU / GPU 共用同一份 RAM，GPU（nvmap）配置需要「真正空閒」的頁面，
`MemAvailable` 裡可回收的檔案快取不保證能及時被回收。實測載入 OWL-ViT（587MB 權重檔）後 `MemFree` 只剩約 100MB，
接著 SAM 的第一個卷積就 OOM（PyTorch 在 OOM 路徑上查 NVML，Jetson 沒有 NVML，於是顯示成
`NVML_SUCCESS == r INTERNAL ASSERT FAILED`）。模型權重已經搬進 GPU 後，檔案快取沒有用處，放掉即可讓出空閒頁面。

只在 Jetson（有 /etc/nv_tegra_release）啟用；WSL / 桌機不受影響。環境變數 DROP_FILE_CACHE=0 可關閉、=1 可強制開啟。
posix_fadvise(DONTNEED) 只丟棄乾淨的快取頁，不需要 root，也不會影響已載入的模型。
"""
import os
from typing import Iterable

from applog import logger


def _enabled() -> bool:
    v = os.environ.get("DROP_FILE_CACHE")
    if v is not None:
        return v != "0"
    return os.path.exists("/etc/nv_tegra_release")


def _meminfo_mb() -> dict:
    out = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":", 1)
                if k in ("MemFree", "MemAvailable", "Cached"):
                    out[k] = int(v.split()[0]) // 1024
    except OSError:
        pass
    return out


def drop_file_cache(paths: Iterable[str]) -> None:
    """對每個路徑（檔案或資料夾，會遞迴）呼叫 posix_fadvise(DONTNEED)。找不到或沒權限的路徑直接略過。"""
    if not _enabled():
        return
    before = _meminfo_mb()
    n = 0
    for p in paths:
        if not p:
            continue
        p = os.path.expanduser(str(p))
        files = [p] if os.path.isfile(p) else [os.path.join(d, f) for d, _, fs in os.walk(p) for f in fs] if os.path.isdir(p) else []
        for fp in files:
            try:
                fd = os.open(fp, os.O_RDONLY | os.O_NONBLOCK)
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                    n += 1
                finally:
                    os.close(fd)
            except OSError:
                continue
    after = _meminfo_mb()
    logger.info(f"[memutil.drop_file_cache]: input=({n} files), output=MemFree {before.get('MemFree')}MB -> {after.get('MemFree')}MB, "
                f"Cached {before.get('Cached')}MB -> {after.get('Cached')}MB")


def hf_cache_dir(model_name: str) -> str:
    """HuggingFace 快取資料夾（例如 google/owlvit-base-patch32 → ~/.cache/huggingface/hub/models--google--owlvit-base-patch32）。"""
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    return os.path.join(hub, "models--" + model_name.replace("/", "--"))


def _read(path: str):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def jetson_stats() -> dict:
    """Jetson 沒有 nvidia-smi，改讀 sysfs：回傳 ram_used_mb / ram_total_mb（CPU 與 GPU 共用）、gpu_temp_c、gpu_clock_mhz、
    gpu_util_pct、power_w（ina3221 的 VDD_IN = 整板輸入功耗）。讀不到的欄位不會出現在結果裡。"""
    import glob
    s = {}
    mi = {}
    for line in (_read("/proc/meminfo") or "").splitlines():
        k, _, v = line.partition(":")
        if k in ("MemTotal", "MemAvailable"):
            mi[k] = int(v.split()[0]) // 1024
    if len(mi) == 2:
        s.update(ram_total_mb=mi["MemTotal"], ram_used_mb=mi["MemTotal"] - mi["MemAvailable"])
    for z in glob.glob("/sys/class/thermal/thermal_zone*"):
        if _read(z + "/type") == "gpu-thermal" and (t := _read(z + "/temp")):
            s["gpu_temp_c"] = int(t) / 1000
    for f in glob.glob("/sys/devices/platform/*/17000000.gpu/devfreq/*/cur_freq"):
        if v := _read(f):
            s["gpu_clock_mhz"] = int(v) / 1e6
    for f in glob.glob("/sys/devices/platform/*/17000000.gpu/load"):
        if v := _read(f):
            s["gpu_util_pct"] = int(v) / 10
    for hw in glob.glob("/sys/class/hwmon/hwmon*"):
        if _read(hw + "/name") == "ina3221":
            for lab in glob.glob(hw + "/in*_label"):
                if _read(lab) == "VDD_IN":
                    i = os.path.basename(lab)[2:].split("_")[0]
                    mv, ma = _read(f"{hw}/in{i}_input"), _read(f"{hw}/curr{i}_input")
                    if mv and ma:
                        s["power_w"] = int(mv) * int(ma) / 1e6
    return s
