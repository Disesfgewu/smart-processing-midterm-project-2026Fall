"""推論服務：持有模型、序列化 GPU 存取、記錄歷史。"""
import io
import subprocess
import threading
import time
from collections import deque
from typing import Dict, List, Optional

import cv2
import numpy as np
import PIL.Image
import PIL.ImageOps
import torch

from applog import log_call, logger
from pipeline import OpenVocabSegmenter
from server import hyde

MAX_IMAGE_SIDE = 4096


class SegmentService:
    def __init__(self, sam_checkpoint="data/models/mobile_sam.pt", owl_engine: Optional[str] = None):
        self.started = time.time()
        t0 = time.perf_counter()
        self.seg = OpenVocabSegmenter(sam_checkpoint=sam_checkpoint, owl_engine=owl_engine)
        self.load_s = time.perf_counter() - t0
        self.backend = "tensorrt" if owl_engine else "pytorch-fp32"
        self.lock = threading.Lock()
        self.history = deque(maxlen=50)
        self.n_requests = 0
        self.warmup()
        logger.info(f"[SegmentService.__init__]: input=(sam_checkpoint={sam_checkpoint}, owl_engine={owl_engine}), "
                    f"output=loaded in {self.load_s:.1f}s on {torch.cuda.get_device_name(0)}")

    def warmup(self, n: int = 3):
        """啟動時先跑幾次，讓 CUDA kernel 與 cuDNN 初始化完成，第一個真實請求才不會被冷啟動拖慢。"""
        t0 = time.perf_counter()
        img = PIL.Image.fromarray((np.random.rand(480, 640, 3) * 255).astype("uint8"))
        self.seg.set_prompts(["an object"])
        for _ in range(n):
            self.seg(img, 0.01)
            self.seg.sam.set_image(np.asarray(img))
            self.seg.sam.predict(box=np.array([100, 100, 300, 300]), multimask_output=False)  # 確保 SAM decoder 也暖機
        self.seg.prompts, self.seg.text_encodings = [], None
        torch.cuda.reset_peak_memory_stats()
        logger.info(f"[SegmentService.warmup]: input=(n={n}), output=done in {(time.perf_counter() - t0) * 1e3:.0f}ms")

    @log_call
    def decode_image(self, data: bytes) -> PIL.Image.Image:
        if len(data) > 30 * 2**20:
            raise ValueError("image too large (>30MB)")
        img = PIL.ImageOps.exif_transpose(PIL.Image.open(io.BytesIO(data))).convert("RGB")
        if max(img.size) > MAX_IMAGE_SIDE:
            img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
        return img

    @log_call
    def run(self, image: PIL.Image.Image, prompts: List[str], threshold: float, nms_iou: float,
            use_hyde: bool, contour_eps: float = 0.002, with_masks: bool = True) -> Dict:
        hyde_info = {"requested": use_hyde, "applied": False, "note": None, "expanded": None}
        if use_hyde:
            try:
                hyde_info["expanded"] = hyde.expand_prompts(prompts)
                hyde_info["applied"] = True
            except NotImplementedError as e:  # 預期內：已在 hyde.py 記錄 WARNING
                hyde_info["note"] = f"{e}，已改用原始提示"

        t_all = time.perf_counter()
        with self.lock:
            t_wait = time.perf_counter()
            if self.seg.prompts != prompts:
                t = time.perf_counter()
                self.seg.set_prompts(prompts)
                torch.cuda.synchronize()
                text_ms = (time.perf_counter() - t) * 1e3
            else:
                text_ms = 0.0
            dets = self.seg(image, threshold, nms_iou, with_masks)
            timing = dict(self.seg.timing)
        lock_wait_ms = (t_wait - t_all) * 1e3

        t = time.perf_counter()
        out_dets = []
        for i, d in enumerate(dets):
            x0, y0, x1, y1 = [float(v) for v in d.box]
            mask_area = d.area if d.mask is not None else None
            out_dets.append({
                "id": i, "label": d.label, "score": d.score, "box": [x0, y0, x1, y1],
                "box_area": int(max(x1 - x0, 0) * max(y1 - y0, 0)), "mask_area": mask_area,
                "contours": [c.astype(int).tolist() for c in d.contours(contour_eps)],
            })
        summary = []
        for p in prompts:
            mine = [d for d in out_dets if d["label"] == p]
            summary.append({"label": p, "count": len(mine), "mask_area": sum(d["mask_area"] for d in mine) if with_masks else None,
                            "box_area": sum(d["box_area"] for d in mine),
                            "mean_score": float(np.mean([d["score"] for d in mine])) if mine else None})
        serialize_ms = (time.perf_counter() - t) * 1e3

        timing.update({"text_encode_ms": text_ms, "lock_wait_ms": lock_wait_ms, "serialize_ms": serialize_ms,
                       "total_ms": (time.perf_counter() - t_all) * 1e3})
        self.n_requests += 1
        result = {"image": {"width": image.width, "height": image.height}, "prompts": prompts, "threshold": threshold,
                  "nms_iou": nms_iou, "with_masks": with_masks, "hyde": hyde_info, "detections": out_dets, "summary": summary,
                  "timing": timing, "backend": self.backend, "memory": self.gpu_stats()}
        self.history.appendleft({"time": time.strftime("%H:%M:%S"), "prompts": prompts, "n": len(out_dets),
                                 "total_ms": timing["total_ms"], "owl_ms": timing["owl_ms"], "sam_ms": timing["sam_ms"]})
        return result

    def gpu_stats(self) -> Dict:
        s = {"device": torch.cuda.get_device_name(0), "torch_allocated_mb": torch.cuda.memory_allocated() / 2**20,
             "torch_peak_mb": torch.cuda.max_memory_allocated() / 2**20}
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,temperature.gpu,utilization.gpu,clocks.sm,power.draw",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=2).stdout.strip().split(",")
            used, total, temp, util, clk, pw = [x.strip() for x in out]
            s.update({"gpu_used_mb": float(used), "gpu_total_mb": float(total), "temp_c": float(temp),
                      "util_pct": float(util), "sm_clock_mhz": float(clk), "power_w": None if pw in ("[N/A]", "N/A") else float(pw)})
        except Exception as e:  # nvidia-smi 不可用時（例如 Jetson）略過
            s["nvidia_smi_error"] = str(e)
        return s
