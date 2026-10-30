"""推論服務：持有模型、序列化 GPU 存取、記錄歷史。"""
import glob
import io
import os
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
    def __init__(self, sam_checkpoint="data/models/mobile_sam.pt", owl_engine: Optional[str] = None, student_checkpoint: Optional[str] = None,
                 nanosam_encoder: Optional[str] = None, nanosam_decoder: Optional[str] = None, nanosam_backend: str = "auto"):
        self.started = time.time()
        t0 = time.perf_counter()
        self.seg = OpenVocabSegmenter(sam_checkpoint=sam_checkpoint, owl_engine=owl_engine, student_checkpoint=student_checkpoint,
                                    nanosam_encoder=nanosam_encoder, nanosam_decoder=nanosam_decoder, nanosam_backend=nanosam_backend)
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
        img.info["original_size"] = img.size          # 縮圖前的尺寸（EXIF 轉正之後）；偵測結果的座標系是縮圖後的尺寸
        if max(img.size) > MAX_IMAGE_SIDE:
            img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
            logger.warning(f"[SegmentService.decode_image]: image downscaled {img.info['original_size']} -> {img.size} (MAX_IMAGE_SIDE={MAX_IMAGE_SIDE}); "
                           f"result coordinates are in the downscaled space, client must rescale")
        return img

    @log_call
    def run(self, image: PIL.Image.Image, prompts: List[str], threshold: float, nms_iou: float,
            use_hyde: bool, contour_eps: float = 0.002, with_masks: bool = True, top_per_group: bool = False) -> Dict:
        groups, hyde_info = self.hyde_groups(prompts, use_hyde)

        t_all = time.perf_counter()
        with self.lock:
            t_wait = time.perf_counter()
            if self.seg.prompts != prompts or self.seg.groups != groups:
                t = time.perf_counter()
                self.seg.set_prompts(prompts, groups)
                torch.cuda.synchronize()
                text_ms = (time.perf_counter() - t) * 1e3
            else:
                text_ms = 0.0
            dets = self.seg(image, threshold, nms_iou, with_masks, top_per_group)
            timing = dict(self.seg.timing)
        lock_wait_ms = (t_wait - t_all) * 1e3

        t = time.perf_counter()
        out_dets = []
        for i, d in enumerate(dets):
            x0, y0, x1, y1 = [float(v) for v in d.box]
            mask_area = d.area if d.mask is not None else None
            out_dets.append({
                "id": i, "label": d.label, "score": d.score, "box": [x0, y0, x1, y1],
                "box_area": int(max(x1 - x0, 0) * max(y1 - y0, 0)), "mask_area": mask_area, "sam_iou": d.sam_iou, "query": d.query, "primary": d.primary,
                "contours": [c.astype(int).tolist() for c in d.contours(contour_eps)],
            })
        summary = []
        for p in prompts:
            mine = [d for d in out_dets if d["label"] == p]
            summary.append({"label": p, "count": len(mine), "mask_area": sum(d["mask_area"] for d in mine) if with_masks else None,
                            "box_area": sum(d["box_area"] for d in mine),
                            "mean_score": float(np.mean([d["score"] for d in mine])) if mine else None,
                            "best_score": self.seg.best_scores.get(p),
                            "queries": groups.get(p) or [p], "primary": self.seg.primary.get(p),
                            "mean_sam_iou": float(np.mean([d["sam_iou"] for d in mine])) if mine and with_masks else None})
        serialize_ms = (time.perf_counter() - t) * 1e3

        timing.update({"text_encode_ms": text_ms, "lock_wait_ms": lock_wait_ms, "serialize_ms": serialize_ms,
                       "total_ms": (time.perf_counter() - t_all) * 1e3})
        self.n_requests += 1
        ow, oh = image.info.get("original_size", image.size)
        # image.width × height 是偵測結果（框 / 輪廓 / 面積）的座標系；original_* 是縮圖前的尺寸
        result = {"image": {"width": image.width, "height": image.height, "original_width": ow, "original_height": oh},
                  "prompts": prompts, "threshold": threshold, "nms_iou": nms_iou, "with_masks": with_masks, "top_per_group": top_per_group, "hyde": hyde_info, "detections": out_dets, "summary": summary,
                  "timing": timing, "backend": self.backend, "sam_encoder": self.seg.sam_encoder, "memory": self.gpu_stats()}
        self.history.appendleft({"time": time.strftime("%H:%M:%S"), "prompts": prompts, "n": len(out_dets),
                                 "total_ms": timing["total_ms"], "owl_ms": timing["owl_ms"], "sam_ms": timing["sam_ms"]})
        return result

    # ---- SAM image encoder：baseline + 自訓練的 checkpoint（可在網頁挑選，第一次選用時才載入）----
    ENCODER_GLOBS = ("data/eval/*.pth", "data/train/*/best.pth")           # 可挑選的位置（白名單；網頁只能選這裡掃到的檔案，不能指定任意路徑）

    def encoder_candidates(self) -> Dict[str, str]:
        """key（相對路徑）→ 絕對路徑。略過 _smoke 之類的測試輸出與標記為 INVALID 的作廢權重（fp16 教師訓練出的，結果是錯的）。"""
        out = {}
        for g in self.ENCODER_GLOBS:
            for f in sorted(glob.glob(g)):
                if "/_" not in f.replace("\\", "/") and "INVALID" not in f:
                    out[f.replace("\\", "/")] = os.path.abspath(f)
        return out

    def sam_encoder_info(self) -> Dict:
        cand, meta = self.encoder_candidates(), self.seg.sam_encoder_meta
        items = [{"key": "mobilesam", "label": "MobileSAM（baseline）", "loaded": True, "epoch": None, "val_loss": None}]
        loaded_paths = {os.path.abspath(m["checkpoint"]): k for k, m in meta.items() if k != "mobilesam"}
        for k, m in meta.items():                                               # 已載入的（含以 SAM_STUDENT 啟動的 "student"）
            if k != "mobilesam":
                items.append({"key": k, "label": m.get("label") or f"自訓練 {k if k != 'student' else m['checkpoint']}",
                              "loaded": True, "epoch": m["epoch"], "val_loss": m["val_loss"]})
        for k, path in cand.items():                                            # 掃到但還沒載入的
            if k not in meta and path not in loaded_paths:
                items.append({"key": k, "label": f"自訓練 {k}", "loaded": False, "epoch": None, "val_loss": None})
        return {"active": self.seg.sam_encoder, "encoders": items, "available": list(self.seg.sam_encoders), "student": getattr(self.seg, "student_info", None)}

    def switch_sam_encoder(self, name: str) -> Dict:
        with self.lock:                                   # 與推論互斥：不會在某一幀中途換 encoder
            if name not in self.seg.sam_encoders:
                cand = self.encoder_candidates()
                if name not in cand:
                    raise ValueError(f"unknown SAM encoder {name!r}")
                self.seg.add_sam_encoder(name, cand[name])          # 第一次選用：載入（約 1 秒、數十 MB 顯存）
            self.seg.set_sam_encoder(name)
        return self.sam_encoder_info()

    def token_count(self, text: str) -> int:
        return len(self.seg.owl.processor.tokenizer(text)["input_ids"])

    def hyde_expand(self, prompts: List[str]) -> Dict:
        """明確的擴展步驟（會呼叫 LLM）：只由 /api/hyde/expand 在開始偵測之前呼叫。"""
        return hyde.expand_prompts(prompts, token_counter=self.token_count)

    def hyde_groups(self, prompts: List[str], use_hyde: bool):
        """偵測路徑用：**只查快取，絕不呼叫 LLM**（LLM 要數秒，會卡住影片 / 攝影機迴圈）。
        回傳 (groups, hyde_info)。查不到快取時退回原始提示並在 hyde_info.note 說明，請求不會失敗也不會變慢。"""
        info = {"requested": use_hyde, "applied": False, "note": None, "expanded": None, "model": hyde.DEFAULT_MODEL, "elapsed_ms": 0, "cached": None}
        if not use_hyde:
            return {}, info
        groups = hyde.lookup_cached(prompts)
        if groups is None:
            logger.warning(f"[SegmentService.hyde_groups]: input=(prompts={prompts}), output=not expanded yet → original prompts (no LLM call on detection path)")
            info["note"] = "尚未擴展這組提示詞（請先按「擴展提示詞」；偵測期間不會呼叫 LLM）"
            return {}, info
        info.update(applied=True, expanded=groups, cached={c: True for c in prompts})
        return groups, info

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
