"""伺服器端來源（攝影機編號、影片檔路徑、GStreamer 管線）→ 持續推論 → MJPEG 串流。

用途：Jetson 接 USB / CSI 攝影機時，由後端直接讀取；瀏覽器以 <img src="/api/camera/stream"> 觀看。
來源字串：  "0"            USB 攝影機（/dev/video0）
            "path/to.mp4"  影片檔（播到結尾會自動重頭）
            "gst:<pipeline>" GStreamer 管線（例如 CSI：nvarguscamerasrc ! ... ! appsink）
"""
import threading
import time
from typing import Dict, List, Optional

import cv2
import PIL.Image

from applog import logger, quiet_var
from overlay import draw_overlay


class CameraWorker:
    def __init__(self, service):
        self.service = service
        self.thread: Optional[threading.Thread] = None
        self.stop_evt = threading.Event()
        self.cond = threading.Condition()
        self.jpeg: Optional[bytes] = None
        self.frame_id = 0
        self.params_lock = threading.Lock()
        self.params: Dict = {}
        self.state: Dict = {"running": False, "error": None}
        self.hyde_info: Dict = {"requested": False}   # 最近一次 HyDE 擴展結果；每幀的 state 都會帶上

    @property
    def running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    @staticmethod
    def _open(source: str):
        if source.startswith("gst:"):
            cap = cv2.VideoCapture(source[4:], cv2.CAP_GSTREAMER)
        elif source.isdigit():
            cap = cv2.VideoCapture(int(source), cv2.CAP_V4L2)
        else:
            cap = cv2.VideoCapture(source)
        return cap

    def start(self, source: str, prompts: List[str], threshold: float, nms_iou: float, style: str, max_side: int, with_masks: bool = True, groups=None, top_per_group: bool = False):
        if self.running:
            raise RuntimeError("camera already running")
        cap = self._open(source)
        if not cap.isOpened():
            logger.error(f"[CameraWorker.start]: input=(source={source}), error=cannot open source")
            raise ValueError(f"cannot open source: {source}")
        is_file = not (source.isdigit() or source.startswith("gst:"))
        self.set_params(prompts=prompts, threshold=threshold, nms_iou=nms_iou, style=style, expected=None, with_masks=with_masks, groups=groups or {}, top_per_group=top_per_group)
        self.stop_evt.clear()
        self.frame_id, self.jpeg = 0, None
        self.state = {"running": True, "error": None, "source": source, "fps": 0.0, "frame_id": 0}
        self.thread = threading.Thread(target=self._loop, args=(cap, source, is_file, max_side), daemon=True, name="camera")
        self.thread.start()
        logger.info(f"[CameraWorker.start]: input=(source={source}, prompts={prompts}, threshold={threshold}, max_side={max_side}), "
                    f"output=started, is_file={is_file}")

    def stop(self):
        was = self.running
        self.stop_evt.set()
        if self.thread:
            self.thread.join(timeout=5)
        self.state["running"] = False
        with self.cond:
            self.cond.notify_all()
        logger.info(f"[CameraWorker.stop]: input=(), output=stopped (was_running={was})")

    def set_params(self, **kw):
        with self.params_lock:
            self.params.update({k: v for k, v in kw.items()})

    def set_prompts(self, prompts: List[str], groups=None):
        self.set_params(prompts=prompts, groups=groups or {})
        logger.info(f"[CameraWorker.set_prompts]: input=(prompts={prompts}), output=queued, applied on next frame")

    def _loop(self, cap, source, is_file, max_side):
        seg, svc = self.service.seg, self.service
        fps, fails, n = 0.0, 0, 0
        try:
            while not self.stop_evt.is_set():
                t0 = time.perf_counter()
                ok, bgr = cap.read()
                if not ok:
                    if is_file:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        fails += 1
                        if fails > 3:  # 連續讀不到（空檔案）
                            raise RuntimeError("source returns no frames")
                        continue
                    fails += 1
                    if fails > 30:
                        raise RuntimeError("camera returns no frames")
                    time.sleep(0.03)
                    continue
                fails = 0
                h, w = bgr.shape[:2]
                if max(h, w) > max_side:
                    s = max_side / max(h, w)
                    bgr = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
                with self.params_lock:
                    p = dict(self.params)
                prompts = p["prompts"]
                # 前 3 幀與每 30 幀完整記錄，其餘只記 WARNING 以上，避免洗版
                quiet_var.set(not (n < 3 or n % 30 == 0))
                with svc.lock:
                    groups = p.get("groups") or {}
                    if seg.prompts != prompts or seg.groups != groups:
                        seg.set_prompts(prompts, groups)
                    dets = seg(PIL.Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), p["threshold"], p["nms_iou"], p.get("with_masks", True), p.get("top_per_group", False))
                    timing = dict(seg.timing)
                dt = time.perf_counter() - t0
                fps = 1 / dt if n == 0 else 0.85 * fps + 0.15 / dt
                vis = draw_overlay(bgr, dets, prompts, timing, fps, p["style"], p.get("expected"))
                ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 80])
                summary = []
                for pr in prompts:
                    mine = [d for d in dets if d.label == pr]
                    summary.append({"label": pr, "count": len(mine), "mask_area": int(sum(d.area for d in mine)) if p.get("with_masks", True) else None,
                                    "box_area": int(sum((d.box[2] - d.box[0]) * (d.box[3] - d.box[1]) for d in mine)),
                                    "mean_score": float(sum(d.score for d in mine) / len(mine)) if mine else None,
                                    "best_score": seg.best_scores.get(pr), "queries": (p.get("groups") or {}).get(pr) or [pr], "primary": seg.primary.get(pr),
                                    "mean_sam_iou": float(sum(d.sam_iou for d in mine) / len(mine)) if mine and p.get("with_masks", True) else None})
                with self.cond:
                    self.jpeg = buf.tobytes()
                    self.frame_id += 1
                    self.state = {"running": True, "error": None, "source": source, "fps": fps, "frame_id": self.frame_id, "hyde": self.hyde_info,
                                  "image": {"width": vis.shape[1], "height": vis.shape[0]}, "prompts": prompts,
                                  "summary": summary, "timing": {**timing, "total_ms": dt * 1e3, "text_encode_ms": 0, "serialize_ms": 0},
                                  "detections": [{"id": i, "label": d.label, "score": d.score, "box": [float(v) for v in d.box],
                                                  "box_area": int((d.box[2] - d.box[0]) * (d.box[3] - d.box[1])), "mask_area": d.area if d.mask is not None else None, "sam_iou": d.sam_iou, "query": d.query, "primary": d.primary, "contours": []}
                                                 for i, d in enumerate(dets)]}
                    self.cond.notify_all()
                n += 1
                quiet_var.set(False)
                if n % 30 == 0:
                    logger.info(f"[CameraWorker._loop]: input=(source={source}), output=frame {n}, fps={fps:.1f}, detections={len(dets)}")
        except Exception as e:
            logger.error(f"[CameraWorker._loop]: input=(source={source}), error={type(e).__name__}: {e}", exc_info=True)
            self.state = {**self.state, "running": False, "error": f"{type(e).__name__}: {e}"}
        finally:
            cap.release()
            self.state["running"] = False
            with self.cond:
                self.cond.notify_all()
            logger.info(f"[CameraWorker._loop]: input=(source={source}), output=loop ended after {n} frames")

    def wait_frame(self, last_id: int, timeout: float = 2.0):
        """回傳 (jpeg, frame_id)；沒有新幀時回傳 (None, last_id)。"""
        with self.cond:
            if self.frame_id == last_id and self.running:
                self.cond.wait(timeout)
            return (self.jpeg, self.frame_id) if self.frame_id != last_id else (None, last_id)
