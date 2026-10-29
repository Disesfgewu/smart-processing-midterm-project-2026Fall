"""文字提示 → NanoOWL 物件框 → SAM 遮罩 的串接（WSL：PyTorch 後端）

Jetson 的 TensorRT 後端：OWL 以 OwlPredictor(image_encoder_engine=...) 切換；SAM 之後在 __init__ 內替換。
"""
import time
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np
import PIL.Image
import torch
from torchvision.ops import nms

from nanoowl.owl_predictor import OwlPredictor
from nanosam.mobile_sam import SamPredictor, sam_model_registry


@dataclass
class Detection:
    label: str
    score: float
    box: np.ndarray   # xyxy（像素）
    mask: np.ndarray  # bool, HxW

    def contours(self, epsilon_ratio=0.0):
        """遮罩 → 輪廓點列表（每個為 Nx2 的 xy 像素座標）。
        epsilon_ratio > 0 時用 Douglas-Peucker 簡化成多邊形（例如 0.005）。"""
        cs, _ = cv2.findContours(self.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cs = [c for c in cs if cv2.contourArea(c) > 20]  # 去掉雜點
        if epsilon_ratio > 0:
            cs = [cv2.approxPolyDP(c, epsilon_ratio * cv2.arcLength(c, True), True) for c in cs]
        return [c[:, 0, :] for c in cs]

    @property
    def area(self):
        return int(self.mask.sum())


class OpenVocabSegmenter:
    def __init__(self, sam_checkpoint="data/models/mobile_sam.pt", owl_name="google/owlvit-base-patch32",
                 owl_engine=None, device="cuda"):
        self.owl = OwlPredictor(owl_name, device=device, image_encoder_engine=owl_engine)
        sam = sam_model_registry["vit_t"](checkpoint=sam_checkpoint).to(device).eval()
        self.sam = SamPredictor(sam)
        self.prompts: List[str] = []
        self.text_encodings = None
        self.timing = {}

    def set_prompts(self, prompts: List[str]):
        """可在執行中隨時呼叫；只重算文字編碼，不需重啟。"""
        self.prompts = list(prompts)
        with torch.no_grad():
            self.text_encodings = self.owl.encode_text(self.prompts)

    @torch.no_grad()
    def __call__(self, image: PIL.Image.Image, threshold=0.1, nms_iou=0.5) -> List[Detection]:
        t0 = time.perf_counter()
        out = self.owl.predict(image=image, text=self.prompts, text_encodings=self.text_encodings,
                               threshold=threshold, pad_square=False)
        boxes, scores, labels = out.boxes, out.scores, out.labels
        if len(boxes):
            keep = nms(boxes, scores, nms_iou)  # OWL 對同一物件常輸出多個重疊框
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        dets = []
        if len(boxes):
            self.sam.set_image(np.asarray(image.convert("RGB")))
            for box, score, label in zip(boxes.cpu().numpy(), scores.cpu().numpy(), labels.cpu().numpy()):
                masks, _, _ = self.sam.predict(box=box, multimask_output=False)
                dets.append(Detection(self.prompts[label], float(score), box, masks[0]))
        torch.cuda.synchronize()
        self.timing = {"owl_ms": (t1 - t0) * 1e3, "sam_ms": (time.perf_counter() - t1) * 1e3}
        return dets
