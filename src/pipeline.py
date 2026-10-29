"""文字提示 → NanoOWL 物件框 → SAM 遮罩 的串接（WSL：PyTorch 後端）

Jetson 的 TensorRT 後端：OWL 以 OwlPredictor(image_encoder_engine=...) 切換；SAM 之後在 __init__ 內替換。
"""
import time
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
import PIL.Image
import torch
from torchvision.ops import nms

from applog import logger, summarize
from nanoowl.owl_predictor import OwlPredictor
from nanosam.mobile_sam import SamPredictor, sam_model_registry


@dataclass
class Detection:
    label: str
    score: float
    box: np.ndarray   # xyxy（像素）
    mask: Optional[np.ndarray]  # bool, HxW；with_masks=False（只要框）時為 None

    def contours(self, epsilon_ratio=0.0):
        """遮罩 → 輪廓點列表（每個為 Nx2 的 xy 像素座標）。
        epsilon_ratio > 0 時用 Douglas-Peucker 簡化成多邊形（例如 0.005）。"""
        if self.mask is None:
            return []
        cs, _ = cv2.findContours(self.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cs = [c for c in cs if cv2.contourArea(c) > 20]  # 去掉雜點
        if epsilon_ratio > 0:
            cs = [cv2.approxPolyDP(c, epsilon_ratio * cv2.arcLength(c, True), True) for c in cs]
        return [c[:, 0, :] for c in cs]

    @property
    def area(self):
        return 0 if self.mask is None else int(self.mask.sum())


class OpenVocabSegmenter:
    def __init__(self, sam_checkpoint="data/models/mobile_sam.pt", owl_name="google/owlvit-base-patch32",
                 owl_engine=None, device="cuda"):
        t0 = time.perf_counter()
        try:
            self.owl = OwlPredictor(owl_name, device=device, image_encoder_engine=owl_engine)
            sam = sam_model_registry["vit_t"](checkpoint=sam_checkpoint).to(device).eval()
            self.sam = SamPredictor(sam)
        except Exception as e:
            logger.error(f"[OpenVocabSegmenter.__init__]: input=(owl={owl_name}, engine={owl_engine}, sam={sam_checkpoint}), "
                         f"error={type(e).__name__}: {e}", exc_info=True)
            raise
        self.prompts: List[str] = []
        self.text_encodings = None
        self.timing = {}
        logger.info(f"[OpenVocabSegmenter.__init__]: input=(owl={owl_name}, engine={owl_engine}, sam={sam_checkpoint}, device={device}), "
                    f"output=models loaded in {(time.perf_counter() - t0):.2f}s")

    def set_prompts(self, prompts: List[str]):
        """可在執行中隨時呼叫；只重算文字編碼，不需重啟。"""
        t0 = time.perf_counter()
        self.prompts = list(prompts)
        with torch.no_grad():
            self.text_encodings = self.owl.encode_text(self.prompts)
        torch.cuda.synchronize()
        logger.info(f"[OpenVocabSegmenter.set_prompts]: input=(prompts={self.prompts}), "
                    f"output=text_encodings{tuple(self.text_encodings.text_embeds.shape)}, elapsed={(time.perf_counter() - t0) * 1e3:.1f}ms")

    @torch.no_grad()
    def __call__(self, image: PIL.Image.Image, threshold=0.1, nms_iou=0.5, with_masks: bool = True) -> List[Detection]:
        logger.info(f"[OpenVocabSegmenter.__call__]: input=(image={summarize(image)}, prompts={self.prompts}, "
                    f"threshold={threshold}, nms_iou={nms_iou}, with_masks={with_masks})")
        t0 = time.perf_counter()
        try:
            # 1. OWL：文字 + 影像 → 候選框
            out = self.owl.predict(image=image, text=self.prompts, text_encodings=self.text_encodings,
                                   threshold=threshold, pad_square=False)
            boxes, scores, labels = out.boxes, out.scores, out.labels
            n_raw = len(boxes)
            logger.info(f"[OwlPredictor.predict]: input=(image={image.size}, threshold={threshold}), "
                        f"output={n_raw} raw boxes, labels={sorted(set(self.prompts[i] for i in labels.tolist()))}")

            # 2. NMS：OWL 對同一物件常輸出多個重疊框
            if n_raw:
                keep = nms(boxes, scores, nms_iou)
                boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            logger.info(f"[torchvision.nms]: input=({n_raw} boxes, iou={nms_iou}), output={len(boxes)} boxes kept")
            torch.cuda.synchronize()
            t1 = time.perf_counter()

            # 3. SAM：影像編碼一次，之後每個框跑 decoder
            dets = []
            if len(boxes) and not with_masks:  # 只要框（計數用）：跳過 SAM，約快 2.5 倍
                dets = [Detection(self.prompts[l], float(sc), b, None)
                        for b, sc, l in zip(boxes.cpu().numpy(), scores.cpu().numpy(), labels.cpu().numpy())]
                logger.info(f"[OpenVocabSegmenter.__call__]: with_masks=False, output={len(dets)} boxes, SAM skipped")
            elif len(boxes):
                self.sam.set_image(np.asarray(image.convert("RGB")))
                logger.info(f"[SamPredictor.set_image]: input=(image={image.size}), output=image embedding ready, "
                            f"elapsed={(time.perf_counter() - t1) * 1e3:.1f}ms")
                for box, score, label in zip(boxes.cpu().numpy(), scores.cpu().numpy(), labels.cpu().numpy()):
                    masks, iou_pred, _ = self.sam.predict(box=box, multimask_output=False)
                    d = Detection(self.prompts[label], float(score), box, masks[0])
                    dets.append(d)
                    logger.info(f"[SamPredictor.predict]: input=(label={d.label}, score={d.score:.2f}, box={box.round(0).astype(int).tolist()}), "
                                f"output=mask_area={d.area}px, sam_iou_pred={float(iou_pred[0]):.2f}")
            torch.cuda.synchronize()
        except Exception as e:
            logger.error(f"[OpenVocabSegmenter.__call__]: input=(image={image.size}, prompts={self.prompts}), "
                         f"error={type(e).__name__}: {e}", exc_info=True)
            raise
        self.timing = {"owl_ms": (t1 - t0) * 1e3, "sam_ms": (time.perf_counter() - t1) * 1e3, "num_raw_boxes": n_raw}
        logger.info(f"[OpenVocabSegmenter.__call__]: output={len(dets)} detections, "
                    f"owl={self.timing['owl_ms']:.1f}ms, sam={self.timing['sam_ms']:.1f}ms")
        return dets
