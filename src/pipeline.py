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
    sam_iou: Optional[float] = None  # SAM 對這個遮罩自評的 IoU（品質預測，0~1）
    query: Optional[str] = None      # 實際命中的 OWL 查詢（HyDE 時為某個描述；否則等於 label）
    primary: bool = False            # 是否為該組（概念）內信心最高的偵測 = 主要輸出

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
                 owl_engine=None, device="cuda", student_checkpoint=None):
        """student_checkpoint：自行蒸餾的 ResNet18 image encoder（training/train_nanosam.py 的 best.pth / checkpoint.pth）。
        給了就和 baseline（MobileSAM 的 TinyViT encoder）一起載入、可用 set_sam_encoder() 即時切換比較；
        prompt encoder / mask decoder 一律用 baseline 的（NanoSAM 本來就共用同一個 decoder）。baseline 權重不會被修改。"""
        t0 = time.perf_counter()
        try:
            self.owl = OwlPredictor(owl_name, device=device, image_encoder_engine=owl_engine)
            sam = sam_model_registry["vit_t"](checkpoint=sam_checkpoint).to(device).eval()
            self.sam = SamPredictor(sam)
            self.sam_encoders = {"mobilesam": sam.image_encoder}               # baseline
            self.sam_encoder = "mobilesam"
            if student_checkpoint:
                from nanosam.models.timm_image_encoder import TimmImageEncoder
                stu = TimmImageEncoder("resnet18")                              # 預設 pretrained=False；權重由 checkpoint 載入，離線也能用
                ck = torch.load(student_checkpoint, map_location="cpu")
                stu.load_state_dict(ck["model"])
                stu.img_size = sam.image_encoder.img_size                       # SamPredictor 需要此屬性（1024）
                self.sam_encoders["student"] = stu.to(device).eval()
                self.student_info = {"checkpoint": student_checkpoint, "epoch": ck.get("epoch"), "val_loss": ck.get("val_loss")}
        except Exception as e:
            logger.error(f"[OpenVocabSegmenter.__init__]: input=(owl={owl_name}, engine={owl_engine}, sam={sam_checkpoint}), "
                         f"error={type(e).__name__}: {e}", exc_info=True)
            raise
        self.prompts: List[str] = []      # 概念（使用者輸入的提示詞）
        self.groups: dict = {}            # 概念 → [OWL 查詢描述...]（HyDE）；沒有則查詢就是概念本身
        self.queries: List[str] = []      # 實際送進 OWL 的查詢（所有概念的描述展平）
        self.query_concept = None         # 查詢索引 → 概念索引
        self.query_scores = {}
        self.primary = {}                 # 概念 → {query, score}：組內信心最高的偵測
        self.text_encodings = None
        self.timing = {}
        self.best_scores = {}  # 每個提示詞的最高分（含低於門檻者），診斷「為什麼沒偵測到」用
        logger.info(f"[OpenVocabSegmenter.__init__]: input=(owl={owl_name}, engine={owl_engine}, sam={sam_checkpoint}, device={device}), "
                    f"output=models loaded in {(time.perf_counter() - t0):.2f}s")

    def set_sam_encoder(self, name: str):
        """切換 SAM 的 image encoder（'mobilesam' = baseline；'student' = 自訓練 ResNet18）。呼叫端需自行避免與推論同時進行。"""
        if name not in self.sam_encoders:
            raise ValueError(f"unknown SAM encoder {name!r}; available: {list(self.sam_encoders)}")
        self.sam.model.image_encoder = self.sam_encoders[name]
        self.sam_encoder = name
        self.sam.reset_image()
        logger.info(f"[OpenVocabSegmenter.set_sam_encoder]: input=(name={name}), output=active encoder = {type(self.sam_encoders[name]).__name__}")

    def set_prompts(self, prompts: List[str], groups: Optional[dict] = None):
        """可在執行中隨時呼叫；只重算文字編碼，不需重啟。
        groups（HyDE）：{概念: [描述...]}，同一物件在各描述中取最高分、結果歸回概念。"""
        t0 = time.perf_counter()
        self.prompts = list(prompts)
        self.groups = {k: list(v) for k, v in (groups or {}).items() if k in self.prompts}
        queries, q2c = [], []
        for ci, p in enumerate(self.prompts):
            for v in (self.groups.get(p) or [p]):
                if v not in queries:
                    queries.append(v); q2c.append(ci)
        self.queries = queries
        self.query_concept = torch.tensor(q2c, dtype=torch.long, device=self.owl.device)
        with torch.no_grad():
            self.text_encodings = self.owl.encode_text(self.queries)
        torch.cuda.synchronize()
        logger.info(f"[OpenVocabSegmenter.set_prompts]: input=(prompts={self.prompts}, groups={self.groups}), "
                    f"output=text_encodings{tuple(self.text_encodings.text_embeds.shape)} for {len(self.queries)} queries, "
                    f"elapsed={(time.perf_counter() - t0) * 1e3:.1f}ms")

    @torch.no_grad()
    def __call__(self, image: PIL.Image.Image, threshold=0.1, nms_iou=0.5, with_masks: bool = True, top_per_group: bool = False) -> List[Detection]:
        logger.info(f"[OpenVocabSegmenter.__call__]: input=(image={summarize(image)}, prompts={self.prompts}, "
                    f"threshold={threshold}, nms_iou={nms_iou}, with_masks={with_masks}, top_per_group={top_per_group})")
        t0 = time.perf_counter()
        try:
            # 1. OWL：文字 + 影像 → 候選框
            # 以極低門檻取得所有候選，先記下各提示詞的最高分，再依使用者門檻過濾
            out = self.owl.predict(image=image, text=self.queries, text_encodings=self.text_encodings,
                                   threshold=min(threshold, 0.01), pad_square=False)
            boxes, scores, qlabels = out.boxes, out.scores, out.labels
            labels = self.query_concept[qlabels] if len(qlabels) else qlabels     # 查詢索引 → 概念索引
            self.query_scores = {q: (float(scores[qlabels == i].max()) if bool((qlabels == i).any()) else None)
                                 for i, q in enumerate(self.queries)}
            self.best_scores = {p: (float(scores[labels == i].max()) if bool((labels == i).any()) else None)
                                for i, p in enumerate(self.prompts)}
            keep_thr = scores >= threshold
            boxes, scores, labels, qlabels = boxes[keep_thr], scores[keep_thr], labels[keep_thr], qlabels[keep_thr]
            n_raw = len(boxes)
            logger.info(f"[OwlPredictor.predict]: input=(image={image.size}, threshold={threshold}, queries={len(self.queries)}), "
                        f"output={n_raw} raw boxes, labels={sorted(set(self.prompts[i] for i in labels.tolist()))}, "
                        f"best_scores={ {k: (round(v, 3) if v is not None else None) for k, v in self.best_scores.items()} }")

            # 2. NMS：OWL 對同一物件常輸出多個重疊框
            if n_raw:
                keep = nms(boxes, scores, nms_iou)
                boxes, scores, labels, qlabels = boxes[keep], scores[keep], labels[keep], qlabels[keep]
            logger.info(f"[torchvision.nms]: input=({n_raw} boxes, iou={nms_iou}), output={len(boxes)} boxes kept")

            # 2b. 組內信心最高者 = 主要輸出（一個概念的所有描述視為一組）；top_per_group 時只留這一個
            lab_np, sc_np = labels.cpu().numpy(), scores.cpu().numpy()
            is_primary = np.zeros(len(lab_np), dtype=bool)
            self.primary = {}
            for ci, pname in enumerate(self.prompts):
                idx = np.where(lab_np == ci)[0]
                if len(idx):
                    j = int(idx[sc_np[idx].argmax()]); is_primary[j] = True
                    self.primary[pname] = {"query": self.queries[int(qlabels[j])], "score": float(sc_np[j])}
            logger.info(f"[OpenVocabSegmenter.primary]: input=(groups={len(self.prompts)}, detections={len(lab_np)}), output={self.primary}")
            if top_per_group and len(lab_np):
                kk = torch.as_tensor(np.where(is_primary)[0], device=boxes.device)
                boxes, scores, labels, qlabels = boxes[kk], scores[kk], labels[kk], qlabels[kk]
                is_primary = np.ones(len(kk), dtype=bool)
            torch.cuda.synchronize()
            t1 = time.perf_counter()

            # 3. SAM：影像編碼一次，之後每個框跑 decoder
            dets = []
            if len(boxes) and not with_masks:  # 只要框（計數用）：跳過 SAM，約快 2.5 倍
                dets = [Detection(self.prompts[l], float(sc), b, None, None, self.queries[q], bool(pr))
                        for b, sc, l, q, pr in zip(boxes.cpu().numpy(), scores.cpu().numpy(), labels.cpu().numpy(), qlabels.cpu().numpy(), is_primary)]
                logger.info(f"[OpenVocabSegmenter.__call__]: with_masks=False, output={len(dets)} boxes, SAM skipped")
            elif len(boxes):
                self.sam.set_image(np.asarray(image.convert("RGB")))
                logger.info(f"[SamPredictor.set_image]: input=(image={image.size}), output=image embedding ready, "
                            f"elapsed={(time.perf_counter() - t1) * 1e3:.1f}ms")
                for box, score, label, ql, pr in zip(boxes.cpu().numpy(), scores.cpu().numpy(), labels.cpu().numpy(), qlabels.cpu().numpy(), is_primary):
                    masks, iou_pred, _ = self.sam.predict(box=box, multimask_output=False)
                    d = Detection(self.prompts[label], float(score), box, masks[0], float(iou_pred[0]), self.queries[ql], bool(pr))
                    dets.append(d)
                    logger.info(f"[SamPredictor.predict]: input=(label={d.label}, query={d.query!r}, score={d.score:.2f}, box={box.round(0).astype(int).tolist()}), "
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
