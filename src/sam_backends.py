"""NanoSAM 的推論後端（image encoder + mask decoder 都是 ONNX 匯出的模型）：TensorRT（Jetson）或 ONNX Runtime（WSL 驗證用）。

目的：Jetson 上用 `03_build_engines.sh` 建好的引擎，要能被 pipeline 載入；WSL 沒有 TensorRT，所以另提供 ONNX Runtime 後端，
用「同一份」前處理 / 後處理驗證邏輯與 ONNX 是否正確（TensorRT 本身只能在 Jetson 實測）。

對外介面與 `nanosam.mobile_sam.SamPredictor` 的用法一致，pipeline 不必區分：
    p = NanoSamPredictor(encoder, decoder, backend="trt" | "onnx")
    p.set_image(rgb_uint8_HxWx3)
    masks, iou, low_res = p.predict(box=xyxy, multimask_output=False)    # masks: (1,H,W) bool；iou: (1,)

做法沿用官方 `nanosam/utils/predictor.py`：
  * 前處理：長邊縮到 1024、保持長寬比，**正規化之後**在右下補 0（官方做法）
  * 框 → 兩個提示點：左上角 label=2、右下角 label=3（SAM 的 box 編碼）；座標乘 1024/長邊
  * decoder 輸出 4 個遮罩，取第 0 個（= multimask_output=False 時 SAM 採用的那一個）；
    裁掉補 0 的部分後雙線性放大到原圖大小，logit > 0 為前景
"""
import time

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F

from applog import logger

RESAMPLE = PIL.Image.BILINEAR       # SamPredictor 的 ResizeLongestSide 用雙線性；官方 NanoSAM 用 PIL 預設（雙三次）。差異見 tests/check_nanosam_backend.py
MEAN = torch.tensor([123.675, 116.28, 103.53])[:, None, None]
STD = torch.tensor([58.395, 57.12, 57.375])[:, None, None]


class _OrtBackend:
    """ONNX Runtime（有 CUDA provider 就用 GPU，否則 CPU）。回傳 torch tensor（CPU）。"""
    name = "onnx"

    def __init__(self, encoder, decoder):
        import onnxruntime as ort
        prov = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
        self.enc = ort.InferenceSession(encoder, providers=prov)
        self.dec = ort.InferenceSession(decoder, providers=prov)
        self.provider = self.enc.get_providers()[0]
        self.device = torch.device("cpu")

    def encode(self, x):
        return self.enc.run(None, {"image": x.numpy()})[0]

    def decode(self, feats, coords, labels):
        iou, low = self.dec.run(None, {"image_embeddings": feats, "point_coords": coords, "point_labels": labels,
                                       "mask_input": np.zeros((1, 1, 256, 256), np.float32), "has_mask_input": np.zeros(1, np.float32)})
        return torch.from_numpy(iou), torch.from_numpy(low)


class _TrtBackend:
    """TensorRT 引擎（torch2trt 的 TRTModule，與官方 Predictor 相同的載入方式）。需在 Jetson（或有 TensorRT 的機器）上使用。"""
    name = "trt"

    def __init__(self, encoder, decoder):
        from nanosam.utils.predictor import load_image_encoder_engine, load_mask_decoder_engine
        self.enc = load_image_encoder_engine(encoder)
        self.dec = load_mask_decoder_engine(decoder)
        self.provider = "TensorRT"
        self.device = torch.device("cuda")

    def encode(self, x):
        return self.enc(x.to(self.device))

    def decode(self, feats, coords, labels):
        z = torch.zeros(1, 1, 256, 256, device=self.device)
        iou, low = self.dec(feats, torch.from_numpy(coords).to(self.device), torch.from_numpy(labels).to(self.device), z,
                            torch.zeros(1, device=self.device))
        return iou, low


class NanoSamPredictor:
    def __init__(self, encoder: str, decoder: str, backend: str = "auto", size: int = 1024):
        if backend == "auto":
            backend = "trt" if encoder.endswith(".engine") else "onnx"
        if backend not in ("trt", "onnx"):
            raise ValueError(f"backend 必須是 trt / onnx / auto，收到 {backend!r}")
        t0 = time.perf_counter()
        try:
            self.be = (_TrtBackend if backend == "trt" else _OrtBackend)(encoder, decoder)
        except Exception as e:
            logger.error(f"[NanoSamPredictor.__init__]: input=(backend={backend}, encoder={encoder}, decoder={decoder}), "
                         f"error={type(e).__name__}: {e}", exc_info=True)
            raise
        self.size, self.backend, self.encoder, self.decoder = size, backend, encoder, decoder
        self.feats = None
        self.image_size = None
        logger.info(f"[NanoSamPredictor.__init__]: input=(backend={backend}, encoder={encoder}, decoder={decoder}), "
                    f"output=provider={self.be.provider}, elapsed={(time.perf_counter() - t0) * 1e3:.0f}ms")

    @property
    def label(self):
        return f"NanoSAM（{'TensorRT' if self.backend == 'trt' else 'ONNX Runtime'}）"

    def reset_image(self):
        self.feats = None

    def _preprocess(self, img: np.ndarray) -> torch.Tensor:
        im = PIL.Image.fromarray(img)
        ar = im.width / im.height
        rw, rh = (self.size, int(self.size / ar)) if ar >= 1 else (int(self.size * ar), self.size)
        t = torch.from_numpy(np.asarray(im.resize((rw, rh), RESAMPLE))).permute(2, 0, 1).float()
        x = torch.zeros((1, 3, self.size, self.size))
        x[0, :, :rh, :rw] = (t - MEAN) / STD
        return x

    def set_image(self, img: np.ndarray):
        h, w = img.shape[:2]
        self.image_size = (h, w)
        self.feats = self.be.encode(self._preprocess(img))

    def predict(self, box, multimask_output=False):
        if self.feats is None:
            raise RuntimeError("先呼叫 set_image")
        h, w = self.image_size
        s = self.size / max(h, w)
        b = np.asarray(box, np.float32).reshape(4)
        coords = (np.array([[[b[0], b[1]], [b[2], b[3]]]], np.float32)) * s          # 左上 / 右下
        labels = np.array([[2, 3]], np.float32)
        iou, low = self.be.decode(self.feats, coords, labels)
        low = low[:, :1].to(self.be.device)                                          # 第 0 個遮罩
        q = self.size // 4                                                           # low-res 遮罩是 256×256（= 1024/4），內容靠左上、右下是補 0 的區域
        lim_x, lim_y = (q, int(q * h / w)) if w > h else (int(q * w / h), q)         # 與官方 upscale_mask 相同
        mask = F.interpolate(low[:, :, :lim_y, :lim_x], (h, w), mode="bilinear")[0] > 0
        return mask.cpu().numpy(), iou[0, :1].float().cpu().numpy(), low[0].cpu().numpy()
