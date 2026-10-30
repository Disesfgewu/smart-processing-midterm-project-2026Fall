"""驗證 NanoSAM 後端（ONNX Runtime）的前後處理：同一張圖、同樣的框，與 PyTorch 的自訓練 encoder + MobileSAM decoder 比遮罩。
    python tests/check_nanosam_backend.py [影像] [--ckpt data/eval/colab80_best.pth]
TensorRT 後端與此共用同一份前後處理，只是 encode / decode 換成引擎；TensorRT 本身只能在 Jetson 實測。
通過條件：每個框的遮罩 IoU ≥ 0.95（ONNX 與 PyTorch 是同一個網路，只差影像縮放的插值方式與數值誤差）。
"""
import argparse
import sys

import numpy as np
import PIL.Image

sys.path.insert(0, "src")
from pipeline import OpenVocabSegmenter  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("image", nargs="?", default="data/samples/cats.jpg")
ap.add_argument("--ckpt", default="data/eval/colab80_best.pth")
ap.add_argument("--encoder", default="data/models/resnet18_image_encoder.selftrain.onnx")
ap.add_argument("--decoder", default="data/models/mobile_sam_mask_decoder.onnx")
a = ap.parse_args()

seg = OpenVocabSegmenter(student_checkpoint=a.ckpt, nanosam_encoder=a.encoder, nanosam_decoder=a.decoder, nanosam_backend="onnx")
img = np.asarray(PIL.Image.open(a.image).convert("RGB")); h, w = img.shape[:2]
print(f"影像 {a.image} {w}x{h}；ONNX provider = {seg.sam_encoders['nanosam'].be.provider}")
# 幾個不同大小 / 位置的框（含貼邊、很小的）
boxes = [np.array(b, np.float32) for b in ([0.05 * w, 0.1 * h, 0.55 * w, 0.95 * h], [0.5 * w, 0.2 * h, 0.95 * w, 0.8 * h],
                                           [0.4 * w, 0.4 * h, 0.5 * w, 0.55 * h], [0, 0, 0.3 * w, 0.3 * h], [0.7 * w, 0.7 * h, w - 1, h - 1])]
res = {}
for name in ("student", "nanosam"):
    seg.set_sam_encoder(name); seg.sam.set_image(img)
    res[name] = [seg.sam.predict(box=b, multimask_output=False) for b in boxes]
ok = True
for i, b in enumerate(boxes):
    (m0, i0, _), (m1, i1, _) = res["student"][i], res["nanosam"][i]
    assert m0.shape == m1.shape == (1, h, w), (m0.shape, m1.shape)
    u = np.logical_or(m0, m1).sum(); iou = np.logical_and(m0, m1).sum() / u if u else 1.0
    good = iou >= 0.95; ok &= bool(good)
    print(f"  框{i} {b.round(0).astype(int).tolist()}  PyTorch {int(m0.sum()):7d}px  ONNX {int(m1.sum()):7d}px  遮罩IoU {iou:.4f}  sam_iou {float(i0[0]):.3f}/{float(i1[0]):.3f}  {'OK' if good else 'FAIL'}")
print("→ 通過" if ok else "→ 不一致：請檢查前處理（補 0 位置 / 正規化）、點座標縮放、遮罩裁切")
sys.exit(0 if ok else 1)
