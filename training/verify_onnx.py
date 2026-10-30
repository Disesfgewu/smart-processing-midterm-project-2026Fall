"""檢查匯出的 ONNX 與 PyTorch checkpoint 的輸出一致（同一張 COCO 影像、同樣的前處理）。
    python training/verify_onnx.py --ckpt data/eval/colab80_best.pth --onnx data/models/resnet18_image_encoder.selftrain.onnx
只讀取檔案，不寫入任何東西。cosine 應 ≈ 1.0、最大差異應很小（fp32 約 1e-3 以下）。
"""
import argparse
import glob

import numpy as np
import onnxruntime as ort
import PIL.Image
import torch

from nanosam.models.timm_image_encoder import TimmImageEncoder

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--onnx", required=True)
ap.add_argument("--images", default="data/coco/val2017")
args = ap.parse_args()

path = sorted(glob.glob(args.images + "/*.jpg"))[-1]
img = PIL.Image.open(path).convert("RGB").resize((1024, 1024))
x = (np.asarray(img, np.float32) - np.array([123.675, 116.28, 103.53], np.float32)) / np.array([58.395, 57.12, 57.375], np.float32)
x = torch.from_numpy(x.transpose(2, 0, 1)[None].copy())

net = TimmImageEncoder("resnet18")
net.load_state_dict(torch.load(args.ckpt, map_location="cpu")["model"])
with torch.no_grad():
    ref = net.eval()(x).numpy()

sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
inp, out = sess.get_inputs()[0], sess.get_outputs()[0]
got = sess.run(None, {inp.name: x.numpy()})[0]

cos = float((ref * got).sum() / (np.linalg.norm(ref) * np.linalg.norm(got)))
print(f"影像 {path}")
print(f"ONNX 輸入 {inp.name} {inp.shape}，輸出 {out.name} {out.shape}；PyTorch 輸出 {ref.shape}")
print(f"cosine {cos:.6f}，最大絕對差 {np.abs(ref - got).max():.2e}，平均絕對差 {np.abs(ref - got).mean():.2e}")
print("→ OK" if cos > 0.9999 else "→ 不一致，請勿使用這個 ONNX")
