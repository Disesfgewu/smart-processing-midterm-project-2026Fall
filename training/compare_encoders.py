"""比較 baseline（MobileSAM TinyViT encoder）與自訓練的 ResNet18 encoder 的分割準確度與 encoder 延遲。

方法與官方 NanoSAM README 的準確度評估相同：用 COCO 標註的「GT 框」當提示，算 SAM 遮罩與 GT 遮罩的 IoU，
依物件大小（COCO 的 area：small < 32²、medium < 96²、large）取平均（mIoU）。mask decoder / prompt encoder 兩邊都用 baseline 的。

    python training/compare_encoders.py                                 # 只量 baseline（先留下基準數字）
    python training/compare_encoders.py --student data/train/resnet18_run1/best.pth
    python training/compare_encoders.py --student A/best.pth --student B/best.pth    # 一次比較多個實驗

需要：data/coco/val2017（影像）、data/coco/instances_val2017.json（./training/download_coco.sh annotations）。
預設只評估「訓練時固定留出的驗證集」（依檔名排序的最後 --val_count 張，與 train_nanosam.py 相同的切法）——這些影像沒參與訓練。
結果另存 JSON 到 data/eval/（不進版控）；baseline 權重只讀不寫。

注意：官方數字是 COCO val2017 全部 5000 張、TensorRT；本腳本預設只用 200 張留出影像、PyTorch 後端、自行蒸餾的 encoder，
**數字不能與官方直接比較**，只能用來比較「我們的 baseline 與我們自訓練的」。
"""
import argparse
import glob
import json
import os
import sys
import time

import cv2
import numpy as np
import PIL.Image
import torch

from nanosam.mobile_sam import SamPredictor, sam_model_registry
from nanosam.models.timm_image_encoder import TimmImageEncoder

OFFICIAL = {"MobileSAM": (0.728, 0.658, 0.759, 0.804), "NanoSAM (ResNet18)": (0.706, 0.624, 0.738, 0.796)}   # All / Small / Medium / Large

ap = argparse.ArgumentParser()
ap.add_argument("--student", action="append", default=[], help="自訓練 encoder 的 checkpoint（可重複給多個）")
ap.add_argument("--sam_checkpoint", default="data/models/mobile_sam.pt")
ap.add_argument("--images", default="data/coco/val2017")
ap.add_argument("--annotations", default="data/coco/instances_val2017.json")
ap.add_argument("--val_count", type=int, default=200, help="取依檔名排序的最後 N 張（= 訓練時留出的驗證集）；0 = 全部")
ap.add_argument("--max_images", type=int, default=None, help="（除錯用）最多評估幾張")
ap.add_argument("--min_area", type=float, default=64, help="忽略 COCO area 小於此值的物件（太小的標註雜訊大）")
ap.add_argument("--out", default=None)
args = ap.parse_args()

if not os.path.exists(args.annotations):
    sys.exit(f"找不到 {args.annotations}：請先 ./training/download_coco.sh annotations")

# ---- 影像清單：與 train_nanosam.py 相同的排序與切法 ----
paths = sorted(glob.glob(os.path.join(args.images, "*.jpg")) + glob.glob(os.path.join(args.images, "*.png")))
paths = paths[-args.val_count:] if args.val_count else paths
if args.max_images:
    paths = paths[:args.max_images]
coco = json.load(open(args.annotations))
id_of = {im["file_name"]: im["id"] for im in coco["images"]}
anns = {}
for a in coco["annotations"]:
    if a["iscrowd"] or not isinstance(a["segmentation"], list) or a["area"] < args.min_area:
        continue
    anns.setdefault(a["image_id"], []).append(a)


def gt_mask(a, h, w):
    m = np.zeros((h, w), np.uint8)
    for poly in a["segmentation"]:
        cv2.fillPoly(m, [np.round(np.array(poly, np.float32).reshape(-1, 2)).astype(np.int32)], 1)
    return m.astype(bool)


def iou(a, b):
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def size_class(area):
    return "small" if area < 32 ** 2 else "medium" if area < 96 ** 2 else "large"


# ---- 模型：同一個 SamPredictor，只換 image encoder ----
device = "cuda"
sam = sam_model_registry["vit_t"](checkpoint=args.sam_checkpoint).to(device).eval()
base_enc = sam.image_encoder
encoders = {"baseline (MobileSAM TinyViT)": base_enc}
for ck in args.student:
    stu = TimmImageEncoder("resnet18")
    c = torch.load(ck, map_location="cpu")
    stu.load_state_dict(c["model"]); stu.img_size = base_enc.img_size
    encoders[f"student: {os.path.basename(os.path.dirname(os.path.abspath(ck)))}/{os.path.basename(ck)} (epoch {c.get('epoch')})"] = stu.to(device).eval()
pred = SamPredictor(sam)

names = list(encoders)
res = {n: {"iou": [], "size": [], "enc_ms": []} for n in names}
agree = {n: [] for n in names[1:]}                    # 與 baseline 遮罩的一致度（IoU）
n_obj = 0
t_start = time.time()
for k, p in enumerate(paths):
    iid = id_of.get(os.path.basename(p))
    objs = anns.get(iid, [])
    if not objs:
        continue
    img = np.asarray(PIL.Image.open(p).convert("RGB")); h, w = img.shape[:2]
    gts = [gt_mask(a, h, w) for a in objs]
    boxes = [np.array([a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]], np.float32) for a in objs]
    masks = {}
    for n in names:
        sam.image_encoder = encoders[n]
        torch.cuda.synchronize(); t0 = time.perf_counter()
        pred.set_image(img)
        torch.cuda.synchronize(); res[n]["enc_ms"].append((time.perf_counter() - t0) * 1e3)
        masks[n] = [pred.predict(box=b, multimask_output=False)[0][0] for b in boxes]
        for m, g, a in zip(masks[n], gts, objs):
            res[n]["iou"].append(iou(m, g)); res[n]["size"].append(size_class(a["area"]))
    for n in names[1:]:
        agree[n] += [iou(ms, mb) for ms, mb in zip(masks[n], masks[names[0]])]
    n_obj += len(objs)
    if (k + 1) % 25 == 0:
        print(f"  {k + 1}/{len(paths)} 張, {n_obj} 個物件, {time.time() - t_start:.0f}s", flush=True)
sam.image_encoder = base_enc

# ---- 報告 ----
def table_row(n):
    iou_a, size_a = np.array(res[n]["iou"]), np.array(res[n]["size"])
    vals = [iou_a.mean()] + [iou_a[size_a == s].mean() if (size_a == s).any() else float("nan") for s in ("small", "medium", "large")]
    return vals, [len(iou_a)] + [int((size_a == s).sum()) for s in ("small", "medium", "large")]


print(f"\n評估影像 {len(paths)} 張（有可用標註的 {sum(1 for p in paths if anns.get(id_of.get(os.path.basename(p))))} 張），物件 {n_obj} 個；GT 框當提示；PyTorch fp32，GTX 1060")
print(f"{'encoder':58s} {'mIoU All':>9s} {'Small':>7s} {'Medium':>7s} {'Large':>7s} | {'encoder ms':>10s} {'與baseline遮罩IoU':>16s}")
out = {"images": len(paths), "objects": n_obj, "val_count": args.val_count, "min_area": args.min_area, "results": {}}
for n in names:
    v, cnt = table_row(n)
    enc = np.array(res[n]["enc_ms"][3:] or res[n]["enc_ms"]).mean()          # 前 3 張暖機不算
    ag = f"{np.mean(agree[n]):.3f}" if n in agree and agree[n] else "—"
    print(f"{n:58s} {v[0]:9.3f} {v[1]:7.3f} {v[2]:7.3f} {v[3]:7.3f} | {enc:10.1f} {ag:>16s}")
    out["results"][n] = {"miou": dict(zip(["all", "small", "medium", "large"], map(float, v))), "counts": dict(zip(["all", "small", "medium", "large"], cnt)),
                         "encoder_ms": float(enc), "agree_with_baseline": float(np.mean(agree[n])) if n in agree and agree[n] else None}
print(f"{'':58s} （物件數 All/Small/Medium/Large = {'/'.join(map(str, table_row(names[0])[1]))}）")
print("\n官方（COCO val2017 全部 5000 張、TensorRT、Jetson）：")
for n, v in OFFICIAL.items():
    print(f"  {n:20s} All {v[0]:.3f}  Small {v[1]:.3f}  Medium {v[2]:.3f}  Large {v[3]:.3f}")
print("→ 官方數字用的影像、後端與訓練資料都不同，**不能直接比較**；上表只用來比較「我們的 baseline」與「我們自訓練的」。")

out_path = args.out or f"data/eval/compare_{time.strftime('%Y%m%d_%H%M%S')}.json"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
json.dump(out, open(out_path, "w"), indent=2, ensure_ascii=False)
print(f"\n結果已存：{out_path}")
