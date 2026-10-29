"""NanoSAM image encoder 蒸餾訓練（PyTorch 教師版，可在沒有 TensorRT 的 WSL 上執行）

與 third_party/nanosam/nanosam/tools/train.py 相同的蒸餾設定（ResNet18 學生 vs MobileSAM ViT-T 教師、
Huber loss、Adam 3e-4），差別只在教師改用 PyTorch（fp16）而非 TensorRT 引擎，
並加入 AMP、梯度累積與每 N 步存檔。checkpoint 格式與官方相同，
可直接用 nanosam.tools.export_image_encoder_onnx 匯出 ONNX。
"""
import argparse
import atexit
import hashlib
import json
import os
import subprocess
import sys
import time

import torch
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from nanosam.datasets.image_folder import ImageFolder
from nanosam.mobile_sam import sam_model_registry
from nanosam.models import create_model

parser = argparse.ArgumentParser()
parser.add_argument("images", type=str, help="訓練影像資料夾（*.jpg / *.png）")
parser.add_argument("output_dir", type=str)
parser.add_argument("--teacher_checkpoint", type=str, default="data/models/mobile_sam.pt")
parser.add_argument("--model_name", type=str, default="resnet18")
parser.add_argument("--student_size", type=int, default=1024)
parser.add_argument("--num_images", type=int, default=None, help="每個 epoch 只用前 N 張（隨機子集）")
parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=4, help="GTX 1060 6GB 建議 2~4")
parser.add_argument("--accum_steps", type=int, default=4, help="梯度累積，等效 batch = batch_size * accum_steps")
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--learning_rate", type=float, default=3e-4)
parser.add_argument("--val_count", type=int, default=200, help="固定留出最後 N 張（依檔名排序）當驗證集，不參與訓練；0 = 不驗證")
parser.add_argument("--max_steps", type=int, default=None, help="（除錯用）訓練 N 個 batch 後結束")
args = parser.parse_args()

# ---- 保護 baseline（MobileSAM）：訓練只「讀」教師權重，絕不寫到 baseline 所在位置 ----
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


out_abs, teacher_abs, models_abs = (os.path.realpath(p) for p in (args.output_dir, args.teacher_checkpoint, "data/models"))
if out_abs == os.path.dirname(teacher_abs) or out_abs == models_abs or out_abs.startswith(models_abs + os.sep):
    sys.exit(f"輸出目錄 {args.output_dir} 位於 baseline 所在的 data/models 底下，會與 baseline 混在一起。請改用 data/train/<run 名稱>")
os.makedirs(args.output_dir, exist_ok=True)
teacher_sha = sha256(args.teacher_checkpoint)
try:
    git_rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip()
except Exception:
    git_rev = ""
info_path = os.path.join(args.output_dir, "run_info.json")           # 每次（含接續）啟動都追加一筆，之後寫報告 / 重現實驗用
info = json.load(open(info_path)) if os.path.exists(info_path) else {"starts": []}
info["starts"].append({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "args": vars(args), "git": git_rev, "teacher_sha256": teacher_sha[:16]})
json.dump(info, open(info_path, "w"), indent=2, ensure_ascii=False)


def _verify_baseline():                                                # 結束時再算一次雜湊，確認 baseline 沒被動到
    ok = sha256(args.teacher_checkpoint) == teacher_sha
    msg = f"baseline {args.teacher_checkpoint} sha256={teacher_sha[:16]}… {'未被改動 ✓' if ok else '**已被改動！**'}"
    print(msg)
    with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
        f.write(msg + "\n")


atexit.register(_verify_baseline)
print(f"run output: {args.output_dir}（baseline {args.teacher_checkpoint} 只讀取，sha256={teacher_sha[:16]}…）")
device = "cuda"

teacher = sam_model_registry["vit_t"](checkpoint=args.teacher_checkpoint).image_encoder
teacher = teacher.half().to(device).eval()
for p in teacher.parameters():
    p.requires_grad_(False)

student = create_model(args.model_name).to(device)
optimizer = torch.optim.Adam(student.parameters(), lr=args.learning_rate)
scaler = torch.amp.GradScaler("cuda")

dataset = ImageFolder(args.images)
order = sorted(range(len(dataset)), key=lambda i: dataset.image_paths[i])        # 依檔名排序，切分固定、可重現
val_idx = order[-args.val_count:] if args.val_count else []
train_idx = order[:-args.val_count] if args.val_count else order
dataset = Subset(dataset, train_idx)
if args.num_images is not None and args.num_images < len(dataset):
    dataset, _ = random_split(dataset, [args.num_images, len(dataset) - args.num_images])
print(f"train images per epoch: {len(dataset)} | validation images: {len(val_idx)}")
loader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size,
                    num_workers=args.num_workers, drop_last=True)
val_loader = None
if val_idx:   # 驗證不做隨機裁切：整張縮放到 1024，結果才可比較
    val_tf = Compose([Resize((1024, 1024)), ToTensor(),
                      Normalize(mean=[123.675 / 255, 116.28 / 255, 103.53 / 255], std=[58.395 / 255, 57.12 / 255, 57.375 / 255])])
    val_loader = DataLoader(Subset(ImageFolder(args.images, transform=val_tf), val_idx), batch_size=args.batch_size, num_workers=2)

ckpt_path = os.path.join(args.output_dir, "checkpoint.pth")
start_epoch, best_val = 0, float("inf")
if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device)
    student.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    start_epoch = ckpt["epoch"] + 1
    best_val = ckpt.get("best_val", float("inf"))
    print(f"resume from epoch {start_epoch}")


def save(epoch):
    torch.save({"model": student.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "best_val": best_val}, ckpt_path)


@torch.no_grad()
def validate():
    student.eval()
    total, n = 0.0, 0
    for image in val_loader:
        image = image.to(device)
        target = teacher(image.half()).float()
        with torch.autocast("cuda", dtype=torch.float16):
            output = student(F.interpolate(image, (args.student_size,) * 2, mode="area"))
        total += float(F.huber_loss(output.float(), target)) * len(image); n += len(image)
    student.train()
    return total / max(n, 1)


step = 0
for epoch in range(start_epoch, args.num_epochs):
    student.train()
    epoch_loss, t0 = 0.0, time.time()
    optimizer.zero_grad()
    for i, image in enumerate(tqdm.tqdm(loader, desc=f"epoch {epoch}")):
        image = image.to(device)
        with torch.no_grad():
            target = teacher(image.half()).float()
        student_in = F.interpolate(image, (args.student_size,) * 2, mode="area")
        with torch.autocast("cuda", dtype=torch.float16):
            output = student(student_in)
        loss = F.huber_loss(output.float(), target)
        scaler.scale(loss / args.accum_steps).backward()
        if (i + 1) % args.accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        epoch_loss += float(loss)
        step += 1
        if args.max_steps and step >= args.max_steps:
            break
    epoch_loss /= max(i + 1, 1)
    val = validate() if val_loader else None
    improved = val is not None and val < best_val
    if improved:
        best_val = val
        torch.save({"model": student.state_dict(), "epoch": epoch, "val_loss": val}, os.path.join(args.output_dir, "best.pth"))
    msg = (f"{epoch} - train {epoch_loss:.6f}" + (f" | val {val:.6f}{' *best' if improved else ''}" if val is not None else "") +
           f" ({time.time() - t0:.0f}s, peak {torch.cuda.max_memory_allocated() / 2**30:.1f}GB)")
    print(msg)
    with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
        f.write(msg + "\n")
    save(epoch)
    if args.max_steps and step >= args.max_steps:
        break
