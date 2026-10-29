"""NanoSAM image encoder 蒸餾訓練（PyTorch 教師版，可在沒有 TensorRT 的 WSL 上執行）

與 third_party/nanosam/nanosam/tools/train.py 相同的蒸餾設定（ResNet18 學生 vs MobileSAM ViT-T 教師、
Huber loss、Adam 3e-4），差別只在教師改用 PyTorch（fp16）而非 TensorRT 引擎，
並加入 AMP、梯度累積與每 N 步存檔。checkpoint 格式與官方相同，
可直接用 nanosam.tools.export_image_encoder_onnx 匯出 ONNX。
"""
import argparse
import os
import time

import torch
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader, random_split

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
parser.add_argument("--max_steps", type=int, default=None, help="（除錯用）訓練 N 個 batch 後結束")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
device = "cuda"

teacher = sam_model_registry["vit_t"](checkpoint=args.teacher_checkpoint).image_encoder
teacher = teacher.half().to(device).eval()
for p in teacher.parameters():
    p.requires_grad_(False)

student = create_model(args.model_name).to(device)
optimizer = torch.optim.Adam(student.parameters(), lr=args.learning_rate)
scaler = torch.amp.GradScaler("cuda")

dataset = ImageFolder(args.images)
if args.num_images is not None and args.num_images < len(dataset):
    dataset, _ = random_split(dataset, [args.num_images, len(dataset) - args.num_images])
print(f"images per epoch: {len(dataset)}")
loader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size,
                    num_workers=args.num_workers, drop_last=True)

ckpt_path = os.path.join(args.output_dir, "checkpoint.pth")
start_epoch = 0
if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device)
    student.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    start_epoch = ckpt["epoch"] + 1
    print(f"resume from epoch {start_epoch}")


def save(epoch):
    torch.save({"model": student.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}, ckpt_path)


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
    msg = f"{epoch} - {epoch_loss:.6f} ({time.time() - t0:.0f}s, peak {torch.cuda.max_memory_allocated() / 2**30:.1f}GB)"
    print(msg)
    with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
        f.write(msg + "\n")
    save(epoch)
    if args.max_steps and step >= args.max_steps:
        break
