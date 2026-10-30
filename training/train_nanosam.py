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
parser.add_argument("--teacher_fp16", action="store_true",
                    help="（不建議）教師用 fp16。MobileSAM 的 TinyViT 在 fp16 下輸出是錯的（實測與 fp32 的 cosine ≈ -0.24；官方 README 也註明此 encoder 必須用 fp32），學生會學到壞掉的目標")
parser.add_argument("--amp", choices=["auto", "on", "off"], default="auto",
                    help="學生是否用 fp16 混合精度。auto = 只在有 Tensor Core 的 GPU（compute capability >= 7.0）開啟；Pascal（GTX 10 系列）上 fp16 反而更慢")
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
# ---- 同一個輸出資料夾同時只能有一個訓練程序（2026-10-29 實際發生過：重複啟動兩個，搶 GPU、速度剩一半，還互相覆蓋 checkpoint）----
lock_path = os.path.join(args.output_dir, ".lock")
if os.path.exists(lock_path):
    try:
        other = int(open(lock_path).read().strip()); os.kill(other, 0)         # 訊號 0 = 只檢查該程序是否存在
        if other != os.getpid():
            sys.exit(f"{args.output_dir} 已經有訓練程序在執行（pid {other}）。同一個資料夾不能同時跑兩個；要停掉它請 kill {other}，"
                     f"或用另一個 RUN 名稱（例如 RUN=resnet18_run2）。")
    except (ValueError, ProcessLookupError):
        pass                                                                    # 殘留的舊鎖（上次異常結束）：忽略
    except PermissionError:
        sys.exit(f"{lock_path} 指到一個存在但無法檢查的程序，請確認沒有其他訓練在跑，再手動刪除這個檔案")
_main_pid = os.getpid()
open(lock_path, "w").write(str(_main_pid))
atexit.register(lambda: os.getpid() == _main_pid and os.path.exists(lock_path) and os.remove(lock_path))    # 只有主程序在結束時移除（DataLoader worker 不動）
teacher_sha = sha256(args.teacher_checkpoint)
try:
    git_rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip()
except Exception:
    git_rev = ""
info_path = os.path.join(args.output_dir, "run_info.json")           # 每次（含接續）啟動都追加一筆，之後寫報告 / 重現實驗用
info = json.load(open(info_path)) if os.path.exists(info_path) else {"starts": []}
# 舊版腳本（沒有 teacher_fp16 這個參數）的教師固定是 fp16，輸出是壞的；用它訓練出的 checkpoint 不可接續，否則會把壞權重帶進新的訓練。
if os.path.exists(os.path.join(args.output_dir, "checkpoint.pth")) and any(st["args"].get("teacher_fp16", True) for st in info["starts"]):
    sys.exit(f"{args.output_dir} 是用 fp16 教師（輸出已證實是壞的）訓練的，不可接續。請把它改名保存（例如 mv {args.output_dir} {args.output_dir}_INVALID_fp16）或換一個 RUN 名稱。")
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
teacher = teacher.to(device).eval()                    # 教師一律 fp32：TinyViT 在 fp16 下輸出是壞的（見 --teacher_fp16 說明）
if args.teacher_fp16:
    print("警告：--teacher_fp16 已開啟，教師目標可能是錯的！")
    teacher = teacher.half()
t_in = (lambda x: x.half()) if args.teacher_fp16 else (lambda x: x)
use_amp = args.amp == "on" or (args.amp == "auto" and torch.cuda.get_device_capability()[0] >= 7)
print(f"教師精度: {'fp16（不建議）' if args.teacher_fp16 else 'fp32'} | 學生混合精度 AMP: {'開' if use_amp else '關'}")
for p in teacher.parameters():
    p.requires_grad_(False)

student = create_model(args.model_name).to(device)
optimizer = torch.optim.Adam(student.parameters(), lr=args.learning_rate)
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

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
    total, cos, n = 0.0, 0.0, 0
    for image in val_loader:
        image = image.to(device)
        target = teacher(t_in(image)).float()
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            output = student(F.interpolate(image, (args.student_size,) * 2, mode="area")).float()
        total += float(F.huber_loss(output, target)) * len(image)
        cos += float(F.cosine_similarity(output.flatten(1), target.flatten(1)).sum()); n += len(image)
    student.train()
    return total / max(n, 1), cos / max(n, 1)             # (Huber loss, 與教師特徵的 cosine：1 = 完全一致、0 = 毫無關係)


@torch.no_grad()
def sanity_check_targets():
    """訓練前先驗證「老師的目標是對的」：loss 下降 ≠ 學對。教師 fp16 壞掉時，目標幾乎與影像無關，loss 一開始就很小、
    學生輕鬆學會，訓練看起來「很順利」卻毫無用處（2026-10-29 實際發生過：與 fp32 的 cosine = -0.24，mIoU = 0）。"""
    batch = next(iter(val_loader if val_loader else loader)).to(device)[:4]
    t = teacher(t_in(batch)).float()
    cross, overall = float(t.std(0).mean()), float(t.std())
    zero = float(F.huber_loss(torch.zeros_like(t), t)); mean = float(F.huber_loss(t.mean(0, keepdim=True).expand_as(t), t))
    print(f"[目標健檢] 教師特徵 mean {float(t.mean()):+.4f} std {overall:.4f} |max| {float(t.abs().max()):.2f}；"
          f"跨影像變化 {cross:.4f}（{cross / max(overall, 1e-9):.0%} of std）")
    print(f"[目標健檢] 參考 loss：全預測 0 = {zero:.2e}；只猜（這批的）平均特徵 = {mean:.2e}。驗證 loss 與 cosine 要明顯優於這兩者才算學到東西")
    problems = []
    if not bool(torch.isfinite(t).all()):
        problems.append("教師特徵含 NaN/inf")
    if cross < 0.1 * overall:
        problems.append(f"教師特徵幾乎不隨影像變化（跨影像 std 只有整體的 {cross / max(overall, 1e-9):.1%}），學生會學到一個固定圖樣")
    if args.teacher_fp16:                                   # 用 fp16 時順便與 fp32 比對
        ref = sam_model_registry["vit_t"](checkpoint=args.teacher_checkpoint).image_encoder.to(device).eval()
        c = float(F.cosine_similarity(t.flatten(1), ref(batch).float().flatten(1)).mean())
        print(f"[目標健檢] fp16 教師與 fp32 教師的 cosine = {c:.3f}")
        if c < 0.99:
            problems.append(f"fp16 教師與 fp32 不一致（cosine {c:.3f}）")
    if problems:
        sys.exit("目標健檢失敗，不開始訓練：" + "；".join(problems) + "。請勿使用 --teacher_fp16。")


sanity_check_targets()
step = 0
for epoch in range(start_epoch, args.num_epochs):
    student.train()
    epoch_loss, t0 = 0.0, time.time()
    optimizer.zero_grad()
    for i, image in enumerate(tqdm.tqdm(loader, desc=f"epoch {epoch}")):
        image = image.to(device)
        with torch.no_grad():
            target = teacher(t_in(image)).float()
        student_in = F.interpolate(image, (args.student_size,) * 2, mode="area")
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
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
    val, val_cos = validate() if val_loader else (None, None)
    improved = val is not None and val < best_val
    if improved:
        best_val = val
        torch.save({"model": student.state_dict(), "epoch": epoch, "val_loss": val}, os.path.join(args.output_dir, "best.pth"))
    msg = (f"{epoch} - train {epoch_loss:.6f}" + (f" | val {val:.6f} cos {val_cos:.3f}{' *best' if improved else ''}" if val is not None else "") +
           f" ({time.time() - t0:.0f}s, peak {torch.cuda.max_memory_allocated() / 2**30:.1f}GB)")
    print(msg)
    if val_cos is not None and epoch == start_epoch and val_cos < 0.3:
        print(f"警告：驗證 cosine 只有 {val_cos:.2f}（學生輸出與教師特徵幾乎不相關）；若之後沒有明顯上升，請停止並檢查。")
    with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
        f.write(msg + "\n")
    save(epoch)
    if args.max_steps and step >= args.max_steps:
        break
