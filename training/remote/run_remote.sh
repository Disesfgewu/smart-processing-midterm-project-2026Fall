#!/usr/bin/env bash
# NanoSAM image encoder 蒸餾訓練（獨立版）：只需要這個資料夾，不需要整個專案。
# 適用：有 NVIDIA 顯卡的 Linux / WSL2 / Google Colab。自動完成「建環境 → 下載權重與資料 → 小測 → 訓練」。
#
#   ./run_remote.sh            # all：setup → data → smoke → train（前景）
#   ./run_remote.sh setup      # 建 venv + PyTorch + nanosam 程式碼，下載並「驗證」baseline 權重（sha256 必須與主機相同）
#   ./run_remote.sh data       # 下載 / 驗證 / 解壓 COCO val2017（5000 張，778MB；可續傳）
#   ./run_remote.sh smoke      # 64 張的小測，並估算一個 epoch 要多久
#   ./run_remote.sh train      # 只訓練（前景）；加 --bg 背景執行
#   ./run_remote.sh status     # 看進度與目前最佳的驗證 loss
#
# 環境變數：RUN(resnet18_run1) EPOCHS(10) BATCH(4) ACCUM(4) VAL(200) WORKERS  PYTHON  IMAGES  COCO_ZIP  MOBILESAM
#           SYSTEM_PYTHON=1   不建虛擬環境，直接用系統 python3（Colab 用；沿用已預裝的 PyTorch，只補裝 timm 等小套件）
#           COCO_SRC    你手動放好的 COCO 來源（val2017.zip 檔案，或內含 val2017.zip / val2017/ 的資料夾，例如 Google Drive）。
#                       有設就「複製到本機磁碟再解壓」，**不再從網路下載**；訓練時讀的全是本機檔案（Drive 讀小檔案很慢）。
#           BACKUP_DIR  每個 epoch 結束就把 checkpoint.pth / best.pth / log.txt / run_info.json 備份到此資料夾（例如 Drive）；
#                       開始訓練時若本機沒有 checkpoint 而這裡有，會自動還原並接續。訓練本身一律在本機磁碟進行。
#           TORCH_SPEC / TORCH_INDEX（自行指定 PyTorch 版本）  DRY_RUN=1（只印出指令）
# 要和主機的結果可比較：BATCH × ACCUM 請維持 16、VAL 請維持 200（切分方式固定，驗證集與主機完全相同）。
# 接續訓練：把主機的 data/train/<RUN>/ 整個複製到這裡同樣的位置，再執行同一個指令即可從上次的 epoch 繼續。
set -uo pipefail
cd "$(dirname "$0")"

RUN=${RUN:-resnet18_run1}; EPOCHS=${EPOCHS:-10}; BATCH=${BATCH:-4}; ACCUM=${ACCUM:-4}; VAL=${VAL:-200}
WORKERS=${WORKERS:-$(( $(nproc) < 8 ? $(nproc) : 8 ))}
IMAGES=${IMAGES:-data/coco/val2017}; VENV=${VENV:-.venv}
NANOSAM_REV=653633614b2eb93b06ba3be9adb2aeffb117bd72          # 與主機 submodule 相同的 commit
MOBILESAM_URL=https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt
MOBILESAM=${MOBILESAM:-data/models/mobile_sam.pt}
COCO_URL=http://images.cocodataset.org/zips/val2017.zip; COCO_ZIP=${COCO_ZIP:-data/coco/val2017.zip}
if [ "${SYSTEM_PYTHON:-0}" = 1 ]; then PYV=$(command -v python3); else PYV=$VENV/bin/python; fi      # SYSTEM_PYTHON=1（Colab）：直接用系統 Python 與已裝好的 PyTorch，不建 venv
mkdir -p .tmp && export TMPDIR="$PWD/.tmp"        # /tmp 在某些機器是記憶體 tmpfs（很小），pip 解壓大型 wheel 會爆；改用資料夾所在的磁碟
PYSTD_FOR_MATH=$([ -x "$VENV/bin/python" ] && echo "$VENV/bin/python" || command -v python3)

say()  { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[警告]\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31m[錯誤]\033[0m %s\n' "$*" >&2; exit 1; }
run()  { if [ "${DRY_RUN:-0}" = 1 ]; then echo "[DRY_RUN] $*"; else "$@"; fi; }

disk_check() {                       # 需要約 8 GB：venv 3.6 + 資料 0.8（zip）+ 0.8（解壓）+ 其他
    local avail; avail=$(df -Pk . | awk 'NR==2{printf "%d", $4/1048576}')
    echo "  磁碟可用空間（$PWD 所在）: ${avail} GB"
    [ "$avail" -ge 8 ] || fail "磁碟可用空間只有 ${avail} GB，至少需要約 8 GB（PyTorch 約 3.6 GB + COCO 約 1.6 GB）。請換一個空間足夠的位置放這個資料夾"
}

gpu_check() {
    command -v nvidia-smi >/dev/null || fail "找不到 nvidia-smi：這台機器沒有 NVIDIA 驅動 / 顯卡（Colab 請選『變更執行階段類型 → GPU』）"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/  GPU: /'
    CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
    echo "  compute capability: ${CC:-（驅動過舊，無法查詢）}"
}

pick_python() {                      # PyTorch 2.5.1 支援 Python 3.9~3.12；太新（例如 3.14）沒有對應的 wheel
    for c in "${PYTHON:-}" python3.10 python3.11 python3.12 python3.9 /usr/bin/python3 python3; do
        [ -n "$c" ] && command -v "$c" >/dev/null 2>&1 || continue
        v=$("$c" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null) || continue
        case "$v" in 3.9|3.10|3.11|3.12) echo "$c"; return 0 ;; esac
    done
    return 1
}

make_venv() {
    [ -x "$PYV" ] && return 0
    if PYBIN=$(pick_python); then
        echo "  使用 $PYBIN（$($PYBIN --version 2>&1)）建立虛擬環境 $VENV"
        run "$PYBIN" -m venv "$VENV" 2>/dev/null && [ -x "$PYV" ] && return 0
        warn "venv 建立失敗（常見：缺 python3-venv / ensurepip），改用 virtualenv"
        rm -rf "$VENV"
        { "$PYBIN" -m virtualenv -q "$VENV" 2>/dev/null || { "$PYBIN" -m pip install -q --user virtualenv 2>/dev/null || "$PYBIN" -m pip install -q --user --break-system-packages virtualenv; } && "$PYBIN" -m virtualenv -q "$VENV"; } && [ -x "$PYV" ] && return 0
    fi
    if command -v conda >/dev/null 2>&1; then
        echo "  找不到合適的系統 Python（需要 3.9~3.12），改用 conda 建立"
        run conda create -y -q -p "$VENV" python=3.10 pip && [ -x "$PYV" ] && return 0
    fi
    fail "無法建立 Python 環境。請安裝 Python 3.10（Ubuntu：sudo apt install python3.10 python3.10-venv），或設定 PYTHON=/path/to/python3.10"
}

download() {                         # download <url> <dest>：可續傳、自動重試
    local url=$1 dest=$2; mkdir -p "$(dirname "$dest")"
    for i in $(seq 1 20); do
        curl -L -C - --fail --connect-timeout 30 --retry 5 --retry-delay 5 -o "$dest.part" "$url" && { mv "$dest.part" "$dest"; return 0; }
        warn "下載中斷（第 $i 次），5 秒後續傳…"; sleep 5
    done
    return 1
}

setup() {
    say "1/5 檢查 GPU 與磁碟"; gpu_check; disk_check
    say "2/5 建立 Python 環境與安裝套件"
    if [ "${DRY_RUN:-0}" = 1 ]; then echo "[DRY_RUN] 建立 $VENV、安裝 torch / timm / numpy<2 …（略過）"; else
        if [ "${SYSTEM_PYTHON:-0}" = 1 ]; then
            echo "  SYSTEM_PYTHON=1：使用系統 $($PYV --version 2>&1)，沿用已安裝的 PyTorch（不重裝、不動 numpy）"
            "$PYV" -c "import torch; print('  現有 torch', torch.__version__)" || fail "系統 Python 沒有 PyTorch"
            "$PYV" -m pip install -q "timm==1.0.11" "huggingface_hub<1.0" pillow tqdm matplotlib || fail "套件安裝失敗"
        else
        make_venv
        "$PYV" -m pip install -q -U pip || fail "pip 更新失敗"
        if [ -n "${TORCH_SPEC:-}" ]; then SPEC=$TORCH_SPEC; INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu121}
        elif [ "${CC%%.*}" -ge 12 ] 2>/dev/null; then
            warn "compute capability $CC（RTX 50 系列）：PyTorch 2.5.1/cu121 不支援，改裝最新版 + cu128（**未經驗證**，若失敗請用 TORCH_SPEC 自行指定）"
            SPEC="torch torchvision"; INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}
        else SPEC="torch==2.5.1 torchvision==0.20.1"; INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu121}; fi
        echo "  安裝 $SPEC（index: $INDEX）— 約 2.5 GB，請耐心等候"
        "$PYV" -m pip install -q --no-cache-dir $SPEC --extra-index-url "$INDEX" || fail "PyTorch 安裝失敗"
        "$PYV" -m pip install -q --no-cache-dir "numpy<2" timm==1.0.11 "huggingface_hub<1.0" pillow tqdm matplotlib || fail "套件安裝失敗"
        fi
        "$PYV" -c "import torch; assert torch.cuda.is_available(), 'torch.cuda.is_available() = False'; print('  torch', torch.__version__, '| CUDA', torch.version.cuda, '|', torch.cuda.get_device_name(0))" \
            || fail "PyTorch 看不到 GPU：驅動可能太舊（cu121 需要 Linux 驅動 >= 525）"
    fi
    say "3/5 取得 NanoSAM 程式碼（commit ${NANOSAM_REV:0:7}）"
    if [ -d nanosam_repo/nanosam/mobile_sam ]; then echo "  已存在"
    elif [ "${DRY_RUN:-0}" = 1 ]; then echo "[DRY_RUN] 下載 nanosam ${NANOSAM_REV:0:7}"
    else
        curl -L --fail --retry 5 -o nanosam.tar.gz "https://github.com/NVIDIA-AI-IOT/nanosam/archive/${NANOSAM_REV}.tar.gz" || fail "下載 nanosam 失敗"
        tar xzf nanosam.tar.gz && rm -rf nanosam_repo && mv "nanosam-${NANOSAM_REV}" nanosam_repo && rm -f nanosam.tar.gz || fail "解壓 nanosam 失敗"
    fi
    say "4/5 baseline 權重（MobileSAM）：下載並驗證 sha256"
    want=$(tr -d ' \n' < BASELINE.sha256 2>/dev/null)
    [ -n "$want" ] || fail "找不到 BASELINE.sha256（請用 make_bundle.sh 產生的完整資料夾）"
    if [ -f "$MOBILESAM" ] && [ "$(sha256sum "$MOBILESAM" | cut -d' ' -f1)" = "$want" ]; then echo "  已存在且雜湊相符 ✓"
    elif [ "${DRY_RUN:-0}" = 1 ]; then echo "[DRY_RUN] 下載 $MOBILESAM_URL 並比對 ${want:0:16}…"
    else
        download "$MOBILESAM_URL" "$MOBILESAM" || fail "下載 mobile_sam.pt 失敗（可手動放到 $MOBILESAM，或設 MOBILESAM=路徑）"
        got=$(sha256sum "$MOBILESAM" | cut -d' ' -f1)
        [ "$got" = "$want" ] && echo "  雜湊相符 ✓（${got:0:16}…）" || fail "baseline 雜湊不符！期望 ${want:0:16}… 實際 ${got:0:16}…；與主機的 baseline 不是同一個檔案，結果將無法比較"
    fi
    say "5/5 匯入測試"
    if [ "${DRY_RUN:-0}" = 1 ]; then echo "[DRY_RUN] 略過"; else
        PYTHONPATH=$PWD/nanosam_repo "$PYV" -c "
import torch, timm, nanosam.mobile_sam, nanosam.models, nanosam.datasets.image_folder
print('  匯入 OK | torch', torch.__version__, '| timm', timm.__version__)" 2>&1 | grep -vE "Warn|warn|register_model"
        [ "${PIPESTATUS[0]}" -eq 0 ] || fail "匯入失敗"
    fi
    echo "環境就緒。"
}

stage_coco() {                       # 把使用者手動放好的 COCO 從 COCO_SRC 複製到本機磁碟
    local src=$COCO_SRC
    say "從 $src 複製 COCO 到本機（$PWD）"
    if [ -d "$src" ] && [ -f "$src/val2017.zip" ]; then src=$src/val2017.zip; fi
    if [ -f "$src" ]; then
        mkdir -p "$(dirname "$COCO_ZIP")"
        echo "  複製 $(du -h "$src" | cut -f1) 的 zip → $COCO_ZIP"
        run cp "$src" "$COCO_ZIP.part" && run mv "$COCO_ZIP.part" "$COCO_ZIP" || fail "複製失敗（本機磁碟空間不足？）"
    elif [ -d "$src/val2017" ] && [ "$(ls "$src/val2017"/*.jpg 2>/dev/null | wc -l)" -ge 5000 ]; then
        echo "  來源是資料夾（5000 個小檔案，經 Drive 會比 zip 慢很多）→ $(dirname "$IMAGES")/"
        mkdir -p "$(dirname "$IMAGES")"; run cp -r "$src/val2017" "$(dirname "$IMAGES")/" || fail "複製失敗"
    else
        fail "COCO_SRC=$COCO_SRC 裡找不到 val2017.zip，也找不到含 5000 張 jpg 的 val2017/。請確認你放的位置（建議放 val2017.zip，複製最快）"
    fi
}

data() {
    say "訓練資料：COCO val2017（5000 張）"
    n=$(ls "$IMAGES"/*.jpg 2>/dev/null | wc -l)
    if [ "$n" -ge 5000 ]; then echo "  $IMAGES 已有 $n 張，略過"; return 0; fi
    if [ -n "${COCO_SRC:-}" ]; then stage_coco || return 1; n=$(ls "$IMAGES"/*.jpg 2>/dev/null | wc -l); [ "$n" -ge 5000 ] && return 0; fi
    if [ ! -f "$COCO_ZIP" ]; then echo "  下載 $COCO_URL（778 MB，可續傳）"; run download "$COCO_URL" "$COCO_ZIP" || fail "下載失敗；可手動把 val2017.zip 放到 $COCO_ZIP（或設 COCO_ZIP=路徑）"; fi
    [ "${DRY_RUN:-0}" = 1 ] && return 0
    PYSTD=$([ -x "$PYV" ] && echo "$PYV" || command -v python3)
    echo "  驗證 zip 完整性…"; "$PYSTD" -c "
import zipfile,sys; z=zipfile.ZipFile('$COCO_ZIP'); b=z.testzip(); print('  zip OK，', sum(i.filename.endswith('.jpg') for i in z.infolist()), '張') if b is None else sys.exit('zip 損壞: '+b)" || { rm -f "$COCO_ZIP"; fail "zip 損壞，已刪除，請重新執行以重新下載"; }
    mkdir -p "$(dirname "$IMAGES")"; "$PYSTD" -c "import zipfile; zipfile.ZipFile('$COCO_ZIP').extractall('$(dirname "$IMAGES")')"
    n=$(ls "$IMAGES"/*.jpg | wc -l); [ "$n" -eq 5000 ] && echo "  解壓完成：$n 張 ✓" || fail "解壓後影像數量 $n（應為 5000）"
}

train_cmd() { echo "$PYV train_nanosam.py $IMAGES data/train/${TRAIN_RUN:-$RUN} --batch_size $BATCH --accum_steps $ACCUM --num_epochs $1 --val_count $VAL --num_workers $WORKERS ${@:2}"; }

smoke() {
    say "小測：64 張，並估算一個 epoch 的時間"
    if [ "${DRY_RUN:-0}" = 1 ]; then echo "[DRY_RUN] TRAIN_RUN=_smoke $(TRAIN_RUN=_smoke train_cmd 1 --num_images 64)"; return 0; fi
    local VAL=16 log; log=$(mktemp); rm -rf data/train/_smoke          # 小測用 16 張驗證（區域變數，不影響後面的正式訓練）
    PYTHONPATH=$PWD/nanosam_repo TRAIN_RUN=_smoke $(TRAIN_RUN=_smoke train_cmd 1 --num_images 64) 2>&1 | tr '\r' '\n' | grep -vE "Warn|warn|register_model|state_dict" > "$log"
    grep -E "^0 - |Error|error|out of memory" "$log" | head -5
    sec=$(grep -E '^0 - ' "$log" | grep -oE '\([0-9]+s' | tr -d '(s' | head -1); rm -rf data/train/_smoke "$log"
    [ -n "$sec" ] || fail "小測失敗，請看上面的訊息"
    est=$("$PYSTD_FOR_MATH" -c "print(round($sec/64*(5000-$VAL)/60))")
    echo "  → 64 張花 ${sec}s（含暖機，偏保守），**一個 epoch（$((5000-VAL)) 張）約 ${est} 分鐘**；$EPOCHS 個 epoch 約 $("$PYSTD_FOR_MATH" -c "print(round($est*$EPOCHS/60,1))") 小時"
    echo "    （主機 GTX 1060 實測約 19~30 分鐘/epoch，可據此判斷這台值不值得）"
}

backup_sync() {                      # 先寫 .tmp 再 mv：複製到一半斷線也不會損壞已有的備份
    [ -n "${BACKUP_DIR:-}" ] || return 0
    local d=data/train/$RUN f; mkdir -p "$BACKUP_DIR/$RUN"
    for f in best.pth log.txt run_info.json checkpoint.pth; do
        [ -f "$d/$f" ] || continue
        cp "$d/$f" "$BACKUP_DIR/$RUN/$f.tmp" && mv -f "$BACKUP_DIR/$RUN/$f.tmp" "$BACKUP_DIR/$RUN/$f" || warn "備份 $f 失敗（Drive 空間不足？）"
    done
    echo "  [備份] $(date +%T) → $BACKUP_DIR/$RUN（$(tail -1 "$d/log.txt" 2>/dev/null | cut -c1-70)）"
}

backup_watch() {                     # 背景：checkpoint.pth 每次更新（= 一個 epoch 結束，且它是 epoch 最後寫的檔）且大小穩定後就備份
    local ck=data/train/$RUN/checkpoint.pth last="" sig a b
    [ -f "$ck" ] && last=$(stat -c '%Y' "$ck")                       # 已存在（接續訓練）的不重複備份
    while :; do
        if [ -f "$ck" ]; then
            sig=$(stat -c '%Y' "$ck")
            if [ "$sig" != "$last" ]; then
                a=$(stat -c '%s' "$ck"); sleep 3; b=$(stat -c '%s' "$ck")
                [ "$a" = "$b" ] && { backup_sync; last=$sig; }
            fi
        fi
        sleep 10
    done
}

backup_restore() {                   # 本機沒有 checkpoint、備份有 → 還原（Colab 換了新的執行階段時）
    [ -n "${BACKUP_DIR:-}" ] || return 0
    local d=data/train/$RUN
    if [ ! -f "$d/checkpoint.pth" ] && [ -f "$BACKUP_DIR/$RUN/checkpoint.pth" ]; then
        mkdir -p "$d"; echo "  從備份還原 $BACKUP_DIR/$RUN → $d"
        cp "$BACKUP_DIR/$RUN"/{checkpoint.pth,best.pth,log.txt,run_info.json} "$d/" 2>/dev/null || true
    elif [ -f "$d/checkpoint.pth" ]; then echo "  本機已有 checkpoint，不從備份還原"
    else echo "  備份資料夾還沒有 $RUN，從頭開始"; fi
}

train() {
    say "訓練 RUN=$RUN，目標 $EPOCHS 個 epoch（有 checkpoint 會自動接續）"
    [ $((BATCH * ACCUM)) -eq 16 ] || warn "BATCH×ACCUM = $((BATCH * ACCUM))（≠ 16）：等效 batch 與主機不同，結果不能直接比較"
    [ "$VAL" -eq 200 ] || warn "VAL=$VAL（≠ 200）：驗證集與主機不同，結果不能直接比較"
    mkdir -p data/train; export PYTHONPATH=$PWD/nanosam_repo
    [ -n "${BACKUP_DIR:-}" ] && { echo "  備份資料夾：$BACKUP_DIR/$RUN（每個 epoch 結束自動備份）"; backup_restore; }
    if [ "${1:-}" = "--bg" ]; then
        [ "${DRY_RUN:-0}" = 1 ] && { echo "[DRY_RUN] nohup $(train_cmd $EPOCHS) > data/train/train.log &"; return 0; }
        [ -n "${BACKUP_DIR:-}" ] && { nohup bash -c "$(declare -f backup_sync backup_watch warn); RUN=$RUN BACKUP_DIR=$BACKUP_DIR; backup_watch" > data/train/backup.log 2>&1 & }
        nohup $(train_cmd $EPOCHS) > data/train/train.log 2>&1 &
        echo "已在背景執行（pid $!）。看進度： ./run_remote.sh status    或    tail -f data/train/train.log"
    else
        local wp=""
        if [ -n "${BACKUP_DIR:-}" ] && [ "${DRY_RUN:-0}" != 1 ]; then backup_watch & wp=$!; fi
        run $(train_cmd $EPOCHS); local rc=$?
        [ -n "$wp" ] && { kill "$wp" 2>/dev/null; wait "$wp" 2>/dev/null; backup_sync; }      # 結束時（不論成功與否）最後再備份一次
        [ $rc -eq 0 ] || fail "訓練異常結束（可直接重跑同一個指令接續）"
        echo; echo "訓練完成。把結果複製回主機："
        echo "  data/train/$RUN/   （best.pth 60MB = 驗證最佳；checkpoint.pth 181MB；log.txt；run_info.json）"
        echo "  主機上比較：python training/compare_encoders.py --student data/train/$RUN/best.pth"
    fi
}

status() {
    say "狀態"; nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader 2>/dev/null | sed 's/^/  GPU: /'
    pgrep -f train_nanosam >/dev/null && echo "  訓練程序：執行中" || echo "  訓練程序：沒有在跑"
    [ -f "data/train/$RUN/log.txt" ] && { echo "  --- data/train/$RUN/log.txt（最後 8 行）"; tail -8 "data/train/$RUN/log.txt" | sed 's/^/  /'; } || echo "  尚無 epoch 紀錄"
}

case "${1:-all}" in
    all)    setup; PYSTD_FOR_MATH=$PYV; data; smoke; train ;;
    setup)  setup ;;
    data)   data ;;
    smoke)  smoke ;;
    train)  shift; train "$@" ;;
    status) status ;;
    *) echo "用法：$0 [all|setup|data|smoke|train [--bg]|status]"; exit 1 ;;
esac
