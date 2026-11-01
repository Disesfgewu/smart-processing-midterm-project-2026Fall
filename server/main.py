"""FastAPI 後端：python -m server.main  （預設 http://127.0.0.1:8000；Jetson 上預設 0.0.0.0 讓區網可連，且有引擎就自動用 TensorRT，見 resolve_backends）"""
import json
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile  # noqa: E402
from fastapi.responses import JSONResponse, Response, StreamingResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from applog import LOG_PATH, log_call, logger, quiet_var, request_id_var  # noqa: E402
from server import hyde  # noqa: E402
from server.camera import CameraWorker  # noqa: E402
from server.service import SegmentService  # noqa: E402

service: Optional[SegmentService] = None
camera: Optional[CameraWorker] = None


def _first_existing(*paths: str) -> Optional[str]:
    return next((p for p in paths if os.path.isfile(p)), None)


def resolve_backends() -> dict:
    """決定 OWL / SAM 用哪個後端。優先序：環境變數（設成空字串 = 明確關閉）> 自動偵測 data/ 裡 Jetson 上建好的 TensorRT 引擎 > PyTorch。
    所以在 Jetson 上建好引擎後，直接 `python -m server.main` 就是全 TensorRT（OWL + NanoSAM）；網頁的「SAM 編碼器」下拉選單仍可切回 MobileSAM / 自訓練 encoder。"""
    d = lambda name: os.path.join(ROOT, "data", name)                                             # noqa: E731
    env = os.environ.get
    owl = env("OWL_ENGINE") if "OWL_ENGINE" in os.environ else _first_existing(d("owl_image_encoder_patch32.engine"))
    if "NANOSAM_ENCODER" in os.environ or "NANOSAM_DECODER" in os.environ:
        enc, dec = env("NANOSAM_ENCODER") or None, env("NANOSAM_DECODER") or None
    else:
        enc = _first_existing(d("resnet18_image_encoder.selftrain.engine"), d("resnet18_image_encoder.engine"))
        dec = _first_existing(d("mobile_sam_mask_decoder.engine")) if enc else None
    nano = bool(enc and dec)
    return {"owl_engine": owl or None, "nanosam_encoder": enc if nano else None, "nanosam_decoder": dec if nano else None,
            "sam_active": env("SAM_ACTIVE") or ("nanosam" if nano else None)}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global service, camera
    logger.info("[lifespan.startup]: input=(), output=loading models")
    be = resolve_backends()
    service = SegmentService(sam_checkpoint=os.environ.get("SAM_CHECKPOINT", os.path.join(ROOT, "data/models/mobile_sam.pt")),
                             owl_engine=be["owl_engine"],
                             student_checkpoint=os.environ.get("SAM_STUDENT") or None,
                             nanosam_encoder=be["nanosam_encoder"],                             # NanoSAM 的引擎（.engine）或 ONNX
                             nanosam_decoder=be["nanosam_decoder"],
                             nanosam_backend=os.environ.get("NANOSAM_BACKEND", "auto"))   # 自訓練的 ResNet18 encoder；不給就只有 baseline
    if be["sam_active"]:                                                                     # 啟動時就選用哪個 SAM encoder（例如 nanosam）
        service.switch_sam_encoder(be["sam_active"])
    print(f"[server] 後端：OWL = {'TensorRT 引擎 ' + os.path.basename(be['owl_engine']) if be['owl_engine'] else 'PyTorch fp32'} | "
          f"SAM = {be['sam_active'] or 'mobilesam (PyTorch fp32)'}", flush=True)
    camera = CameraWorker(service)
    import threading
    threading.Thread(target=hyde.ensure_running, name="ollama-autostart", daemon=True).start()   # 不阻塞啟動；Ollama 沒在跑就自動拉起
    logger.info("[lifespan.startup]: input=(), output=ready")
    print("[server] 就緒，可以開啟網頁了。", flush=True)
    yield
    camera.stop()
    hyde.stop_started()
    logger.info("[lifespan.shutdown]: input=(), output=bye")


app = FastAPI(title="Open-vocab detection + segmentation", lifespan=lifespan)


@app.middleware("http")
async def log_api_call(request: Request, call_next):
    rid = uuid.uuid4().hex[:8]
    request_id_var.set(rid)
    t0 = time.perf_counter()
    quiet = request.url.path in ("/api/stats", "/api/logs", "/api/camera/state", "/api/camera/stream")  # 前端輪詢的 API 不逐筆寫 log，避免洗版
    if not quiet:
        logger.info(f"[API {request.method} {request.url.path}]: input=(query={dict(request.query_params)}, "
                    f"content_type={request.headers.get('content-type', '-')}, length={request.headers.get('content-length', '-')})")
    try:
        response = await call_next(request)
    except Exception as e:
        logger.error(f"[API {request.method} {request.url.path}]: error={type(e).__name__}: {e}", exc_info=True)
        return JSONResponse({"error": str(e), "request_id": rid}, status_code=500, headers={"X-Request-ID": rid})
    response.headers["X-Request-ID"] = rid
    if not quiet:
        logger.info(f"[API {request.method} {request.url.path}]: output=(status={response.status_code}, "
                    f"elapsed={(time.perf_counter() - t0) * 1e3:.1f}ms)")
    return response


def parse_prompts(raw: str) -> List[str]:
    """接受 JSON 陣列字串或以逗號／換行分隔的文字。"""
    raw = raw.strip()
    try:
        data = json.loads(raw)
        items = data if isinstance(data, list) else [str(data)]
    except json.JSONDecodeError:
        items = raw.replace("\n", ",").split(",")
    out, seen = [], set()
    for p in (str(x).strip() for x in items):
        if p and p not in seen:
            out.append(p); seen.add(p)
    return out


@app.get("/api/health")
@log_call(name="api.health")
def health():
    return {"status": "ok", "backend": service.backend, "device": service.gpu_stats()["device"],
            "model_load_s": service.load_s, "uptime_s": time.time() - service.started, "requests": service.n_requests}


@app.get("/api/stats")
def stats():
    return {**service.gpu_stats(), "requests": service.n_requests, "uptime_s": time.time() - service.started}


@app.post("/api/hyde/expand")
@log_call(name="api.hyde.expand")
def hyde_expand(prompts: str = Form(...)):
    """HyDE 擴展（會呼叫 LLM，數秒～十幾秒）。前端必須在「開始偵測之前」呼叫；偵測端點只讀快取，不會呼叫 LLM。"""
    plist = parse_prompts(prompts)
    if not plist or len(plist) > 20:
        raise HTTPException(400, "prompts must have 1~20 items")
    try:
        return service.hyde_expand(plist)
    except hyde.HydeError as e:
        raise HTTPException(503, str(e))


@app.get("/api/sam/encoder")
def sam_encoder_get():
    return service.sam_encoder_info()


@app.post("/api/sam/encoder")
@log_call(name="api.sam.encoder")
def sam_encoder_set(name: str = Form(...)):
    """切換 SAM image encoder（mobilesam = baseline；student = 自訓練）。camera 串流中請先停止，避免兩邊用不同 encoder 比較。"""
    if camera.running:
        raise HTTPException(409, "伺服器串流進行中，請先停止再切換 SAM encoder")
    try:
        return service.switch_sam_encoder(name)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/hyde/status")
def hyde_status():
    return hyde.status()


@app.get("/api/history")
@log_call(name="api.history")
def history():
    return list(service.history)


@app.get("/api/logs")
def logs(lines: int = 120, request_id: Optional[str] = None):
    try:
        with open(LOG_PATH, "rb") as f:  # 只讀檔尾，避免日誌變大後每次輪詢都讀整檔
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - (3_000_000 if request_id else 400_000)))
            rows = f.read().decode("utf-8", "replace").splitlines()
            if size > len(rows) and rows:
                rows = rows[1:]  # 第一行可能被截斷
    except FileNotFoundError:
        rows = []
    if request_id:
        rows = [r for r in rows if f"| {request_id} |" in r]
    return {"path": LOG_PATH, "lines": rows[-max(1, min(lines, 1000)):]}


@app.post("/api/segment")
@log_call(name="api.segment")
def segment(image: UploadFile = File(...), prompts: str = Form(...), threshold: float = Form(0.1),
            nms_iou: float = Form(0.5), use_hyde: bool = Form(False), quiet: bool = Form(False), with_masks: bool = Form(True), top_per_group: bool = Form(False)):
    quiet_var.set(quiet)  # 連續影格（攝影機／影片）由前端取樣記錄，其餘只記 WARNING 以上
    plist = parse_prompts(prompts)
    if not plist:
        raise HTTPException(400, "prompts is empty")
    if len(plist) > 20:
        raise HTTPException(400, "too many prompts (max 20)")
    if not 0 < threshold < 1 or not 0 < nms_iou <= 1:
        raise HTTPException(400, "threshold must be in (0,1) and nms_iou in (0,1]")
    t_req = time.perf_counter()
    try:
        img = service.decode_image(image.file.read())
    except Exception as e:
        raise HTTPException(400, f"cannot read image: {e}")
    decode_ms = (time.perf_counter() - t_req) * 1e3
    result = service.run(img, plist, threshold, nms_iou, use_hyde, with_masks=with_masks, top_per_group=top_per_group)
    # timing.total_ms 只含 run()（推論 + 序列化），不含圖片解碼與回應組裝。要讓畫面數字貼近使用者實際等待的時間，另外提供：
    #   decode_ms = 圖片解碼＋縮放；server_ms = 本函式從開始到 run() 回傳（解碼 + 推論 + 序列化 + 組裝回應）。
    # 瀏覽器端的 client_ms − server_ms ≈ 上傳 + FastAPI 編 JSON + 下載 + 解析（server_ms 不含 FastAPI 把 dict 編成 JSON 的時間）。
    result["timing"]["decode_ms"] = decode_ms
    result["timing"]["server_ms"] = (time.perf_counter() - t_req) * 1e3
    result["request_id"] = request_id_var.get()
    return result


def check_params(prompts: str, threshold: float, nms_iou: float) -> List[str]:
    plist = parse_prompts(prompts)
    if not plist or len(plist) > 20:
        raise HTTPException(400, "prompts must have 1~20 items")
    if not 0 < threshold < 1 or not 0 < nms_iou <= 1:
        raise HTTPException(400, "threshold must be in (0,1) and nms_iou in (0,1]")
    return plist


@app.post("/api/camera/start")
@log_call(name="api.camera.start")
def camera_start(source: str = Form("0"), prompts: str = Form(...), threshold: float = Form(0.1), nms_iou: float = Form(0.5),
                 style: str = Form("both"), max_side: int = Form(640), with_masks: bool = Form(True), use_hyde: bool = Form(False), top_per_group: bool = Form(False)):
    plist = check_params(prompts, threshold, nms_iou)
    if style not in ("mask", "contour", "both", "box"):
        raise HTTPException(400, "bad style")
    groups, hyde_info = service.hyde_groups(plist, use_hyde)
    if camera.running:
        camera.stop()
    try:
        camera.start(source.strip(), plist, threshold, nms_iou, style, max(160, min(max_side, 1920)), with_masks, groups, top_per_group)
    except ValueError as e:
        raise HTTPException(400, str(e))
    camera.hyde_info = hyde_info
    return {"started": True, "source": source, "hyde": hyde_info}


@app.post("/api/camera/stop")
@log_call(name="api.camera.stop")
def camera_stop():
    camera.stop()
    return {"stopped": True}


@app.post("/api/camera/prompts")
@log_call(name="api.camera.prompts")
def camera_prompts(prompts: str = Form(...), threshold: float = Form(0.1), nms_iou: float = Form(0.5), style: str = Form("both"),
                   expected: str = Form(""), with_masks: bool = Form(True), use_hyde: bool = Form(False), top_per_group: bool = Form(False)):
    """執行中更換提示詞與參數，不需重啟串流。expected 為 JSON：{提示詞: 預期數量}。"""
    plist = check_params(prompts, threshold, nms_iou)
    exp = json.loads(expected) if expected.strip() else None
    camera.set_params(threshold=threshold, nms_iou=nms_iou, style=style, expected=exp, with_masks=with_masks, top_per_group=top_per_group)
    groups, hyde_info = service.hyde_groups(plist, use_hyde)
    camera.set_prompts(plist, groups)
    camera.hyde_info = hyde_info
    return {"prompts": plist, "hyde": hyde_info}


@app.get("/api/camera/state")
def camera_state():
    return {**camera.state, "running": camera.running}


@app.get("/api/camera/stream")
async def camera_stream(request: Request):
    import asyncio

    async def gen():
        last = -1
        while camera.running and not await request.is_disconnected():
            jpeg, last = await asyncio.get_event_loop().run_in_executor(None, camera.wait_frame, last, 1.0)
            if jpeg:
                yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n"
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


app.mount("/", StaticFiles(directory=os.path.join(ROOT, "frontend"), html=True), name="frontend")

def _lan_ips() -> list:
    """這台機器的區網 IPv4。先用「對外連線」取得預設路由的來源 IP（UDP connect 不會真的送封包），
    再補上主機名稱解析到的位址；排除 127.* 與 docker 的 172.17.*。"""
    import socket
    ips = set()
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sk:
            sk.connect(("10.255.255.255", 1))
            ips.add(sk.getsockname()[0])
    except OSError:
        pass
    try:
        ips.update(i[4][0] for i in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET))
    except OSError:
        pass
    return sorted(ip for ip in ips if not ip.startswith(("127.", "172.17.", "0.")))


def _urls(host: str, port: int) -> str:
    if host in ("0.0.0.0", "::"):
        lan = "、".join(f"http://{ip}:{port}" for ip in _lan_ips()) or f"http://<這台的 IP>:{port}"
        return f"區網內其他裝置開 {lan}（本機：http://127.0.0.1:{port}）。注意：沒有登入驗證，只在可信任的網路使用"
    return (f"http://{host}:{port}（只有本機可連。要讓區網其他裝置連：HOST=0.0.0.0 python -m server.main；"
            f"或用 SSH 轉發 ssh -L {port}:127.0.0.1:{port} <使用者>@<這台的 IP>）")


def _port_free(host: str, port: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sk:
        sk.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sk.bind((host, port))
            return True
        except OSError:
            return False


def pick_port(host: str) -> int:
    """在載入模型「之前」決定埠（載完才發現綁不上要白等十幾秒）。PORT 有指定就一定用它，被佔直接報錯；
    沒指定就從 8000 起找第一個空的（同一台機器上常有別的網頁服務佔著 8000）。"""
    if "PORT" in os.environ:
        port = int(os.environ["PORT"])
        if not _port_free(host, port):
            sys.exit(f"[server] 埠 {port} 已被佔用。用 `ss -ltnp | grep :{port}` 看是誰，或換一個：PORT={port + 1} python -m server.main")
        return port
    for port in range(8000, 8011):
        if _port_free(host, port):
            if port != 8000:
                print(f"[server] 埠 8000 已被佔用，改用 {port}（要固定埠請設 PORT=…）", flush=True)
            return port
    sys.exit("[server] 8000~8010 都被佔用，請用 PORT=… 指定")


if __name__ == "__main__":
    import uvicorn
    # Jetson 是用來示範 / 量測的機台，預設讓區網其他裝置可連；其他環境（WSL / 桌機）仍預設只開本機。HOST=127.0.0.1 可改回只開本機
    host = os.environ.get("HOST", "0.0.0.0" if os.path.exists("/etc/nv_tegra_release") else "127.0.0.1")
    port = pick_port(host)
    print(f"[server] 載入模型中（Jetson 約 10~30 秒）… 網頁：{_urls(host, port)}", flush=True)
    uvicorn.run("server.main:app", host=host, port=port, log_level="warning")
