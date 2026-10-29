"""FastAPI 後端：python -m server.main  （預設 http://127.0.0.1:8000）"""
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    global service, camera
    logger.info("[lifespan.startup]: input=(), output=loading models")
    service = SegmentService(sam_checkpoint=os.environ.get("SAM_CHECKPOINT", os.path.join(ROOT, "data/models/mobile_sam.pt")),
                             owl_engine=os.environ.get("OWL_ENGINE") or None,
                             student_checkpoint=os.environ.get("SAM_STUDENT") or None)   # 自訓練的 ResNet18 encoder；不給就只有 baseline
    camera = CameraWorker(service)
    import threading
    threading.Thread(target=hyde.ensure_running, name="ollama-autostart", daemon=True).start()   # 不阻塞啟動；Ollama 沒在跑就自動拉起
    logger.info("[lifespan.startup]: input=(), output=ready")
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
    try:
        img = service.decode_image(image.file.read())
    except Exception as e:
        raise HTTPException(400, f"cannot read image: {e}")
    result = service.run(img, plist, threshold, nms_iou, use_hyde, with_masks=with_masks, top_per_group=top_per_group)
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.main:app", host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 8000)), log_level="warning")
