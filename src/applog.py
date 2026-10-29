"""全專案共用的 logger 與 @log_call 裝飾器。

格式：  時間 | 等級 | request_id | [函式名]: input=..., output=...
        失敗時 [函式名]: input=..., error=ExceptionType: message（含 traceback）

所有紀錄寫入專案根目錄的 app.log（輪替 5MB x 3）並同步輸出到終端機。
"""
import contextvars
import functools
import inspect
import logging
import os
import time
from logging.handlers import RotatingFileHandler

LOG_PATH = os.environ.get("APP_LOG", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.log"))
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")
# 攝影機連續推論時設為 True：只保留 WARNING 以上，避免每幀十幾行洗掉 app.log（每 N 幀會取樣關閉一次）
quiet_var: contextvars.ContextVar[bool] = contextvars.ContextVar("quiet", default=False)


class _RequestIdFilter(logging.Filter):
    def filter(self, record):
        record.request_id = request_id_var.get()
        return not (quiet_var.get() and record.levelno < logging.WARNING)


def _setup() -> logging.Logger:
    root = logging.getLogger("app")
    if root.handlers:
        return root
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-5s | %(request_id)s | %(message)s")
    for h in (RotatingFileHandler(LOG_PATH, maxBytes=5_000_000, backupCount=3, encoding="utf-8"), logging.StreamHandler()):
        h.setFormatter(fmt)
        h.addFilter(_RequestIdFilter())
        root.addHandler(h)
    root.propagate = False
    return root


logger = _setup()


def summarize(v, limit: int = 300):
    """把大型物件縮成可讀的一行（不把整張影像或陣列寫進 log）。"""
    try:
        import numpy as np
        import PIL.Image
        import torch
        if isinstance(v, np.ndarray):
            return f"ndarray(shape={v.shape}, dtype={v.dtype})"
        if isinstance(v, PIL.Image.Image):
            return f"PIL.Image(size={v.size}, mode={v.mode})"
        if isinstance(v, torch.Tensor):
            return f"Tensor(shape={tuple(v.shape)}, dtype={v.dtype}, device={v.device})"
    except ImportError:
        pass
    if isinstance(v, (bytes, bytearray)):
        return f"bytes(len={len(v)})"
    if isinstance(v, dict):
        s = "{" + ", ".join(f"{k}: {summarize(x, 120)}" for k, x in list(v.items())[:12]) + ("…" if len(v) > 12 else "") + "}"
    elif isinstance(v, (list, tuple)):
        s = f"[{', '.join(summarize(x, 120) for x in v[:6])}{f', …(+{len(v) - 6})' if len(v) > 6 else ''}]"
    elif hasattr(v, "__dataclass_fields__"):
        s = f"{type(v).__name__}(" + ", ".join(f"{k}={summarize(getattr(v, k), 80)}" for k in v.__dataclass_fields__) + ")"
    elif hasattr(v, "filename") and hasattr(v, "file"):  # UploadFile
        s = f"UploadFile(filename={v.filename})"
    elif isinstance(v, float):
        s = f"{v:.4g}"
    else:
        s = repr(v)
    return s if len(s) <= limit else s[:limit] + f"…(len={len(s)})"


def log_call(fn=None, *, name: str = None):
    """記錄函式的輸入、輸出（或錯誤）與耗時：  [name]: input=..., output=..., elapsed=..ms"""
    def deco(f):
        label = name or f"{f.__module__.split('.')[-1]}.{f.__qualname__}"
        sig = inspect.signature(f)

        def fmt_input(args, kwargs):
            try:
                bound = sig.bind_partial(*args, **kwargs)
                items = {k: v for k, v in bound.arguments.items() if k != "self"}
            except TypeError:
                items = {"args": args, "kwargs": kwargs}
            return ", ".join(f"{k}={summarize(v)}" for k, v in items.items()) or "-"

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            inp, t0 = fmt_input(args, kwargs), time.perf_counter()
            try:
                out = f(*args, **kwargs)
            except Exception as e:
                logger.error(f"[{label}]: input=({inp}), error={type(e).__name__}: {e}", exc_info=True)
                raise
            logger.info(f"[{label}]: input=({inp}), output={summarize(out)}, elapsed={(time.perf_counter() - t0) * 1e3:.1f}ms")
            return out
        return wrapper
    return deco(fn) if fn else deco
