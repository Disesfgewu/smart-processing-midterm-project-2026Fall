"""HyDE 風格的提示擴展：用本機 Ollama 的小型 LLM，把使用者提示改寫成數個「短的英文外觀描述」。

為什麼需要：OWL-ViT 的文字編碼器是英文 CLIP，只認得普通的視覺概念——
  * 中文、型號、行話（"螺絲"、"M4 hex nut"）分數很低甚至完全偵測不到；
  * 文字上限 16 token（含首尾特殊 token），所以描述必須很短，不能像一般 HyDE 生成整段文章。
做法：每個原始提示（概念）→ LLM 產生 n 個短描述 → 全部作為 OWL 查詢，同一物件取分數最高者並歸回原概念。

環境變數：OLLAMA_URL（預設 http://127.0.0.1:11434）、HYDE_MODEL、HYDE_VARIANTS（預設 3）、HYDE_TIMEOUT（秒）
"""
import json
import os
import re
import shutil
import subprocess
import threading
import time
from collections import OrderedDict
from typing import Callable, Dict, List, Optional

import requests

from applog import logger

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
DEFAULT_MODEL = os.environ.get("HYDE_MODEL", "qwen3:4b")
N_VARIANTS = int(os.environ.get("HYDE_VARIANTS", "5"))   # 每個提示詞產生幾個類似詞句；加上原始提示共一組 N+1 個
TIMEOUT_S = float(os.environ.get("HYDE_TIMEOUT", "90"))
AUTOSTART = os.environ.get("HYDE_AUTOSTART", "1") != "0"   # Ollama 沒在跑時，由 server 自動啟動
OLLAMA_LOG = os.environ.get("OLLAMA_LOG", "/tmp/ollama.log")
MAX_WORDS = 8          # 每個描述最多幾個字（OWL 上限 16 token）
OWL_MAX_TOKENS = 16

SYSTEM = (
    "You help an open-vocabulary object detector (OWL-ViT, a CLIP-based model). "
    "It only understands plain English visual descriptions, not part numbers, brand names or jargon. "
    "Given an object name (it may be in any language), output a group of similar phrases that all refer to the same kind of object, "
    "so the detector can try each of them.\n"
    "Output format: JSON only, exactly {\"phrases\": [\"phrase 1\", \"phrase 2\", ...]} with the requested number of phrases.\n"
    "Rules:\n"
    "1. Translate the name into English.\n"
    "2. Phrase 1 is the plain, common English name with an article, e.g. \"a hex nut\".\n"
    "3. The other phrases are synonyms, alternative common names, or short visual descriptions (shape, material, color) of the same kind of object. "
    "Every phrase must be clearly different from the others.\n"
    f"4. Every phrase has at most {MAX_WORDS} words. No sentences, no punctuation, no part numbers, no sizes, no brands.\n"
    "5. Never describe the background or a different kind of object.\n"
    "Example for \"screw\" with 3 phrases: {\"phrases\": [\"a screw\", \"a metal screw\", \"a threaded fastener\"]}"
)


def _schema(n: int) -> Dict:
    return {"type": "object", "properties": {"phrases": {"type": "array", "items": {"type": "string"}, "minItems": n, "maxItems": n}},
            "required": ["phrases"]}


class HydeError(RuntimeError):
    """Ollama 不可用、模型不存在、逾時或輸出無法使用。"""


_cache: "OrderedDict[tuple, List[str]]" = OrderedDict()
_cache_lock = threading.Lock()


# ---------------- Ollama 服務的自動啟動（只管理「由 server 自己啟動」的那一個） ----------------
_proc: Optional[subprocess.Popen] = None
_proc_lock = threading.Lock()
_last_try = 0.0


def find_binary() -> Optional[str]:
    cand = os.path.expanduser("~/.local/ollama/bin/ollama")
    return cand if os.access(cand, os.X_OK) else shutil.which("ollama")


def reachable(timeout: float = 1.5) -> bool:
    try:
        return requests.get(f"{OLLAMA_URL}/api/version", timeout=timeout).ok
    except Exception:
        return False


def ensure_running(wait_s: float = 25.0, cooldown_s: float = 0.0) -> bool:
    """Ollama 已在跑 → True；否則（AUTOSTART 且找得到執行檔）啟動並等待就緒。失敗回傳 False（不丟例外）。"""
    global _proc, _last_try
    if reachable():
        return True
    if not AUTOSTART:
        return False
    with _proc_lock:
        if reachable():
            return True
        if time.time() - _last_try < cooldown_s:
            return False
        _last_try = time.time()
        binary = find_binary()
        if not binary:
            logger.warning("[hyde.ensure_running]: input=(), output=ollama binary not found (run installation/install_ollama.sh)")
            return False
        if _proc is None or _proc.poll() is not None:
            env = {**os.environ, "OLLAMA_HOST": OLLAMA_URL.replace("http://", "").replace("https://", "")}
            _proc = subprocess.Popen([binary, "serve"], env=env, stdout=open(OLLAMA_LOG, "ab"), stderr=subprocess.STDOUT, start_new_session=True)
            logger.info(f"[hyde.ensure_running]: input=(binary={binary}), output=started ollama serve pid={_proc.pid}, log={OLLAMA_LOG}")
        t0 = time.time()
        while time.time() - t0 < wait_s:
            if reachable():
                logger.info(f"[hyde.ensure_running]: output=ollama ready after {time.time() - t0:.1f}s")
                return True
            if _proc.poll() is not None:
                logger.error(f"[hyde.ensure_running]: output=ollama exited with code {_proc.returncode}; see {OLLAMA_LOG}")
                return False
            time.sleep(0.3)
    logger.warning(f"[hyde.ensure_running]: output=ollama not ready after {wait_s}s")
    return False


def stop_started():
    """server 結束時，只關掉由 server 啟動的 Ollama（使用者自己啟動的不動）。"""
    global _proc
    if _proc is not None and _proc.poll() is None:
        _proc.terminate()
        try:
            _proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            _proc.kill()
        logger.info("[hyde.stop_started]: input=(), output=stopped the ollama started by this server")
    _proc = None


def status() -> Dict:
    """Ollama 是否可連線、設定的模型是否已下載（前端用來決定要不要啟用 HyDE 開關）。"""
    info = {"url": OLLAMA_URL, "model": DEFAULT_MODEL, "reachable": False, "model_ready": False, "models": [], "variants": N_VARIANTS,
            "autostart": AUTOSTART, "binary": find_binary(), "started_by_server": _proc is not None and _proc.poll() is None}
    if not reachable(1.0):
        if AUTOSTART and not _proc_lock.locked():   # 狀態輪詢時順便嘗試拉起（30 秒冷卻；啟動中就略過，絕不阻塞）
            ensure_running(wait_s=0.0, cooldown_s=30.0)
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        r.raise_for_status()
        info["reachable"] = True
        info["models"] = [m["name"] for m in r.json().get("models", [])]
        info["model_ready"] = any(n == DEFAULT_MODEL or n.split(":")[0] == DEFAULT_MODEL for n in info["models"])
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"
    return info


def _clean(phrase: str) -> str:
    p = re.sub(r"[^A-Za-z0-9 '\-]", " ", phrase)           # OWL 只吃英文字母；去掉標點與非英文字元
    return re.sub(r"\s+", " ", p).strip().lower()


def _valid(phrase: str, token_counter: Optional[Callable[[str], int]]) -> bool:
    if not phrase or len(phrase.split()) > MAX_WORDS:
        return False
    return token_counter is None or token_counter(phrase) <= OWL_MAX_TOKENS


def _ask(concept: str, n: int, model: str) -> List[str]:
    body = {
        "model": model, "stream": False, "format": _schema(n), "think": False, "keep_alive": "10m",
        "options": {"temperature": 0.3, "num_predict": 40 + 24 * n, "seed": 7},
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": f'Object: "{concept}"\nGive exactly {n} phrases.'}],
    }
    if not reachable() and not ensure_running():          # 擴展是「開始偵測之前」的明確步驟，這裡可以等服務啟動
        raise HydeError(f"無法連線 Ollama（{OLLAMA_URL}）。"
                        + ("找不到 ollama 執行檔，請先執行 ./installation/install_ollama.sh" if not find_binary() else
                           f"已嘗試自動啟動但失敗，請看 {OLLAMA_LOG}，或手動執行 ./installation/start_ollama.sh"))
    try:
        r = requests.post(f"{OLLAMA_URL}/api/chat", json=body, timeout=TIMEOUT_S)
    except requests.exceptions.ConnectionError:
        raise HydeError(f"無法連線 Ollama（{OLLAMA_URL}）；請先執行 ./installation/start_ollama.sh")
    except requests.exceptions.Timeout:
        raise HydeError(f"Ollama 逾時（>{TIMEOUT_S:.0f}s）")
    if r.status_code == 404:
        raise HydeError(f"Ollama 找不到模型 {model}；請先 ollama pull {model}")
    if not r.ok:
        raise HydeError(f"Ollama 錯誤 {r.status_code}: {r.text[:200]}")
    content = r.json().get("message", {}).get("content", "")
    try:
        return [str(x) for x in json.loads(content)["phrases"]]
    except Exception:
        raise HydeError(f"LLM 輸出不是預期的 JSON：{content[:120]!r}")


def lookup_cached(prompts: List[str], n: int = N_VARIANTS, model: str = DEFAULT_MODEL) -> Optional[Dict[str, List[str]]]:
    """只讀快取：全部提示詞都有擴展結果才回傳 {提示詞: [描述...]}，否則 None。**絕不呼叫 LLM**，供偵測路徑使用。"""
    out = {}
    with _cache_lock:
        for c in prompts:
            hit = _cache.get((c, n, model))
            if hit is None:
                return None
            out[c] = list(hit)
    return out


def expand_prompts(prompts: List[str], n: int = N_VARIANTS, model: str = DEFAULT_MODEL,
                   token_counter: Optional[Callable[[str], int]] = None) -> Dict:
    """【會呼叫 LLM，可能需要數秒～十幾秒；只能在「開始偵測之前」由 /api/hyde/expand 呼叫，不可放進偵測迴圈。】
    回傳 {"groups": {原提示: [描述...]}, "model", "elapsed_ms", "cached": {原提示: bool}, "dropped": {原提示: [被丟棄的描述]}}。

    groups 的第一個描述一定是「原提示本身（若為英文且合法）」或 LLM 給的標準名稱，之後是替代描述。
    失敗時丟出 HydeError（呼叫端改用原始提示並回報原因）。
    """
    t0 = time.perf_counter()
    groups, cached, dropped = {}, {}, {}
    for concept in prompts:
        key = (concept, n, model)
        with _cache_lock:
            hit = _cache.get(key)
            if hit is not None:
                _cache.move_to_end(key)
        if hit is not None:
            groups[concept], cached[concept] = list(hit), True
            continue
        t1 = time.perf_counter()
        raw = _ask(concept, n, model)
        cleaned, bad = [], []
        for p in [_clean(x) for x in raw]:
            if p in cleaned:
                continue
            (cleaned if _valid(p, token_counter) else bad).append(p)
        orig = _clean(concept)
        if orig and orig == concept.strip().lower() and _valid(orig, token_counter) and orig not in cleaned:
            cleaned.insert(0, orig)          # 英文原提示保留，作為基準查詢
        if not cleaned:
            raise HydeError(f"「{concept}」的描述全部不合格（過長或非英文）：{raw}")
        groups[concept], cached[concept], dropped[concept] = cleaned[: n + 1], False, bad
        with _cache_lock:
            _cache[key] = groups[concept]
            while len(_cache) > 128:
                _cache.popitem(last=False)
        logger.info(f"[hyde.expand_prompts]: input=(concept={concept!r}, n={n}, model={model}), output={groups[concept]}, "
                    f"dropped={bad}, elapsed={(time.perf_counter() - t1) * 1e3:.0f}ms")
    return {"groups": groups, "model": model, "elapsed_ms": (time.perf_counter() - t0) * 1e3, "cached": cached, "dropped": dropped}
