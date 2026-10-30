"use strict";
const $ = (id) => document.getElementById(id);
const COLORS = ["#16a34a", "#f97316", "#3b82f6", "#d946ef", "#06b6d4", "#eab308"];
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const state = {
  mode: "image", file: null, src: null, srcW: 0, srcH: 0, result: null, hidden: new Set(), paths: [],
  busy: false, lastRid: null, runs: [], latencies: [], expected: {},
  loop: { on: false, n: 0, fps: 0 },          // 瀏覽器攝影機 / 影片分析迴圈
  video: { url: null, timeline: [] }, srv: { on: false, timer: null },
};

const fmt = (v, d = 1) => (v == null || Number.isNaN(v) ? "—" : Number(v).toFixed(d));
const fmtInt = (v) => (v == null ? "—" : Math.round(v).toLocaleString());
const el = (tag, props = {}, ...kids) => { const e = Object.assign(document.createElement(tag), props); kids.forEach((k) => e.append(k)); return e; };
const colorOf = (label, prompts) => COLORS[Math.max(0, prompts.indexOf(label)) % COLORS.length];
const maxSide = () => +$("maxside").value || 640;

// ================= HyDE：一律在「開始偵測之前」完成擴展 =================
// 擴展要呼叫 LLM（每個提示詞數秒），放進偵測迴圈會卡住影片 / 攝影機。所以：
//  * 按下開始（執行 / 開始分析 / 開啟攝影機 / 開始串流）時先擴展完，才送第一幀；
//  * 偵測用的提示詞是「擴展當下凍結的那一組」（state.hyde.prompts），偵測期間改文字不會立刻生效；
//  * 偵測中要換提示詞（HyDE 開著）：按「套用」→ 暫停偵測 → 擴展 → 繼續。HyDE 關閉時仍即時生效。
state.hyde = { key: "", prompts: [], ok: false, info: null, promise: null };
state.hydeOn = false;   // 目前這輪偵測是否使用 HyDE（開始 / 套用時決定）
state.paused = false;   // 套用提示詞期間，迴圈暫停送幀
state.inflight = 0;     // 進行中的偵測請求數；套用時要等它歸零才開始擴展，確保擴展與偵測不重疊
const promptKey = (ps) => ps.join("\u0001");
const isRunning = () => state.loop.on || state.srv.on;
function activeSet() {
  return state.hydeOn ? { prompts: state.hyde.prompts, hyde: true } : { prompts: parsePrompts($("prompts").value), hyde: false };
}
function hydePending() {
  const ck = $("hyde").checked;
  return ck !== state.hydeOn || (ck && promptKey(parsePrompts($("prompts").value)) !== state.hyde.key);
}
function updateHydeChip() {
  const st = $("hyde-state"), b = $("btn-hyde"), running = isRunning(), pending = hydePending(), expanding = !!state.hyde.promise;
  b.disabled = expanding || (!$("hyde").checked && !state.hydeOn);
  b.textContent = running && pending ? "套用提示詞（暫停偵測）" : "擴展提示詞";
  b.classList.toggle("primary", running && pending);
  if (expanding) st.textContent = "HyDE 擴展中…（完成前不會開始偵測）";
  else if (!$("hyde").checked && !state.hydeOn) st.textContent = "";
  else if (pending && running) st.textContent = "⚠ 提示詞已變更：目前偵測仍使用舊提示詞，按「套用」才會生效";
  else if (pending) st.textContent = "尚未擴展：按開始時會先擴展完才偵測";
  else if (state.hydeOn && state.hyde.info) st.textContent = `✓ 已擴展（${state.hyde.info.model} · 耗時 ${fmt(state.hyde.info.elapsed_ms, 0)} ms）`;
  st.className = "muted" + (pending && running ? " st-ng" : "");
}
async function ensureExpanded() {
  if (state.hyde.promise) return state.hyde.promise;
  const ck = $("hyde").checked, ps = parsePrompts($("prompts").value);
  if (!ck) { state.hydeOn = false; updateHydeChip(); return true; }
  if (!ps.length) { showError("請至少輸入一個提示詞"); return false; }
  if (state.hyde.ok && state.hyde.key === promptKey(ps)) { state.hydeOn = true; updateHydeChip(); return true; }
  state.hyde.promise = (async () => {
    $("busy").textContent = "HyDE 擴展中…（約每個提示詞 3–4 秒；完成後才開始偵測）"; $("busy").hidden = false; updateHydeChip();
    try {
      const fd = new FormData(); fd.append("prompts", JSON.stringify(ps));
      const res = await fetch("/api/hyde/expand", { method: "POST", body: fd });
      const b = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(b.detail || `HTTP ${res.status}`);
      Object.assign(state.hyde, { key: promptKey(ps), prompts: ps, ok: true, info: b }); state.hydeOn = true; showError("");
      drawHydeCard(b.groups, null);
      return true;
    } catch (e) { showError("HyDE 擴展失敗：" + e.message + "。可取消勾選「使用 HyDE」後直接偵測。"); return false; }
    finally { state.hyde.promise = null; $("busy").hidden = true; updateHydeChip(); }
  })();
  return state.hyde.promise;
}
async function applyPrompts() {                     // 偵測中換提示詞：暫停 → 擴展 → 繼續（伺服器串流則重啟串流）
  if (state.hyde.promise) return;
  if (state.srv.on) { await serverStop(true); if (await ensureExpanded()) await serverStart(); return; }
  if (state.loop.on) {
    state.paused = true;                                  // 1) 不再送新幀
    try { while (state.inflight > 0) await sleep(25); await ensureExpanded(); }   // 2) 等進行中的那一幀做完，3) 才開始擴展
    finally { state.paused = false; updateHydeChip(); }
    return;
  }
  await ensureExpanded();
}
const waitIfPaused = async () => { while (state.paused && state.loop.on) await sleep(100); };

function parsePrompts(text) {
  const seen = new Set();
  return text.split(/[\n,，]/).map((s) => s.trim()).filter((s) => s && !seen.has(s) && seen.add(s));
}
function showError(msg, warn = false) { const e = $("error"); e.hidden = !msg; e.textContent = msg || ""; e.classList.toggle("warn", warn); }

// ================= 來源模式 =================
function setMode(m) {
  stopAll();
  state.mode = m; state.result = null;
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("on", t.dataset.mode === m));
  ["image", "video", "cam", "server"].forEach((x) => ($("panel-" + x).hidden = x !== m));
  $("card-timeline").hidden = m !== "video" || !state.video.timeline.length;
  $("btn-run").hidden = m !== "image"; $("btn-bench").hidden = m !== "image";
  $("bench-n").closest("label").hidden = m !== "image";
  $("stream").hidden = m !== "server"; $("canvas").hidden = m === "server";
  $("empty").hidden = m === "image" ? !!state.src : m === "server";
  $("empty").textContent = { image: "尚未選擇圖片", video: "尚未選擇影片", cam: "尚未開啟攝影機", server: "" }[m];
  showError("");
}
document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => setMode(t.dataset.mode)));

function stopAll() {
  state.loop.on = false;
  if (state.camStream) { state.camStream.getTracks().forEach((t) => t.stop()); state.camStream = null; }
  if (state.srv.on) serverStop(true);
  $("btn-camstart").disabled = false; $("btn-camstop").disabled = true;
  $("btn-vstart").disabled = !state.video.url; $("btn-vstop").disabled = true; state.paused = false; updateHydeChip(); rebuildCheckRows();
}

// ================= 取得影格 / 呼叫後端 =================
function grab(srcEl, w, h) {
  const s = Math.min(1, maxSide() / Math.max(w, h)), c = document.createElement("canvas");
  c.width = Math.round(w * s); c.height = Math.round(h * s);
  c.getContext("2d").drawImage(srcEl, 0, 0, c.width, c.height);
  return c;
}
const toBlob = (canvas) => new Promise((r) => canvas.toBlob(r, "image/jpeg", 0.85));

async function callSegment(blob, quiet = false) {
  const { prompts, hyde } = activeSet();
  if (!prompts.length) throw new Error("請至少輸入一個提示詞");
  const fd = new FormData();
  fd.append("image", blob, blob.name || "frame.jpg");
  fd.append("prompts", JSON.stringify(prompts));
  fd.append("threshold", $("threshold").value);
  fd.append("nms_iou", $("nms").value);
  fd.append("use_hyde", hyde ? "true" : "false");   // 偵測端點只查快取，不會呼叫 LLM
  fd.append("quiet", quiet ? "true" : "false");
  fd.append("with_masks", $("masks").checked ? "true" : "false");
  fd.append("top_per_group", $("top1").checked ? "true" : "false");
  const t0 = performance.now();
  state.inflight++;
  let res, body, rid;
  try { res = await fetch("/api/segment", { method: "POST", body: fd }); rid = res.headers.get("X-Request-ID"); body = await res.json().catch(() => ({})); }
  finally { state.inflight--; }
  if (!res.ok) throw new Error(body.detail || body.error || `HTTP ${res.status}`);
  body.client_ms = performance.now() - t0; body.request_id = body.request_id || rid;
  return body;
}
// 後端對過大的圖（長邊 > 4096）會先縮小再偵測，回傳的框 / 輪廓 / 面積都在縮小後的座標系。
// 這裡一次換算成「畫面上那張圖」的座標系，下游（畫圖、面積、比例尺、報表）就都一致。影片 / 攝影機的影格本來就一樣大，不受影響。
function toDisplayCoords(r, w, h) {
  const sw = r.image.width, sh = r.image.height;
  if (!sw || !sh || (Math.abs(sw - w) <= 1 && Math.abs(sh - h) <= 1)) return;
  const sx = w / sw, sy = h / sh, sa = sx * sy;
  r.processed = { width: sw, height: sh };
  r.detections.forEach((d) => {
    d.box = [d.box[0] * sx, d.box[1] * sy, d.box[2] * sx, d.box[3] * sy]; d.box_area *= sa;
    if (d.mask_area != null) d.mask_area *= sa;
    d.contours = d.contours.map((c) => c.map(([x, y]) => [x * sx, y * sy]));
  });
  r.summary.forEach((s) => { s.box_area *= sa; if (s.mask_area != null) s.mask_area *= sa; });
  r.image = { width: w, height: h };
  if (Math.abs(sx - sy) / sx > 0.02) showError(`後端與瀏覽器看到的影像長寬比不同（${sw}×${sh} vs ${w}×${h}），疊圖可能變形；常見原因是 EXIF 方向`, true);
}
function accept(r, src, w, h, { history = true } = {}) {
  toDisplayCoords(r, w, h);
  state.src = src; state.srcW = w; state.srcH = h; state.result = r; state.lastRid = r.request_id;
  if (r.hyde.requested && !r.hyde.applied) showError(`HyDE：${r.hyde.note}`, true);
  if (history) pushRun(r);
  render(r);
}

// ================= 圖片 =================
function setFile(file) {
  if (!file || !file.type.startsWith("image/")) return showError("請選擇圖片檔");
  showError(""); state.file = file;
  const img = new Image();
  img.onload = () => {
    state.src = img; state.srcW = img.naturalWidth; state.srcH = img.naturalHeight; state.result = null;
    $("drop").classList.add("has");
    $("drop-text").textContent = `${file.name} · ${img.naturalWidth}×${img.naturalHeight} · ${(file.size / 1024).toFixed(0)} KB（點擊更換）`;
    $("empty").hidden = true; $("btn-run").disabled = $("btn-bench").disabled = false; draw();
  };
  img.src = URL.createObjectURL(file);
}
const dropZone = (zoneId, onFile) => {
  const z = $(zoneId);
  ["dragenter", "dragover"].forEach((t) => z.addEventListener(t, (e) => { e.preventDefault(); z.classList.add("over"); }));
  ["dragleave", "drop"].forEach((t) => z.addEventListener(t, (e) => { e.preventDefault(); z.classList.remove("over"); }));
  z.addEventListener("drop", (e) => onFile(e.dataTransfer.files[0]));
};
$("file").addEventListener("change", (e) => setFile(e.target.files[0])); dropZone("drop", setFile);
document.addEventListener("paste", (e) => { const f = [...(e.clipboardData?.files || [])].find((x) => x.type.startsWith("image/")); if (f && state.mode === "image") setFile(f); });

async function run() {
  if (!state.file || state.busy) return;
  state.busy = true; showError(""); $("btn-run").disabled = true;
  try {
    if (!(await ensureExpanded())) return;              // HyDE：先擴展完，才開始偵測
    $("busy").textContent = "推論中…"; $("busy").hidden = false;
    accept(await callSegment(state.file), state.src, state.srcW, state.srcH); refreshLogs(); }
  catch (e) { showError(e.message); }
  finally { state.busy = false; $("busy").hidden = true; $("btn-run").disabled = false; }
}
$("btn-run").addEventListener("click", run);

// ================= 重複量測（圖片） =================
const stats = (a) => {
  const s = [...a].sort((x, y) => x - y), mean = a.reduce((x, y) => x + y, 0) / a.length;
  return { mean, median: s[Math.floor((s.length - 1) / 2)], p95: s[Math.floor(0.95 * (s.length - 1))], min: s[0], max: s[s.length - 1],
           std: Math.sqrt(a.reduce((x, y) => x + (y - mean) ** 2, 0) / a.length) };
};
async function bench() {
  if (!state.file || state.busy) return;
  const n = Math.max(2, Math.min(200, +$("bench-n").value || 10));
  state.busy = true; showError(""); $("btn-bench").disabled = $("btn-run").disabled = true;
  const totals = [], owl = [], sam = [], client = []; let last;
  try {
    if (!(await ensureExpanded())) return;

    for (let i = 0; i < n; i++) {
      $("busy").hidden = false; $("busy").textContent = `量測中 ${i + 1}/${n}`;
      last = await callSegment(state.file, i % 10 !== 0);
      if (i > 0) { totals.push(last.timing.total_ms); owl.push(last.timing.owl_ms); sam.push(last.timing.sam_ms); client.push(last.client_ms); } // 第 1 次丟棄
      pushRun(last);
    }
    accept(last, state.src, state.srcW, state.srcH, { history: false }); renderBench({ totals, owl, sam, client, n });
  } catch (e) { showError(e.message); }
  finally { state.busy = false; $("busy").hidden = true; $("busy").textContent = "推論中…"; $("btn-bench").disabled = $("btn-run").disabled = false; refreshLogs(); }
}
$("btn-bench").addEventListener("click", bench);
function renderBench(b) {
  const kv = $("bench-kv"); kv.replaceChildren();
  const add = (k, v) => kv.append(el("dt", { textContent: k }), el("dd", { textContent: v }));
  add("有效樣本", `${b.totals.length}（共 ${b.n} 次，捨棄第 1 次）`);
  for (const [name, arr] of [["後端總計", b.totals], ["OWL", b.owl], ["SAM", b.sam], ["含網路往返", b.client]]) {
    const s = stats(arr);
    add(name + " mean / median / p95", `${fmt(s.mean)} / ${fmt(s.median)} / ${fmt(s.p95)} ms`);
    add(name + " min – max (std)", `${fmt(s.min)} – ${fmt(s.max)} (${fmt(s.std)}) ms`);
  }
  add("等效 FPS（後端總計）", fmt(1000 / stats(b.totals).mean));
  $("card-bench").hidden = false;
}

// ================= 影片檔（瀏覽器端逐段取樣） =================
const vid = $("vid");
function setVideo(file) {
  if (!file || !file.type.startsWith("video/")) return showError("請選擇影片檔");
  showError(""); stopAll();
  if (state.video.url) URL.revokeObjectURL(state.video.url);
  state.video.url = URL.createObjectURL(file); state.video.timeline = []; $("card-timeline").hidden = true;
  vid.src = state.video.url;
  vid.onloadedmetadata = () => {
    $("vdrop-text").textContent = `${file.name} · ${vid.videoWidth}×${vid.videoHeight} · ${vid.duration.toFixed(1)} 秒（點擊更換）`;
    $("drop-video").classList.add("has"); $("btn-vstart").disabled = false; $("empty").hidden = true;
    vid.currentTime = 0;
    vid.onseeked = () => { vid.onseeked = null; const c = grab(vid, vid.videoWidth, vid.videoHeight); state.src = c; state.srcW = c.width; state.srcH = c.height; draw(); };
  };
  vid.onerror = () => showError("瀏覽器無法解碼這個影片（請改用 H.264 mp4 或 webm）");
}
$("vfile").addEventListener("change", (e) => setVideo(e.target.files[0])); dropZone("drop-video", setVideo);

const seekTo = (t) => new Promise((res) => {
  if (Math.abs(vid.currentTime - t) < 1e-3) return res();
  vid.addEventListener("seeked", () => res(), { once: true }); vid.currentTime = t;
});
async function analyzeVideo() {
  if (!state.video.url || state.loop.on || state.hyde.promise) return;
  $("btn-vstart").disabled = true;
  if (!(await ensureExpanded())) { $("btn-vstart").disabled = false; return; }   // HyDE：先擴展完，才開始分析
  const step = Math.max(0.05, +$("v-step").value || 0.5), t0 = Math.max(0, +$("v-from").value || 0), dur = vid.duration;
  state.loop = { on: true, n: 0, fps: 0 }; state.video.timeline = []; showError("");
  $("btn-vstart").disabled = true; $("btn-vstop").disabled = false; $("v-prog").hidden = false; $("v-prog").max = dur; $("card-timeline").hidden = false;
  // 準備某個時間點的影格（seek → 擷取 → JPEG）。推論第 N 幀的同時就開始準備第 N+1 幀。
  const prepare = async (t) => { await seekTo(Math.min(t, dur - 0.01)); await sleep(30); const c = grab(vid, vid.videoWidth, vid.videoHeight); return { t, c, blob: await toBlob(c) }; };
  try {
    let cur = t0 < dur ? await prepare(t0) : null;
    for (let i = 0; cur && state.loop.on; i++) {
      await waitIfPaused();
      const tf = performance.now(), nextT = cur.t + step, nextP = nextT < dur ? prepare(nextT) : Promise.resolve(null);
      const r = await callSegment(cur.blob, i >= 3 && i % 10 !== 0);
      if (!state.loop.on) break;
      accept(r, cur.c, cur.c.width, cur.c.height, { history: i % 5 === 0 });
      state.video.timeline.push({ t: cur.t, counts: Object.fromEntries(r.summary.map((s) => [s.label, s.count])), areas: Object.fromEntries(r.summary.map((s) => [s.label, s.mask_area ?? 0])), ms: r.timing.total_ms });
      $("v-prog").value = cur.t; $("v-info").textContent = `分析中 ${cur.t.toFixed(1)} / ${dur.toFixed(1)} 秒 · 已處理 ${state.video.timeline.length} 個取樣點`;
      renderTimeline(); if (i % 5 === 0) refreshLogs();
      cur = await nextP; loopFps(tf);
    }
    $("v-info").textContent = state.loop.on ? `完成：${state.video.timeline.length} 個取樣點` : `已停止：${state.video.timeline.length} 個取樣點`;
  } catch (e) { showError(e.message); }
  finally { state.loop.on = false; $("btn-vstart").disabled = false; $("btn-vstop").disabled = true; $("v-prog").hidden = true; refreshLogs(); }
}
$("btn-vstart").addEventListener("click", analyzeVideo);
$("btn-vstop").addEventListener("click", () => { state.loop.on = false; });

function loopFps(tStart) { const dt = performance.now() - tStart; state.loop.fps = state.loop.n++ === 0 ? 1000 / dt : 0.85 * state.loop.fps + 0.15 * (1000 / dt); $("loop-fps").textContent = `${fmt(state.loop.fps)} FPS（含網路）`; }

function renderTimeline() {
  const tl = state.video.timeline, c = $("timeline"), g = c.getContext("2d"), W = c.width, H = c.height, pad = { l: 36, r: 10, t: 10, b: 22 };
  const css = getComputedStyle(document.documentElement), ink = css.getPropertyValue("--muted").trim(), line = css.getPropertyValue("--line").trim();
  g.clearRect(0, 0, W, H); if (!tl.length) return;
  const labels = Object.keys(tl[tl.length - 1].counts), tmax = Math.max(tl[tl.length - 1].t, 1e-6), cmax = Math.max(1, ...tl.flatMap((x) => Object.values(x.counts)));
  const X = (t) => pad.l + (t / tmax) * (W - pad.l - pad.r), Y = (v) => H - pad.b - (v / cmax) * (H - pad.t - pad.b);
  g.font = "11px system-ui"; g.fillStyle = ink; g.strokeStyle = line; g.lineWidth = 1;
  for (let v = 0; v <= cmax; v += Math.max(1, Math.ceil(cmax / 5))) { g.beginPath(); g.moveTo(pad.l, Y(v)); g.lineTo(W - pad.r, Y(v)); g.stroke(); g.fillText(v, 8, Y(v) + 4); }
  g.fillText("0s", pad.l, H - 6); g.fillText(tmax.toFixed(1) + "s", W - pad.r - 30, H - 6);
  labels.forEach((l, i) => {
    g.strokeStyle = COLORS[i % COLORS.length]; g.lineWidth = 2; g.beginPath();
    tl.forEach((p, k) => { const x = X(p.t), y = Y(p.counts[l] ?? 0); k ? g.lineTo(x, y) : g.moveTo(x, y); }); g.stroke();
  });
  const lg = $("tl-legend"); lg.replaceChildren();
  labels.forEach((l, i) => lg.append(el("span", { className: "chip" }, el("i", { className: "sw", style: `background:${COLORS[i % COLORS.length]}` }), l)));
  const th = $("tbl-tl").tHead, tb = $("tbl-tl").tBodies[0]; th.replaceChildren(); tb.replaceChildren();
  th.append(el("tr", {}, el("th", { textContent: "時間(s)" }), ...labels.map((l) => el("th", { className: "num", textContent: l + " 數量" })), el("th", { className: "num", textContent: "後端 ms" })));
  tl.slice(-8).reverse().forEach((p) => tb.append(el("tr", {}, el("td", { textContent: p.t.toFixed(2) }), ...labels.map((l) => el("td", { className: "num", textContent: p.counts[l] ?? 0 })), el("td", { className: "num", textContent: fmt(p.ms) }))));
}
$("btn-csv").addEventListener("click", () => {
  const tl = state.video.timeline; if (!tl.length) return;
  const labels = Object.keys(tl[tl.length - 1].counts), q = (s) => `"${String(s).replace(/"/g, '""')}"`;
  const rows = [["time_s", ...labels.map((l) => q(l + " count")), ...labels.map((l) => q(l + " mask_area_px")), "backend_ms"].join(",")];
  tl.forEach((p) => rows.push([p.t.toFixed(3), ...labels.map((l) => p.counts[l] ?? 0), ...labels.map((l) => p.areas[l] ?? 0), p.ms.toFixed(1)].join(",")));
  const a = el("a", { href: URL.createObjectURL(new Blob(["﻿" + rows.join("\n")], { type: "text/csv" })), download: "timeline.csv" }); a.click();
});

// ================= 瀏覽器攝影機 =================
async function listCams() {
  try {
    const devs = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput"), sel = $("cam-dev"), cur = sel.value;
    sel.replaceChildren(el("option", { value: "", textContent: "預設攝影機" }));
    devs.forEach((d, i) => sel.append(el("option", { value: d.deviceId, textContent: d.label || `攝影機 ${i + 1}` })));
    sel.value = cur;
  } catch { /* 沒有權限或不支援 */ }
}
async function camStart() {
  if (!navigator.mediaDevices?.getUserMedia) return showError("此瀏覽器或網址不允許使用攝影機（需 localhost 或 HTTPS）");
  showError(""); $("btn-camstart").disabled = true;
  if (!(await ensureExpanded())) { $("btn-camstart").disabled = false; return; }   // HyDE：先擴展完，才開啟攝影機迴圈
  try {
    state.camStream = await navigator.mediaDevices.getUserMedia({ video: { deviceId: $("cam-dev").value ? { exact: $("cam-dev").value } : undefined, width: { ideal: 1280 } }, audio: false });
  } catch (e) { $("btn-camstart").disabled = false; return showError("無法開啟攝影機：" + e.message); }
  listCams();
  const cam = $("cam"); cam.srcObject = state.camStream; await cam.play(); $("empty").hidden = true;
  $("btn-camstop").disabled = false; state.loop = { on: true, n: 0, fps: 0 };
  try {
    for (let i = 0; state.loop.on; i++) {
      await waitIfPaused();
      const tf = performance.now(), c = grab(cam, cam.videoWidth, cam.videoHeight);
      const r = await callSegment(await toBlob(c), i >= 3 && i % 10 !== 0);
      if (!state.loop.on) break;
      accept(r, c, c.width, c.height, { history: i % 10 === 0 }); loopFps(tf); if (i % 10 === 0) refreshLogs();
    }
  } catch (e) { showError(e.message); }
  finally { stopAll(); }
}
$("btn-camstart").addEventListener("click", camStart);
$("btn-camstop").addEventListener("click", stopAll);

// ================= 伺服器來源（攝影機 / 影片路徑，MJPEG） =================
function serverForm(extra = {}) {
  const fd = new FormData();
  const act = activeSet();
  fd.append("prompts", JSON.stringify(act.prompts));
  fd.append("threshold", $("threshold").value); fd.append("nms_iou", $("nms").value); fd.append("style", $("style").value); fd.append("with_masks", $("masks").checked ? "true" : "false"); fd.append("use_hyde", act.hyde ? "true" : "false"); fd.append("top_per_group", $("top1").checked ? "true" : "false");
  Object.entries(extra).forEach(([k, v]) => fd.append(k, v)); return fd;
}
async function serverStart() {
  showError(""); $("btn-srvstart").disabled = true;
  if (!(await ensureExpanded())) { $("btn-srvstart").disabled = false; return; }   // HyDE：先擴展完，才啟動串流
  try {
    const res = await fetch("/api/camera/start", { method: "POST", body: serverForm({ source: $("srv-src").value, max_side: maxSide() }) });
    const b = await res.json().catch(() => ({})); if (!res.ok) throw new Error(b.detail || `HTTP ${res.status}`);
  } catch (e) { $("btn-srvstart").disabled = false; return showError("無法啟動：" + e.message); }
  state.srv.on = true; $("btn-srvstop").disabled = false; $("empty").hidden = true;
  $("stream").src = "/api/camera/stream?t=" + Date.now();
  state.srv.timer = setInterval(pollServer, 400);
}
async function pollServer() {
  try {
    const s = await (await fetch("/api/camera/state")).json();
    if (s.error) { showError("伺服器來源錯誤：" + s.error); return serverStop(false); }
    if (!s.running || !s.summary) return;
    $("loop-fps").textContent = `${fmt(s.fps)} FPS（後端迴圈）`;
    const r = { ...s, hyde: s.hyde || { requested: false }, backend: "server", client_ms: null, request_id: "camera", threshold: +$("threshold").value };
    state.result = r; render(r, { skipDraw: true });
  } catch { /* 暫時失敗略過 */ }
}
async function serverStop(quiet) {
  clearInterval(state.srv.timer); state.srv.on = false; $("stream").removeAttribute("src");
  try { await fetch("/api/camera/stop", { method: "POST" }); } catch { /* 忽略 */ }
  $("btn-srvstart").disabled = false; $("btn-srvstop").disabled = true; if (!quiet) refreshLogs();
}
$("btn-srvstart").addEventListener("click", serverStart);
$("btn-srvstop").addEventListener("click", () => serverStop(false));
async function pushServerParams() {
  if (!state.srv.on) return;
  const ps = activeSet().prompts;
  const exp = Object.fromEntries(Object.entries(state.expected).filter(([k, v]) => v !== "" && ps.includes(k)).map(([k, v]) => [k, +v]));
  try { await fetch("/api/camera/prompts", { method: "POST", body: serverForm({ expected: JSON.stringify(exp) }) }); } catch { /* 忽略 */ }
}

// ================= 輸入控制項 =================
function renderPromptChips() {
  const box = $("prompt-chips"); box.replaceChildren();
  parsePrompts($("prompts").value).forEach((p, i) => box.append(el("span", { className: "chip" }, el("i", { className: "sw", style: `background:${COLORS[i % COLORS.length]}` }), p)));
}
let autoTimer = null;
$("prompts").addEventListener("input", () => {
  renderPromptChips(); rebuildCheckRows();
  clearTimeout(autoTimer);
  updateHydeChip();
  if (state.srv.on) { if (!state.hydeOn) autoTimer = setTimeout(pushServerParams, 400); }   // HyDE 關閉：串流即時換提示；開啟：按「套用」
  else if ($("auto").checked && state.mode === "image" && state.file) autoTimer = setTimeout(run, 700);
  // 瀏覽器攝影機／影片（HyDE 關閉）：迴圈每一幀都會讀取提示詞，下一幀即生效；HyDE 開啟則需按「套用」
});
const onParam = () => { if (state.srv.on) pushServerParams(); else if ($("auto").checked && state.mode === "image" && state.file) run(); };
$("threshold").addEventListener("input", (e) => { $("th-out").textContent = (+e.target.value).toFixed(2); });
$("nms").addEventListener("input", (e) => { $("nms-out").textContent = (+e.target.value).toFixed(2); });
$("threshold").addEventListener("change", onParam); $("nms").addEventListener("change", onParam);
$("style").addEventListener("change", () => { draw(); onParam(); });
$("masks").addEventListener("change", onParam); $("top1").addEventListener("change", onParam);
// SAM image encoder 切換：baseline MobileSAM + 後端掃到的自訓練 checkpoint（data/eval/*.pth、data/train/*/best.pth）；第一次選用才載入
const encLabels = {};
const encText = (e) => `${e.label}${e.epoch != null ? ` · epoch ${e.epoch}` : ""}${e.val_loss != null ? ` · val ${(+e.val_loss).toFixed(6)}` : ""}${e.loaded || e.key === "mobilesam" ? "" : "（未載入）"}`;
async function samEncInit() {
  try {
    const j = await (await fetch("/api/sam/encoder")).json(), sel = $("sam-enc");
    for (const e of j.encoders || []) encLabels[e.key] = encText(e);
    $("sam-enc-wrap").hidden = !(j.encoders && j.encoders.length > 1);
    sel.replaceChildren(...(j.encoders || []).map((e) => el("option", { value: e.key, textContent: encText(e) })));
    sel.value = j.active;
  } catch { /* 後端沒有這個 API 就不顯示 */ }
}
$("sam-enc").addEventListener("change", async (e) => {
  showError(""); const sel = e.target; sel.disabled = true;
  try {
    const fd = new FormData(); fd.append("name", sel.value);
    const res = await fetch("/api/sam/encoder", { method: "POST", body: fd }), b = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(b.detail || `HTTP ${res.status}`);
    await samEncInit();                                       // 重新整理選單（剛載入的會顯示 epoch / val loss）
    if (state.mode === "image" && state.file) run();          // 圖片模式：切換後立刻用新 encoder 重跑同一張圖，方便比較
  } catch (err) { showError("切換 SAM 編碼器失敗：" + err.message); samEncInit(); }
  finally { sel.disabled = false; }
});
async function hydeStatus() {
  const b = $("hyde-badge");
  try {
    const h = await (await fetch("/api/hyde/status")).json();
    if (!h.reachable) { b.className = "badge warn"; b.textContent = h.binary ? "Ollama 啟動中…" : "未安裝 Ollama"; b.title = (h.binary ? "後端會自動啟動；若一直失敗請執行 ./installation/start_ollama.sh，並看 /tmp/ollama.log\n" : "請執行 ./installation/install_ollama.sh\n") + (h.error || ""); }
    else if (!h.model_ready) { b.className = "badge warn"; b.textContent = `未下載 ${h.model}`; b.title = "ollama pull " + h.model; }
    else { b.className = "badge ok"; b.textContent = `Ollama · ${h.model}`; b.title = `每個提示詞產生 ${h.variants} 個描述`; }
  } catch { b.className = "badge warn"; b.textContent = "狀態未知"; }
}
$("hyde").addEventListener("change", () => { showError(""); updateHydeChip(); if (!isRunning() && $("auto").checked && state.mode === "image" && state.file) run(); });
$("btn-hyde").addEventListener("click", applyPrompts);

// ================= 繪圖 =================
const canvas = $("canvas"), ctx = canvas.getContext("2d");
function draw() {
  if (!state.src || state.mode === "server") return;
  const W = state.srcW, H = state.srcH; canvas.width = W; canvas.height = H;
  ctx.drawImage(state.src, 0, 0, W, H); state.paths = [];
  const r = state.result; if (!r) { drawCalibration(); return; }
  const style = $("style").value, lw = Math.max(2, Math.round(Math.max(W, H) / 400)), calK = calScale(r.image);
  for (const d of r.detections) {
    const color = colorOf(d.label, r.prompts), path = new Path2D(), thick = r.hyde && r.hyde.applied && d.primary ? 2 : 1;
    d.contours.forEach((c) => { c.forEach(([x, y], i) => (i ? path.lineTo(x, y) : path.moveTo(x, y))); path.closePath(); });
    state.paths.push({ d, path });
    if (state.hidden.has(d.label)) continue;
    if (d.mask_area != null && (style === "mask" || style === "both")) { ctx.globalAlpha = style === "both" ? 0.3 : 0.45; ctx.fillStyle = color; ctx.fill(path, "evenodd"); ctx.globalAlpha = 1; }
    if (d.mask_area != null && (style === "contour" || style === "both")) { ctx.lineWidth = lw * thick; ctx.strokeStyle = color; ctx.lineJoin = "round"; ctx.stroke(path); }
    if (style === "box" || d.mask_area == null) { ctx.lineWidth = lw; ctx.strokeStyle = color; const [x0, y0, x1, y1] = d.box; ctx.strokeRect(x0, y0, x1 - x0, y1 - y0); }
    const fs = Math.max(12, Math.round(Math.max(W, H) / 50)); ctx.font = `600 ${fs}px system-ui, sans-serif`;
    const star = r.hyde && r.hyde.applied && d.primary, text = `${star ? "★ " : ""}${d.label} ${d.score.toFixed(2)}${calK && d.mask_area != null ? ` · ${(d.mask_area * calK * calK).toFixed(0)} mm²` : ""}`, tw = ctx.measureText(text).width, [x0, y0] = d.box, ty = Math.max(y0, fs + 4);
    ctx.fillStyle = color; ctx.fillRect(x0, ty - fs - 4, tw + 8, fs + 6); ctx.fillStyle = "#fff"; ctx.fillText(text, x0 + 4, ty - 3);
  }
  drawCalibration();
}
canvas.addEventListener("mousemove", (e) => {
  const rect = canvas.getBoundingClientRect(), sx = canvas.width / rect.width, x = (e.clientX - rect.left) * sx, y = (e.clientY - rect.top) * sx;
  const hit = [...state.paths].reverse().find(({ d, path }) => !state.hidden.has(d.label) && ctx.isPointInPath(path, x, y, "evenodd")), tip = $("tip");
  if (!hit) { tip.hidden = true; return; }
  const d = hit.d; tip.textContent = `${d.label} · 分數 ${d.score.toFixed(2)} · 遮罩 ${fmtInt(d.mask_area)} px · 框 ${fmtInt(d.box_area)} px`; tip.hidden = false;
  const st = $("stage").getBoundingClientRect();
  tip.style.left = Math.min(e.clientX - st.left + 12, st.width - tip.offsetWidth - 4) + "px"; tip.style.top = Math.max(e.clientY - st.top - 34, 4) + "px";
});
canvas.addEventListener("mouseleave", () => { $("tip").hidden = true; });

// ================= 顯示結果 =================
function render(r, { skipDraw = false } = {}) {
  state.hidden = new Set([...state.hidden].filter((l) => r.prompts.includes(l)));
  if (!skipDraw) draw();
  renderLegend(r); renderSummary(r); renderTiming(r); renderCheck(r); renderHyde(r);
}
function renderLegend(r) {
  const box = $("legend"); box.replaceChildren();
  r.summary.forEach((s) => {
    const chip = el("span", { className: "chip" + (state.hidden.has(s.label) ? " off" : ""), role: "button", tabIndex: 0, title: "點擊顯示／隱藏" },
      el("i", { className: "sw", style: `background:${colorOf(s.label, r.prompts)}` }), `${s.label} × ${s.count}`);
    const toggle = () => { state.hidden.has(s.label) ? state.hidden.delete(s.label) : state.hidden.add(s.label); draw(); renderLegend(r); };
    chip.addEventListener("click", toggle); chip.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggle(); } });
    box.append(chip);
  });
}
function renderSummary(r) {
  const tb = $("tbl-summary").tBodies[0]; tb.replaceChildren();
  r.summary.forEach((s) => {
    const ratio = s.box_area && s.mask_area != null ? (s.mask_area / s.box_area) * 100 : null;
    const miss = s.count === 0;
    tb.append(el("tr", {}, el("td", { textContent: s.label }), el("td", { className: "num" + (miss ? " st-ng" : ""), textContent: s.count }),
      el("td", { className: "num", textContent: s.best_score == null ? "<0.01" : s.best_score.toFixed(2) }), el("td", { className: "num", textContent: fmt(s.mean_score, 2) }),
      el("td", { className: "num", textContent: fmtInt(s.mask_area) }), el("td", { className: "num", textContent: fmtInt(s.box_area) }),
      el("td", { className: "num", textContent: ratio == null ? "—" : ratio.toFixed(0) + "%" }), el("td", { className: "num", textContent: fmt(s.mean_sam_iou, 2) })));
  });
  const td = $("tbl-det").tBodies[0]; td.replaceChildren();
  r.detections.forEach((d) => td.append(el("tr", {}, el("td", { textContent: d.id }), el("td", { textContent: d.label }), el("td", { textContent: (d.primary && r.hyde && r.hyde.applied ? "★ " : "") + (d.query && d.query !== d.label ? d.query : "—"), className: "muted" }), el("td", { className: "num", textContent: d.score.toFixed(3) }),
    el("td", { textContent: d.box.map((v) => Math.round(v)).join(", ") }), el("td", { className: "num", textContent: fmtInt(d.mask_area) }),
    el("td", { className: "num", textContent: fmt(d.sam_iou, 3) }), el("td", { className: "num", textContent: d.contours.reduce((a, c) => a + c.length, 0) }))));
  renderHint(r);
}
// 沒有偵測結果時說明原因（SAM 只在 OWL 有給框時才會執行）
function renderHint(r) {
  const msgs = [], th = r.threshold ?? +$("threshold").value, bad = r.prompts.filter((p) => /[^\x00-\x7F]/.test(p));
  if (bad.length && !(r.hyde && r.hyde.applied)) msgs.push(`OWL 的文字編碼器只認英文：「${bad.join("、")}」這類非英文提示詞幾乎偵測不到，請改用英文（例如 a screw）。`);
  const miss = r.summary.filter((s) => s.count === 0);
  if (!r.detections.length) msgs.push(`OWL 在門檻 ${(+th).toFixed(2)} 下沒有找到任何物件，所以 SAM 沒有執行（沒有框就沒有遮罩，面積與 SAM 品質都會是空的）。`);
  if (miss.length) msgs.push(`沒偵測到：${miss.map((s) => `${s.label}（最高分 ${s.best_score == null ? "—" : s.best_score.toFixed(2)}${s.best_score != null && s.best_score < th ? "，低於門檻" : ""}）`).join("、")}。可降低門檻，或改用 "a …" / 更具體的英文描述。`);
  $("hint").hidden = !msgs.length; $("hint").textContent = msgs.join("\n");
}
function renderTiming(r) {
  const t = r.timing, other = Math.max(0, t.total_ms - t.text_encode_ms - t.owl_ms - t.sam_ms - t.serialize_ms);
  $("t-total").textContent = fmt(t.total_ms); $("t-fps").textContent = fmt(1000 / t.total_ms); $("req-id").textContent = `request ${r.request_id}`;
  const parts = [["文字編碼", t.text_encode_ms, "#a78bfa"], ["OWL（含 NMS）", t.owl_ms, "#3b82f6"], ["SAM", t.sam_ms, "#f97316"], ["序列化", t.serialize_ms, "#14b8a6"], ["其他", other, "#94a3b8"]];
  const stack = $("stack"), lg = $("stack-legend"); stack.replaceChildren(); lg.replaceChildren();
  parts.forEach(([n, v, c]) => { stack.append(el("div", { style: `width:${(v / t.total_ms) * 100}%;background:${c}`, title: `${n} ${fmt(v)} ms` })); lg.append(el("span", { className: "chip" }, el("i", { className: "sw", style: `background:${c}` }), `${n} ${fmt(v)}`)); });
  const kv = $("timing-kv"); kv.replaceChildren(); const add = (k, v) => kv.append(el("dt", { textContent: k }), el("dd", { textContent: v }));
  add("含網路往返", r.client_ms == null ? "—（伺服器端迴圈）" : `${fmt(r.client_ms)} ms`);
  add("連續處理速度", $("loop-fps").textContent || "—");
  add("影像大小", r.processed ? `${r.image.width}×${r.image.height}（後端縮至 ${r.processed.width}×${r.processed.height} 處理，座標已換算）` : `${r.image.width}×${r.image.height}`); add("原始框 → NMS 後", `${t.num_raw_boxes ?? "—"} → ${r.detections.length}`);
  add("SAM 編碼器", r.sam_encoder ? (encLabels[r.sam_encoder] || r.sam_encoder) : "—");
  add("SAM 平均每個框", r.detections.length ? `${fmt(t.sam_ms / r.detections.length)} ms（含整張影像編碼）` : "—"); add("後端", r.backend);
  state.latencies.push(t.total_ms); if (state.latencies.length > 40) state.latencies.shift(); spark();
}
function pushRun(r) {
  state.runs.unshift({ time: new Date().toLocaleTimeString(), prompts: r.prompts.join(", "), n: r.detections.length, owl: r.timing.owl_ms, sam: r.timing.sam_ms, total: r.timing.total_ms });
  state.runs.length = Math.min(state.runs.length, 15);
  const tb = $("tbl-hist").tBodies[0]; tb.replaceChildren();
  state.runs.forEach((x) => tb.append(el("tr", {}, el("td", { textContent: x.time }), el("td", { textContent: x.prompts }), el("td", { className: "num", textContent: x.n }),
    el("td", { className: "num", textContent: fmt(x.owl) }), el("td", { className: "num", textContent: fmt(x.sam) }), el("td", { className: "num", textContent: fmt(x.total) }))));
}
function spark() {
  const c = $("spark"), g = c.getContext("2d"), a = state.latencies, w = c.width, h = c.height; g.clearRect(0, 0, w, h); if (!a.length) return;
  const max = Math.max(...a) * 1.1, bw = w / 40; g.fillStyle = getComputedStyle(document.documentElement).getPropertyValue("--accent").trim() || "#2563eb";
  a.forEach((v, i) => { const bh = (v / max) * (h - 4); g.fillRect(i * bw + 1, h - bh, bw - 2, bh); });
}

function drawHydeCard(groups, r) {
  const info = state.hyde.info, card = $("card-hyde"), body = $("hyde-body"); card.hidden = false; body.replaceChildren();
  $("hyde-meta").textContent = info ? `${info.model} · 擴展耗時 ${fmt(info.elapsed_ms, 0)} ms${Object.values(info.cached || {}).every(Boolean) ? "（快取）" : ""}` : "";
  Object.entries(groups).forEach(([label, qs]) => {
    const dets = r ? r.detections.filter((d) => d.label === label) : [], win = new Set(dets.map((d) => d.query));
    const s = r && r.summary.find((x) => x.label === label), pr = s && s.primary ? `　★ 主要輸出：「${s.primary.query}」${s.primary.score.toFixed(2)}` : "";
    body.append(el("div", { className: "hy-row" }, el("b", { textContent: r ? `${label} → ${s ? s.count : 0} 個${pr}` : label }),
      ...qs.map((q) => el("span", { className: "chip" + (win.has(q) ? " win" : ""), title: win.has(q) ? "有物件由這個描述命中" : "" }, q))));
  });
}
function renderHyde(r) {
  const h = r.hyde || {}, card = $("card-hyde");
  if (!h.requested) { card.hidden = true; return; }
  if (!h.applied) { card.hidden = false; $("hyde-meta").textContent = ""; $("hyde-body").replaceChildren(el("div", { className: "muted", textContent: h.note || "未套用" })); return; }
  drawHydeCard(h.expanded, r);
}

// ================= 製造情境：零件檢查（工單 / 面積 mm² / 判定 / 紀錄） =================
state.areaMin = {}; state.areaMax = {}; state.cal = null; state.calMode = false; state.calPts = [];
state.log = []; state.lastLogSig = ""; state.lastLogAt = 0;
const WO_PRESETS = {
  "範本｜螺絲包（螺絲 4 / 螺帽 4 / 墊片 4）": [{ prompt: "a screw", expected: 4 }, { prompt: "a nut", expected: 4 }, { prompt: "a washer", expected: 4 }],
  "範本｜電子零件盤（電阻 10 / 電容 2 / LED 1）": [{ prompt: "a resistor", expected: 10 }, { prompt: "a capacitor", expected: 2 }, { prompt: "a led", expected: 1 }],
  "範本｜貓與遙控器（測試用，對應範例圖）": [{ prompt: "a cat", expected: 2 }, { prompt: "a remote control", expected: 2 }],
};
// localStorage 可能被封鎖（私密視窗等），一律包 try/catch，失敗時功能照常、只是不保存
const lsGet = (k, d) => { try { const v = localStorage.getItem(k); return v ? JSON.parse(v) : d; } catch { return d; } };
const lsSet = (k, v) => { try { localStorage.setItem(k, JSON.stringify(v)); return true; } catch { return false; } };
const isNum = (v) => v !== "" && v != null && !Number.isNaN(+v);

// ---- 工單 ----
function refreshWoSelect(selected) {
  const sel = $("wo-sel"), saved = lsGet("workorders", {}); sel.replaceChildren(el("option", { value: "", textContent: "（選擇工單）" }));
  Object.keys(WO_PRESETS).forEach((n) => sel.append(el("option", { value: "p:" + n, textContent: n })));
  Object.keys(saved).forEach((n) => sel.append(el("option", { value: "s:" + n, textContent: "已存｜" + n })));
  if (selected) sel.value = selected; $("wo-del").disabled = !sel.value.startsWith("s:");
}
function currentWorkOrder() {
  return checkPrompts().map((p) => ({ prompt: p, expected: isNum(state.expected[p]) ? +state.expected[p] : null,
    areaMin: isNum(state.areaMin[p]) ? +state.areaMin[p] : null, areaMax: isNum(state.areaMax[p]) ? +state.areaMax[p] : null }));
}
function applyWorkOrder() {
  const v = $("wo-sel").value; if (!v) return showError("請先選擇一個工單");
  const name = v.slice(2), parts = v.startsWith("p:") ? WO_PRESETS[name] : lsGet("workorders", {})[name];
  if (!parts) return showError("找不到這個工單");
  showError(""); state.expected = {}; state.areaMin = {}; state.areaMax = {};
  parts.forEach((x) => { state.expected[x.prompt] = x.expected ?? ""; state.areaMin[x.prompt] = x.areaMin ?? ""; state.areaMax[x.prompt] = x.areaMax ?? ""; });
  state.woName = name.replace(/^範本｜/, ""); $("wo-name").value = state.woName;
  const ta = $("prompts"); ta.value = parts.map((x) => x.prompt).join("\n"); ta.dispatchEvent(new Event("input", { bubbles: true }));   // 走既有流程（HyDE 待套用、自動執行、串流更新）
  rebuildCheckRows(true); updateCheck(state.result);
}
$("wo-apply").addEventListener("click", applyWorkOrder);
$("wo-sel").addEventListener("change", () => { $("wo-del").disabled = !$("wo-sel").value.startsWith("s:"); });
$("wo-save").addEventListener("click", () => {
  const name = $("wo-name").value.trim(); if (!name) return showError("請先輸入工單名稱");
  const parts = currentWorkOrder(); if (!parts.length) return showError("沒有可儲存的零件（請先輸入提示詞）");
  const saved = lsGet("workorders", {}); saved[name] = parts;
  if (!lsSet("workorders", saved)) return showError("無法儲存：瀏覽器封鎖了本機儲存（私密視窗？）", true);
  state.woName = name; showError(""); refreshWoSelect("s:" + name);
});
$("wo-del").addEventListener("click", () => {
  const v = $("wo-sel").value; if (!v.startsWith("s:")) return;
  const saved = lsGet("workorders", {}); delete saved[v.slice(2)]; lsSet("workorders", saved); refreshWoSelect();
});
$("wo-name").addEventListener("input", () => { state.woName = $("wo-name").value.trim(); });

// ---- 比例尺（mm / 像素）：以「校準當下畫面的最大邊」為基準，畫面縮放時自動換算 ----
const curRefMax = () => (state.mode === "server" && state.result ? Math.max(state.result.image.width, state.result.image.height) : Math.max(state.srcW, state.srcH) || (state.result ? Math.max(state.result.image.width, state.result.image.height) : maxSide()));
function calScale(img) { return state.cal && img ? state.cal.mmPerPx * state.cal.refMax / Math.max(img.width, img.height) : null; }   // 此畫面的 mm / 像素
function updateCalState() {
  $("cal-state").textContent = state.cal ? `已校準：${state.cal.mmPerPx.toPrecision(4)} mm/px（畫面最大邊 ${state.cal.refMax}px）→ 1 px² = ${(state.cal.mmPerPx ** 2).toPrecision(3)} mm²` : "未校準：面積只能以像素顯示，無法設定面積範圍";
  $("cal-two").disabled = state.mode === "server";
  $("cal-two").title = state.mode === "server" ? "伺服器串流模式請手動輸入比例尺" : "在影像上點兩個已知實際距離的點";
}
function setCal(mmPerPx) {
  state.cal = mmPerPx > 0 ? { mmPerPx, refMax: curRefMax() } : null; lsSet("calibration", state.cal);
  $("cal-mm").value = state.cal ? +mmPerPx.toPrecision(5) : ""; updateCalState();
  if (state.result) { draw(); updateCheck(state.result); }
}
$("cal-mm").addEventListener("change", () => setCal(+$("cal-mm").value));
$("cal-two").addEventListener("click", () => {
  if (!state.src) return showError("請先載入圖片或開始影片 / 攝影機，才能在影像上校準");
  showError(""); state.calMode = true; state.calPts = []; $("cal-box").hidden = false; $("cal-ok").disabled = true;
  $("cal-msg").textContent = "請在影像上點第 1 個點"; draw();
});
const calCancel = () => { state.calMode = false; state.calPts = []; $("cal-box").hidden = true; draw(); };
$("cal-cancel").addEventListener("click", calCancel);
$("cal-ok").addEventListener("click", () => {
  const dist = +$("cal-dist").value, [p, q] = state.calPts;
  if (!(dist > 0) || !q) return $("cal-msg").textContent = "請輸入大於 0 的實際距離";
  const px = Math.hypot(q.x - p.x, q.y - p.y); if (px < 2) return $("cal-msg").textContent = "兩點太近，請重新選點";
  state.calMode = false; $("cal-box").hidden = true; setCal(dist / px); state.calPts = []; draw();
});
canvas.addEventListener("click", (e) => {
  if (!state.calMode || state.calPts.length >= 2) return;
  const rect = canvas.getBoundingClientRect(), k = canvas.width / rect.width;
  state.calPts.push({ x: (e.clientX - rect.left) * k, y: (e.clientY - rect.top) * k });
  $("cal-msg").textContent = state.calPts.length === 1 ? "請點第 2 個點" : "已選兩點，輸入實際距離後按確定"; $("cal-ok").disabled = state.calPts.length < 2; draw();
});
function drawCalibration() {                       // 校準中的點與連線（疊在影像上）
  if (!state.calPts.length) return;
  ctx.save(); ctx.lineWidth = Math.max(2, canvas.width / 300); ctx.strokeStyle = "#ef4444"; ctx.fillStyle = "#ef4444";
  state.calPts.forEach((p) => { ctx.beginPath(); ctx.arc(p.x, p.y, ctx.lineWidth * 2.5, 0, 7); ctx.fill(); });
  if (state.calPts.length === 2) { ctx.beginPath(); ctx.moveTo(state.calPts[0].x, state.calPts[0].y); ctx.lineTo(state.calPts[1].x, state.calPts[1].y); ctx.stroke(); }
  ctx.restore();
}

// ---- 檢查表：每一列 = 一種零件 ----
let checkKey = "";
const checkPrompts = () => (isRunning() && state.result ? state.result.prompts : parsePrompts($("prompts").value));
function rebuildCheckRows(force = false) {
  const ps = checkPrompts(), key = ps.join("\u0001"), tb = $("tbl-check").tBodies[0];
  if (key === checkKey && !force) return;            // 提示詞沒變就不重建，避免打字時輸入框被覆蓋
  checkKey = key; tb.replaceChildren();
  if (!ps.length) { tb.append(el("tr", {}, el("td", { colSpan: 7, className: "muted", textContent: "輸入提示詞後顯示" }))); return; }
  ps.forEach((p) => {
    const mk = (store, ph, step) => {
      const inp = el("input", { type: "number", min: 0, step, placeholder: ph, value: store[p] ?? "" });
      inp.addEventListener("input", () => { store[p] = inp.value; updateCheck(state.result); if (store === state.expected) pushServerParams(); });
      return inp;
    };
    const row = el("tr", {}, el("td", { textContent: p }), el("td", { className: "num" }, mk(state.expected, "—", 1)),
      el("td", { className: "num area-cell" }, mk(state.areaMin, "—", 1)), el("td", { className: "num area-cell" }, mk(state.areaMax, "—", 1)),
      el("td", { className: "num cnt" }), el("td", { className: "num area" }), el("td", { className: "st" }));
    row.dataset.label = p; tb.append(row);       // dataset 是唯讀屬性，不能用 Object.assign 指定
  });
}
function renderCheck(r) { rebuildCheckRows(); updateCheck(r); }

// 評估一個結果 → 每種零件的數量 / 面積 / 判定，以及整體 PASS / FAIL / 無法判定
function evaluate(r) {
  const parts = []; let any = false, fail = false, incomplete = false;
  checkPrompts().forEach((p) => {
    const has = !!r && r.prompts.includes(p), dets = has ? r.detections.filter((d) => d.label === p) : [], scale = calScale(r && r.image);
    const areas = dets.map((d) => (d.mask_area == null || !scale ? null : d.mask_area * scale * scale)), areaKnown = scale && areas.every((a) => a != null);
    const exp = state.expected[p], lo = state.areaMin[p], hi = state.areaMax[p], hasArea = isNum(lo) || isNum(hi);
    const total = has && areaKnown ? areas.reduce((a, b) => a + b, 0) : null, msgs = [], warns = [];
    if (has && isNum(exp)) { any = true; const d = dets.length - +exp; if (d) { fail = true; msgs.push(d < 0 ? `缺 ${-d}` : `多 ${d}`); } }
    if (has && hasArea) {
      any = true;
      if (!areaKnown) { if (dets.length) { incomplete = true; warns.push(!scale ? "需校準比例尺" : "需計算遮罩"); } }
      else { const bad = areas.filter((a) => (isNum(lo) && a < +lo) || (isNum(hi) && a > +hi)).length; if (bad) { fail = true; msgs.push(`${bad} 件面積超出範圍`); } }
    }
    let status = "—", cls = "st muted";
    if (has && (isNum(exp) || hasArea)) { if (msgs.length) { status = "NG（" + msgs.join("、") + "）"; cls = "st st-ng"; } else if (warns.length) { status = "⚠ " + warns[0]; cls = "st st-warn"; } else { status = "OK"; cls = "st st-ok"; } }
    parts.push({ label: p, has, areasPx: dets.map((d) => d.mask_area), count: has ? dets.length : null, expected: isNum(exp) ? +exp : null, areaMin: isNum(lo) ? +lo : null, areaMax: isNum(hi) ? +hi : null, areas, total, status, cls });
  });
  return { parts, verdict: !any ? "idle" : fail ? "ng" : incomplete ? "inc" : "ok" };
}
const VERDICT_TEXT = { idle: "未設定", ok: "PASS", ng: "FAIL", inc: "無法判定" };
function updateCheck(r) {
  const ev = evaluate(r), rows = [...$("tbl-check").tBodies[0].rows].filter((x) => x.dataset.label);
  rows.forEach((row) => {
    const p = ev.parts.find((x) => x.label === row.dataset.label); if (!p) return;
    row.querySelector(".cnt").textContent = p.count ?? "—";
    row.querySelector(".area").textContent = p.total != null ? `${p.total.toFixed(1)} mm²` : (p.has && r.detections.some((d) => d.label === p.label && d.mask_area != null) ? `${fmtInt(r.detections.filter((d) => d.label === p.label).reduce((a, d) => a + d.mask_area, 0))} px` : "—");
    const st = row.querySelector(".st"); st.textContent = p.status; st.className = p.cls;
  });
  const v = $("verdict"); v.className = "verdict " + ev.verdict; v.textContent = VERDICT_TEXT[ev.verdict];
  if (r && $("log-auto").checked) autoLog(r, ev);
}

// ---- 檢查紀錄 ----
function snapshot() {                              // 目前畫面（含偵測疊圖）縮成 JPEG，供報表附圖；失敗不影響記錄
  try {
    const src = state.mode === "server" ? $("stream") : canvas, w = src.naturalWidth || src.width, h = src.naturalHeight || src.height;
    if (!w || !h) return null;
    const k = Math.min(1, 380 / Math.max(w, h)), c = document.createElement("canvas"); c.width = Math.round(w * k); c.height = Math.round(h * k);
    c.getContext("2d").drawImage(src, 0, 0, c.width, c.height); return c.toDataURL("image/jpeg", 0.72);
  } catch { return null; }
}
function logRecord(r, ev) {
  state.log.push({ thumb: snapshot(), t: new Date(), mode: ({ image: "圖片", video: "影片", cam: "瀏覽器攝影機", server: "伺服器來源" })[state.mode], wo: state.woName || "（未命名）", verdict: VERDICT_TEXT[ev.verdict], parts: ev.parts });
  if (state.log.length > 2000) state.log.shift(); renderLog();
}
function autoLog(r, ev) {                           // 只在「判定或各零件數量改變」時記錄，並限制頻率，避免連續影格洗版
  const sig = ev.verdict + "|" + ev.parts.map((p) => `${p.label}:${p.count}:${p.status}`).join(","), now = performance.now();
  if (sig === state.lastLogSig || now - state.lastLogAt < 500) return;
  state.lastLogSig = sig; state.lastLogAt = now; logRecord(r, ev);
}
function renderLog() {
  $("log-n").textContent = state.log.length; const tb = $("tbl-log").tBodies[0]; tb.replaceChildren();
  state.log.slice(-30).reverse().forEach((x) => {
    const sum = x.parts.reduce((a, p) => a + (p.total ?? 0), 0), any = x.parts.some((p) => p.total != null);
    tb.append(el("tr", {}, el("td", { textContent: x.t.toLocaleTimeString() }), el("td", { textContent: x.mode }), el("td", { textContent: x.wo }),
      el("td", { textContent: x.verdict, className: x.verdict === "PASS" ? "st-ok" : x.verdict === "FAIL" ? "st-ng" : "muted" }),
      el("td", { textContent: x.parts.map((p) => `${p.label} ${p.count ?? "—"}/${p.expected ?? "—"}`).join("、") }), el("td", { className: "num", textContent: any ? sum.toFixed(1) : "—" })));
  });
}
function buildLogCsv() {
  const q = (s) => `"${String(s).replace(/"/g, '""')}"`, labels = [];
  state.log.forEach((x) => x.parts.forEach((p) => { if (!labels.includes(p.label)) labels.push(p.label); }));
  const head = ["time", "source", "work_order", "verdict", ...labels.flatMap((l) => [q(l + " count"), q(l + " expected"), q(l + " total_area_mm2"), q(l + " status")])].join(",");
  const rows = state.log.map((x) => [x.t.toISOString(), q(x.mode), q(x.wo), x.verdict, ...labels.flatMap((l) => { const p = x.parts.find((y) => y.label === l); return p ? [p.count ?? "", p.expected ?? "", p.total != null ? p.total.toFixed(2) : "", q(p.status)] : ["", "", "", ""]; })].join(","));
  return "﻿" + [head, ...rows].join("\n");
}
const esc = (v) => String(v).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const stat = (a) => (a.length ? { n: a.length, mean: a.reduce((x, y) => x + y, 0) / a.length, min: Math.min(...a), max: Math.max(...a) } : null);
function buildReportHtml() {
  const L = state.log, labels = []; L.forEach((x) => x.parts.forEach((p) => { if (!labels.includes(p.label)) labels.push(p.label); }));
  const calibrated = L.length > 0 && L.every((x) => x.parts.every((p) => p.areas.every((a) => a != null))), unit = calibrated ? "mm²" : "px";
  const cnt = { PASS: 0, FAIL: 0, "無法判定": 0, "未設定": 0 }; L.forEach((x) => { cnt[x.verdict] = (cnt[x.verdict] || 0) + 1; });
  const judged = cnt.PASS + cnt.FAIL, rate = judged ? ((cnt.PASS / judged) * 100).toFixed(1) + "%" : "—";
  const f = (v, d = 1) => (v == null ? "—" : Number(v).toFixed(d));
  const fmtT = (d) => d.toLocaleString("zh-TW", { hour12: false });
  const perPart = labels.map((lb) => {
    const rows = L.map((x) => x.parts.find((p) => p.label === lb)).filter((p) => p && p.count != null);
    const counts = rows.map((p) => p.count), exps = [...new Set(rows.map((p) => p.expected).filter((v) => v != null))];   // 預期值可能中途改過：每筆紀錄各用當時的預期值
    const areas = rows.flatMap((p) => (calibrated ? p.areas : p.areasPx)).filter((a) => a != null), cs = stat(counts), as = stat(areas);
    const judged = rows.filter((p) => p.expected != null), ok = judged.length ? judged.filter((p) => p.count === p.expected).length : null, rg = rows.find((p) => p.areaMin != null || p.areaMax != null);
    return `<tr><td>${esc(lb)}</td><td class="n">${exps.length ? exps.join("、") : "—"}</td><td class="n">${f(cs?.mean)} / ${cs?.min ?? "—"} / ${cs?.max ?? "—"}</td>
      <td class="n">${ok == null ? "—" : `${ok}/${judged.length}（${((ok / judged.length) * 100).toFixed(0)}%）`}</td>
      <td class="n">${as ? `${f(as.mean)} / ${f(as.min)} / ${f(as.max)}` : "—"}</td><td class="n">${as ? as.n : "—"}</td>
      <td class="n">${rg ? `${rg.areaMin ?? "—"} ~ ${rg.areaMax ?? "—"} mm²` : "—"}</td></tr>`;
  }).join("");
  let withImg = { FAIL: 0, other: 0 };
  const detail = L.map((x, i) => {
    const want = x.verdict === "FAIL" ? withImg.FAIL++ < 50 : withImg.other++ < 10, img = want && x.thumb ? `<img src="${x.thumb}" alt="">` : "";
    return `<tr class="${x.verdict === "FAIL" ? "bad" : ""}"><td class="n">${i + 1}</td><td>${fmtT(x.t)}</td><td>${esc(x.mode)}</td><td class="v ${x.verdict}">${x.verdict}</td>
      <td>${x.parts.map((p) => `${esc(p.label)}：${p.count ?? "—"}${p.expected != null ? `/${p.expected}` : ""}${p.total != null ? `，${p.total.toFixed(1)} mm²` : p.areasPx.length ? `，${fmtInt(p.areasPx.reduce((a, b) => a + b, 0))} px` : ""}${p.status !== "—" && p.status !== "OK" ? `（${esc(p.status)}）` : ""}`).join("<br>")}</td><td>${img}</td></tr>`;
  }).join("");
  const settings = `threshold ${$("threshold").value}、NMS IoU ${$("nms").value}、HyDE ${state.hydeOn ? "開（" + (state.hyde.info?.model || "") + "）" : "關"}、計算遮罩 ${$("masks").checked ? "是" : "否（只有框）"}`;
  return `<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8"><title>零件檢查報表</title><style>
  body{font:14px/1.6 system-ui,"Noto Sans TC","Microsoft JhengHei",sans-serif;color:#1b1f27;max-width:980px;margin:24px auto;padding:0 16px}
  h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:26px 0 8px;border-bottom:2px solid #e3e6ec;padding-bottom:4px}
  table{border-collapse:collapse;width:100%;font-size:13px}th,td{border:1px solid #d5d9e0;padding:5px 8px;text-align:left;vertical-align:top}th{background:#f3f5f8}
  td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}.kv{display:grid;grid-template-columns:max-content 1fr;gap:2px 16px}.kv dt{color:#667085}.kv dd{margin:0}
  .v.PASS{color:#166534;font-weight:700}.v.FAIL{color:#b91c1c;font-weight:700}tr.bad{background:#fff5f5}img{max-width:190px;height:auto;border-radius:4px;display:block}
  .cards{display:flex;gap:12px;flex-wrap:wrap}.card{border:1px solid #d5d9e0;border-radius:8px;padding:8px 16px}.card b{font-size:22px;display:block}
  .note{color:#667085;font-size:12px}@media print{body{margin:0}tr{page-break-inside:avoid}h2{page-break-after:avoid}}</style></head><body>
  <h1>製造現場零件檢查報表</h1><div class="note">產生時間：${fmtT(new Date())}</div>
  <h2>基本資訊</h2><dl class="kv"><dt>工單</dt><dd>${esc(state.woName || "（未命名）")}</dd>
    <dt>檢查期間</dt><dd>${L.length ? fmtT(L[0].t) + " ～ " + fmtT(L[L.length - 1].t) : "—"}</dd>
    <dt>零件（提示詞）</dt><dd>${labels.map(esc).join("、") || "—"}</dd><dt>偵測設定</dt><dd>${esc(settings)}</dd>
    <dt>比例尺</dt><dd>${state.cal ? `${state.cal.mmPerPx.toPrecision(4)} mm/px（畫面最大邊 ${state.cal.refMax}px）` : "未校準（面積單位為像素）"}</dd>
    <dt>面積單位</dt><dd>${unit}</dd></dl>
  <h2>總覽</h2><div class="cards"><div class="card">檢查筆數<b>${L.length}</b></div><div class="card">PASS<b style="color:#166534">${cnt.PASS}</b></div>
    <div class="card">FAIL<b style="color:#b91c1c">${cnt.FAIL}</b></div><div class="card">無法判定 / 未設定<b>${cnt["無法判定"] + cnt["未設定"]}</b></div><div class="card">合格率（PASS / 可判定）<b>${rate}</b></div></div>
  <h2>各零件統計</h2><table><thead><tr><th>零件</th><th class="n">預期數量</th><th class="n">偵測數量<br>平均 / 最小 / 最大</th><th class="n">數量符合</th>
    <th class="n">單件面積（${unit}）<br>平均 / 最小 / 最大</th><th class="n">樣本（個）</th><th class="n">設定的面積範圍</th></tr></thead><tbody>${perPart || '<tr><td colspan="7">尚無紀錄</td></tr>'}</tbody></table>
  <h2>逐筆明細</h2><table><thead><tr><th class="n">#</th><th>時間</th><th>來源</th><th>判定</th><th>各零件（偵測/預期，合計面積）</th><th>畫面</th></tr></thead><tbody>${detail || '<tr><td colspan="6">尚無紀錄</td></tr>'}</tbody></table>
  <p class="note">說明：面積由 SAM 遮罩的像素數乘以比例尺的平方換算，假設鏡頭垂直拍攝平面；未校準時以像素表示。準確度取決於比例尺校準、鏡頭畸變與遮罩品質。
  失敗案例附圖最多 50 張、其餘判定最多 10 張。數量符合率以有設定預期數量的紀錄計算。</p></body></html>`;
}
$("log-report").addEventListener("click", () => {
  if (!state.log.length) return showError("還沒有紀錄：請先按「記錄目前結果」，或勾選「結果改變時自動記錄」");
  showError(""); const url = URL.createObjectURL(new Blob([buildReportHtml()], { type: "text/html" }));
  if (!window.open(url, "_blank")) el("a", { href: url, download: "inspection_report.html" }).click();     // 彈出視窗被擋時改為下載
});
$("log-add").addEventListener("click", () => { if (!state.result) return showError("還沒有偵測結果可以記錄"); showError(""); logRecord(state.result, evaluate(state.result)); $("log-details").open = true; });
$("log-clear").addEventListener("click", () => { state.log = []; state.lastLogSig = ""; renderLog(); });
$("log-csv").addEventListener("click", () => {
  if (!state.log.length) return showError("還沒有紀錄可以下載");
  el("a", { href: URL.createObjectURL(new Blob([buildLogCsv()], { type: "text/csv" })), download: "inspection_log.csv" }).click();
});

// ================= 即時 GPU 數據 =================
async function pollStats() {
  try {
    const s = await (await fetch("/api/stats")).json();
    $("chip-status").querySelector(".dot").className = "dot ok"; $("chip-status").lastChild.textContent = `已連線 · 請求 ${s.requests} 次`;
    const pct = s.gpu_total_mb ? (s.gpu_used_mb / s.gpu_total_mb) * 100 : 0;
    $("m-mem").style.width = pct + "%"; $("m-mem-t").textContent = s.gpu_total_mb ? `${fmtInt(s.gpu_used_mb)} / ${fmtInt(s.gpu_total_mb)} MB` : "—";
    const kv = $("gpu-kv"); kv.replaceChildren(); const add = (k, v) => kv.append(el("dt", { textContent: k }), el("dd", { textContent: v }));
    add("PyTorch 已配置 / 峰值", `${fmtInt(s.torch_allocated_mb)} / ${fmtInt(s.torch_peak_mb)} MB`); add("GPU 使用率", s.util_pct != null ? `${s.util_pct}%` : "—");
    add("溫度", s.temp_c != null ? `${s.temp_c} °C` : "—"); add("SM 時脈", s.sm_clock_mhz != null ? `${s.sm_clock_mhz} MHz` : "—"); add("功耗", s.power_w != null ? `${s.power_w} W` : "—");
    $("chip-device").textContent = s.device;
  } catch { $("chip-status").querySelector(".dot").className = "dot bad"; $("chip-status").lastChild.textContent = "無法連線"; }
}

// ================= 日誌 =================
async function refreshLogs() {
  try {
    const q = $("log-filter").checked && state.lastRid && state.lastRid !== "camera" ? `&request_id=${state.lastRid}` : "";
    const j = await (await fetch(`/api/logs?lines=200${q}`)).json();
    $("log-path").textContent = j.path.split("/").slice(-1)[0];
    const pre = $("log"), stick = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 30; pre.replaceChildren();
    j.lines.forEach((ln) => {
      const m = ln.match(/^(\S+ \S+) \| (\w+)\s*\| (\S+) \| (.*)$/), row = el("div");
      if (m) { row.className = "lv-" + m[2]; row.append(m[1].slice(11, 23) + " " + m[2].padEnd(7), el("span", { className: "rid", textContent: m[3] + " " }), m[4]); } else row.textContent = ln;
      pre.append(row);
    });
    if (stick) pre.scrollTop = pre.scrollHeight;
  } catch { /* 忽略 */ }
}
$("btn-log").addEventListener("click", refreshLogs); $("log-filter").addEventListener("change", refreshLogs);

// ================= 啟動 =================
(async function init() {
  state.cal = lsGet("calibration", null); if (state.cal && !(state.cal.mmPerPx > 0)) state.cal = null; if (state.cal) $("cal-mm").value = +state.cal.mmPerPx.toPrecision(5);
  refreshWoSelect(); updateCalState(); rebuildCheckRows(true); samEncInit();
  setMode("image"); renderPromptChips(); updateHydeChip(); listCams(); hydeStatus(); setInterval(hydeStatus, 15000);
  try { const h = await (await fetch("/api/health")).json(); $("chip-backend").textContent = h.backend; $("chip-device").textContent = h.device; } catch { /* pollStats 會顯示 */ }
  pollStats(); setInterval(pollStats, 1000);
  refreshLogs(); setInterval(() => { if ($("log-auto").checked && !state.busy) refreshLogs(); }, 2000);
})();
