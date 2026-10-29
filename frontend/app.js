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
  $("btn-vstart").disabled = !state.video.url; $("btn-vstop").disabled = true;
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
  const prompts = parsePrompts($("prompts").value);
  if (!prompts.length) throw new Error("請至少輸入一個提示詞");
  const fd = new FormData();
  fd.append("image", blob, blob.name || "frame.jpg");
  fd.append("prompts", JSON.stringify(prompts));
  fd.append("threshold", $("threshold").value);
  fd.append("nms_iou", $("nms").value);
  fd.append("use_hyde", $("hyde").checked ? "true" : "false");
  fd.append("quiet", quiet ? "true" : "false");
  fd.append("with_masks", $("masks").checked ? "true" : "false");
  const t0 = performance.now();
  const res = await fetch("/api/segment", { method: "POST", body: fd });
  const rid = res.headers.get("X-Request-ID");
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.detail || body.error || `HTTP ${res.status}`);
  body.client_ms = performance.now() - t0; body.request_id = body.request_id || rid;
  return body;
}
function accept(r, src, w, h, { history = true } = {}) {
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
  state.busy = true; showError(""); $("busy").hidden = false; $("btn-run").disabled = true;
  try { accept(await callSegment(state.file), state.src, state.srcW, state.srcH); refreshLogs(); }
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
  if (!state.video.url || state.loop.on) return;
  const step = Math.max(0.05, +$("v-step").value || 0.5), t0 = Math.max(0, +$("v-from").value || 0), dur = vid.duration;
  state.loop = { on: true, n: 0, fps: 0 }; state.video.timeline = []; showError("");
  $("btn-vstart").disabled = true; $("btn-vstop").disabled = false; $("v-prog").hidden = false; $("v-prog").max = dur; $("card-timeline").hidden = false;
  // 準備某個時間點的影格（seek → 擷取 → JPEG）。推論第 N 幀的同時就開始準備第 N+1 幀。
  const prepare = async (t) => { await seekTo(Math.min(t, dur - 0.01)); await sleep(30); const c = grab(vid, vid.videoWidth, vid.videoHeight); return { t, c, blob: await toBlob(c) }; };
  try {
    let cur = t0 < dur ? await prepare(t0) : null;
    for (let i = 0; cur && state.loop.on; i++) {
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
  try {
    state.camStream = await navigator.mediaDevices.getUserMedia({ video: { deviceId: $("cam-dev").value ? { exact: $("cam-dev").value } : undefined, width: { ideal: 1280 } }, audio: false });
  } catch (e) { $("btn-camstart").disabled = false; return showError("無法開啟攝影機：" + e.message); }
  listCams();
  const cam = $("cam"); cam.srcObject = state.camStream; await cam.play(); $("empty").hidden = true;
  $("btn-camstop").disabled = false; state.loop = { on: true, n: 0, fps: 0 };
  try {
    for (let i = 0; state.loop.on; i++) {
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
  fd.append("prompts", JSON.stringify(parsePrompts($("prompts").value)));
  fd.append("threshold", $("threshold").value); fd.append("nms_iou", $("nms").value); fd.append("style", $("style").value); fd.append("with_masks", $("masks").checked ? "true" : "false");
  Object.entries(extra).forEach(([k, v]) => fd.append(k, v)); return fd;
}
async function serverStart() {
  showError(""); $("btn-srvstart").disabled = true;
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
    const r = { ...s, hyde: { requested: false }, backend: "server", client_ms: null, request_id: "camera", threshold: +$("threshold").value };
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
  const ps = parsePrompts($("prompts").value);
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
  renderPromptChips();
  clearTimeout(autoTimer);
  if (state.srv.on) autoTimer = setTimeout(pushServerParams, 400);                       // 伺服器串流：即時換提示
  else if ($("auto").checked && state.mode === "image" && state.file) autoTimer = setTimeout(run, 700);
  // 瀏覽器攝影機／影片：迴圈每一幀都會重新讀取提示詞，下一幀即生效
});
const onParam = () => { if (state.srv.on) pushServerParams(); else if ($("auto").checked && state.mode === "image" && state.file) run(); };
$("threshold").addEventListener("input", (e) => { $("th-out").textContent = (+e.target.value).toFixed(2); });
$("nms").addEventListener("input", (e) => { $("nms-out").textContent = (+e.target.value).toFixed(2); });
$("threshold").addEventListener("change", onParam); $("nms").addEventListener("change", onParam);
$("style").addEventListener("change", () => { draw(); onParam(); });
$("masks").addEventListener("change", onParam);
$("hyde").addEventListener("change", () => { showError($("hyde").checked ? "HyDE 後端尚未實作：請求會送出 use_hyde=true，但伺服器會改用原始提示並回報原因。" : "", true); });

// ================= 繪圖 =================
const canvas = $("canvas"), ctx = canvas.getContext("2d");
function draw() {
  if (!state.src || state.mode === "server") return;
  const W = state.srcW, H = state.srcH; canvas.width = W; canvas.height = H;
  ctx.drawImage(state.src, 0, 0, W, H); state.paths = [];
  const r = state.result; if (!r) return;
  const style = $("style").value, lw = Math.max(2, Math.round(Math.max(W, H) / 400));
  for (const d of r.detections) {
    const color = colorOf(d.label, r.prompts), path = new Path2D();
    d.contours.forEach((c) => { c.forEach(([x, y], i) => (i ? path.lineTo(x, y) : path.moveTo(x, y))); path.closePath(); });
    state.paths.push({ d, path });
    if (state.hidden.has(d.label)) continue;
    if (d.mask_area != null && (style === "mask" || style === "both")) { ctx.globalAlpha = style === "both" ? 0.3 : 0.45; ctx.fillStyle = color; ctx.fill(path, "evenodd"); ctx.globalAlpha = 1; }
    if (d.mask_area != null && (style === "contour" || style === "both")) { ctx.lineWidth = lw; ctx.strokeStyle = color; ctx.lineJoin = "round"; ctx.stroke(path); }
    if (style === "box" || d.mask_area == null) { ctx.lineWidth = lw; ctx.strokeStyle = color; const [x0, y0, x1, y1] = d.box; ctx.strokeRect(x0, y0, x1 - x0, y1 - y0); }
    const fs = Math.max(12, Math.round(Math.max(W, H) / 50)); ctx.font = `600 ${fs}px system-ui, sans-serif`;
    const text = `${d.label} ${d.score.toFixed(2)}`, tw = ctx.measureText(text).width, [x0, y0] = d.box, ty = Math.max(y0, fs + 4);
    ctx.fillStyle = color; ctx.fillRect(x0, ty - fs - 4, tw + 8, fs + 6); ctx.fillStyle = "#fff"; ctx.fillText(text, x0 + 4, ty - 3);
  }
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
  renderLegend(r); renderSummary(r); renderTiming(r); renderCheck(r);
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
    tb.append(el("tr", {}, el("td", { textContent: s.label }), el("td", { className: "num", textContent: s.count }), el("td", { className: "num", textContent: fmtInt(s.mask_area) }),
      el("td", { className: "num", textContent: fmtInt(s.box_area) }), el("td", { className: "num", textContent: ratio == null ? "—" : ratio.toFixed(0) + "%" }), el("td", { className: "num", textContent: fmt(s.mean_score, 2) })));
  });
  const td = $("tbl-det").tBodies[0]; td.replaceChildren();
  r.detections.forEach((d) => td.append(el("tr", {}, el("td", { textContent: d.id }), el("td", { textContent: d.label }), el("td", { className: "num", textContent: d.score.toFixed(3) }),
    el("td", { textContent: d.box.map((v) => Math.round(v)).join(", ") }), el("td", { className: "num", textContent: fmtInt(d.mask_area) }), el("td", { className: "num", textContent: d.contours.reduce((a, c) => a + c.length, 0) }))));
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
  add("影像大小", `${r.image.width}×${r.image.height}`); add("原始框 → NMS 後", `${t.num_raw_boxes ?? "—"} → ${r.detections.length}`);
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

// ================= 製造情境：數量檢查 =================
let checkKey = "";
function renderCheck(r) {
  const key = r.prompts.join("\u0001"), tb = $("tbl-check").tBodies[0];
  if (key !== checkKey) {                                        // 提示詞改變才重建輸入框，避免打字時被覆蓋
    checkKey = key; tb.replaceChildren();
    r.prompts.forEach((p) => {
      const inp = el("input", { type: "number", min: 0, step: 1, placeholder: "—", value: state.expected[p] ?? "" });
      inp.addEventListener("input", () => { state.expected[p] = inp.value; updateCheck(state.result); pushServerParams(); });
      const row = el("tr", {}, el("td", { textContent: p }), el("td", { className: "num" }, inp), el("td", { className: "num cnt" }), el("td", { className: "st" }));
      row.dataset.label = p;   // dataset 是唯讀屬性，不能用 Object.assign 指定
      tb.append(row);
    });
  }
  updateCheck(r);
}
function updateCheck(r) {
  if (!r) return;
  const rows = [...$("tbl-check").tBodies[0].rows].filter((x) => x.dataset.label); let any = false, allOk = true;
  rows.forEach((row) => {
    const s = r.summary.find((x) => x.label === row.dataset.label), n = s ? s.count : 0, exp = state.expected[row.dataset.label];
    row.querySelector(".cnt").textContent = n; const st = row.querySelector(".st");
    if (exp === "" || exp == null) { st.textContent = "—"; st.className = "st muted"; return; }
    any = true; const diff = n - +exp;
    if (diff === 0) { st.textContent = "OK"; st.className = "st st-ok"; } else { allOk = false; st.textContent = diff < 0 ? `NG（缺 ${-diff}）` : `NG（多 ${diff}）`; st.className = "st st-ng"; }
  });
  const v = $("verdict"); v.className = "verdict " + (!any ? "idle" : allOk ? "ok" : "ng"); v.textContent = !any ? "未設定" : allOk ? "PASS" : "FAIL";
}

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
  setMode("image"); renderPromptChips(); listCams();
  try { const h = await (await fetch("/api/health")).json(); $("chip-backend").textContent = h.backend; $("chip-device").textContent = h.device; } catch { /* pollStats 會顯示 */ }
  pollStats(); setInterval(pollStats, 1000);
  refreshLogs(); setInterval(() => { if ($("log-auto").checked && !state.busy) refreshLogs(); }, 2000);
})();
