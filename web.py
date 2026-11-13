#!/usr/bin/env python3
"""isophote-web: tiny local web UI for isophote.py. No dependencies."""
import io
import base64
import json
import threading
import uuid
import numpy as np
import cv2
from http.server import HTTPServer, BaseHTTPRequestHandler
from isophote import run as paint_run, flow_field, smooth_flow, \
    painterly_base, render_strokes, canvas_relief

JOBS = {}
LOCK = threading.Lock()


def paint_to_png(img_bgr, params):
    """In-memory version of run(): returns encoded png bytes."""
    brush_len = params["len"]; brush_w = params["width"]
    layers = int(params["layers"]); relief = float(params["relief"])
    base_radius = int(params["base_radius"]); flow_smooth = int(params["flow_smooth"])
    seed = int(params["seed"]); max_size = 1400

    img = img_bgr
    h, w = img.shape[:2]
    scale = max(h, w) / max_size
    if scale > 1.0:
        img = cv2.resize(img, (int(w / scale), int(h / scale)),
                         interpolation=cv2.INTER_AREA)
    np.random.seed(seed)

    base = painterly_base(img, radius=base_radius)
    angle, coh = flow_field(img, sigma=2.5)
    angle, coh = smooth_flow(angle, coh, r=flow_smooth)
    n_strokes = int(img.shape[0] * img.shape[1] / 140)
    painted = render_strokes(img, base, angle, n_strokes, brush_len, brush_w,
                             layers=layers, seed=seed)
    if relief > 0:
        shade, _ = canvas_relief(np.zeros_like(coh), painted.shape[:2],
                                 strength=relief)
        painted = (painted.astype(np.float32) * shade).astype(np.uint8)
    ok, buf = cv2.imencode(".png", painted)
    return buf.tobytes()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence
        pass

    def _send(self, code, body, ctype="text/html"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/test_photo.jpg"):
            try:
                with open("test_photo.jpg", "rb") as f:
                    self._send(200, f.read(), "image/jpeg")
            except OSError:
                self._send(404, b"missing", "text/plain")
        elif self.path == "/" or self.path.startswith("/index"):
            self._send(200, PAGE.encode())
        elif self.path.startswith("/job/"):
            jid = self.path.split("/")[2].split("?")[0]
            with LOCK:
                job = JOBS.get(jid)
            if not job:
                self._send(404, b'{"status":"gone"}', "application/json")
                return
            resp = {"status": job["status"]}
            if job["status"] == "done":
                resp["image"] = "data:image/png;base64," + \
                    base64.b64encode(job["png"]).decode()
                resp["secs"] = job.get("secs")
            self._send(200, json.dumps(resp).encode(), "application/json")
        else:
            self._send(404, b"nope")

    def do_POST(self):
        if self.path != "/paint":
            self._send(404, b"nope")
            return
        ln = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(ln))
        img_bytes = base64.b64decode(body["image"])
        params = body["params"]
        arr = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        jid = uuid.uuid4().hex[:10]
        with LOCK:
            JOBS[jid] = {"status": "running"}

        def work():
            import time
            t0 = time.time()
            png = paint_to_png(img_bgr, params)
            with LOCK:
                JOBS[jid] = {"status": "done", "png": png,
                             "secs": round(time.time() - t0, 1)}

        threading.Thread(target=work, daemon=True).start()
        self._send(200, json.dumps({"job": jid}).encode(), "application/json")


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>isophote</title>
<style>
:root {
  --bg: rgb(17,17,17); --surface: rgb(28,28,30); --elev: rgb(44,44,46);
  --txt: rgb(204,204,204); --txt2: rgb(142,142,147); --txt3: rgb(99,99,102);
  --accent: rgb(10,132,255); --mod: rgb(255,214,10);
  --border: rgba(255,255,255,0.06);
}
* { box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  font-size: 14px; color: var(--txt); background: var(--bg); margin: 0;
  height: 100vh; display: flex; flex-direction: column; overflow: hidden; }
header { height: 44px; flex: none; display: flex; align-items: center;
  gap: 16px; padding: 0 16px; background: var(--surface);
  border-bottom: 1px solid var(--border); }
header .logo { font-weight: 600; color: #fff; letter-spacing: 0.3px; font-size: 13px; }
header .btn { background: var(--elev); color: var(--txt); border: 1px solid var(--border);
  padding: 6px 14px; border-radius: 6px; cursor: pointer; font-size: 13px;
  transition: background .12s; }
header .btn:hover { background: #555; }
header .btn.primary { background: var(--accent); color: #fff; border: none; }
header .btn.primary:hover { background: rgb(50,150,255); }
header .btn.primary:disabled { opacity: 0.4; cursor: default; }
header .spacer { flex: 1; }
#status { color: var(--txt3); font-size: 12px; }
main { flex: 1; display: flex; min-height: 0; }
/* ---- canvas area ---- */
#stage { flex: 1; position: relative; overflow: hidden;
  display: flex; align-items: center; justify-content: center; }
#stage img { max-width: 92%; max-height: 92%; border-radius: 2px; }
#viewBtns { position: absolute; top: 18px; left: 50%; transform: translateX(-50%);
  display: none; gap: 0; z-index: 5; background: rgba(28,28,30,0.85);
  border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
#viewBtns .vbtn { background: transparent; color: var(--txt2); border: none;
  padding: 7px 18px; cursor: pointer; font-size: 12.5px; transition: all .12s; }
#viewBtns .vbtn:hover { color: var(--txt); }
#viewBtns .vbtn.on { background: var(--elev); color: #fff; }
/* image hover actions (enlarge + close, close on the right) */
#imgWrap { position: relative; display: flex; align-items: center;
  justify-content: center; max-width: 96%; max-height: 96%; }
#imgWrap img { max-width: 100%; max-height: 96vh; border-radius: 2px;
  transition: filter .18s; }
#imgWrap:hover img { filter: brightness(0.92); }
#imgActions { position: absolute; top: 10px; right: 10px; display: flex;
  gap: 8px; opacity: 0; transition: opacity .15s; }
#imgWrap:hover #imgActions { opacity: 1; }
.abtn { width: 32px; height: 32px; border-radius: 50%; border: none;
  background: rgba(20,20,22,0.55); color: rgba(255,255,255,0.75);
  font-size: 14px; cursor: pointer; display: flex; align-items: center;
  justify-content: center; backdrop-filter: blur(6px);
  box-shadow: 0 1px 6px rgba(0,0,0,0.35);
  transition: background .12s, color .12s, transform .12s; }
.abtn:hover { background: rgba(20,20,22,0.85); color: #fff; transform: scale(1.06); }
.abtn svg { width: 14px; height: 14px; stroke: currentColor;
  stroke-width: 2; fill: none; stroke-linecap: round; stroke-linejoin: round; }
#stage.zoomed #imgWrap { max-width: none; max-height: none; }
#stage.zoomed #imgWrap img { max-width: none; max-height: none;
  max-width: 96vw; max-height: 96vh; width: auto; height: auto;
  cursor: move; }
#drop { position: absolute; inset: 10%; border: 2px dashed var(--txt3);
  border-radius: 12px; display: flex; flex-direction: column; gap: 8px;
  align-items: center; justify-content: center; cursor: pointer;
  color: var(--txt2); transition: border-color .15s, background .15s; }
#drop:hover, #drop.drag { border-color: var(--accent); background: rgba(10,132,255,0.04); }
#drop .big { font-size: 15px; color: var(--txt); }
#drop .small { font-size: 12px; color: var(--txt3); }
/* ---- side panel ---- */
#panel { width: 300px; flex: none; background: var(--surface);
  border-left: 1px solid var(--border); overflow-y: auto; padding: 16px;
  display: flex; flex-direction: column; gap: 4px; }
#panel.hidden { display: none; }
.preset-row { display: flex; gap: 6px; margin-bottom: 12px; }
.presets { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 8px; }
.chip { padding: 5px 12px; border-radius: 14px; background: var(--elev);
  border: 1px solid var(--border); color: var(--txt2); font-size: 12px;
  cursor: pointer; transition: all .12s; }
.chip:hover { color: var(--txt); }
.chip.on { background: var(--accent); color: #fff; border-color: transparent; }
.slider { padding: 8px 0; }
.slider .row { display: flex; justify-content: space-between;
  align-items: center; margin-bottom: 4px; }
.slider label { font-size: 11px; letter-spacing: 0.2px; color: var(--txt2);
  font-weight: 500; }
.slider .v { font-size: 12px; color: var(--txt3); font-variant-numeric: tabular-nums; }
.slider .v.changed { color: var(--mod); }
.slider input[type=range] { width: 100%; height: 22px; -webkit-appearance: none;
  background: transparent; cursor: pointer; margin: 0; }
.slider input[type=range]::-webkit-slider-runnable-track { height: 3px;
  background: var(--elev); border-radius: 2px; }
.slider input[type=range]::-webkit-slider-thumb { -webkit-appearance: none;
  width: 15px; height: 15px; border-radius: 50%; background: #fff;
  margin-top: -6px; box-shadow: 0 1px 4px rgba(0,0,0,0.5); }
.slider input[type=range]::-moz-range-track { height: 3px;
  background: var(--elev); border-radius: 2px; }
.slider input[type=range]::-moz-range-thumb { width: 15px; height: 15px;
  border-radius: 50%; background: #fff; border: none;
  box-shadow: 0 1px 4px rgba(0,0,0,0.5); }
.dot { width: 5px; height: 5px; border-radius: 50%; background: var(--mod);
  display: inline-block; margin-left: 5px; vertical-align: 1px; }
input[type=file] { display: none; }
</style></head>
<body>
<header>
  <div class="logo">isophote</div>
  <div class="spacer"></div>
  <div id="status"></div>
  <button class="btn primary" id="export" style="display:none">Export</button>
</header>
<main>
  <div id="stage">
    <div id="viewBtns" style="display:none">
      <button class="vbtn on" id="viewOriginal">Input</button>
      <button class="vbtn" id="viewResult">Result</button>
    </div>
    <div id="imgWrap" style="display:none">
      <img id="solo">
      <div id="imgActions">
        <button class="abtn" id="enlarge" title="Enlarge">
          <svg viewBox="0 0 24 24"><path d="M9 3H3v6M15 21h6v-6M3 3l7 7M21 21l-7-7"/></svg>
        </button>
        <button class="abtn" id="clear" title="Remove image">
          <svg viewBox="0 0 24 24"><path d="M6 6l12 12M18 6L6 18"/></svg>
        </button>
      </div>
    </div>
    <div id="drop">
      <div class="big">Drop a photo</div>
      <div class="small">or click to browse &middot; stays on your machine</div>
    </div>
  </div>
  <div id="panel">
    <div class="presets">
      <div class="chip on" data-p="art">Art</div>
      <div class="chip" data-p="medium">Medium</div>
      <div class="chip" data-p="bold">Impasto</div>
      <div class="chip" data-p="pastel">Pastel</div>
    </div>
    <div class="slider"><div class="row"><label>Stroke length<span class="dot" id="d_len" style="visibility:hidden"></span></label><span class="v" id="v_len">45</span></div>
      <input type="range" id="len" min="10" max="90" value="45"></div>
    <div class="slider"><div class="row"><label>Stroke width<span class="dot" id="d_width" style="visibility:hidden"></span></label><span class="v" id="v_width">3</span></div>
      <input type="range" id="width" min="2" max="14" value="3"></div>
    <div class="slider"><div class="row"><label>Layers<span class="dot" id="d_layers" style="visibility:hidden"></span></label><span class="v" id="v_layers">4</span></div>
      <input type="range" id="layers" min="1" max="6" value="4"></div>
    <div class="slider"><div class="row"><label>Flow coherence<span class="dot" id="d_flow_smooth" style="visibility:hidden"></span></label><span class="v" id="v_flow_smooth">8</span></div>
      <input type="range" id="flow_smooth" min="2" max="24" value="8"></div>
    <div class="slider"><div class="row"><label>Base flatten<span class="dot" id="d_base_radius" style="visibility:hidden"></span></label><span class="v" id="v_base_radius">6</span></div>
      <input type="range" id="base_radius" min="2" max="16" value="6"></div>
    <div class="slider"><div class="row"><label>Canvas relief<span class="dot" id="d_relief" style="visibility:hidden"></span></label><span class="v" id="v_relief">0.10</span></div>
      <input type="range" id="relief" min="0" max="0.6" step="0.05" value="0.10"></div>
    <div class="slider"><div class="row"><label>Seed<span class="dot" id="d_seed" style="visibility:hidden"></span></label><span class="v" id="v_seed">42</span></div>
      <input type="range" id="seed" min="1" max="200" value="42"></div>
  </div>
</main>
<input type="file" id="file" accept="image/*">
<script>
const PRESETS = {
  art:    {len:45, width:3, layers:4, flow_smooth:8, base_radius:6, relief:0.10},
  medium: {len:30, width:4, layers:3, flow_smooth:8, base_radius:6, relief:0.15},
  bold:   {len:60, width:5, layers:2, flow_smooth:8, base_radius:8, relief:0.10},
  pastel: {len:30, width:5, layers:3, flow_smooth:8, base_radius:9, relief:0.00},
};
const DEFAULTS = {len:45, width:3, layers:4, flow_smooth:8, base_radius:6, relief:0.10, seed:42};
const KEYS = ["len","width","layers","flow_smooth","base_radius","relief","seed"];
let srcB64 = null, origURL = null, paintedURL = null, presetOn = "art";
const $ = id => document.getElementById(id);
for (const k of KEYS) {
  const el = $(k);
  el.addEventListener("input", () => {
    $("v_"+k).textContent = el.value;
    const def = DEFAULTS[k];
    $("d_"+k).style.visibility = (+el.value !== def) ? "visible" : "hidden";
    if (+el.value !== def) presetOn = null;
    scheduleLive();
  });
}
document.querySelectorAll(".chip").forEach(c => c.onclick = () => {
  document.querySelectorAll(".chip").forEach(x=>x.classList.remove("on"));
  c.classList.add("on");
  presetOn = c.dataset.p;
  const p = PRESETS[presetOn];
  for (const k in p) { $(k).value = p[k]; $("v_"+k).textContent = p[k];
    $("d_"+k).style.visibility = "hidden"; }
  scheduleLive();
});
/* drop zone */
const stage = $("stage"), drop = $("drop");
drop.onclick = () => $("file").click();
["dragenter","dragover"].forEach(ev => stage.addEventListener(ev, e => {
  e.preventDefault(); drop.classList.add("drag");
}));
["dragleave","dragend"].forEach(ev => stage.addEventListener(ev, () =>
  drop.classList.remove("drag")));
stage.addEventListener("drop", e => { e.preventDefault();
  drop.classList.remove("drag");
  if (e.dataTransfer.files[0]) load(e.dataTransfer.files[0]); });
$("file").onchange = e => { if (e.target.files[0]) load(e.target.files[0]); };
function load(f) {
  const r = new FileReader();
  r.onload = () => { srcB64 = r.result.split(",")[1];
    origURL = r.result;
    drop.style.display = "none";
    $("imgWrap").style.display = "flex";
    $("viewBtns").style.display = "flex";
    $("export").style.display = "";
    $("status").textContent = f.name;
    showOrig();
    scheduleLive();
  };
  r.readAsDataURL(f);
}
/* view switching: Input / Result segmented control above the image */
function showOrig() {
  $("solo").src = origURL; $("viewOriginal").classList.add("on");
  $("viewResult").classList.remove("on");
}
function showResult() {
  if (!paintedURL) return;
  $("solo").src = paintedURL; $("viewResult").classList.add("on");
  $("viewOriginal").classList.remove("on");
}
$("viewOriginal").onclick = showOrig;
$("viewResult").onclick = showResult;
/* image actions: clear + enlarge */
$("clear").onclick = () => {
  srcB64 = origURL = paintedURL = null;
  clearTimeout(liveTimer); liveQueue = false;
  $("imgWrap").style.display = "none";
  $("viewBtns").style.display = "none";
  $("export").style.display = "none";
  drop.style.display = "flex";
  $("status").textContent = "";
};
/* pan when zoomed */
let zoomed = false, panX = 0, panY = 0;
$("enlarge").onclick = () => {
  zoomed = !zoomed;
  panX = panY = 0;
  $("stage").classList.toggle("zoomed", zoomed);
  $("enlarge").innerHTML = zoomed
    ? '<svg viewBox="0 0 24 24"><path d="M21 9h-6V3M3 15h6v6M21 3l-7 7M3 21l7-7"/></svg>'
    : '<svg viewBox="0 0 24 24"><path d="M9 3H3v6M15 21h6v-6M3 3l7 7M21 21l-7-7"/></svg>';
  setTransform();
};
function setTransform() {
  $("solo").style.transform =
    `translate(${panX}px, ${panY}px) scale(${zoomed ? 2 : 1})`;
}
/* drag to pan when zoomed */
let panning = false, px0 = 0, py0 = 0;
$("solo").onpointerdown = e => {
  if (!zoomed) return;
  panning = true; px0 = e.clientX - panX; py0 = e.clientY - panY;
  $("solo").setPointerCapture(e.pointerId);
};
$("solo").onpointermove = e => {
  if (!panning) return;
  panX = e.clientX - px0; panY = e.clientY - py0;
  setTransform();
};
$("solo").onpointerup = () => panning = false;
/* live preview: debounce paint jobs while a photo is loaded */
let liveTimer = null, liveQueue = false, liveBusy = false, lastParams = null;
function scheduleLive() {
  if (!srcB64) return;
  clearTimeout(liveTimer);
  liveTimer = setTimeout(livePaint, 350);
}
function currentParams() {
  const p = {};
  for (const k of ["len","width","layers","flow_smooth","base_radius","relief","seed"])
    p[k] = parseFloat($(k).value);
  return p;
}
async function livePaint() {
  const params = currentParams();
  lastParams = JSON.stringify(params);
  if (liveBusy) { liveQueue = true; return; }
  liveBusy = true;
  $("status").textContent = "painting…";
  try {
    const resp = await fetch("/paint", {method:"POST",
      headers:{"Content-Type":"application/json"},
      body: JSON.stringify({image: srcB64, params})});
    const {job} = await resp.json();
    for (;;) {
      const j = await (await fetch("/job/"+job)).json();
      if (j.status !== "done") { await new Promise(r => setTimeout(r, 400)); continue; }
      paintedURL = j.image;
      if ($("solo").style.display !== "none")
        $("solo").src = paintedURL;
      showResultLight();
      $("status").textContent = j.secs + "s";
      break;
    }
  } catch (e) { $("status").textContent = "error: " + e.message; }
  liveBusy = false;
  if (liveQueue) { liveQueue = false; scheduleLive(); }
}
function showResultLight() {
  $("viewResult").classList.add("on");
  $("viewOriginal").classList.remove("on");
}
/* save */
$("export").onclick = () => {
  if (!paintedURL) return;
  const a = document.createElement("a");
  a.href = paintedURL; a.download = "painted.png"; a.click();
};
</script>
</body></html>"""

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8799)
    a = ap.parse_args()
    print(f"isophote web UI -> http://localhost:{a.port}")
    HTTPServer(("127.0.0.1", a.port), Handler).serve_forever()
