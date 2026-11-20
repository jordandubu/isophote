// viewer.js — the canvas: original underneath, painted result on top, with
// Original / Split / Result comparison, wheel + pinch zoom, drag pan.
import { $ } from "./dom.js";

export function createViewer() {
  const viewport = $("viewport"), content = $("content"), orig = $("orig"), canvas = $("res");
  const ctx = canvas.getContext("2d");
  const divider = $("divider"), grip = divider.querySelector(".grip");
  const segBtns = document.querySelectorAll("#viewSeg button");
  const MODES = ["orig", "split", "result"];

  let ref = null;            // {w,h}: layout aspect
  let res = null, kind = null;
  let mode = "result", held = false, split = 0.5;
  let fitMode = true, z = 1, tx = 0, ty = 0, fw = 0, fh = 0;

  /* ---- layout ---- */
  function layout() {
    const sw = viewport.clientWidth, sh = viewport.clientHeight;
    if (!ref || !sw || !sh) return;
    const a = ref.w / ref.h;
    fw = Math.min(sw * 0.94, sh * 0.94 * a);
    fh = fw / a;
    content.style.width = fw + "px";
    content.style.height = fh + "px";
    if (fitMode) { z = 1; tx = (sw - fw) / 2; ty = (sh - fh) / 2; }
    apply();
  }
  function apply() {
    content.style.transform = `translate(${tx}px,${ty}px) scale(${z})`;
    content.style.setProperty("--z", z);
    const native = res ? res.w : ref ? ref.w : fw;
    $("zoomPct").textContent = Math.round(z * fw / native * 100) + "%";
  }
  new ResizeObserver(layout).observe(viewport);

  function zoomAt(cx, cy, factor) {
    const z2 = Math.min(32, Math.max(0.25, z * factor));
    tx = cx - (cx - tx) * (z2 / z);
    ty = cy - (cy - ty) * (z2 / z);
    z = z2; fitMode = false;
    apply();
  }
  const local = (e) => { const r = viewport.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };

  function fit() { fitMode = true; layout(); }
  function actualSize(cx, cy) {
    if (!res) return;
    if (cx == null) { cx = viewport.clientWidth / 2; cy = viewport.clientHeight / 2; }
    zoomAt(cx, cy, (res.w / fw) / z);
  }

  viewport.addEventListener("wheel", (e) => {
    if (!ref) return;
    e.preventDefault();
    const [cx, cy] = local(e);
    zoomAt(cx, cy, Math.exp(-e.deltaY * (e.ctrlKey ? 0.01 : 0.0015)));
  }, { passive: false });

  const pts = new Map();
  let lastDist = 0, lastMid = null;
  const pinch = () => {
    const [a, b] = [...pts.values()];
    return { d: Math.hypot(a.x - b.x, a.y - b.y), mx: (a.x + b.x) / 2, my: (a.y + b.y) / 2 };
  };
  viewport.addEventListener("pointerdown", (e) => {
    if (!ref || e.target.closest(".grip")) return;
    viewport.setPointerCapture(e.pointerId);
    pts.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (pts.size === 2) { const p = pinch(); lastDist = p.d; lastMid = [p.mx, p.my]; }
    viewport.classList.add("grabbing");
  });
  viewport.addEventListener("pointermove", (e) => {
    const p = pts.get(e.pointerId);
    if (!p) return;
    const dx = e.clientX - p.x, dy = e.clientY - p.y;
    p.x = e.clientX; p.y = e.clientY;
    if (pts.size === 1) {
      tx += dx; ty += dy; fitMode = false; apply();
    } else if (pts.size === 2) {
      const n = pinch(), r = viewport.getBoundingClientRect();
      tx += n.mx - lastMid[0]; ty += n.my - lastMid[1];
      zoomAt(n.mx - r.left, n.my - r.top, n.d / lastDist);
      lastDist = n.d; lastMid = [n.mx, n.my];
    }
  });
  const up = (e) => { pts.delete(e.pointerId); if (!pts.size) viewport.classList.remove("grabbing"); };
  viewport.addEventListener("pointerup", up);
  viewport.addEventListener("pointercancel", up);
  viewport.addEventListener("dblclick", (e) => {
    if (!ref || e.target.closest(".grip")) return;
    if (fitMode) { const [cx, cy] = local(e); actualSize(cx, cy); } else fit();
  });

  // split divider drag
  let dragging = false;
  grip.addEventListener("pointerdown", (e) => { e.stopPropagation(); grip.setPointerCapture(e.pointerId); dragging = true; });
  grip.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    const r = content.getBoundingClientRect();
    split = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width));
    applyMode();
  });
  grip.addEventListener("pointerup", () => { dragging = false; });
  grip.addEventListener("pointercancel", () => { dragging = false; });

  /* ---- modes ---- */
  function applyMode() {
    const eff = held ? "orig" : mode;
    canvas.style.visibility = eff === "orig" || !res ? "hidden" : "visible";
    canvas.style.clipPath = eff === "split" ? `inset(0 0 0 ${split * 100}%)` : "none";
    divider.hidden = eff !== "split" || !res;
    divider.style.left = split * 100 + "%";
    segBtns.forEach((b) => b.classList.toggle("on", b.dataset.mode === mode));
  }
  segBtns.forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode)));
  function setMode(m) { if (MODES.includes(m)) { mode = m; applyMode(); } }

  /* ---- content ---- */
  function draw() {
    if (!res) return;
    const bm = res.bitmap(kind);
    if (canvas.width !== bm.width || canvas.height !== bm.height) { canvas.width = bm.width; canvas.height = bm.height; }
    else ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(bm, 0, 0);
  }

  return {
    setSource(source) {
      ref = source.ref;
      res = null;
      orig.src = source.origURL;
      $("stageEmpty").hidden = true;
      viewport.hidden = false;
      fitMode = true;
      layout();
      applyMode();
    },
    clear() {
      ref = null; res = null;
      orig.removeAttribute("src");
      viewport.hidden = true;
      $("stageEmpty").hidden = false;
      $("outRes").textContent = "";
      applyMode();
    },
    setResult(r) { res = r; draw(); applyMode(); apply(); },
    setKind(k) { kind = k; draw(); },
    currentBitmap() { return res ? res.bitmap(kind) : null; },
    setMode, get mode() { return mode; },
    hold(on) { held = on; applyMode(); },
    fit, actualSize,
    setBusy(text) { const b = $("busy"); b.hidden = !text; b.textContent = text || ""; },
    setInfo(text) { $("outRes").textContent = text; },
  };
}
