// main.js — wiring: source loading, paint scheduling (draft while dragging,
// full quality when idle), header actions, shortcuts, Blender hookup.
import { $ } from "./dom.js";
import { DEFAULTS, PAINT_KEYS, ANIM_KEYS, exportSettings, importSettings } from "./schema.js";
import { createStore } from "./store.js";
import { buildPanel } from "./controls.js";
import { PaintClient, CANCELLED, makeSource, paint, draftUseful, bitmapToPngBlob } from "./engine.js";
import { createViewer } from "./viewer.js";
import { createAnim } from "./anim.js";
import { createBlender } from "./blender.js";

const store = createStore(DEFAULTS);
const viewer = createViewer();
const client = new PaintClient(new URL("./worker.js", import.meta.url));

let source = null;        // loaded image (or Blender channel queue)
let result = null;        // newest painted Result (draft or final) of `source`
let activeKind = null;    // channel shown (multi sources)
let previewAuto = false;  // jumped to the relief preview once
let dirty = false;        // settings changed since the last final paint
let version = 0;          // bumped on every settings change
let finalTimer = 0, rafId = 0;

/* ---------- toast ---------- */
let toastTimer = 0;
function toast(text, ms = 5000) {
  const t = $("toast");
  t.textContent = text; t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.hidden = true; }, ms);
}
$("toast").onclick = () => { $("toast").hidden = true; };

/* ---------- persisted settings (Blender bridge on/off) ---------- */
const settings = { addonBlender: false, sendAnimated: false, sendHeight: true };
try { Object.assign(settings, JSON.parse(localStorage.getItem("isophote_settings") || "{}")); } catch { /* corrupt: defaults */ }
const saveSettings = () => { try { localStorage.setItem("isophote_settings", JSON.stringify(settings)); } catch { /* private mode */ } };
// the add-on opens the site with ?bridge=1 -> force the bridge on
if (new URLSearchParams(location.search).has("bridge")) { settings.addonBlender = true; saveSettings(); }

/* ---------- painting ---------- */
function onPaintError(e) {
  viewer.setBusy(null);
  toast("paint failed: " + (e.message || e));
}

function applyResult(res) {
  if (source.multi) {
    // first time impasto goes above 0 with no colour map loaded, jump to the
    // clay relief preview once (the user can switch back and it stays)
    if (!res.kinds.includes("preview")) previewAuto = false;
    else if (!previewAuto && !source.kinds.includes("color")) { activeKind = "preview"; previewAuto = true; }
    if (!res.kinds.includes(activeKind)) activeKind = source.kinds[0];
  }
  const old = result;
  result = res;
  viewer.setKind(activeKind);
  viewer.setResult(res);
  if (!source.shown) { source.shown = true; viewer.setMode("result"); }
  if (!res.draft) viewer.setBusy(null);
  $("status").textContent = res.draft ? "" : (res.ms / 1000).toFixed(2) + "s";
  viewer.setInfo(`${res.w}×${res.h}` + (res.g.workLong > res.g.outLong ? ` · painted at ${res.g.workLong}px` : "") + (res.draft ? " · draft" : ""));
  old?.close();
  syncUi();
}

async function runDraft() {
  const src = source;
  if (!src) return;
  try {
    const res = await paint(client, src, store.values, { draft: true });
    if (src !== source || !dirty) { res.close(); return; }    // superseded, or the final already landed
    applyResult(res);
  } catch (e) {
    if (e !== CANCELLED && src === source) onPaintError(e);
  }
}

async function runFinal() {
  const src = source;
  if (!src) return null;
  const ver = version;
  viewer.setBusy("painting…");
  try {
    const res = await paint(client, src, store.values);
    if (src !== source || ver !== version) { res.close(); return null; }
    dirty = false;
    applyResult(res);
    return res;
  } catch (e) {
    if (e !== CANCELLED && src === source) onPaintError(e);
    return null;
  }
}

// settings changed: fast low-res draft now (while dragging), full paint once idle
function requestPaint({ draft = true } = {}) {
  if (!source) return;
  anim.invalidate();
  version++;
  dirty = true;
  viewer.setBusy("painting…");
  cancelAnimationFrame(rafId);
  clearTimeout(finalTimer);
  const useDraft = draft && draftUseful(store.values, source.ref);
  if (useDraft) rafId = requestAnimationFrame(runDraft);
  finalTimer = setTimeout(runFinal, useDraft ? 250 : 80);
}

// resolves with a settled final Result (null if superseded)
async function ensureFinal() {
  if (!source) return null;
  if (result && !result.draft && !dirty) return result;
  clearTimeout(finalTimer);
  cancelAnimationFrame(rafId);
  return runFinal();
}

store.subscribe((changed) => {
  if (changed.some((k) => PAINT_KEYS.includes(k))) {
    requestPaint({ draft: !(changed.length === 1 && changed[0] === "imgscale") });
  } else if (changed.every((k) => k === "fps")) {
    anim.restartTimer();
  } else if (changed.some((k) => ANIM_KEYS.includes(k))) {
    anim.invalidate();
  }
});

/* ---------- sources ---------- */
function installSource(s) {
  anim.invalidate(false);
  clearTimeout(finalTimer);
  cancelAnimationFrame(rafId);
  const oldSource = source, oldResult = result;
  source = s; result = null;
  activeKind = s.kinds[0];
  previewAuto = false;
  dirty = true;
  version++;
  viewer.setSource(s);
  viewer.setKind(activeKind);
  viewer.setBusy(null);
  $("status").textContent = "";
  document.body.classList.add("has-image");
  oldResult?.close();
  oldSource?.close();
  syncUi();
}

async function loadFile(file) {
  try {
    installSource(await makeSource({ image: file }, ["image"], false));
    requestPaint();
  } catch (e) { toast(e.message || String(e)); }
}

async function loadQueue(kinds, blobs, normalSpace) {
  installSource(await makeSource(blobs, kinds, true, normalSpace));
  await ensureFinal();
}

function clearSource() {
  anim.invalidate(false);
  clearTimeout(finalTimer);
  cancelAnimationFrame(rafId);
  result?.close();
  source?.close();
  source = result = null;
  viewer.clear();
  viewer.setBusy(null);
  $("status").textContent = "";
  document.body.classList.remove("has-image");
  syncUi();
}

/* the channel dropdown lives in the Blender section (blender.refresh) */
function syncUi() {
  blender.refresh();
}

function setKind(k) {
  activeKind = k;
  viewer.setKind(k);
  syncUi();
}

/* ---------- panel (the Blender section mounts into it) ---------- */
buildPanel($("panel"), store, { onCopy: copySettings, onPaste: pasteSettings });

/* ---------- animation + Blender ---------- */
const anim = createAnim({
  strip: $("strip"), store,
  hooks: {
    paintFrame: (i) => paint(client, source, store.values, { frame: i }),
    key: () => `${source && source.id}|${store.values.frames}|${store.values.boil}`,
    show: (f) => viewer.setResult(f),
    restore: () => { if (result) viewer.setResult(result); },
    settle: ensureFinal,
    error: (e) => toast("animation failed: " + (e.message || e)),
  },
});

const blender = createBlender({
  source: () => source, result: () => result, activeKind: () => activeKind,
  setKind, loadQueue, ensureFinal, anim, toast, sendHeight: () => $("sendHeight").checked,
});
blender.setEnabled(settings.addonBlender);

/* ---------- header ---------- */
$("stageEmpty").onclick = () => $("file").click();
$("file").onchange = (e) => { if (e.target.files[0]) loadFile(e.target.files[0]); e.target.value = ""; };
$("clearBtn").onclick = clearSource;

async function exportPng() {
  if (!source) return;
  await ensureFinal();
  const bm = viewer.currentBitmap();
  if (!bm) return;
  const a = document.createElement("a");
  a.href = URL.createObjectURL(await bitmapToPngBlob(bm));
  a.download = "isophote.png";
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 10000);
}
$("exportBtn").onclick = exportPng;

const menu = $("menu"), sendMenu = $("sendMenu");
$("settingsBtn").onclick = (e) => { e.stopPropagation(); sendMenu.hidden = true; menu.hidden = !menu.hidden; };
$("sendCaret").onclick = (e) => { e.stopPropagation(); menu.hidden = true; sendMenu.hidden = !sendMenu.hidden; };
document.addEventListener("click", (e) => {
  if (!e.target.closest("#menu")) menu.hidden = true;
  if (!e.target.closest("#sendMenu")) sendMenu.hidden = true;
});
$("sendAnimated").checked = settings.sendAnimated;
$("sendAnimated").onchange = (e) => { settings.sendAnimated = e.target.checked; saveSettings(); };
$("sendHeight").checked = settings.sendHeight;
$("sendHeight").onchange = (e) => { settings.sendHeight = e.target.checked; saveSettings(); };
$("sendBtn").onclick = () => { sendMenu.hidden = true; blender.send($("sendAnimated").checked); };
$("addonBlender").checked = settings.addonBlender;
$("addonBlender").onchange = (e) => {
  settings.addonBlender = e.target.checked;
  saveSettings();
  blender.setEnabled(settings.addonBlender);
};

$("adjustBtn").onclick = (e) => { e.stopPropagation(); document.body.classList.toggle("sheet"); };
$("center").addEventListener("pointerdown", () => document.body.classList.remove("sheet"));

/* ---------- settings clipboard ---------- */
function legacyCopy(text) {
  const ta = document.createElement("textarea");
  ta.value = text; ta.style.cssText = "position:fixed;opacity:0";
  document.body.appendChild(ta); ta.select();
  let ok = false;
  try { ok = document.execCommand("copy"); } catch { /* unsupported */ }
  ta.remove();
  return ok;
}
async function copySettings() {
  const json = JSON.stringify(exportSettings(store.values));
  try { await navigator.clipboard.writeText(json); }
  catch { if (!legacyCopy(json)) { toast("copy failed"); return; } }
  toast("settings copied", 1500);
}
function applyPasted(text) {
  try {
    const obj = JSON.parse(text);
    if (obj && typeof obj === "object" && "len" in obj) { store.apply(importSettings(obj)); return true; }
  } catch { /* not settings JSON */ }
  return false;
}
async function pasteSettings() {
  try {
    if (applyPasted(await navigator.clipboard.readText())) toast("settings applied", 1500);
    else toast("clipboard has no isophote settings");
  } catch { toast("can't read the clipboard here — press Ctrl+V on the page instead"); }
}
syncUi();

window.addEventListener("paste", (e) => {
  if (e.target.closest("input[type=text],textarea")) return;
  const file = [...(e.clipboardData?.files || [])].find((f) => f.type.startsWith("image/"));
  if (file) { e.preventDefault(); loadFile(file); return; }
  const text = e.clipboardData?.getData("text");
  if (text && applyPasted(text)) { e.preventDefault(); toast("settings applied", 1500); }
});

/* ---------- drop ---------- */
const center = $("center");
["dragenter", "dragover"].forEach((ev) => center.addEventListener(ev, (e) => { e.preventDefault(); center.classList.add("drag"); }));
["dragleave", "dragend"].forEach((ev) => center.addEventListener(ev, () => center.classList.remove("drag")));
center.addEventListener("drop", (e) => {
  e.preventDefault(); center.classList.remove("drag");
  const f = e.dataTransfer.files[0];
  if (f) loadFile(f);
});

/* ---------- shortcuts ---------- */
window.addEventListener("keydown", (e) => {
  const typing = e.target.closest("input[type=text],textarea,select");
  const mod = e.ctrlKey || e.metaKey;
  if (e.key === "Escape") { menu.hidden = true; sendMenu.hidden = true; document.body.classList.remove("sheet"); return; }
  if (mod) {
    if (typing) return;
    const k = e.key.toLowerCase();
    if (k === "z") { e.preventDefault(); e.shiftKey ? store.redo() : store.undo(); }
    else if (k === "y") { e.preventDefault(); store.redo(); }
    else if (k === "s") { e.preventDefault(); exportPng(); }
    return;
  }
  if (typing || e.altKey) return;
  if (e.key === "\\") { if (!e.repeat) viewer.hold(true); e.preventDefault(); }
  else if (e.key === " " && !e.target.closest("button,summary,a")) { e.preventDefault(); if (source) anim.toggle(); }
  else if (e.key === "r" || e.key === "R") { document.querySelector(".dice")?.click(); }
  else if (e.key === "f" || e.key === "F") viewer.fit();
  else if (e.key === "1") viewer.setMode("orig");
  else if (e.key === "2") viewer.setMode("split");
  else if (e.key === "3") viewer.setMode("result");
});
window.addEventListener("keyup", (e) => { if (e.key === "\\") viewer.hold(false); });
window.addEventListener("blur", () => viewer.hold(false));

$("zoomFit").onclick = () => viewer.fit();
$("zoom100").onclick = () => viewer.actualSize();

/* ---------- default sample ---------- */
// skip it if the bridge already delivered Blender's image (poll can complete
// before this fetch resolves; the bridge claims its queue before its awaits)
fetch("test_photo.jpg").then((r) => (r.ok ? r.blob() : null)).then((b) => {
  if (b && !source && !blender.claimed()) return loadFile(b);
});
