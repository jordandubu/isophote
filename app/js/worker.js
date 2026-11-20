// worker.js — hosts isophote_core.wasm off the main thread (classic worker).
// Inputs are cached by key so repeated paints of one image send nothing.
"use strict";

let wasm = null;
const inputs = new Map();

// the page sends the wasm URL in an {op:"init"} message (a worker's own
// location is not reliable when the host wraps it in a blob: URL)
let onUrl;
const ready = new Promise((resolve) => { onUrl = resolve; }).then(async (url) => {
  try {
    wasm = (await WebAssembly.instantiateStreaming(fetch(url), { env: {} })).instance;
  } catch (e) {   // servers without the application/wasm mime type
    wasm = (await WebAssembly.instantiate(await (await fetch(url)).arrayBuffer(), { env: {} })).instance;
  }
});

const x = () => wasm.exports;

// core slab is BGRA; the page wants RGBA
function toRgba(bytes) {
  const out = new Uint8Array(bytes.length);
  for (let i = 0; i < bytes.length; i += 4) {
    out[i] = bytes[i + 2]; out[i + 1] = bytes[i + 1]; out[i + 2] = bytes[i]; out[i + 3] = bytes[i + 3];
  }
  return out.buffer;
}

function dims(w, h) {
  const d = x().wasm_paint_out_dims(w, h);
  return { ow: Number(d & 0xffffffffn), oh: Number(d >> 32n) };
}

function paintSingle(inp, params) {
  const { w, h } = inp, bgra = inp.data[0];
  const pb = new TextEncoder().encode(JSON.stringify(params));
  const inPtr = x().wasm_alloc(bgra.length);
  const pPtr = x().wasm_alloc(pb.length);
  const outPtr = x().wasm_alloc(bgra.length);   // output is at most the input size
  try {
    new Uint8Array(x().memory.buffer, inPtr, bgra.length).set(bgra);
    new Uint8Array(x().memory.buffer, pPtr, pb.length).set(pb);
    x().wasm_paint(w, h, inPtr, outPtr, pPtr, pb.length);
    const { ow, oh } = dims(w, h);
    const out = new Uint8Array(x().memory.buffer, outPtr, ow * oh * 4).slice();
    return { outs: [toRgba(out)], w: ow, h: oh, heightScale: 0 };
  } finally {
    x().wasm_free(inPtr, bgra.length);
    x().wasm_free(pPtr, pb.length);
    x().wasm_free(outPtr, bgra.length);
  }
}

// Coherent multi-channel paint (stroke replay): data[0] is the reference
// (drives flow + strokes), the rest are painted with the SAME strokes. The
// core appends an impasto height frame + a display-only relief preview.
function paintMulti(inp, params) {
  const { w, h, data } = inp, k = data.length;
  const frame = w * h * 4;
  const blob = new Uint8Array(frame * k);
  data.forEach((b, i) => {
    if (b.length !== frame) throw new Error("channel " + i + " size mismatch");
    blob.set(b, i * frame);
  });
  const pb = new TextEncoder().encode(JSON.stringify(params));
  const outLen = frame * (k + 2);
  const inPtr = x().wasm_alloc(blob.length);
  const pPtr = x().wasm_alloc(pb.length);
  const outPtr = x().wasm_alloc(outLen);
  try {
    new Uint8Array(x().memory.buffer, inPtr, blob.length).set(blob);
    new Uint8Array(x().memory.buffer, pPtr, pb.length).set(pb);
    const ret = x().wasm_paint_multi(w, h, k, inPtr, outPtr, pPtr, pb.length);
    if (ret > 0) {   // core wrote an error message at out_ptr
      throw new Error(new TextDecoder().decode(new Uint8Array(x().memory.buffer, outPtr, ret)) || ("wasm_paint_multi failed: " + ret));
    }
    if (ret < 0) throw new Error("wasm_paint_multi failed: " + ret);
    const nOut = x().wasm_paint_multi_frames();
    const { ow, oh } = dims(w, h);
    const fo = ow * oh * 4;
    const out = new Uint8Array(x().memory.buffer, outPtr, nOut * fo).slice();
    const outs = [];
    for (let i = 0; i < nOut; i++) outs.push(toRgba(out.subarray(i * fo, (i + 1) * fo)));
    return { outs, w: ow, h: oh, heightScale: x().wasm_paint_multi_height_scale() };
  } finally {
    x().wasm_free(inPtr, blob.length);
    x().wasm_free(pPtr, pb.length);
    x().wasm_free(outPtr, outLen);
  }
}

self.onmessage = async (e) => {
  const m = e.data;
  if (m.op === "init") { onUrl(m.wasm); return; }
  if (m.op === "clear") { inputs.clear(); return; }
  try {
    await ready;
    if (m.input) inputs.set(m.key, m.input);
    const inp = inputs.get(m.key);
    if (!inp) throw new Error("worker input missing");
    const t0 = performance.now();
    const r = m.op === "multi" ? paintMulti(inp, m.params) : paintSingle(inp, m.params);
    r.ms = Math.round(performance.now() - t0);
    r.id = m.id; r.ok = true;
    self.postMessage(r, r.outs);
  } catch (err) {
    self.postMessage({ id: m.id, ok: false, error: String((err && err.message) || err) });
  }
};
