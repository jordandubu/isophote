// wasm.js — glue: instantiate isophote_core.wasm, expose wasmPaint(rgba,
// w, h, params) -> {paintedURL, ms}. Keeps the app's UI code untouched.
"use strict";

let wasm = null;          // WebAssembly.Instance
let wasmReady = null;     // promise

async function loadWasm() {
  const resp = await fetch("isophote_core.wasm");
  const { instance } = await WebAssembly.instantiateStreaming(resp, { env: {} });
  wasm = instance;
}
wasmReady = loadWasm().catch(e => {
  document.getElementById("status").textContent = "wasm load failed: " + e.message;
  throw e;
});

function alloc(len) {
  return wasm.exports.wasm_alloc(len);
}

// bgra: Uint8Array (w*h*4) BGRA (page already swapped R/B); returns
// { url, ms, w, h } — BGRA→BMP via canvas putImageData on the fly.
async function wasmPaint(bgra, w, h, params) {
  await wasmReady;
  const t0 = performance.now();
  // input is already BGRA (page swaps on decode)
  const pj = JSON.stringify(params);
  const pb = new TextEncoder().encode(pj);

  const inPtr = alloc(bgra.length);
  const pPtr = alloc(pb.length);
  // output is at most the input size (no upscale in the core)
  const outPtr = alloc(bgra.length);
  try {
    new Uint8Array(wasm.exports.memory.buffer, inPtr, bgra.length).set(bgra);
    new Uint8Array(wasm.exports.memory.buffer, pPtr, pb.length).set(pb);
    wasm.exports.wasm_paint(w, h, inPtr, outPtr, pPtr, pb.length);
    const dims = wasm.exports.wasm_paint_out_dims(w, h);
    const outH = Number(dims >> 32n);
    const outW = Number(dims & 0xffffffffn);
    const outLen = outW * outH * 4;
    const out = new Uint8Array(wasm.exports.memory.buffer, outPtr, outLen).slice();
    const url = slabToUrl(out, outW, outH);
    return { url, ms: Math.round(performance.now() - t0), w: outW, h: outH };
  } finally {
    wasm.exports.wasm_free(inPtr, bgra.length);
    wasm.exports.wasm_free(pPtr, pb.length);
    wasm.exports.wasm_free(outPtr, bgra.length);  // allocation was input-size
  }
}

// one BGRA slab -> PNG data URL (BGRA->RGBA swap + putImageData)
function slabToUrl(bytes, w, h) {
  const rgbaOut = new ImageData(w, h);
  const od = rgbaOut.data;
  for (let i = 0; i < w * h; i++) {
    od[i * 4] = bytes[i * 4 + 2];     // R
    od[i * 4 + 1] = bytes[i * 4 + 1]; // G
    od[i * 4 + 2] = bytes[i * 4];     // B
    od[i * 4 + 3] = bytes[i * 4 + 3];
  }
  const c = document.createElement("canvas");
  c.width = w; c.height = h;
  c.getContext("2d").putImageData(rgbaOut, 0, 0);
  return c.toDataURL("image/png");
}

// Multi-channel coherent paint (stroke replay): bgras[0] is the reference
// (drives flow + strokes), the rest are painted with the SAME strokes.
// Returns { urls: [png data URLs (reference first, plus a gray impasto
// height map and a display-only relief preview when params.impasto > 0)],
// ms, heightScale (UV thickness of a white height pixel; 0 = no height),
// w, h (frame dims: params.out_size area-downscales the painted frames in the core) }.
// All input frames must be the same w*h already.
async function wasmPaintMulti(bgras, w, h, params) {
  await wasmReady;
  if (!bgras.length) throw new Error("no channels");
  const t0 = performance.now();
  const k = bgras.length;
  const frame = w * h * 4;
  const blob = new Uint8Array(frame * k);
  bgras.forEach((b, i) => { if (b.length !== frame) throw new Error("channel " + i + " size mismatch"); blob.set(b, i * frame); });
  const pb = new TextEncoder().encode(JSON.stringify(params));

  const inPtr = alloc(blob.length);
  const pPtr = alloc(pb.length);
  const outLen = frame * (k + 2);  // room for the trailing impasto height + preview frames
  const outPtr = alloc(outLen);
  try {
    new Uint8Array(wasm.exports.memory.buffer, inPtr, blob.length).set(blob);
    new Uint8Array(wasm.exports.memory.buffer, pPtr, pb.length).set(pb);
    const ret = wasm.exports.wasm_paint_multi(w, h, k, inPtr, outPtr, pPtr, pb.length);
    if (ret > 0) {
      // core wrote an error message at out_ptr (zero-padded UTF-8)
      const msg = new TextDecoder().decode(
        new Uint8Array(wasm.exports.memory.buffer, outPtr, ret));
      throw new Error(msg || ("wasm_paint_multi failed: " + ret));
    }
    if (ret < 0) throw new Error("wasm_paint_multi failed: " + ret);
    // snapshot out of wasm memory before decoding (buffer can detach)
    const nOut = wasm.exports.wasm_paint_multi_frames();
    const dims = wasm.exports.wasm_paint_out_dims(w, h);
    const oh = Number(dims >> 32n), ow = Number(dims & 0xffffffffn);
    const fo = ow * oh * 4;
    const out = new Uint8Array(wasm.exports.memory.buffer, outPtr, nOut * fo).slice();
    const urls = [];
    for (let i = 0; i < nOut; i++)
      urls.push(slabToUrl(out.subarray(i * fo, (i + 1) * fo), ow, oh));
    return { urls, ms: Math.round(performance.now() - t0), w: ow, h: oh,
             heightScale: wasm.exports.wasm_paint_multi_height_scale() };
  } finally {
    wasm.exports.wasm_free(inPtr, blob.length);
    wasm.exports.wasm_free(pPtr, pb.length);
    wasm.exports.wasm_free(outPtr, outLen);
  }
}
