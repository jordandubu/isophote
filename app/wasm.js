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
    const out = new Uint8Array(wasm.exports.memory.buffer, outPtr, outLen);
    // BGRA -> RGBA for putImageData
    const rgbaOut = new ImageData(outW, outH);
    const od = rgbaOut.data;
    for (let i = 0; i < outW * outH; i++) {
      od[i * 4] = out[i * 4 + 2];     // R
      od[i * 4 + 1] = out[i * 4 + 1]; // G
      od[i * 4 + 2] = out[i * 4];     // B
      od[i * 4 + 3] = out[i * 4 + 3];
    }
    const c = document.createElement("canvas");
    c.width = outW; c.height = outH;
    c.getContext("2d").putImageData(rgbaOut, 0, 0);
    return { url: c.toDataURL("image/png"), ms: Math.round(performance.now() - t0), w: outW, h: outH };
  } finally {
    wasm.exports.wasm_free(inPtr, bgra.length);
    wasm.exports.wasm_free(pPtr, pb.length);
    wasm.exports.wasm_free(outPtr, bgra.length);
  }
}
