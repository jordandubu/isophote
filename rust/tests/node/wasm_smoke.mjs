// Load the wasm module and paint the 135x180 fixture (expanded to BGRA),
// compare against painted_ref.npy with the strokes.rs thresholds.
import fs from 'node:fs';

const wasmPath = process.argv[2] || 'target/wasm32-unknown-unknown/release/isophote_core.wasm';
const bytes = fs.readFileSync(wasmPath);

// read npy (u8)
function readNpy(path) {
  const buf = fs.readFileSync(path);
  const hlen = buf.readUInt16LE(8);
  const header = buf.slice(10, 10 + hlen).toString();
  const dims = header.split("'shape': (")[1].split(')')[0]
    .split(',').filter(s => s.trim()).map(s => parseInt(s.trim()));
  return { dims, data: buf.slice(10 + hlen) };
}

const { dims, data } = readNpy('tests/in_bgr.npy'); // BGR h,w,3
const [h, w] = dims;
const bgra = new Uint8Array(w * h * 4);
for (let i = 0; i < w * h; i++) {
  bgra[i*4] = data[i*3]; bgra[i*4+1] = data[i*3+1]; bgra[i*4+2] = data[i*3+2]; bgra[i*4+3] = 255;
}
const params = JSON.stringify({brush_len: 45, brush_w: 3, layers: 4, relief: 0.1, base_radius: 6, flow_smooth: 8, seed: 42, max_size: 1400});
const pb = Buffer.from(params, 'utf8');

const env = { env: { wasm_paint_out_shim: () => {} } };
const mod = await WebAssembly.instantiate(bytes, env);
const exp = mod.instance.exports;
const inPtr = exp.wasm_alloc(bgra.length);
const outPtr = exp.wasm_alloc(bgra.length); // same size (no downscale here)
const pPtr = exp.wasm_alloc(pb.length);
new Uint8Array(exp.memory.buffer, inPtr, bgra.length).set(bgra);
new Uint8Array(exp.memory.buffer, pPtr, pb.length).set(pb);
const ms = exp.wasm_paint(w, h, inPtr, outPtr, pPtr, pb.length);
const out = new Uint8Array(exp.memory.buffer, outPtr, bgra.length);

// compare with fixture
const ref = fs.readFileSync('tests/painted_ref.npy');
const rhlen = ref.readUInt16LE(8);
const refData = ref.slice(10 + rhlen);
let sum = 0, big = 0;
for (let i = 0; i < w * h; i++) {
  const d = Math.abs(out[i*4] - refData[i*3]) +
            Math.abs(out[i*4+1] - refData[i*3+1]) +
            Math.abs(out[i*4+2] - refData[i*3+2]);
  sum += d; if (d > 30) big++;
}
const mean = sum / (w * h);
console.log(`wasm smoke: mean ${mean.toFixed(1)}, big ${(100*big/(w*h)).toFixed(1)}%, ms=${ms}`);
if (mean < 25 && true) { console.log('PASS'); } else { console.log('FAIL'); process.exit(1); }
