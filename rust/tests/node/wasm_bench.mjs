// A/B bench: node tests/node/wasm_bench.mjs <base.wasm> <new.wasm> <bench.bgra>
// bench.bgra = 768x1024 BGRA. Checks byte identity at impasto 0, closeness with impasto.
import fs from 'node:fs';

const [baseP, newP, imgP] = process.argv.slice(2);
if (!baseP || !newP || !imgP) { console.error('usage: wasm_bench.mjs base.wasm new.wasm bench.bgra'); process.exit(2); }
const W = 768, H = 1024;
const img = fs.readFileSync(imgP);
const baseBytes = fs.readFileSync(baseP), newBytes = fs.readFileSync(newP);

const s = 1024 / 720;
const art = { len: 40, width: 9, layers: 2, flow_smooth: 8, base_radius: 6, relief: 0.10 };
const bold = { len: 44, width: 26, layers: 2, flow_smooth: 8, base_radius: 8, relief: 0.10 };
const mk = (o, impasto, seed = 42, knife = false) => ({
  brush_len: Math.round(o.len * s), brush_w: Math.round(o.width * s), layers: o.layers, relief: o.relief,
  base_radius: Math.round(o.base_radius * s), flow_smooth: Math.round(o.flow_smooth * s), seed,
  max_size: 1024, out_size: 1024, look_scale: 1, impasto, impasto_brush: knife ? 1 : 0,
});

async function inst(bytes) { return (await WebAssembly.instantiate(bytes, { env: {} })).instance.exports; }

// returns { ms, frames: [Uint8Array] }
function run(e, kind, params) {
  const pb = Buffer.from(JSON.stringify(params));
  const k = kind === 'single' ? 1 : 2;
  const n = img.length;
  const ip = e.wasm_alloc(n * k), op = e.wasm_alloc(n * (k + 2)), pp = e.wasm_alloc(pb.length);
  for (let i = 0; i < k; i++) new Uint8Array(e.memory.buffer, ip + i * n, n).set(img);
  new Uint8Array(e.memory.buffer, pp, pb.length).set(pb);
  const t = performance.now();
  const r = kind === 'single'
    ? e.wasm_paint(W, H, ip, op, pp, pb.length)
    : e.wasm_paint_multi(W, H, k, ip, op, pp, pb.length);
  const ms = performance.now() - t;
  if (r !== 0) throw new Error('paint failed ' + r);
  const dims = e.wasm_paint_out_dims(W, H);
  const oh = Number(dims >> 32n), ow = Number(dims & 0xffffffffn);
  const nOut = kind === 'single' ? 1 : e.wasm_paint_multi_frames();
  const fo = ow * oh * 4;
  const out = new Uint8Array(e.memory.buffer, op, nOut * fo).slice();
  e.wasm_free(ip, n * k); e.wasm_free(op, n * (k + 2)); e.wasm_free(pp, pb.length);
  const frames = [];
  for (let i = 0; i < nOut; i++) frames.push(out.subarray(i * fo, (i + 1) * fo));
  return { ms, frames };
}

async function timed(bytes, kind, mkParams) {
  const e = await inst(bytes);
  run(e, kind, { ...mkParams(), seed: 43 });          // warm-up, different cache key
  return run(e, kind, mkParams());
}

const cases = [
  ['art imp0', 'single', () => mk(art, 0)],
  ['art imp1', 'single', () => mk(art, 1)],
  ['bold imp0', 'single', () => mk(bold, 0)],
  ['bold imp1', 'single', () => mk(bold, 1)],
  ['multi bold imp0', 'multi', () => ({ ...mk(bold, 0), kinds: ['color', 'normal'] })],
  ['multi bold imp1', 'multi', () => ({ ...mk(bold, 1), kinds: ['color', 'normal'] })],
];

let fail = false;
let coldBold1 = null;
for (const [name, kind, mkp] of cases) {
  const a = await timed(baseBytes, kind, mkp);
  const b = await timed(newBytes, kind, mkp);
  if (name === 'bold imp1') coldBold1 = b.frames[0];
  const imp = mkp().impasto > 0;
  let note = '';
  if (a.frames.length !== b.frames.length) { note = ` FRAME COUNT ${a.frames.length} vs ${b.frames.length}`; fail = true; }
  else if (!imp) {
    const same = a.frames.every((f, i) => Buffer.compare(f, b.frames[i]) === 0);
    note = same ? ' identical' : ' DIFFERS'; if (!same) fail = true;
  } else {
    const ds = a.frames.map((f, i) => {
      let sum = 0, mx = 0; const g = b.frames[i];
      for (let j = 0; j < f.length; j++) { const d = Math.abs(f[j] - g[j]); sum += d; if (d > mx) mx = d; }
      return { mean: sum / f.length, max: mx };
    });
    note = ' diff ' + ds.map(d => `${d.mean.toFixed(3)}/${d.max}`).join(' ');
    if (ds.some(d => d.mean > 0.5)) { note += ' TOO FAR'; fail = true; }
  }
  console.log(`${name.padEnd(18)} ${a.ms.toFixed(0).padStart(6)} ${b.ms.toFixed(0).padStart(6)} ${(b.ms / a.ms).toFixed(2)}${note}`);
}

// cache cases: same instance, bold single
{
  const e = await inst(newBytes);
  run(e, 'single', { ...mk(bold, 0.5), seed: 43 });
  const t0 = run(e, 'single', mk(bold, 0.5));
  const t1 = run(e, 'single', mk(bold, 1.0));
  const t2 = run(e, 'single', mk(bold, 1.0, 42, true));
  console.log(`cache: cold(0.5) ${t0.ms.toFixed(0)}ms  strength(1.0) ${t1.ms.toFixed(0)}ms  knife ${t2.ms.toFixed(0)}ms`);
  if (Buffer.compare(t1.frames[0], coldBold1) !== 0) { console.log('cache: cached imp1 output differs from cold'); fail = true; }
}
process.exit(fail ? 1 : 0);
