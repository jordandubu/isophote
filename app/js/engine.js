// engine.js — everything between the settings and the wasm worker:
// paint geometry, source images, input scaling, the cancellable worker
// client, and PNG helpers.
import { IMGSCALE_PX, sscaleFromPos } from "./schema.js";

export const LOOK_REF = 1024;      // long edge at which the default brushwork is defined
export const DRAFT_LONG = 512;     // long edge of the fast preview painted while dragging

/* ---------- geometry ---------- */

// Image Scale = output resolution only. out = what the slider asks for;
// work = what the core paints at (then area-downscales to out). Work never
// drops below LOOK_REF/strokeScale, so the brushwork keeps one look at any
// out. A draft paints straight at DRAFT_LONG (look scales with it).
export function paintGeom(v, ref, draft = false) {
  const L = Math.max(ref.w, ref.h);
  const t = IMGSCALE_PX[Math.min(v.imgscale, IMGSCALE_PX.length - 1)] ?? 1024;
  const finalOut = t > 0 ? t : L;
  const ss = sscaleFromPos(v.sscale);
  let outLong, workLong;
  if (draft) {
    outLong = workLong = Math.min(DRAFT_LONG, finalOut);
  } else {
    const need = Math.min(LOOK_REF, LOOK_REF / ss);
    const shrink = Math.max(1, Math.ceil(need / finalOut - 1e-9));   // integer: exact box filter
    outLong = finalOut;
    workLong = finalOut * shrink;
  }
  return {
    outLong, workLong, look: ss * workLong / LOOK_REF,
    workW: Math.max(1, Math.round(ref.w * workLong / L)),
    workH: Math.max(1, Math.round(ref.h * workLong / L)),
  };
}

// a draft is only worth it when the real paint is clearly bigger
export const draftUseful = (v, ref) => paintGeom(v, ref).workLong > DRAFT_LONG * 1.4;

// strokes are sized in pixels; 720 = reference size the default sliders were tuned at
export function coreParams(v, g) {
  const s = sscaleFromPos(v.sscale) * g.workLong / 720;
  return {
    brush_len: Math.max(4, Math.round(v.len * s)),
    brush_w: Math.max(1, Math.round(v.width * s)),
    layers: Math.round(v.layers),
    relief: v.relief,
    base_radius: Math.max(2, Math.round(v.base_radius * s)),
    flow_smooth: Math.max(2, Math.round(v.flow_smooth * s)),
    seed: Math.round(v.seed),
    max_size: g.workLong,
    out_size: g.outLong,
    look_scale: g.look,
    impasto: v.impasto || 0,
    impasto_knife: !!v.knife,
  };
}

// Boil: share of strokes repainted on frame i (the rest hold still);
// 100% = a fresh seed every frame.
export function frameParams(v, i) {
  const boil = v.boil / 100;
  return boil >= 1 ? { seed: Math.round(v.seed) + i } : { boil, boil_frame: i };
}

/* ---------- sources ---------- */

let nextSourceId = 1;

// blobs: {kind: Blob}; kinds: order (kinds[0] is the reference that drives
// flow + strokes). multi = a Blender channel queue; otherwise one photo.
// normalSpace: "object" | "tangent", how the Blender normal map is encoded.
export async function makeSource(blobs, kinds, multi, normalSpace = "object") {
  const bitmaps = {};
  try {
    for (const k of kinds) bitmaps[k] = await createImageBitmap(blobs[k]);
  } catch (e) {
    Object.values(bitmaps).forEach((b) => b.close());
    throw new Error("could not decode image");
  }
  // reference rect = the channel with the LONGEST edge (first wins ties)
  let best = kinds[0];
  for (const k of kinds) {
    if (Math.max(bitmaps[k].width, bitmaps[k].height) > Math.max(bitmaps[best].width, bitmaps[best].height)) best = k;
  }
  return {
    id: nextSourceId++, multi, kinds, bitmaps, normalSpace, shown: false,
    ref: { w: bitmaps[best].width, h: bitmaps[best].height },
    origURL: URL.createObjectURL(blobs[kinds[0]]),
    close() {
      Object.values(bitmaps).forEach((b) => b.close());
      URL.revokeObjectURL(this.origURL);
    },
  };
}

// bitmap -> BGRA bytes at exactly w×h (browser filtering when it scales)
export function bitmapToBgra(bitmap, w, h) {
  const c = document.createElement("canvas");
  c.width = w; c.height = h;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(bitmap, 0, 0, w, h);
  const d = ctx.getImageData(0, 0, w, h).data;
  const out = new Uint8Array(d.length);
  for (let i = 0; i < d.length; i += 4) {
    out[i] = d[i + 2]; out[i + 1] = d[i + 1]; out[i + 2] = d[i]; out[i + 3] = d[i + 3];
  }
  return out;
}

// What the worker needs for a paint. Single photo: the core only downsizes,
// so the full-size source goes in unless we must paint ABOVE source res (or
// draft). Multi: every channel is stretched to the reference working rect.
function inputsFor(source, g, draft) {
  if (!source.multi) {
    const b = source.bitmaps[source.kinds[0]];
    const scaled = draft || g.workLong > Math.max(b.width, b.height);
    const w = scaled ? g.workW : b.width, h = scaled ? g.workH : b.height;
    return { key: `${source.id}:s:${w}x${h}`,
             build: async () => ({ w, h, data: [bitmapToBgra(b, w, h)] }) };
  }
  const w = g.workW, h = g.workH;
  return { key: `${source.id}:m:${w}x${h}`,
           build: async () => ({ w, h, data: source.kinds.map((k) => bitmapToBgra(source.bitmaps[k], w, h)) }) };
}

/* ---------- results ---------- */

export class Result {
  constructor(o) { Object.assign(this, o); }
  index(kind) { const i = this.kinds.indexOf(kind); return i < 0 ? 0 : i; }
  bitmap(kind) { return this.bitmaps[this.index(kind)]; }
  close() { this.bitmaps.forEach((b) => b.close()); this.bitmaps = []; }
}

/* ---------- worker client ---------- */

export const CANCELLED = Object.freeze({ cancelled: true });

// One worker, one job at a time. A newer request replaces a queued one; a
// running "final" paint is cancelled outright (the worker is terminated and
// respawned, since a wasm call can't be interrupted); a running draft is
// short, so it is left to finish and the newest request waits behind it.
export class PaintClient {
  constructor(url) { this.url = url; this.nextId = 0; this.spawn(); }

  spawn() {
    const w = new Worker(this.url);
    w.postMessage({ op: "init", wasm: new URL("../isophote_core.wasm", import.meta.url).href });
    this.worker = w;
    this.sent = new Set();     // input keys the live worker already holds
    this.cur = null;
    this.pending = null;
    w.onmessage = (e) => this.onMessage(w, e.data);
    w.onerror = (e) => {
      if (this.worker !== w) return;
      const job = this.cur, pend = this.pending;
      this.cur = this.pending = null;
      w.terminate();
      this.spawn();
      job?.reject(new Error(e.message || "paint worker crashed"));
      if (pend) this.start(pend);
    };
  }

  kill() {
    const pend = this.pending;
    this.worker.onmessage = this.worker.onerror = null;
    this.worker.terminate();
    this.spawn();
    if (pend) this.start(pend);
  }

  // req: { op: "paint"|"multi", key, build(), params }
  run(req, { final = false } = {}) {
    return new Promise((resolve, reject) => {
      const job = { req, final, resolve, reject, dead: false, posted: false, id: 0 };
      const cur = this.cur;
      if (!cur) { this.start(job); return; }
      if (cur.final) {
        if (this.pending) { this.pending.reject(CANCELLED); this.pending = null; }
        cur.dead = true;
        if (cur.posted) this.kill(); else this.cur = null;
        cur.reject(CANCELLED);
        this.start(job);
        return;
      }
      if (this.pending) this.pending.reject(CANCELLED);
      this.pending = job;
    });
  }

  pump() {
    if (!this.cur && this.pending) { const j = this.pending; this.pending = null; this.start(j); }
  }

  async start(job) {
    this.cur = job;
    job.id = ++this.nextId;
    try {
      const key = job.req.key;
      let input = null;
      if (!this.sent.has(key)) {
        input = await job.req.build();
        if (job.dead || this.cur !== job) return;      // cancelled while building
        if (this.sent.size >= 6) { this.worker.postMessage({ op: "clear" }); this.sent.clear(); }
        this.sent.add(key);
      }
      job.posted = true;
      const msg = { id: job.id, op: job.req.op, key, params: job.req.params };
      const transfer = [];
      if (input) { msg.input = input; for (const d of input.data) transfer.push(d.buffer); }
      this.worker.postMessage(msg, transfer);
    } catch (e) {
      if (this.cur === job) { this.cur = null; this.pump(); }
      job.reject(e);
    }
  }

  onMessage(w, msg) {
    if (w !== this.worker) return;
    const job = this.cur;
    if (!job || job.id !== msg.id) return;
    this.cur = null;
    this.pump();
    if (!msg.ok) { job.reject(new Error(msg.error)); return; }
    Promise.all(msg.outs.map((b) =>
      createImageBitmap(new ImageData(new Uint8ClampedArray(b), msg.w, msg.h), { premultiplyAlpha: "none" })))
      .then((bitmaps) => job.resolve({ bitmaps, w: msg.w, h: msg.h, ms: msg.ms, heightScale: msg.heightScale }),
            (e) => job.reject(e));
  }
}

// Paint the source with the current settings. frame != null = boil frame i.
export async function paint(client, source, v, { draft = false, frame = null } = {}) {
  const g = paintGeom(v, source.ref, draft);
  const params = { ...coreParams(v, g), ...(frame == null ? {} : frameParams(v, frame)) };
  if (source.multi) { params.kinds = source.kinds; params.normal_tangent = source.normalSpace === "tangent"; }
  const inp = inputsFor(source, g, draft);
  const r = await client.run({ op: source.multi ? "multi" : "paint", key: inp.key, build: inp.build, params },
                             { final: !draft });
  // multi results carry a trailing impasto height + display-only preview when impasto > 0
  const kinds = !source.multi ? ["image"]
    : r.bitmaps.length > source.kinds.length ? [...source.kinds, "height", "preview"] : source.kinds;
  return new Result({ kinds, bitmaps: r.bitmaps, w: r.w, h: r.h, ms: r.ms, heightScale: r.heightScale || 0, g, draft });
}

/* ---------- PNG helpers ---------- */

export function bitmapToPngBlob(bitmap) {
  const c = document.createElement("canvas");
  c.width = bitmap.width; c.height = bitmap.height;
  c.getContext("2d").drawImage(bitmap, 0, 0);
  return new Promise((res, rej) => c.toBlob((b) => (b ? res(b) : rej(new Error("png encode failed"))), "image/png"));
}

export function blobToB64(blob) {
  return new Promise((res) => {
    const r = new FileReader();
    r.onload = () => res(r.result.split(",")[1]);
    r.readAsDataURL(blob);
  });
}

export const bitmapToB64 = async (bitmap) => blobToB64(await bitmapToPngBlob(bitmap));

export function b64ToBlob(b64, type) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Blob([bytes], { type });
}
